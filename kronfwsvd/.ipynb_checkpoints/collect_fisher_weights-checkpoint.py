import argparse
from collections import defaultdict
from functools import reduce
from typing import Any, Dict, Union

import torch
from datasets import load_from_disk
from safetensors.torch import save_file as safe_save
from torch import nn
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

set_seed(42)

import gc

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

from accelerate.accelerator import AcceleratorState

def print_once(*args, **kwargs):
    if AcceleratorState.is_main_process:
        print(*args, **kwargs)

import re

def get_module_by_name(module, access_string):
    return reduce(getattr, access_string.split('.'), module)

class CustomTrainer(Trainer):
    def make_grad_bank(self):
        #self.mass = {}         # [layer_name] = list of grads per step
        #self.mass_avg = {}     # unused but preserved
        #self.mass_w_avg = {}   # unused but preserved
        self.avg_counter = 0   # count of accum steps
        self.output_grad_dir = None  # set via save_tensors()
        self._step_idx = 0     # for naming saved files
        self._accum_step_counter = 0  # 🔧 ← ИНИЦИАЛИЗАЦИЯ счетчика аккумуляции

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_samples: int = None) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            loss = loss.mean()

        self.accelerator.backward(loss)

        # Очистка лишнего сразу после backward


        # Увеличиваем локальный счётчик аккумуляции
        self._accum_step_counter += 1

        # Если мы дошли до конца аккумуляции
        if self._accum_step_counter == self.args.gradient_accumulation_steps:
            grad_dict = {}

            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.requires_grad:
                    grad = module.weight.grad
                    if grad is not None:
                        grad_cpu = grad.detach().cpu() / self.args.gradient_accumulation_steps

                        grad_dict[name.replace('.', '_')] = grad_cpu.clone()
                        #print(f"[DEBUG] grad {name} step {self._step_idx}:", grad_cpu[:2, :2], flush=True)

            # Save to disk
            if self.output_grad_dir is not None:
                os.makedirs(self.output_grad_dir, exist_ok=True)
                step_path = os.path.join(self.output_grad_dir, f"grad_step_{self._step_idx:04d}.safetensors")
                try:
                    safe_save(grad_dict, step_path)
                    print(f"[Grad] Saved gradients to {step_path}", flush=True)
                except Exception as e:
                    print(f"[Error] Failed to save gradients at step {self._step_idx}: {e}", flush=True)

            self._step_idx += 1
            self.avg_counter += 1
            self._accum_step_counter = 0  # 🔁 сбрасываем после сохранения

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        gc.collect()

        return loss.detach()


def tokenize(example, tokenizer, max_length=3096):
    prompt = example.get("text") or example.get("content")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = -100
    return {k: v.squeeze(0) for k, v in tokens.items()}


def tokenize1(example, tokenizer, max_length=4096):
    prompt = example.get("text") or example.get("content")
    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].clone()
    # Mask padding tokens in labels
    if tokenizer.pad_token_id is not None:
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = -100
    return {k: v.squeeze(0) for k, v in tokens.items()}

def training_process(path_to_output, batch_size, learning_rate, model_name, layer_pattern=None, max_length=3096):
    dataset = load_from_disk("../fineweb/sample/66K")
    dataset = dataset.select(range(25600))
    # model_name = "unsloth/llama-2-7b"
    #model_name = "unsloth/Meta-Llama-3.1-8B"
    # model_name = "Qwen/Qwen3-8B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if "gemma" not in model_name else "eager",
    )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if hasattr(model.config, "gradient_checkpointing"):
        model.config.gradient_checkpointing = True
        model.config.use_cache = False
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    for param in model.parameters():
        param.requires_grad = False

    if layer_pattern is None:
        #layer_pattern = r"model\.layers\.(?:[0-9]|10)\."
        layer_pattern = r"layers.[0-9]+.(mlp|self_attn).[a-z_]*_proj"

    unfrozen_count = 0
    for name, param in model.named_parameters():
        if re.search(layer_pattern, name):
            param.requires_grad = True
            unfrozen_count += 1
            print_once(f"[INFO] Unfrozen: {name}")

    if unfrozen_count == 0:
        print_once(f"[WARNING] No parameters matched the pattern '{layer_pattern}'. Check your regex.")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_once(f"[INFO] Total params: {total:,} | Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

    dataset = dataset.map(lambda x: tokenize(x, tokenizer, max_length=max_length), remove_columns=dataset.column_names)
    dataset.set_format(type="torch")

    print ("dataset 10", dataset[10])

    training_args = TrainingArguments(
        output_dir=path_to_output,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=1,
        save_steps=None,
        eval_steps=None,
        save_strategy='no',
        remove_unused_columns=True,
        gradient_accumulation_steps = 128,
        bf16=True,
        report_to='none',
        seed=42,
        overwrite_output_dir=True,
        max_grad_norm=1.0,
        gradient_checkpointing=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.make_grad_bank()
    trainer.output_grad_dir = path_to_output  # ✅ Установить путь для сохранения
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with gradient collection.')
    parser.add_argument('--model_name', type=str, required=True, help='HF link to model to tune', default='unsloth/llama-2-7b')
    parser.add_argument('--path_to', type=str, required=True, help='Output directory')
    parser.add_argument('--size_of', type=int, required=True, help='Train batch size per device')
    parser.add_argument('--max_length', type=int, required=True, help='Max training seq length')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--layer_pattern', type=str, required=False, help='Regex to match layers')
    args = parser.parse_args()

    print(f"Training of model {args.model_name} started:\nOutput: {args.path_to}\nBatch size/device: {args.size_of}\nLearning rate: {args.lr}")
    training_process(args.path_to, args.size_of, args.lr, args.model_name, args.layer_pattern, args.max_length)