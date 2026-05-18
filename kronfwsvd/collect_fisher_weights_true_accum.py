"""
Collect TRUE-accumulated per-step gradient samples for Fisher / Kronecker
factor estimation.

Variant of collect_fisher_weights.py with true gradient accumulation:
optimizer.step / lr_scheduler.step / optimizer.zero_grad are gated by the
accumulation cycle (called only at accum-boundary, not on every microbatch).

At the accumulation boundary, module.weight.grad contains the SUM of all
microbatch grads in the cycle (because zero_grad has not run yet). Dividing
by gradient_accumulation_steps yields the true mean grad of that window —
not a sample of the last microbatch.

This is the only semantic difference from collect_fisher_weights.py. All
CLI, output naming, file format, and class naming are preserved 1:1 so the
downstream get_kron_factors_llama.py is unaware of the variant.

Output format (consumed by get_kron_factors_llama.py):
    grad_step_{N:04d}.safetensors
        keys: name.replace('.', '_') for each unfrozen nn.Linear,
              e.g. "model_layers_5_mlp_gate_proj"
        values: bfloat16 CPU tensors of shape (out_features, in_features)
"""

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import transformers
from datasets import load_from_disk
from safetensors.torch import save_file as safe_save
from torch import nn
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, set_seed)

set_seed(42)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,max_split_size_mb:512"
)


def stamp() -> str:
    return time.strftime("%H:%M:%S")


LAYER_INDEX_RE = re.compile(r"\.layers\.(\d+)\.")


def extract_layer_idx(param_name: str) -> Optional[int]:
    m = LAYER_INDEX_RE.search(param_name)
    return int(m.group(1)) if m else None


class CustomTrainer(Trainer):
    def make_grad_bank(self):
        self.avg_counter = 0
        self.output_grad_dir: Optional[str] = None
        self._step_idx = 0
        self._accum_step_counter = 0
        self._max_gpu_mem_gb = 0.0

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_samples: int = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            loss = loss.mean()

        self.accelerator.backward(loss)

        self._accum_step_counter += 1

        if self._accum_step_counter == self.args.gradient_accumulation_steps:
            grad_dict: Dict[str, torch.Tensor] = {}
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and module.weight.requires_grad:
                    grad = module.weight.grad
                    if grad is not None:
                        # At accum boundary, .grad contains the SUM over all
                        # microbatches of this cycle (zero_grad has not run).
                        # Divide by accum_steps for the true mean grad.
                        grad_bf16 = (
                            grad.detach()
                            / self.args.gradient_accumulation_steps
                        ).to(torch.bfloat16).cpu().contiguous()
                        grad_dict[name.replace('.', '_')] = grad_bf16.clone()

            if self.output_grad_dir is not None:
                os.makedirs(self.output_grad_dir, exist_ok=True)
                step_path = os.path.join(
                    self.output_grad_dir,
                    f"grad_step_{self._step_idx:04d}.safetensors",
                )
                safe_save(grad_dict, step_path)

                gpu_mem_peak_gb = max(
                    (
                        torch.cuda.max_memory_allocated(d) / (1024 ** 3)
                        for d in range(torch.cuda.device_count())
                    ),
                    default=0.0,
                )
                self._max_gpu_mem_gb = max(self._max_gpu_mem_gb, gpu_mem_peak_gb)
                print(
                    f"[{stamp()}] [collect] step {self._step_idx} saved | "
                    f"path={step_path} | grads_saved={len(grad_dict)} | "
                    f"peak GPU={gpu_mem_peak_gb:.1f} GB",
                    flush=True,
                )
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)

            self._step_idx += 1
            self.avg_counter += 1
            self._accum_step_counter = 0

            # TRUE accumulation: optimizer / scheduler / zero_grad ONLY at
            # accumulation boundary, after grads have been read out and saved.
            # This is the only behavioral change vs collect_fisher_weights.py.
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        torch.cuda.empty_cache()
        gc.collect()

        return loss.detach()


def tokenize(example, tokenizer, max_length: int):
    prompt = example.get("text") or example.get("content")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    tokens["labels"] = tokens["input_ids"].clone()
    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = -100
    return {k: v.squeeze(0) for k, v in tokens.items()}


def sanity_check_unfrozen(
    model: nn.Module, layer_start: int, layer_end: int,
) -> None:
    """Print unfreeze summary; abort if zero trainable params or out-of-range."""
    unfrozen = [n for n, p in model.named_parameters() if p.requires_grad]
    if not unfrozen:
        print(
            f"[{stamp()}] [collect] FATAL: 0 trainable params in "
            f"[{layer_start},{layer_end}). Aborting.",
            flush=True,
        )
        sys.exit(1)

    by_layer: Dict[int, int] = {}
    out_of_range = []
    for n in unfrozen:
        idx = extract_layer_idx(n)
        if idx is None:
            out_of_range.append(n)
            continue
        if idx < layer_start or idx >= layer_end:
            out_of_range.append(n)
            continue
        by_layer[idx] = by_layer.get(idx, 0) + 1

    if out_of_range:
        print(
            f"[{stamp()}] [collect] FATAL: {len(out_of_range)} unfrozen params "
            f"are outside [{layer_start},{layer_end}) or have no layer index. "
            f"First: {out_of_range[0]}. Aborting.",
            flush=True,
        )
        sys.exit(1)

    decoders = sorted(by_layer.keys())
    print(
        f"[{stamp()}] [collect] sanity: {len(unfrozen)} trainable params | "
        f"{len(decoders)} decoders | idx {decoders[0]}..{decoders[-1]}",
        flush=True,
    )
    for n in unfrozen[:3]:
        print(f"[{stamp()}] [collect]   first: {n}", flush=True)
    if len(unfrozen) > 6:
        print(
            f"[{stamp()}] [collect]   ... ({len(unfrozen) - 6} more) ...",
            flush=True,
        )
    for n in unfrozen[-3:]:
        print(f"[{stamp()}] [collect]   last:  {n}", flush=True)
    counts = sorted(set(by_layer.values()))
    print(
        f"[{stamp()}] [collect] params/decoder counts (distinct): {counts}",
        flush=True,
    )


def training_process(args) -> None:
    print(
        f"[{stamp()}] [collect] starting (TRUE ACCUM) | model={args.model_name} "
        f"layers=[{args.layer_start},{args.layer_end}) "
        f"max_steps={args.max_steps} batch={args.per_device_batch_size} "
        f"grad_accum={args.grad_accum} max_length={args.max_length}",
        flush=True,
    )
    print(
        f"[{stamp()}] [collect] transformers={transformers.__version__} "
        f"torch={torch.__version__} cuda_devices={torch.cuda.device_count()}",
        flush=True,
    )

    dataset = load_from_disk(args.dataset_path)
    if args.dataset_size > 0 and args.dataset_size < len(dataset):
        dataset = dataset.select(range(args.dataset_size))
    print(
        f"[{stamp()}] [collect] loaded dataset {args.dataset_path} "
        f"size={len(dataset)}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        #device_map="auto",
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )
    model = model.cuda()

    print(f"[{stamp()}] [collect] model loaded", flush=True)
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        per_dev: Dict[Any, int] = {}
        for _, dev in model.hf_device_map.items():
            per_dev[dev] = per_dev.get(dev, 0) + 1
        print(
            f"[{stamp()}] [collect] hf_device_map (modules per device): {per_dev}",
            flush=True,
        )
    else:
        print(
            f"[{stamp()}] [collect] WARNING: no hf_device_map — Trainer may "
            f"try to relocate model",
            flush=True,
        )

    model.enable_input_require_grads()
    #model.gradient_checkpointing_enable(
    #    gradient_checkpointing_kwargs={"use_reentrant": False}
    #)
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = False

    layer_pattern = args.layer_pattern
    if layer_pattern is None:
        layer_pattern = r"layers\.[0-9]+\.(mlp|self_attn)\.[a-z_]*_proj"
    layer_re = re.compile(layer_pattern)

    for name, param in model.named_parameters():
        if not layer_re.search(name):
            continue
        idx = extract_layer_idx(name)
        if idx is None:
            continue
        if idx < args.layer_start or idx >= args.layer_end:
            continue
        param.requires_grad = True

    sanity_check_unfrozen(model, args.layer_start, args.layer_end)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{stamp()}] [collect] total params: {total:,} | "
        f"trainable: {trainable:,} ({100 * trainable / total:.4f}%)",
        flush=True,
    )

    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, max_length=args.max_length),
        remove_columns=dataset.column_names,
    )
    dataset.set_format(type="torch")

    grad_dir = Path(args.grad_dir)
    grad_dir.mkdir(parents=True, exist_ok=True)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else (grad_dir / "_trainer")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args_kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        learning_rate=args.lr,
        num_train_epochs=1,
        save_steps=None,
        eval_steps=None,
        save_strategy="no",
        remove_unused_columns=True,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        report_to="none",
        seed=42,
        overwrite_output_dir=True,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )
    if args.max_steps is not None and args.max_steps > 0:
        training_args_kwargs["max_steps"] = args.max_steps

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.make_grad_bank()
    trainer.output_grad_dir = str(grad_dir)

    print(
        f"[{stamp()}] [collect] starting trainer.train() | "
        f"grad_dir={grad_dir} output_dir={output_dir}",
        flush=True,
    )

    run_t0 = time.time()
    trainer.train()
    run_dt = time.time() - run_t0

    print(
        f"[{stamp()}] [collect] done | total time={run_dt:.1f}s | "
        f"saved={trainer._step_idx} step files | "
        f"peak GPU={trainer._max_gpu_mem_gb:.1f} GB",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-step gradient samples (TRUE accumulation) for Kronecker factor estimation."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="HF model identifier (e.g. unsloth/llama-2-7b)")
    parser.add_argument("--grad_dir", type=str, required=True,
                        help="Output dir for grad_step_*.safetensors")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Trainer output_dir (default: <grad_dir>/_trainer)")
    parser.add_argument("--layer_start", type=int, required=True,
                        help="Inclusive lower bound on decoder index")
    parser.add_argument("--layer_end", type=int, required=True,
                        help="Exclusive upper bound on decoder index")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=384,
                        help="gradient_accumulation_steps")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Trainer max_steps; -1 = full dataset")
    parser.add_argument("--max_length", type=int, default=3096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer_pattern", type=str, default=None,
                        help="Regex on param name to filter projection types "
                             "(default: mlp/self_attn projections only)")
    parser.add_argument("--dataset_path", type=str,
                        default="/workspace-SR004.nfs2/data/fineweb/sample/66K")
    parser.add_argument("--dataset_size", type=int, default=38400,
                        help="Cap on dataset samples; <=0 → full dataset")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        help="HF attn_implementation: flash_attention_2 / sdpa / eager")
    args = parser.parse_args()

    if args.layer_end <= args.layer_start:
        sys.exit(
            f"--layer_end ({args.layer_end}) must be > "
            f"--layer_start ({args.layer_start})"
        )

    training_process(args)


if __name__ == "__main__":
    main()
