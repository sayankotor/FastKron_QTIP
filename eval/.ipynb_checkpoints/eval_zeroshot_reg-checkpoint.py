import argparse
import os
import random

import torch
import glog
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', type=str, required=True,
                    help='Path to HuggingFace model (e.g., meta-llama/Llama-2-7b-hf)')
parser.add_argument('--tokenizer', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--tasks', type=str, required=True,
                    help='Comma-separated list of evaluation tasks (e.g., arc_easy,boolq)')
parser.add_argument('--output_path', type=str, default='.')
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--apply_chat_template', action='store_true')
parser.add_argument('--fewshot_as_multiturn', action='store_true')
parser.add_argument('--max_mem_ratio', type=float, default=0.7)

def main(args):
    glog.info(f'Loading model from {args.hf_path}')
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer_path = args.hf_path if args.tokenizer is None else args.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    glog.info('Model and tokenizer loaded.')

    task_names = args.tasks.split(",")

    lm_eval_model = HFLM(model,
                         tokenizer=tokenizer,
                         batch_size=args.batch_size)

   
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn
    )

    for key in results['results']:
        print(key)
        print()
        print(results['results'][key])
        print()
        print()

    if args.output_path:
        torch.save(results, args.output_path)
        glog.info(f"Saved results to {args.output_path}")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
