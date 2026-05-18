import argparse
import json
import os
import random
from datetime import datetime, timezone

import datasets
import glog
import torch
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
#from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import sys
_FASTKRON_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FASTKRON_ROOT not in sys.path:
    sys.path.insert(0, _FASTKRON_ROOT)
sys.path.insert(0, "../lm-evaluation-harness")

from lib.linear import QuantizedLinear
from lib.utils.unsafe_import import model_from_hf_path


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--tokenizer', default=None, type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str,
                    help='Optional path for torch.save of full lm_eval results (.pt)')
parser.add_argument('--num_fewshot', type=int, default=0,
                    help="N-shot count override; pass -1 to use each task's YAML "
                         "default (e.g. gsm8k=5, ifeval=0)")
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--apply_chat_template', action='store_true')
parser.add_argument('--fewshot_as_multiturn', action='store_true')
parser.add_argument('--manifest_model', action='store_true')
parser.add_argument('--max_mem_ratio', type=float, default=0.7)
parser.add_argument('--output_json', type=str, default=None,
                    help='Atomic JSON dump of results plus provenance metadata')


def main(args):
    if args.manifest_model:
        model, model_str = model_from_hf_path(args.hf_path,
                                          max_mem_ratio=args.max_mem_ratio,
                                          device_map='balanced')

    else:
        model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
        )

    #model_str = "unsloth/llama-2-7b"

    # manifest for faster inference
    # use for codebooks without kernel support
    if args.manifest_model:
        for module in model.modules():
            if isinstance(module, QuantizedLinear):
                module.mode = 'train-fixW'

    tokenizer = AutoTokenizer.from_pretrained(model_str if args.tokenizer is
                                              None else args.tokenizer)

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    task_names = args.tasks.split(",")

    lm_eval_model = HFLM(model,
                         tokenizer=tokenizer,
                         batch_size=args.batch_size)

    fewshot_arg = None if args.num_fewshot < 0 else args.num_fewshot

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        limit=args.limit,
        num_fewshot=fewshot_arg,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs="max_gen_toks=640",
    )
    for key in results['results']:
        print(key)
        print()
        print(results['results'][key])
        print()
        print()

    if args.output_path is not None:
        torch.save(results, args.output_path)

    if args.output_json is not None:
        summary = {
            "_meta": {
                "hf_path": args.hf_path,
                "tokenizer": args.tokenizer,
                "tasks": task_names,
                "num_fewshot": args.num_fewshot,
                "num_fewshot_resolved": fewshot_arg,
                "apply_chat_template": args.apply_chat_template,
                "fewshot_as_multiturn": args.fewshot_as_multiturn,
                "manifest_model": args.manifest_model,
                "batch_size": args.batch_size,
                "max_mem_ratio": args.max_mem_ratio,
                "seed": args.seed,
                "limit": args.limit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "results": results.get("results", {}),
            "configs": results.get("configs", {}),
            "versions": results.get("versions", {}),
            "n-samples": results.get("n-samples", {}),
        }
        text = json.dumps(summary, indent=2, default=str)
        tmp = args.output_json + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, args.output_json)
        glog.info(f'wrote results JSON to {args.output_json}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
