# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

FastKron is a research implementation of Fisher-guided post-training quantization (PTQ) for LLaMA and Qwen3 LLMs. It plugs into the **QTIP** quantization framework and the **YAQA** Hessian pipeline, replacing YAQA's power-iteration "Sketch A" Kronecker-factor estimator with a Lanczos-based estimator (FastKron, in `kronfwsvd/`). Output is QTIP-quantized weights at 2- or 4-bit using a bitshift trellis codebook.

## End-to-end pipeline

The pipeline has four stages, run as separate scripts:

1. **Collect Hessian / Fisher factors** — either:
   - `hessian_llama/get_hess_llama.py` (Sketch A or B, distributed via `torchrun` + FSDP, used for the YAQA baseline), or
   - `kronfwsvd/collect_fisher_weights.py` then `kronfwsvd/get_kron_factors_llama.py` (FastKron — collects per-layer gradient minibatches via a `CustomTrainer`, then runs Lanczos SVD on the gradient stack with CuPy).
2. **Quantize + (optional) finetune** — `quantize_llama/quantize_finetune_llama.py` walks decoder layers across all visible GPUs, quantizes each `q/k/v/o/up/gate/down` projection with `lib/codebook/bitshift.py` + `lib/algo/finetune.py`, and writes per-layer `.pt` files to `--save_path`.
3. **HF-ize** — `quantize_llama/hfize_llama.py`, `hfize_qwen.py`, or `hfize_qwen1.py` reload the per-layer `.pt` files into a custom `model.llama.LlamaForCausalLM` / `model.qwen.Qwen3ForCausalLM` (which use `lib.linear.QuantizedLinear` in place of `nn.Linear`) and save a HuggingFace-compatible checkpoint.
4. **Evaluate** — `eval/eval_ppl.py` (wikitext2 + c4 perplexity), `eval/eval_zeroshot.py` (lm-evaluation-harness: arc/boolq/piqa/hellaswag), `eval/eval_kl.py`, `eval/eval_ppl_qwen.py`.

The wrapper shell scripts run stages 2–4 together:
- `./run_quantizer.sh <BASE_MODEL> <HESS_PATH> <FOLDER_NAME> [TOKENIZER]` — LLaMA, K=4 (4-bit).
- `./run_quantizer_qwen.sh <BASE_MODEL> <HESS_PATH> <FOLDER_NAME> [TOKENIZER]` — Qwen3, K=2 (2-bit), uses `hfize_qwen1.py`.
- `./run_quantizer-eval-only.sh` — skips quantization, only re-runs hfize + eval.

Both wrappers hardcode `SAVE_DIR="../yaqa-quantization/${FOLDER_NAME}"` — see the next section.

## The `../yaqa-quantization` sibling directory

Almost every entry-point script in this repo does `sys.path.append("../yaqa-quantization")` and writes outputs there. Two consequences:

- Scripts must be run with the working directory set such that `../yaqa-quantization` resolves to the YAQA checkout. The shell wrappers assume the cwd is the repo root.
- `lib/utils/unsafe_import.py` (used by all eval scripts) hardcodes a path for Qwen3 quantized configs: `../yaqa-quantization/qwen3_sketchA_2048_qw_2_vika/config.pt`. If that file is missing, Qwen3 evaluation will fail to load. Don't generalize this without understanding why — it's a workaround for `AutoConfig` not preserving `_name_or_path` correctly on Qwen3 quantized checkpoints.

## QTIP CUDA kernels — the compilation gotcha

`qtip-kernels/` is a CUDA extension built per matrix shape. `lib/codebook/__init__.py` registers a Torch op for every `(M, N, K, bitrate)` tuple in its `kernels` list, and each tuple must have a corresponding kernel in `qtip-kernels/src/wrapper.cpp` and `src/qtip_torch.cu`. If a model has a projection shape that isn't in the list, you'll see:

```
AttributeError: '_OpNamespace' 'quip_lib' object has no attribute 'decompress_matvec_qtip_<M>_1_<K>_<bitrate>'
```

To add a new shape:
1. Add the `(M, 1, K, bitrate)` tuple to `kernels` in `lib/codebook/__init__.py`.
2. Add matching kernel definitions in `qtip-kernels/src/wrapper.cpp` and `src/qtip_torch.cu` (follow the existing pattern).
3. Reinstall: `cd qtip-kernels && python setup.py install`.

`lib.linear.QuantizedLinear` has a `mode='train-fixW'` ("manifest") path that materializes the dequantized weight in BF/FP16 and bypasses the kernel — eval scripts pass `--manifest` / `--manifest_model` to enable it for shapes without compiled kernels.

## Common commands

Install:
```bash
pip install -r requirements.txt              # if present in the YAQA repo
cd fast-hadamard-transform && pip install -v . && cd ..
cd qtip-kernels && python setup.py install && cd ..
```

Hessians (YAQA Sketch A baseline, 4× GPUs):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  hessian_llama/get_hess_llama.py \
  --save_path <OUT> --orig_model unsloth/llama-2-7b \
  --batch_size 6 --hessian_sketch A --power_iters 4 --ctx_size 4096 --n_seqs 4096
```

Hessians (FastKron):
```bash
python kronfwsvd/collect_fisher_weights.py --model_name <MODEL> --path_to <OUT> --size 1 --lr 1e-4
python kronfwsvd/get_kron_factors_llama.py --model_name <MODEL>
```

Single-test perplexity sanity check (after a full quantize+hfize):
```bash
python eval/eval_ppl.py --hf_path <HF_DIR> --manifest
```

There is no test suite, no linter config, and no CI.

## Code layout (non-obvious bits)

- `model/llama.py`, `model/qwen.py` — forks of the HF transformers model classes, modified to swap `nn.Linear` projections for `lib.linear.QuantizedLinear` and to use `model/cache_utils.py`. Used only for *inference of quantized checkpoints*, not for training/calibration.
- `lib/algo/finetune.py` — the per-decoder-layer LDLQ + optional finetune routine called from `quantize_finetune_llama.py`.
- `lib/algo/ldlq.py` — the LDLQ projection used inside the codebook search.
- `lib/codebook/bitshift.py` — the trellis-bitshift codebook (`L`, `K`, `V`, `tlut_bits`, `decode_mode` are its key knobs; the shell wrappers fix `L=16, V=2, td_x=td_y=16, decode_mode=quantlut_sym, tlut_bits=9` and vary only `K` for bitwidth).
- `hessian_llama/custom_linear_A.py` / `custom_linear_B.py` — the two Sketch variants, hooked into `llama_hess.py`'s forked LLaMA forward.
- `kronfwsvd/get_kron_factors_llama.py` — uses **CuPy + `cupyx.scipy.sparse.linalg.svds`** (Lanczos), not PyTorch, for the Kronecker factor SVD. Requires CuPy installed.

## Branching / git

The `main` branch carries the public README and the model/qtip-kernels modules. Active work happens on topic branches (current: `quantize_big_model`). Recent history shows the repo is being reorganized — the `model/` and `qtip-kernels/` trees were pulled in from yaqa-quantization in `00c99fe`.
