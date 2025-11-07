# FastKron + YAQA Quantization

This repository contains scripts for reproducing our experiments on **post-training quantization (PTQ)** of LLaMA and Qwen models using  
- **Sketch A** Hessian factors (baseline, from YAQA), and  
- **FastKron** Hessian factors (our method).  

We follow the YAQA pipeline with QTIP quantization and replace the Kronecker-factor estimation step with FastKron.

---

## 0. Installation

Install [QTIP](https://github.com/Cornell-RelaxML/qtip/tree/main):


git clone https://github.com/Cornell-RelaxML/qtip.git
cd qtip
pip install -e .


## 1. Hessians with Sketch A (YAQA baseline)

To reproduce the baseline YAQA factors, run:

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  hessian_llama/get_hess_llama.py \
  --save_path <PATH_TO_SAVE> \
  --orig_model unsloth/llama-2-7b \
  --batch_size 6 \
  --hessian_sketch A \
  --power_iters 4 \
  --ctx_size 4096 \
  --n_seqs 4096

## 2. Hessians with FastKron

FastKron replaces power-iteration with a Lanczos-based estimator.

## 2a. Collect calibration minibatches


python kronfwsvd/collect_fisher_weights.py \
  --model_name <ORIG_MODEL_PATH> \
  --path_to <PATH_TO_SAVE> \
  --size 1 \
  --lr 1e-4
  
## 2b. Run FastKron factor estimation

python kronfwsvd/get_kron_factors_llama.py \
--model_name <ORIG_MODEL_PATH> \


## 3. Quantization and Evaluation

Quantize the model with QTIP and evaluate downstream tasks:

./run_quantizer.sh \
  <ORIG_MODEL_PATH> \
  <PATH_TO_HESSIANS> \
  <TOKENIZER_PATH>
