#!/bin/bash
set -e

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <base_model_path> <hess_path> <folder_name> [tokenizer_path]"
    exit 1
fi

BASE_MODEL="$1"
HESS_PATH="$2"
FOLDER_NAME="$3"
TOKENIZER_PATH="${4:-}"   # необязательный аргумент


SAVE_DIR="../yaqa-quantization/${FOLDER_NAME}"
HF_DIR="${SAVE_DIR}_hf"

# 1. Квантовка и (опциональное) дообучение
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 quantize_llama/quantize_finetune_llama.py \
    --base_model "${BASE_MODEL}" \
    --hess_path "${HESS_PATH}" \
    --save_path "${SAVE_DIR}" \
    --codebook bitshift \
    --scale_override 0.9 \
    --ft_epochs 0 \
    --decode_mode quantlut_sym \
    --tlut_bits 9 \
    --L 16 \
    --K 4 \
    --V 2 \
    --td_x 16 \
    --td_y 16

# 2. HuggingFace-совместимая конвертация
python quantize_llama/hfize_llama.py \
    --quantized_path "${SAVE_DIR}" \
    --hf_output_path "${HF_DIR}"

# 3. Оценка perplexity
python eval/eval_ppl.py \
    --hf_path "${HF_DIR}" \
    --manifest

# 4. Zero-shot оценка
if [[ -z "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_PATH="${HF_DIR}"
fi

CUDA_VISIBLE_DEVICES=1 python3 eval/eval_zeroshot.py \
  --hf_path "${HF_DIR}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --tasks arc_challenge,arc_easy,hellaswag,boolq,piqa \
  --num_fewshot 0 \
  --batch_size 4 \
  --manifest_model

