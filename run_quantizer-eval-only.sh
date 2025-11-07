#!/bin/bash
set -e

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 <base_model_path> <hess_path> <folder_name> [tokenizer_path]"
    exit 1
fi

BASE_MODEL="$1"
HESS_PATH="$2"
FOLDER_NAME="$3"
TOKENIZER_PATH="${4:-}"   


SAVE_DIR="../yaqa-quantization/${FOLDER_NAME}"
HF_DIR="${SAVE_DIR}_hf"




python quantize_llama/hfize_llama.py \
    --quantized_path "${SAVE_DIR}" \
    --hf_output_path "${HF_DIR}"


python eval/eval_ppl.py \
    --hf_path "${HF_DIR}" \
    --manifest


if [[ -z "${TOKENIZER_PATH}" ]]; then
  TOKENIZER_PATH="${HF_DIR}"
fi

# CUDA_VISIBLE_DEVICES=1 \
python3 eval/eval_zeroshot.py \
  --hf_path "${HF_DIR}" \
  --tokenizer "${TOKENIZER_PATH}" \
  --tasks arc_challenge,arc_easy,hellaswag,boolq,piqa \
  --num_fewshot 0 \
  --batch_size 4 \
  --manifest_model

