# Qwen3.5-27B quantized (QTIP) — load & evaluate

Python: `/home/jovyan/.mlspace/envs/qtip_clean/bin/python`

## Checkpoints (where they live)

On Hugging Face:
- **4-bit:** `Sayankotor/qwen35-27b-4bit` — https://huggingface.co/Sayankotor/qwen35-27b-4bit
- **2-bit:** `Sayankotor/qwen35-27b-FastKron-2bit` — https://huggingface.co/Sayankotor/qwen35-27b-FastKron-2bit

Local copies (same weights):
- 4-bit: `/home/jovyan/shares/SR004.nfs2/chekalina/qwen35_27b_run1/quant_k4_hf`
- 2-bit: `/home/jovyan/shares/SR004.nfs2/chekalina/qwen35_27b_run1/quant_k2_hf`

## Code

- `model/qwen3_5_quantized.py` — the quantized 27B model class.
- `eval/eval_ppl_compressed.py` — perplexity (wikitext2 + c4).   # validation: PPL
- `eval/eval_zeroshot.py` — task eval (gsm8k, ifeval, ...).      # validation: benchmarks

## Load the model from HF

```python
import torch
from transformers import AutoConfig, AutoTokenizer
from lib.linear.quantized_linear import QuantizedLinear
from model.qwen3_5_quantized import (
    Qwen3_5QuantizedForConditionalGeneration, materialize_hadK)

# pick the checkpoint: 4-bit or 2-bit
HF = "Sayankotor/qwen35-27b-4bit"                 # or "Sayankotor/qwen35-27b-FastKron-2bit"
cfg = AutoConfig.from_pretrained(HF, trust_remote_code=True)

model = Qwen3_5QuantizedForConditionalGeneration.from_pretrained(
    HF, config=cfg, torch_dtype=torch.bfloat16, device_map="auto",
    low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
materialize_hadK(model)                           # required after load
for m in model.modules():
    if isinstance(m, QuantizedLinear):
        m.mode = "train-fixW"                     # unpack weights once (fast)

tok = AutoTokenizer.from_pretrained(HF)
ids = tok("The history of quantization", return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=64, do_sample=False, use_cache=True)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Validation — perplexity (wikitext2 + c4)

```bash
PY=/home/jovyan/.mlspace/envs/qtip_clean/bin/python
cd /home/jovyan/shares/SR004.nfs2/chekalina/FastKron_QTIP

# 4-bit
CUDA_VISIBLE_DEVICES=0 $PY eval/eval_ppl_compressed.py \
  --hf_path Sayankotor/qwen35-27b-4bit \
  --seqlen 4096 --output_json ppl_27b_4bit.json

# 2-bit
CUDA_VISIBLE_DEVICES=0 $PY eval/eval_ppl_compressed.py \
  --hf_path Sayankotor/qwen35-27b-FastKron-2bit \
  --seqlen 4096 --output_json ppl_27b_2bit.json
```

## Validation — gsm8k

```bash
PY=/home/jovyan/.mlspace/envs/qtip_clean/bin/python
cd /home/jovyan/shares/SR004.nfs2/chekalina/FastKron_QTIP

# 4-bit
CUDA_VISIBLE_DEVICES=0 $PY eval/eval_zeroshot.py \
  --hf_path Sayankotor/qwen35-27b-4bit --tokenizer Sayankotor/qwen35-27b-4bit \
  --tasks gsm8k --num_fewshot -1 --batch_size 16 --output_json gsm8k_27b_4bit.json

# 2-bit
CUDA_VISIBLE_DEVICES=0 $PY eval/eval_zeroshot.py \
  --hf_path Sayankotor/qwen35-27b-FastKron-2bit --tokenizer Sayankotor/qwen35-27b-FastKron-2bit \
  --tasks gsm8k --num_fewshot -1 --batch_size 16 --output_json gsm8k_27b_2bit.json
```

Notes:
- `--num_fewshot -1` uses the task default (gsm8k = 5-shot).
- No `--limit` = full set (1319 samples); add `--limit N` for a quick check.
- Add more tasks with a comma, e.g. `--tasks gsm8k,ifeval`.
- If ifeval errors on nltk:
  `python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`
