# FastKron: Fast and Accurate Fisher-Guided Quantization

This repository contains scripts and results for **post-training quantization (PTQ)** of Qwen3.5* models with linear attention layers.

Checkpoints are available on Hf🤗 :

[qwen35-122b-a3b_2_4_bit (MOE)](https://huggingface.co/Sayankotor/qwen35-122b-a10b-k4e2)

[qwen35-35b-a3b_2_4_bit (MOE)](https://huggingface.co/Sayankotor/qwen35-35b-a3b-FastKron-mixed)

[qwen35-27b-a3b_2_bit](https://huggingface.co/Sayankotor/qwen35-27b-FastKron-2bit)

[qwen35-27b-a3b_4_bit](https://huggingface.co/Sayankotor/qwen35-27b-FastKron-4bit)



---

## Publication and Methodology

Detailed methodology, implementation, and experimental results are presented in our paper:

> [Fast and Accurate Fisher-Guided Quantization via Efficient Kronecker Factor
Approximation.], ACL'2026 
> V. Chekalina, T.Gerasin. A.Kuznetsov, E.Frolov

---

## Results for released Qwen3.5-27B quantized models

### 📊 Zero-shot results on Image Benchmarks 

| Accuracy | MMBench | MME | MMMU | OCRBench |
| :--- | :--- | :--- | :--- | :--- |
| Mode | No-thinking | No-thinking | No-thinking | No-thinking |
| tok_gen | 1024 | 1024 | 1024 | 1024 |
| Примеров | 1500 | 2374 | 900 | 1000 |
| Full | 0.898 ± 0.02 | 0.728 ± 0.02 | 0.520 ± 0.03 | 0.852 ± 0.02 |
| K4 | 0.892 ± 0.02 | 0.762 ± 0.02 | 0.518 ± 0.03 | 0.860 ± 0.02 |
| K2 | 0.890 ± 0.02 | 0.739 ± 0.02 | 0.523 ± 0.03 | 0.829 ± 0.02 |
| Model card (27B) | 0.926 | — | 0.823 | 0.894 |

---

### 📊 Perplexity Evaluation
Perplexity (ctx 4096):

| model | wikitext2 | c4 |
| :--- | :--- | :--- |
| bf16 | 6.720 | 8.809 |
| 4-bit (k4) | 6.949 | 8.871 |
| 2-bit (k2) | 8.390 | 9.969 |

---

### 📊 Common-Sense Reasoning
Common-sense, 0-shot, without chat template (acc_norm; boolq — acc):

| model | arc_c | arc_e | boolq | piqa | hellaswag | AVG |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| full | 0.613 | 0.796 | 0.767 | 0.823 | 0.834 | **0.767** |
| 4-bit | 0.613 | 0.811 | 0.783 | 0.819 | 0.831 | **0.771** |
| 2-bit | 0.619 | 0.827 | 0.739 | 0.814 | 0.793 | **0.758** |

---

### 📊 Reasoning & Instruction Performance
Reasoning / instruction (`--no_thinking`, chat template, fewshot_as_multiturn; gsm8k 5-shot/512 tok, ifeval 0-shot/1280 tok):

| task | full | 4-bit | 2-bit |
| :--- | :--- | :--- | :--- |
| gsm8k (strict) | **0.961** | **0.961** | 0.933 |
| ifeval (prompt / inst strict) | 0.867 / 0.909 | 0.867 / 0.911 | 0.852 / 0.899 |


## Results for released Qwen3.5-35B quantized models

### 📊 Reasoning & Instruction Performance
Qwen3.5-35B-A3B (`--no_thinking`, chat template, fewshot_as_multiturn; gsm8k 5-shot/512 tok, ifeval 0-shot/1280 tok)

| task | full | mixed 2/4-bit | Δ |
| :--- | :--- | :--- | :--- |
| gsm8k (strict) | 0.891 | 0.868 | −2.27 |
| ifeval (prompt / inst strict) | 0.850 / 0.897 | 0.861 / 0.905 | +1.11 / +0.84 |

---

### 📊 Perplexity Evaluation
Qwen3.5-35B-A3B (MoE) — perplexity (ctx 4096), mixed: 4-bit non-experts / 2-bit experts:

| model | wikitext2 | c4 |
| :--- | :--- | :--- |
| bf16 | 6.257 | 9.203 |
| 2/4-bit | 7.002 | 9.863 |
| Δ | +0.745 | +0.660 |


## Inference

When downloading from HF, you also need to pull the modelling.py file.

📊 Speed — Qwen3.5-35B-A3B (decode, bs=1, A100, decode-only)markdown
| mode | ms/tok | tok/s | memory |
| :--- | :--- | :--- | :--- |
| bf16 (uncompressed) | 82.1 | 12.18 | 70.5 GB |
| mixed 2/4-bit, kernel eager | 423.8 | 2.36 | 12.5 GB |
| mixed, kernel + CUDA-graph | 72.0 | 13.9 | 12.8 GB |
| mixed, grouped GEMM + CUDA-graph | 43.5 | 23.0 | 13.0 GB |

## Installation process

### Essential libraries
```
pip install -r requirements.txt
```
the inference pipeline was additionally tested on transformers==4.57.1 and torch==2.5.1+cu124, other versions may work but are not guaranteed
### Install `fast_hadamard_transform`
```
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git fast-hadamard-transform
cd fast-hadamard-transform
pip install -v .
```
### Install the `qtip-kernels` submodule
```
cd qtip-kernels
python setup.py install
```

Important: for the kernels to work, they need to be compiled for specific matrix sizes and codebook settings. 
Otherwise, you may get an error like `AttributeError: '_OpNamespace' 'quip_lib' object has no attribute 'decompress_matvec_qtip_4096_1_12288_2'. Did you mean: 'decompress_matvec_qtip_4096_1_4096_2'?`
For example, the model you want to run may not have the appropriate precompiled dimentions (4096x12288 in the error above). In that case:
- navigate to `qtip-kernels/src`
- add the kernels to be compiled to `wrapper.cpp` and `qtip_torch.cu` in the same notation as all the others in the same file, and add the dimentions of your kernels to the `kernels` array in `/lib/codebook/__init__.py` file
- reinstall the library


## Example

Below is an inference example for Qwen3 quantized model.

```
from transformers import AutoTokenizer, AutoConfig
from model.qwen import Qwen3ForCausalLM # from FastKron package
from tqdm import tqdm

path = '/path/to/qwen3_kronfwsvd_2048_qw_2bit_hf'
device = 'cuda:0'

model = Qwen3ForCausalLM.from_pretrained(path, config = AutoConfig.from_pretrained(path)).to(device)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

prompt = 'What is 2+2?'
prompt = tokenizer.apply_chat_template([
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': prompt},
], tokenize=False)

print('prompt:', prompt)
for _ in tqdm(range(1)):
    res = model.generate(
        tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    )
print('result:', tokenizer.batch_decode(res.cpu()))
```

## How to add a kernel for a layer with an unseen shape

For kernel-level details and how to measure speed see
[README_KERNELS.md](README_KERNELS.md).

# Quantization from scratch:

### 0. Installation

Install the required [QTIP framework](https://github.com/Cornell-RelaxML/qtip/tree/main):

```bash
git clone [https://github.com/Cornell-RelaxML/qtip.git](https://github.com/Cornell-RelaxML/qtip.git)
cd qtip
pip install -e .
```

### 1. Hessians with FastKron

FastKron Hessian estimator.

#### 1a. Collect calibration minibatches

```
python kronfwsvd/collect_fisher_weights.py \
  --model_name <ORIG_MODEL_PATH> \
  --path_to <PATH_TO_SAVE> \
  --size 1 \
  --lr 1e-4
```
  
#### 1b. Run FastKron factor estimation
```
python kronfwsvd/get_kron_factors_llama.py \
--model_name <ORIG_MODEL_PATH> \
```

#### 2. Quantization and Evaluation

Quantize the model with QTIP and evaluate downstream tasks:
```
./run_quantizer.sh \
  <ORIG_MODEL_PATH> \
  <PATH_TO_HESSIANS> \
  <TOKENIZER_PATH>
```


# Quantizing large models (chunked pipeline)

For models larger than 10B parameters use the chunked pipeline (`quantize_big_model` branch): model layers are
split into chunks, and Kronecker-Fisher factors are computed one chunk at a time.
GPU memory stays bounded regardless of model size. This prevents excessive memory usage (at the cost of being
slower).

### Stage 1 — collect per-chunk gradients and compute Kronecker factors
bash run_experiment_7b_true_accum.sh <MODEL_NAME>

Key parameters inside the script (override via env vars or edit lines 25–28):

NUM_LAYERS=32         # total layers in the model
CHUNK_SIZE=16         # how many layers per chunk
DATASET_SIZE=9600     # number of calibration sequences
GRAD_ACCUM=64         # gradient-accumulation steps
MAX_LENGTH=2048       # sequence length
LR=1e-7               # learning rate


Outputs Kronecker factors to `<run_dir>/factors/`.

### Stage 2 — quantize from precomputed factors + convert to HF

SAVE_DIR_BASE=<output_dir> bash run_quantizer_qwen.sh 
<MODEL_NAME> 
<FACTORS_DIR> 
<OUTPUT_FOLDER_NAME>

This runs `quantize_finetune_llama.py` → `hfize_qwen2.py` → perplexity and
zero-shot evaluation in sequence. Override the bitrate via `--K 2` or `--K 4`.

### Recommended settings per model size

| Model | NUM_LAYERS | CHUNK_SIZE | GRAD_ACCUM | MAX_LENGTH |
|---|---|---|---|---|
| Llama-2 7B | 32 | 16 | 64 | 2048 |
| Qwen2.5-32B | 64 | 16 | 96 | 3096 |


# Results for previously released Qwen 2.5 32B quantized model

## 📊 Zero-shot results — Qwen-2.5 32B PTQ no fine-tuning

### Perplexity

| | wikitext2 ↓ | c4 ↓ |
|---|---|---|
| Baseline (BF16) | 4.6701 | 8.5833 |
| 4-bit FastKron | 4.7944 | 8.6555 |
| 2-bit FastKron | 6.2502 | 9.7388 |

### Zero-shot tasks

| | arc_c ↑ | arc_e ↑ | boolq ↑ | hellaswag ↑ | piqa ↑ | winogrande ↑ | AVG ↑ |
|---|---|---|---|---|---|---|---|
| Baseline (BF16) | 0.5307 | 0.8085 | 0.8713 | 0.6498 | 0.8199 | 0.7522 | **0.7387** |
| 4-bit FastKron | 0.5205 | 0.7950 | 0.8722 | 0.6482 | 0.8166 | 0.7545 | **0.7345** |
| 2-bit FastKron | 0.4633 | 0.7837 | 0.8700 | 0.6052 | 0.8003 | 0.7443 | **0.7111** |

### GSM8K & IFEval

| | gsm8k ↑ | ifeval ↑ |
|---|---|---|
| Baseline (BF16) | 0.8294 | 0.3660 |
| 4-bit FastKron | 0.8673 | 0.3641 |
| 2-bit FastKron | 0.7953 | 0.3401 |

### Model size

| | Disk size | Compression |
|---|---|---|
| Baseline (BF16) | 64 GB | 1× |
| 4-bit FastKron | 17 GB | 3.8× |
| 2-bit FastKron | 11 GB | 5.8× |

> `gsm8k` — exact_match (strict-match), 5-shot
> `ifeval` — prompt_level_strict_acc, 0-shot



## 📊 Zero-shot results — LLaMA-3 8B PTQ no fine-tuning

### 🟡 4-bit Quantization

| Method             | Steps | ARC_c ↑ | BoolQ ↑ | PIQA ↑ | ARC_e ↑ | HSwag ↑ | AVG ↑  | GPU/h ↓ | Tokens ↓ |
|---------------------|-------|---------|---------|--------|---------|---------|--------|---------|-----------|
| 16 bit (baseline)   | –     | **0.5171** | **0.8409** | **0.7986** | **0.8177** | **0.5908** | **0.7131** | –    | –      |
| 4-bit Sketch A      | 4096  | **0.5136** | **0.8443** | 0.7997 | 0.8198 | **0.5865** | 0.7127 | 92   | 16 M   |
| 4-bit FastKron      | 75    | 0.5116 | 0.8438 | **0.8025** | **0.8207** | 0.5863 | **0.7129** | 9.5  | 712 K  |
| 4-bit No Hess       | –     | 0.5119 | 0.8415 | 0.7959 | 0.8097 | 0.5859 | 0.7112 | –    | –      |


### 🟠 2-bit Quantization

| Method             | Steps | ARC_c ↑ | BoolQ ↑ | PIQA ↑ | ARC_e ↑ | HSwag ↑ | AVG ↑  | GPU/h ↓ | Tokens ↓ |
|---------------------|-------|---------|---------|--------|---------|---------|--------|---------|-----------|
| 2-bit Sketch A      | 4096  | **0.4312** | 0.7567 | 0.7647 | 0.7391 | **0.5259** | 0.6435 | 92   | 16 M   |
| 2-bit FastKron      | 100   | 0.4277 | **0.7646** | **0.7661** | **0.7468** | 0.5159 | **0.6442** | 11.5 | 950 K |
| 2-bit No Hess       | –     | 0.2363 | 0.6336 | 0.6554 | 0.5108 | 0.3620 | 0.5094 | –    | –     |



## 📊 Zero-shot results — Qwen-3 8B PTQ no fine-tuning

### 🟡 4-bit Quantization

| Method             | Steps | ARC_c ↑ | BoolQ ↑ | PIQA ↑ | ARC_e ↑ | HSwag ↑ | AVG ↑  | GPU/h ↓ | Tokens ↓ |
|---------------------|-------|---------|---------|--------|---------|---------|--------|---------|-----------|
| 16 bit (baseline)   | –     | **0.5563** | **0.8682** | **0.7677** | **0.8354** | **0.5708** | **0.7197** | –   | –     |
| 4-bit Sketch A      | 4096  | **0.5503** | 0.8611 | 0.7612 | 0.8324 | 0.5601 | **0.7132** | 84  | 8 M   |
| 4-bit FastKron      | 150   | 0.5469 | 0.8667 | 0.7601 | **0.8287** | **0.5637** | **0.7132** | 42  | 712 K |
| 4-bit No Hess       | –     | 0.5467 | **0.8675** | **0.7622** | 0.8312 | 0.5585 | **0.7132** | –   | –     |


### 🟠 2-bit Quantization

| Method             | Steps | ARC_c ↑ | BoolQ ↑ | PIQA ↑ | ARC_e ↑ | HSwag ↑ | AVG ↑  | GPU/h ↓ | Tokens ↓ |
|---------------------|-------|---------|---------|--------|---------|---------|--------|---------|-----------|
| 2-bit Sketch A      | 4096  | 0.4536 | 0.7782 | **0.7435** | **0.7797** | 0.4611 | 0.6432 | 84  | 8 M   |
| 2-bit FastKron      | 150   | **0.4616** | 0.8416 | 0.7334 | 0.7702 | **0.4853** | **0.6584** | 42  | 712 K |
| 2-bit No Hess       | –     | 0.3993 | **0.8675** | 0.7743 | 0.7003 | 0.4758 | 0.6434 | –   | –     |


## 📊 Zero-shot results — LLaMA-2 7B

### 🟡 4-bit Quantization

| Method             | Steps | ARC_c ↑ | BoolQ ↑ | PIQA ↑ | ARC_e ↑ | HSwag ↑ | AVG ↑  | GPU/h ↓ | Tokens ↓ |
|---------------------|-------|---------|---------|--------|---------|---------|--------|---------|-----------|
| 16 bit (baseline)   | –     | **0.4325** | **0.7767** | **0.7774** | **0.7617** | **0.5721** | **0.6640** | –  | –       |
| 4-bit Sketch A      | 4096  | 0.4274 | 0.7688 | 0.7752 | **0.7613** | **0.5672** | 0.6599 | 50 | 16 M    |
| 4-bit FastKron      | 75    | 0.4283 | 0.7792 | **0.7802** | 0.7610 | 0.5660 | 0.6629 | 5  | 712 K   |
| 4-bit No Hess       | –     | **0.4352** | **0.7875** | 0.7742 | 0.7609 | 0.5628 | **0.6641** | –  | –       |


### 🟠 2-bit Quantization

| Method             | Steps | ARC_c ↑ | BoolQ ↑ | PIQA ↑ | ARC_e ↑ | HSwag ↑ | AVG ↑  | GPU/h ↓ | Tokens ↓ |
|---------------------|-------|---------|---------|--------|---------|---------|--------|---------|-----------|
| 2-bit Sketch A      | 4096  | 0.3805 | 0.7333 | 0.7562 | **0.7192** | **0.5227** | 0.6223 | 50 | 16 M    |
| 2-bit FastKron      | 150   | **0.3843** | **0.7510** | **0.7600** | 0.7112 | 0.5139 | **0.6240** | 6  | 1400 K |
| 2-bit No Hess       | –     | 0.2210 | 0.6355 | 0.6306 | 0.5152 | 0.3422 | 0.4689 | –  | –       |

