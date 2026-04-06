# FastKron + YAQA Quantization: High-Speed LLM Compression

This repository contains scripts for reproducing our experiments on **post-training quantization (PTQ)** of LLaMA and Qwen models using:

-   **Sketch A** Hessian factors (baseline, from the YAQA framework), and
-   **FastKron** Hessian factors (our method).

We follow the YAQA pipeline with QTIP quantization and replace the Kronecker-factor estimation step with our accelerated method, **FastKron**.

---

## Publication and Methodology

Detailed methodology, implementation, and experimental results are presented in our paper:

> [Fast and Accurate Fisher-Guided Quantization via Efficient Kronecker Factor
Approximation.], ACL'2026 Main
> V. Chekalina, T.Gerasin. M.Kurkin, A.Kuznetsov, E.Frolov*

---

# Inference

Checkpoints are available on Hf🤗 : [FastKron Hugging Face Collection](https://huggingface.co/collections/timo13113/test-collection)
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

# Quantization from scratch:

### 0. Installation

Install the required [QTIP framework](https://github.com/Cornell-RelaxML/qtip/tree/main):

```bash
git clone [https://github.com/Cornell-RelaxML/qtip.git](https://github.com/Cornell-RelaxML/qtip.git)
cd qtip
pip install -e .
```

### 1. Hessians with Sketch A (YAQA baseline)

To reproduce the baseline YAQA factors, run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  hessian_llama/get_hess_llama.py \
  --save_path <PATH_TO_SAVE> \
  --orig_model unsloth/llama-2-7b \
  --batch_size 6 \
  --hessian_sketch A \
  --power_iters 4 \
  --ctx_size 4096 \
  --n_seqs 4096
```

### 2. Hessians with FastKron

FastKron replaces power-iteration with a Lanczos-based estimator.

#### 2a. Collect calibration minibatches

```
python kronfwsvd/collect_fisher_weights.py \
  --model_name <ORIG_MODEL_PATH> \
  --path_to <PATH_TO_SAVE> \
  --size 1 \
  --lr 1e-4
```
  
#### 2b. Run FastKron factor estimation
```
python kronfwsvd/get_kron_factors_llama.py \
--model_name <ORIG_MODEL_PATH> \
```

#### 3. Quantization and Evaluation

Quantize the model with QTIP and evaluate downstream tasks:
```
./run_quantizer.sh \
  <ORIG_MODEL_PATH> \
  <PATH_TO_HESSIANS> \
  <TOKENIZER_PATH>
```



## 📊 Zero-shot results — LLaMA-3 8B

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



## 📊 Zero-shot results — Qwen-3 8B

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

