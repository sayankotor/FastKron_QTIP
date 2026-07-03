# FastKron: Fast and Accurate Fisher-Guided Quantization

Post-training quantization (PTQ) of LLaMA / Qwen models via **trellis-coded quantization guided** with **second-order information**.

> **Paper:** *Fast and Accurate Fisher-Guided Quantization via Efficient Kronecker Factor Approximation.* ACL 2026 Main.
> V. Chekalina, T. Gerasin, M. Kurkin, A. Kuznetsov, E. Frolov.
> 
> https://aclanthology.org/2026.acl-long.1805/

> **HF Checkpoints:** [FastKron HF Collection](https://huggingface.co/collections/timo13113/test-collection)
> 
> [FastKron Qwen2.5-32B 2-bit](https://huggingface.co/Sayankotor/Qwen2.5-32B-FastKron-2bit)
>
> [FastKron Qwen2.5-32B 4-bit](https://huggingface.co/Sayankotor/Qwen2.5-32B-FastKron-4bit)

---

## Qwen3.5-27B — FastKron Quantization Results

> ⚠️ **Gated DeltaNet attention.** Qwen3.5-27B uses a **hybrid Gated DeltaNet + Gated Attention** stack (linear-attention layers with a fixed recurrent state, not standard softmax self-attention). FastKron quantizes it end-to-end — demonstrating that our Fisher-guided trellis quantization is **not tied to standard-attention architectures** and transfers to linear-attention / hybrid models.


**Perplexity**

| | wikitext2 ↓ | c4 ↓ |
|---|---|---|
| Baseline (BF16) | 6.7202 | 8.8092 |
| 4-bit FastKron | 6.9444 | 8.8716 |
| 2-bit FastKron | 8.2054 | 9.8243 |

**Zero-shot tasks (row acc)**

| | arc_c ↑ | arc_e ↑ | boolq ↑ | hellaswag ↑ | piqa ↑ | AVG ↑ |
|---|---|---|---|---|---|---|
| Baseline (BF16) | 0.5900 | 0.8481 | 0.7670 | 0.6377 | 0.8107 | **0.7315** |
| 4-bit FastKron | 0.5904 | 0.8514 | 0.7780 | 0.6335 | 0.8102 | **0.7327** |
| 2-bit FastKron | 0.6015 | 0.8540 | 0.7590 | 0.5984 | 0.8085 | **0.7243** |

**GSM8K & IFEval · Model size**

| | gsm8k ↑ | ifeval ↑ | 
|---|---|---|
| Baseline (BF16) | 0.3556 | 0.3995 | 
| 4-bit FastKron | 0.2805 | 0.3162 | 
| 2-bit FastKron | 0.1190 | 0.1539 | 


**Model size**

| | size | compression |
|---|---|---|
| Baseline (BF16) | 52 GB | 1× |
| 4-bit FastKron | 12.2 GB | 4.3× |
| 2-bit FastKron | 6.1 GB | 9.2× |

---


## Qwen-2.5 32B — FastKron Quantization Results

**HF checkpoints:** [FastKron Qwen2.5-32B 2-bit] (https://huggingface.co/Sayankotor/Qwen2.5-32B-FastKron-2bit) · [FastKron Qwen2.5-32B 4-bit](https://huggingface.co/Sayankotor/Qwen2.5-32B-FastKron-4bit)

**Perplexity**

| | wikitext2 ↓ | c4 ↓ |
|---|---|---|
| Baseline (BF16) | 4.6701 | 8.5833 |
| 4-bit FastKron | 4.7944 | 8.6555 |
| 2-bit FastKron | 6.2502 | 9.7388 |

**Zero-shot tasks**

| | arc_c ↑ | arc_e ↑ | boolq ↑ | hellaswag ↑ | piqa ↑ | winogrande ↑ | AVG ↑ |
|---|---|---|---|---|---|---|---|
| Baseline (BF16) | 0.5307 | 0.8085 | 0.8713 | 0.6498 | 0.8199 | 0.7522 | **0.7387** |
| 4-bit FastKron | 0.5205 | 0.7950 | 0.8722 | 0.6482 | 0.8166 | 0.7545 | **0.7345** |
| 2-bit FastKron | 0.4633 | 0.7837 | 0.8700 | 0.6052 | 0.8003 | 0.7443 | **0.7111** |

**GSM8K & IFEval · Model size**

| | gsm8k ↑ | ifeval ↑ | disk | compression |
|---|---|---|---|---|
| Baseline (BF16) | 0.8294 | 0.3660 | 64 GB | 1× |
| 4-bit FastKron | 0.8673 | 0.3641 | 17 GB | 3.8× |
| 2-bit FastKron | 0.7953 | 0.3401 | 11 GB | 5.8× |

> `gsm8k` — exact_match (strict), 5-shot · `ifeval` — prompt_level_strict_acc, 0-shot

---


## Qwen3.5-32B — FastKron Quantization Results

**HF checkpoints:** `⟨add link⟩` (2-bit) · `⟨add link⟩` (4-bit)

**Results** *(fill in measured numbers):*

| | wikitext2 ↓ | c4 ↓ | zero-shot AVG ↑ | gsm8k ↑ | disk | compression |
|---|---|---|---|---|---|---|
| Baseline (BF16) | TBD | TBD | TBD | TBD | TBD | 1× |
| 4-bit FastKron | TBD | TBD | TBD | TBD | TBD | ~3.8× |
| 2-bit FastKron | TBD | TBD | TBD | TBD | TBD | ~5.8× |

---

## Kernels

Quantized checkpoints ship with custom **QTIP decompression kernels** (CUDA C++) for fast trellis-decode-and-matvec inference.

Kernels are compiled **per matrix shape + codebook setting**. If a shape is missing you'll see e.g.
`'quip_lib' object has no attribute 'decompress_matvec_qtip_4096_1_12288_2'`.

To add a shape:
1. add the kernel in `qtip-kernels/src/wrapper.cpp` and `qtip_torch.cu` (same notation as existing entries);
2. add its dimensions to the `kernels` array in `lib/codebook/__init__.py`;
3. reinstall the submodule.

Kernel-level details and speed measurement: [README_KERNELS.md](README_KERNELS.md).

Install:
```bash
# fast Hadamard transform
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git && cd fast-hadamard-transform && pip install -v . && cd ..
# QTIP kernels
cd qtip-kernels && python setup.py install && cd ..
```

---

## Inference

```bash
pip install -r requirements.txt   # tested on transformers==4.57.1, torch==2.5.1+cu124
```

```python
from transformers import AutoTokenizer, AutoConfig
from model.qwen import Qwen3ForCausalLM  # from FastKron

path, device = '/path/to/qwen3_kronfwsvd_2048_qw_2bit_hf', 'cuda:0'
model = Qwen3ForCausalLM.from_pretrained(path, config=AutoConfig.from_pretrained(path)).to(device)
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

prompt = tok.apply_chat_template(
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'What is 2+2?'}], tokenize=False)
out = model.generate(tok.encode(prompt, return_tensors='pt').to(device))
print(tok.batch_decode(out.cpu()))
```

---

## Quantize from scratch

Install [QTIP](https://github.com/Cornell-RelaxML/qtip): `git clone … && cd qtip && pip install -e .`

**1. Hessians.** Baseline (YAQA Sketch A):
```bash
torchrun --standalone --nproc-per-node=4 hessian_llama/get_hess_llama.py \
  --save_path <OUT> --orig_model <MODEL> --hessian_sketch A --power_iters 4 \
  --batch_size 6 --ctx_size 4096 --n_seqs 4096
```
FastKron (Lanczos, replaces power-iteration):
```bash
python kronfwsvd/collect_fisher_weights.py --model_name <MODEL> --path_to <OUT> --size 1 --lr 1e-4
python kronfwsvd/get_kron_factors_llama.py --model_name <MODEL>
```

**2. Quantize + eval:**
```bash
./run_quantizer.sh <MODEL> <HESSIANS> <TOKENIZER>
```

**Large models (>10B):** use the `quantize_big_model` branch — chunked, block-by-block calibration to cap memory (slower).

---

## Other released results

<details>
<summary><b>LLaMA-3 8B / Qwen-3 8B / LLaMA-2 7B</b> — FastKron matches Sketch A at ~10–20× fewer GPU-hours</summary>

FastKron reaches Sketch-A accuracy using ~700K–1.4M calibration tokens vs 8–16M, at a fraction of the GPU-hours. Full per-task tables (ARC, BoolQ, PIQA, HellaSwag): see paper / git history.

Key point: at **2-bit**, `No Hess` collapses (e.g. LLaMA-2 7B AVG 0.47) while FastKron holds baseline-level accuracy (0.62) — the second-order signal is what makes low-bit work.
</details>
