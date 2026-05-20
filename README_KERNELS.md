# Quantization & inference details

## Quantization

- Batches 1–3: `lr = 1e-4`
- Batch 4: `lr = 1e-6`

## Inference

### 1) with `--manifest` flag

Quantized weight matriх are converted back to bfloat16 on every forward pass.
Speed is the same as the unquantized model. Use this mode for accuracy validation or when fast
CUDA kernels are unavailable.

### 2) with `--manifest kernel` flag

Quantized matrix-vector products are computed directly on packed weights using
custom CUDA kernels (`qtip_kernels`). Memory usage stays compressed (5.8× smaller
for 2-bit), but inference speed is currently lower than baseline due to Python
and kernel-launch overhead around each projection.

## C++ code structure — where to plug in kernels for a new layer shape

```
Python (lib/codebook/__init__.py)
    ↓ torch.library.define("quip_lib::decompress_matvec_qtip_M_1_N_K")
    ↓ torch.library.impl(...) ← Python wrapper: alloc + reshape
    ↓ getattr(qtip_kernels, "decompress_matvec_16_9_K_1_M_1_N")
    ↓
wrapper.cpp (pybind11 → C++)
    ↓ declarations for ~87 functions covering all (M, K, bitrate) combinations
    ↓
qtip_torch.cu (host side, reading tensors)
    ↓ extract data_ptr, check shapes / dtypes
    ↓ call the template instantiation:
    ↓
inference.cu (kernel template)
    ↓ kernel_decompress_matvec<L=16, S=9, R=bitrate, V=1, M, N=1, K>   ← single template
    ↓ compute_block_count(M) → gridSize
    ↓ kernel<<<gridSize, BLOCK_SIZE>>>(out, compressed, x, codebook)
    ↓
CUDA on GPU
```

## Files to edit when adding a new shape

```
FastKron_QTIP/qtip-kernels/
├── setup.py            # build config (touch only when changing flags)
└── src/
    ├── inference.cu    # STEP 1: template void ...<...,M,1,K>()
    │
    ├── qtip_torch.cu   # STEP 2: decompress_matvec_16_9_<bit>_1_<M>_1_<K>(...)
    │
    └── wrapper.cpp     # STEP 3: m.def("decompress_matvec_...", &fn)
```

## How to measure kernel speed

```
python /workspace-SR004.nfs2/chekalina/FastKron_QTIP/qtip-kernels/test_kernel_speed.py
```

## How to measure model speed

```
CUDA_VISIBLE_DEVICES=0 python /workspace-SR004.nfs2/chekalina/FastKron_QTIP/qwen25_32b_run2/bench_inference.py \
    --hf_path /workspace-SR004.nfs2/chekalina/yaqa-quantization/qwen25_32b_run2_quantized_hf \
    --tokenizer /workspace-SR004.nfs2/chekalina/yaqa-quantization/qwen25_32b_run2_quantized_hf \
    --manifest kernel \
    --prompt_len 256 \
    --gen_tokens 128 \
    --trials 3
```
