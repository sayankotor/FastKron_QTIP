"""Microbenchmark for qtip_kernels decompress_matvec.

Compares two shapes:
  * (27648, 5120) — Qwen2.5-32B MLP gate/up shape (M >> K, was 1.17 ms before
    making BLOCK_COUNT shape-aware in inference.cu).
  * (8192, 8192)  — square shape; should NOT regress.

Run with:
    cd qtip-kernels && python test_kernel_speed.py
"""

import torch
import qtip_kernels

# Kernel template constants (must match qtip_torch.cu instantiations).
L, S, R, V = 16, 9, 2, 1
N = 1  # batch dim (matvec)

SHAPES = [
    # (M,     K,    expected_ms_after_fix, label)
    (27648, 5120, 0.30, "Qwen-32B MLP (M>>K)  decompress_matvec_16_9_2_1_27648_1_5120"),
    (8192,  8192, None, "square 8192x8192    decompress_matvec_16_9_2_1_8192_1_8192"),
]

WARMUP = 5
ITERS = 100


def prepare(M, K):
    """Allocate tensors matching what qtip_torch.cu checks."""
    out = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    # compressed: 1D int32, numel * 32 == R * M * K  =>  numel = R*M*K/32
    n_int32 = R * M * K // 32
    compressed = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (n_int32,), dtype=torch.int32, device="cuda",
    )
    x = (torch.randn((K, N), dtype=torch.float16, device="cuda") / 16).clamp(-1, 1).contiguous()
    codebook = (torch.randn((1 << (S + V),), dtype=torch.float16, device="cuda") / 16).clamp(-1, 1)
    return out, compressed, x, codebook


def bench(fn, args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / iters


def main():
    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}")
    print(f"  L={L}, S={S}, R={R}, V={V}, N={N}")
    print(f"  warmup={WARMUP} iters={ITERS}")
    print()

    for M, K, target, label in SHAPES:
        fn_name = f"decompress_matvec_{L}_{S}_{R}_{V}_{M}_{N}_{K}"
        fn = getattr(qtip_kernels, fn_name)

        out, compressed, x, codebook = prepare(M, K)
        ms = bench(fn, (out, compressed, x, codebook))

        # Theoretical HBM-bound lower bound: compressed weight bytes / BW
        compressed_bytes = R * M * K // 8
        hbm_bw_a100 = 1.55e12  # SXM4 80GB ≈ 1.55 TB/s; PCIe is ~1.94
        min_ms = compressed_bytes / hbm_bw_a100 * 1e3
        gbps = (compressed_bytes / 1e9) / (ms / 1e3)

        target_str = f" target<{target:.2f}ms" if target is not None else ""
        print(
            f"{label}"
            f"\n  M={M:>5}  K={K:>5}  compressed={compressed_bytes/1e6:6.1f} MB"
            f"\n  per-call:  {ms:7.3f} ms   {gbps:6.1f} GB/s   HBM-min: {min_ms:5.3f} ms"
            f"   ratio-vs-min: {ms/min_ms:5.1f}x{target_str}"
        )
        print()


if __name__ == "__main__":
    main()
