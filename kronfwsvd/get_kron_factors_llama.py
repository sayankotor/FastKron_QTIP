"""
Compute top-k Kronecker factors of per-layer gradient covariance for QTIP.

Outputs (per layer, per projection):
    {kron_dir}/model_layers_{idx}_{block}_{proj}_proj.safetensors
        XF:  (m, m) float32   — left/output Hessian factor
        YF:  (n, n) float32   — right/input Hessian factor
        s:   (top_k,) float32 — top singular values (debug only; reader ignores)

These files are consumed unchanged by lib/algo/finetune.py:170 in the
quantize_finetune_llama.py pipeline (Hin <- YF, Hout <- XF).

Designed for chunked layer processing on large models (e.g. Qwen2.5-32B):
the python script knows nothing about chunks; --layer_start/--layer_end
restrict which layer indices are processed in this invocation, and the
output directory is shared across chunk runs.
"""

import argparse
import logging
import os
import re
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import cupy as cp
import torch
import torch.multiprocessing as mp
from cupyx.scipy.sparse.linalg import LinearOperator, svds
from multiprocessing import SimpleQueue
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from torch.multiprocessing import Process
from tqdm.auto import tqdm


LAYER_NAME_RE = re.compile(r"^model_layers_(\d+)_(self_attn|mlp)_[a-z_]+_proj$")


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def extract_layer_idx(layer_name: str) -> Optional[int]:
    m = LAYER_NAME_RE.match(layer_name)
    return int(m.group(1)) if m else None


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "kron_factors.log"
    logger = logging.getLogger("KroneckerLogger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def load_grad_index(
    grad_dir: Path,
    layer_start: int,
    layer_end: int,
    logger: logging.Logger,
) -> Dict[str, List[str]]:
    """Build {layer_name: [file_path, ...]} for layers in [layer_start, layer_end).

    Reads only safetensors headers — no tensor data is materialized.
    """
    files = sorted(
        p for p in grad_dir.iterdir()
        if p.is_file() and p.suffix == ".safetensors"
    )
    if not files:
        raise RuntimeError(f"No .safetensors files found in {grad_dir}")

    layer_to_files: Dict[str, List[str]] = {}
    seen_unrecognized: set = set()

    for path in tqdm(files, desc=f"Indexing grads in {grad_dir.name}"):
        try:
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    idx = extract_layer_idx(key)
                    if idx is None:
                        if key not in seen_unrecognized:
                            logger.warning(f"skipping unrecognized key: {key}")
                            seen_unrecognized.add(key)
                        continue
                    if idx < layer_start or idx >= layer_end:
                        continue
                    layer_to_files.setdefault(key, []).append(str(path))
        except SafetensorError as e:
            logger.warning(f"failed to read {path}: {e}")
            continue

    return layer_to_files


def get_kron_factors_worker(
    layer_name: str,
    file_paths: List[str],
    layer_idx: int,
    total_layers: int,
    top_k: int,
    grad_chunk_size: int,
    max_grads: Optional[int],
    debug_matvec: bool,
    device_id: int,
    kron_dir: Path,
    logger: logging.Logger,
) -> bool:
    """Compute top-k Kronecker factors for one layer.

    On compute success: writes <kron_dir>/<layer_name>.safetensors atomically
    and returns True. On compute failure: logs and returns False without
    writing. Save errors propagate (chunk-level assert relies on file
    presence to detect fatal issues).
    """
    cp.cuda.Device(device_id).use()

    pfx = (
        f"[GPU-{device_id}] [layer {layer_idx}/{total_layers}] {layer_name}"
    )

    if max_grads is not None:
        file_paths = file_paths[:max_grads]
    k_total = len(file_paths)
    if k_total == 0:
        print(f"[{stamp()}] {pfx} | NO grad files — skipping", flush=True)
        logger.error(f"[{layer_name}] no gradient files — skipping")
        return False

    print(f"[{stamp()}] {pfx} | starting", flush=True)
    print(f"[{stamp()}] {pfx} | indexed K={k_total} grad files", flush=True)

    with safe_open(file_paths[0], framework="pt", device="cpu") as f:
        first = f.get_tensor(layer_name)
    m, n = first.shape
    del first

    bf16_gb = (k_total * m * n * 2) / 1e9
    print(
        f"[{stamp()}] {pfx} | loading grads to CPU ({bf16_gb:.1f} GB bf16)",
        flush=True,
    )

    load_t0 = time.time()
    big = torch.empty((k_total, m, n), dtype=torch.bfloat16)
    for i, path in enumerate(file_paths):
        with safe_open(path, framework="pt", device="cpu") as f:
            t = f.get_tensor(layer_name)
        if tuple(t.shape) != (m, n):
            raise RuntimeError(
                f"shape mismatch at {path}:{layer_name}: got {tuple(t.shape)}, "
                f"expected ({m},{n})"
            )
        big[i].copy_(t.to(torch.bfloat16))
    load_dt = time.time() - load_t0

    print(
        f"[{stamp()}] {pfx} | loaded in {load_dt:.1f}s, "
        f"peak RSS={peak_rss_gb():.1f} GB",
        flush=True,
    )
    logger.info(
        f"[{layer_name}] gpu={device_id} loaded k={k_total} "
        f"shape=({m},{n}) bf16 → {bf16_gb:.2f} GB CPU in {load_dt:.1f}s"
    )

    matvec_count = [0]

    def matvec(vec):
        matvec_count[0] += 1
        if debug_matvec:
            print(
                f"[{stamp()}] [GPU-{device_id}] matvec #{matvec_count[0]}",
                flush=True,
            )
        V = vec.reshape(n, n, order="F")
        result = cp.zeros((m, m), dtype=cp.float32)
        for ci in range(0, k_total, grad_chunk_size):
            sub_fp32_np = big[ci:ci + grad_chunk_size].to(torch.float32).numpy()
            sub_cp = cp.asarray(sub_fp32_np)
            for j in range(sub_cp.shape[0]):
                G = sub_cp[j].reshape(m, n, order="F")
                result += G @ V @ G.T
            del sub_cp
        return (result / k_total).T.ravel()

    def r_matvec(vec):
        V = vec.reshape(m, m, order="F")
        result = cp.zeros((n, n), dtype=cp.float32)
        for ci in range(0, k_total, grad_chunk_size):
            sub_fp32_np = big[ci:ci + grad_chunk_size].to(torch.float32).numpy()
            sub_cp = cp.asarray(sub_fp32_np)
            for j in range(sub_cp.shape[0]):
                G = sub_cp[j].reshape(m, n, order="F")
                result += G.T @ V @ G
            del sub_cp
        return (result / k_total).T.ravel()

    print(
        f"[{stamp()}] {pfx} | running svds (top_k={top_k}, m={m}, n={n})",
        flush=True,
    )

    svd_t0 = time.time()
    try:
        linop = LinearOperator(
            shape=(m * m, n * n),
            matvec=matvec,
            rmatvec=r_matvec,
            dtype=cp.float32,
        )
        u, s, vt = svds(linop, k=top_k, return_singular_vectors=True)

        sidx = cp.argsort(-s)
        s = s[sidx]
        u = u[:, sidx]
        v = vt[sidx, :].T

        XF = (u[:, 0] * s[0]).reshape(m, m, order="F")
        YF = v[:, 0].reshape(n, n, order="F")

        XF_t = torch.from_numpy(XF.get())
        YF_t = torch.from_numpy(YF.get())
        s_t = torch.from_numpy(s.get())
    except Exception:
        print(
            f"[{stamp()}] {pfx} | SVD FAILED — see log",
            flush=True,
        )
        logger.error(f"[{layer_name}] SVD failure on gpu {device_id}")
        logger.error(traceback.format_exc())
        del big
        cp.get_default_memory_pool().free_all_blocks()
        return False
    svd_dt = time.time() - svd_t0

    print(
        f"[{stamp()}] {pfx} | svds done in {svd_dt:.1f}s "
        f"(matvec calls={matvec_count[0]}), top sv={float(s[0]):.4f}",
        flush=True,
    )

    out_path = kron_dir / f"{layer_name}.safetensors"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    save_file({"XF": XF_t, "YF": YF_t, "s": s_t}, str(tmp_path))
    os.replace(tmp_path, out_path)

    print(
        f"[{stamp()}] {pfx} | saved to {out_path}",
        flush=True,
    )
    logger.info(
        f"[{layer_name}] gpu={device_id} done load={load_dt:.1f}s "
        f"svd={svd_dt:.1f}s matvec_calls={matvec_count[0]} | "
        f"top sv={float(s[0]):.4f} | peak RSS={peak_rss_gb():.2f} GB"
    )

    del big
    cp.get_default_memory_pool().free_all_blocks()
    return True


def gpu_worker(
    gpu_id: int,
    task_queue: SimpleQueue,
    top_k: int,
    grad_chunk_size: int,
    max_grads: Optional[int],
    debug_matvec: bool,
    kron_dir: str,
) -> None:
    cp.cuda.Device(gpu_id).use()
    kron_dir_path = Path(kron_dir)
    logger = setup_logger(kron_dir_path)
    print(f"[{stamp()}] [GPU-{gpu_id}] worker started", flush=True)
    logger.info(f"[worker {gpu_id}] started")

    while True:
        task = task_queue.get()
        if task is None:
            print(f"[{stamp()}] [GPU-{gpu_id}] worker exiting", flush=True)
            logger.info(f"[worker {gpu_id}] received stop signal")
            return
        layer_idx, total_layers, layer_name, file_paths = task
        get_kron_factors_worker(
            layer_name=layer_name,
            file_paths=file_paths,
            layer_idx=layer_idx,
            total_layers=total_layers,
            top_k=top_k,
            grad_chunk_size=grad_chunk_size,
            max_grads=max_grads,
            debug_matvec=debug_matvec,
            device_id=gpu_id,
            kron_dir=kron_dir_path,
            logger=logger,
        )


def run_parallel_kron(
    layer_to_files: Dict[str, List[str]],
    kron_dir: Path,
    layer_start: int,
    layer_end: int,
    top_k: int,
    grad_chunk_size: int,
    max_grads: Optional[int],
    debug_matvec: bool,
    num_devices: int,
    logger: logging.Logger,
) -> bool:
    kron_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(layer_to_files.items())
    M = len(tasks)

    print(
        f"[{stamp()}] kron run starting | layers=[{layer_start},{layer_end}) "
        f"| num_layers={M} | num_devices={num_devices} | max_grads={max_grads}",
        flush=True,
    )
    logger.info(
        f"kron run starting | layers=[{layer_start},{layer_end}) "
        f"num_layers={M} num_devices={num_devices} max_grads={max_grads} "
        f"grad_chunk_size={grad_chunk_size} top_k={top_k} debug_matvec={debug_matvec}"
    )

    run_t0 = time.time()

    task_queue: SimpleQueue = SimpleQueue()
    processes: List[Process] = []
    for gpu_id in range(num_devices):
        p = Process(
            target=gpu_worker,
            args=(
                gpu_id, task_queue, top_k, grad_chunk_size, max_grads,
                debug_matvec, str(kron_dir),
            ),
        )
        p.start()
        processes.append(p)

    for i, (layer_name, file_paths) in enumerate(tasks, start=1):
        task_queue.put((i, M, layer_name, file_paths))
    for _ in range(num_devices):
        task_queue.put(None)

    for p in processes:
        p.join()

    nonzero = [(p.pid, p.exitcode) for p in processes if p.exitcode != 0]
    if nonzero:
        print(
            f"[{stamp()}] kron run FAILED | workers exited non-zero: {nonzero}",
            flush=True,
        )
        logger.error(f"workers exited non-zero: {nonzero}")
        return False

    run_dt = time.time() - run_t0
    print(
        f"[{stamp()}] kron run done | total time={run_dt:.1f}s "
        f"| peak RSS={peak_rss_gb():.1f} GB",
        flush=True,
    )
    logger.info(
        f"kron run done | total time={run_dt:.1f}s peak RSS={peak_rss_gb():.2f} GB"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Kronecker factors of gradient covariance per layer."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model identifier (logging only, no path use)")
    parser.add_argument("--grad_dir", type=str, required=True,
                        help="Directory with grad_step_*.safetensors files (one chunk)")
    parser.add_argument("--kron_dir", type=str, required=True,
                        help="Output directory for per-layer .safetensors factors "
                             "(shared across chunk runs)")
    parser.add_argument("--layer_start", type=int, required=True,
                        help="Inclusive lower bound on layer index")
    parser.add_argument("--layer_end", type=int, required=True,
                        help="Exclusive upper bound on layer index")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of singular values for svds")
    parser.add_argument("--max_grads", type=int, default=100,
                        help="Cap on gradient samples per layer")
    parser.add_argument("--grad_chunk_size", type=int, default=4,
                        help="Grad samples uploaded to GPU per matvec batch")
    parser.add_argument("--num_devices", type=int, default=1,
                        help="GPUs to use (one worker per device)")
    parser.add_argument("--debug_matvec", action="store_true",
                        help="Print 'matvec #N' on each linop iteration (very verbose)")
    args = parser.parse_args()

    if args.layer_end <= args.layer_start:
        sys.exit(
            f"--layer_end ({args.layer_end}) must be > "
            f"--layer_start ({args.layer_start})"
        )

    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    grad_dir = Path(args.grad_dir)
    kron_dir = Path(args.kron_dir)
    kron_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(kron_dir)
    logger.info(
        f"model={args.model_name} grad_dir={grad_dir} kron_dir={kron_dir} "
        f"layers=[{args.layer_start},{args.layer_end}) top_k={args.top_k} "
        f"max_grads={args.max_grads} grad_chunk_size={args.grad_chunk_size} "
        f"num_devices={args.num_devices} debug_matvec={args.debug_matvec}"
    )

    layer_to_files = load_grad_index(
        grad_dir, args.layer_start, args.layer_end, logger,
    )
    logger.info(f"indexed {len(layer_to_files)} layer keys from {grad_dir}")
    for layer_name, files in layer_to_files.items():
        logger.info(f"  {layer_name}: {len(files)} grad file(s)")

    if not layer_to_files:
        logger.error(
            f"no layer keys found in [{args.layer_start},{args.layer_end}). "
            f"check --grad_dir / --layer_start / --layer_end"
        )
        sys.exit(1)

    ok = run_parallel_kron(
        layer_to_files=layer_to_files,
        kron_dir=kron_dir,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        top_k=args.top_k,
        grad_chunk_size=args.grad_chunk_size,
        max_grads=args.max_grads,
        debug_matvec=args.debug_matvec,
        num_devices=args.num_devices,
        logger=logger,
    )
    if not ok:
        logger.error("kron computation failed (worker exit codes non-zero)")
        sys.exit(1)

    logger.info(f"done. final peak RSS={peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
