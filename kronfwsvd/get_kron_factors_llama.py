import os
import logging
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Dict, List, Any

import cupy as cp
import torch
from cupyx.scipy.sparse.linalg import LinearOperator, svds
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm

from multiprocessing import SimpleQueue
from torch.multiprocessing import Process

from safetensors import safe_open, SafetensorError

import time



def setup_logger(save_dir: Path) -> logging.Logger:
    """
    Set up a logger that writes both to a file and to the console.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "kron_factors.log"
    logger = logging.getLogger("KroneckerLogger")
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def get_kron_factors_worker(
    layer_name: str,
    list_of_grads: List[Any],
    top_k: int = 1,
    device_id: int = 0,
    save_dir: str = "fisher_factors"
) -> bool:
    save_dir_path = Path(save_dir)
    logger = setup_logger(save_dir_path)

    print(f"[DEBUG] {layer_name} starting on GPU {device_id}")
    cp.cuda.Device(device_id).use()
    print(f"[DEBUG] {layer_name} active CuPy device: {cp.cuda.runtime.getDevice()}")
    start_time = time.time()
    try:
        m, n = list_of_grads[0].shape
        grad_vectors = [cp.asarray(grad).reshape(m, n, order="F") for grad in list_of_grads]
        k = len(grad_vectors)
        print (layer_name, k)
    
        def matvec(vec):
            print ("matvec", flush = True)
            V = vec.reshape(n, n, order="F")
            result = cp.zeros((m, m), dtype=cp.float32)
            for G in grad_vectors:
                result += G @ V @ G.T
            return (result / k).T.ravel()

        def r_matvec(vec):
            V = vec.reshape(m, m, order="F")
            result = cp.zeros((n, n), dtype=cp.float32)
            for G in grad_vectors:
                result += G.T @ V @ G
            return (result / k).T.ravel()
        
        
        start_time = time.time()

        linop = LinearOperator(
            shape=(m * m, n * n),
            matvec=matvec,
            rmatvec=r_matvec,
            dtype=cp.float32,
        )

        u, s, vt = svds(linop, k=top_k, return_singular_vectors=True)
        
        #sidx = cp.argsort(-s)
        #s = s[sidx]
        #u = u[:, sidx]
        #v = vt[sidx, :].T

        #XF = (u[:, 0] * s[0]).reshape(m, m, order="F").get()
        #YF = v[:, 0].reshape(n, n, order="F").get()

        
        sidx = cp.argsort(-s)
        s = s[sidx]
        u = u[:, sidx]
        v = vt[sidx, :].T

        print(f"✔ Layer {layer_name} on device {device_id} done | singular values: {s}")

        XF = (u[:, 0] * s[0]).reshape(m, m, order="F")
        YF = v[:, 0].reshape(n, n, order="F")

        XF_tensor = torch.from_numpy(XF.get())
        YF_tensor = torch.from_numpy(YF.get())
        s_tensor = torch.from_numpy(s.get())

        end_time = time.time()
        print(f"Время выполнения: {end_time - start_time:.2f} секунд")


        #save_file(
        #    {"XF": XF_tensor, "YF": YF_tensor, "s": s_tensor},
        #    str(save_dir_path / f"{layer_name.replace('.', '_')}.safetensors")
        #)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The task took {elapsed_time:.2f} seconds to complete.")
        logger.info(
            f"✔ Saved factors for {layer_name} on device {device_id} | top singular value: {s[0]:.4f}"
        )
        return True

    except Exception as e:
        logger.error(f"Error in layer {layer_name} on device {device_id}")
        logger.error(traceback.format_exc())
        return False

    finally:
        cp.get_default_memory_pool().free_all_blocks()


def load_all_gradients(base_path: str, model_name: str) -> Dict[str, List[Any]]:
    base_dir = Path(base_path) / model_name
    grad_filenames = sorted(
        [f for f in os.listdir(base_dir) if f.endswith(".safetensors") and (base_dir / f).is_file()]
    )

    all_grads: Dict[str, List[Any]] = {}
    for grad_filename in tqdm(grad_filenames, desc="Loading gradients"):
        try:
            grad_file = load_file(filename=str(base_dir / grad_filename))
            for layer_name, gradient_tensor in grad_file.items():
                all_grads.setdefault(layer_name, []).append(gradient_tensor)
        except SafetensorError as e:
            print(f"[WARNING] Failed to load {grad_filename}: {e}")
            continue  # optionally skip to the next file

        except Exception as e:
            print(f"[ERROR] Unexpected error loading {grad_filename}: {e}")
            continue
    return all_grads


from torch.multiprocessing import Process, Queue
import torch

def gpu_worker(gpu_id: int, task_queue: SimpleQueue, top_k: int, save_dir: str):
    cp.cuda.Device(gpu_id).use()
    print(f"[WORKER {gpu_id}] started", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            print(f"[WORKER {gpu_id}] exiting.", flush=True)
            break

        layer_name, grads = task
        try:
            result = get_kron_factors_worker(
                layer_name=layer_name,
                list_of_grads=grads,
                top_k=top_k,
                device_id=gpu_id,
                save_dir=save_dir
            )
            print(f"[WORKER {gpu_id}] {layer_name} done: {result}", flush=True)
        except Exception as e:
            print(f"[WORKER {gpu_id}] ERROR processing {layer_name}: {e}", flush=True)
            traceback.print_exc()

from torch.multiprocessing import Process, SimpleQueue

def run_parallel_kron(
    all_grads: Dict[str, List[Any]],
    top_k: int = 1,
    num_devices: int = 4,
    save_dir: str = "fisher_factors"
) -> None:
    save_dir_path = Path(save_dir)
    logger = setup_logger(save_dir_path)
    logger.info(f"Starting parallel computation on {len(all_grads)} layers...")

    os.makedirs(save_dir, exist_ok=True)

    task_queue = SimpleQueue()
    processes = []

    for gpu_id in range(num_devices):
        p = Process(target=gpu_worker, args=(gpu_id, task_queue, top_k, save_dir))
        p.start()
        processes.append(p)

    # Enqueue tasks
    for layer_name, grad_tensors in all_grads.items():
        #grads_np = [g.to(dtype=torch.float32).cpu().numpy() for g in grad_tensors] 
        grads_np = [g.to(dtype=torch.float32).cpu().numpy() for g in grad_tensors[:100]]
        task_queue.put((layer_name, grads_np))

    for _ in range(num_devices):
        task_queue.put(None)  # stop signals

    for p in processes:
        p.join()

    logger.info("All GPU workers finished.")


def get_already_processed_layers(output_dir: Path) -> set:
    return {
        f.stem
        for f in output_dir.glob("*.safetensors")
        if f.name != "kron_factors.log"
    }

if __name__ == "__main__":
    base_path = "../yaqa-quantization/kronfwsvd/grads_output/"

    parser = argparse.ArgumentParser(description='Train a model with gradient collection.')
    parser.add_argument('--model_name', type=str, required=True, help='HF link to model to tune', default='unsloth/llama-2-7b')
    args = parser.parse_args()
    model_name = args.model_name

    top_k = 2
    max_workers = 1
    save_dir = Path(base_path) / model_name / "fisher_factors_output_150"
    
    logger = setup_logger(save_dir)
    logger.info(f"Loading gradients for {model_name}...")
    all_grads = load_all_gradients(base_path, model_name)

    filtered_grads = all_grads#{k: v for k, v in all_grads.items() if k.replace(".", "_") not in get_already_processed_layers(save_dir)}

    #logger.info(f"Found already processed layers, processing these layers: {filtered_grads.keys()}")

    logger.info("Summary of gradients:")
    for layer_name, grads in filtered_grads.items():
        logger.info(f"  Layer: {layer_name} — {len(grads)} grad(s) | Shape: {grads[0].shape}")

    logger.info(f"Launching Kronecker factor computation using {max_workers} workers...")
    run_parallel_kron(filtered_grads, top_k=top_k, num_devices=1, save_dir=str(save_dir))
    logger.info("Done.")
