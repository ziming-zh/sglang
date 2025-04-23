import torch
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass

# Mapping PyTorch dtypes to NumPy dtypes
TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
}

@dataclass
class Task:
    task_id: int
    layer_id: int
    x_shm_name: str
    x_shape: tuple
    x_dtype: torch.dtype
    topk_weights_shm_name: str
    topk_weights_shape: tuple
    topk_weights_dtype: torch.dtype
    topk_ids_shm_name: str
    topk_ids_shape: tuple
    topk_ids_dtype: torch.dtype

def create_shared_memory_tensor(tensor):
    """Create shared memory and store a tensor in it."""
    shm = SharedMemory(create=True, size=tensor.numel() * tensor.element_size())
    np_array = np.ndarray(tensor.shape, dtype=TORCH_TO_NUMPY_DTYPE[tensor.dtype], buffer=shm.buf)
    np_array[:] = tensor.numpy()  # Copy tensor data into shared memory
    return shm

def load_shared_memory_tensor(shm_name, shape, dtype):
    """Load a tensor from shared memory."""
    np_dtype = TORCH_TO_NUMPY_DTYPE[dtype]  # Convert torch dtype to numpy dtype
    shm = SharedMemory(name=shm_name)
    np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
    return torch.tensor(np_array, dtype=dtype), shm

def cpu_offload_worker(task):
    """Worker function that processes a single task (runs in a separate process)."""
    try:
        task_start = time.time()
        print(f"[Worker] Task {task.task_id} started at {task_start}")

        # Load all tensors from shared memory
        x_remote_cpu, x_shm = load_shared_memory_tensor(task.x_shm_name, task.x_shape, task.x_dtype)
        topk_weights, topk_weights_shm = load_shared_memory_tensor(task.topk_weights_shm_name, task.topk_weights_shape, task.topk_weights_dtype)
        topk_ids, topk_ids_shm = load_shared_memory_tensor(task.topk_ids_shm_name, task.topk_ids_shape, task.topk_ids_dtype)

        # Perform CPU computation (in-place update on x_remote_cpu)
        x_remote_cpu.mul_(2)  # Modify in-place

        task_end = time.time()
        print(f"[Worker] Task {task.task_id} completed at {task_end}")

        # Cleanup: Only close, DO NOT unlink (main process will unlink)
        x_shm.close()
        topk_weights_shm.close()
        topk_ids_shm.close()

        return (task.task_id, task_start, task_end)
    except Exception as e:
        print(f"[Worker Error] Task {task.task_id} failed: {e}")
        return (task.task_id, None, None)

def main():
    num_tasks = 10
    shared_memories = {}

    with ProcessPoolExecutor() as executor:
        task_futures = {}

        # Submit tasks using shared memory
        for i in range(num_tasks):
            x_remote_cpu = torch.rand(100, 100, dtype=torch.float32)
            topk_weights_remote_cpu = torch.rand(10, dtype=torch.float32)
            topk_ids_remote_cpu = torch.randint(0, 100, (10,), dtype=torch.int32)

            x_shm = create_shared_memory_tensor(x_remote_cpu)
            topk_weights_shm = create_shared_memory_tensor(topk_weights_remote_cpu)
            topk_ids_shm = create_shared_memory_tensor(topk_ids_remote_cpu)

            task = Task(
                task_id=i, layer_id=0,
                x_shm_name=x_shm.name, x_shape=x_remote_cpu.shape, x_dtype=x_remote_cpu.dtype,
                topk_weights_shm_name=topk_weights_shm.name, topk_weights_shape=topk_weights_remote_cpu.shape, topk_weights_dtype=topk_weights_remote_cpu.dtype,
                topk_ids_shm_name=topk_ids_shm.name, topk_ids_shape=topk_ids_remote_cpu.shape, topk_ids_dtype=topk_ids_remote_cpu.dtype
            )

            # Submit task to process pool
            future = executor.submit(cpu_offload_worker, task)
            task_futures[i] = future
            shared_memories[i] = (x_shm, topk_weights_shm, topk_ids_shm)

        # Retrieve results
        task_timestamps = {}
        for task_id, future in task_futures.items():
            task_id, task_start, task_end = future.result()
            task_received = time.time()
            task_timestamps[task_id] = {
                'task_sent': task_timestamps.get(task_id, {}).get('task_sent', None),
                'task_start': task_start,
                'task_end': task_end,
                'task_received': task_received
            }

            # Load and verify in-place updated result
            result_tensor, shm = load_shared_memory_tensor(shared_memories[task_id][0].name, (100, 100), torch.float32)
            print(f"Received Task {task_id}, result shape: {result_tensor.shape}, first element: {result_tensor[0, 0]}")

            # Cleanup shared memory
            shm.close()
            shm.unlink()  # Unlink after reading

        # Cleanup shared memory
        for shm_tuple in shared_memories.values():
            for shm in shm_tuple:
                try:
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass  # If already unlinked, ignore

    # Print timing results
    for task_id, timestamps in task_timestamps.items():
        print(f"Task {task_id}: "
              f"Sent at {timestamps['task_sent']}, "
              f"Started at {timestamps['task_start']}, Ended at {timestamps['task_end']}, "
              f"Received at {timestamps['task_received']}, "
              f"Total round-trip latency: {timestamps['task_received'] - timestamps['task_start']:.4f}s")

if __name__ == "__main__":
    main()
