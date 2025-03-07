import torch
import torch.multiprocessing as mp
import time
import numpy as np
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

def cpu_offload_worker(task_pipe):
    """Worker process using shared memory for in-place tensor updates."""
    try:
        while True:
            task = task_pipe.recv()
            if task is None:
                break  # Stop the worker

            task_start = time.time()
            print(f"[Worker] Task {task.task_id} started at {task_start}")

            # Load all tensors from shared memory
            x_remote_cpu, x_shm = load_shared_memory_tensor(task.x_shm_name, task.x_shape, task.x_dtype)
            topk_weights, topk_weights_shm = load_shared_memory_tensor(task.topk_weights_shm_name, task.topk_weights_shape, task.topk_weights_dtype)
            topk_ids, topk_ids_shm = load_shared_memory_tensor(task.topk_ids_shm_name, task.topk_ids_shape, task.topk_ids_dtype)

            # Perform CPU computation (in-place update on x_remote_cpu)
            # x_remote_cpu.mul_(2)  # Modify in-place

            task_end = time.time()
            print(f"[Worker] Task {task.task_id} completed at {task_end}")

            # Notify the main process that computation is done
            task_pipe.send((task.task_id, task_start, task_end))

            # Cleanup: Close shared memory objects
            x_shm.close()
            topk_weights_shm.close()
            topk_ids_shm.close()

    finally:
        print("[Worker] Cleaning up before exit")
        task_pipe.close()

def main():
    mp.set_start_method("fork")
    num_tasks = 10
    parent_task_pipe, child_task_pipe = mp.Pipe()

    worker = mp.Process(target=cpu_offload_worker, args=(child_task_pipe,))
    worker.start()

    task_timestamps = {}
    shared_memories = {}

    try:
        # Send tasks using shared memory
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

            task_sent = time.time()
            parent_task_pipe.send(task)
            task_timestamps[i] = {'task_sent': task_sent, 'x_shm_name': x_shm.name}

            shared_memories[i] = (x_shm, topk_weights_shm, topk_ids_shm)  # Store references for cleanup

        # Receive results
        for _ in range(num_tasks):
            task_id, task_start, task_end = parent_task_pipe.recv()
            task_received = time.time()
            task_timestamps[task_id].update({
                'task_start': task_start,
                'task_end': task_end,
                'task_received': task_received
            })

            # Load and verify in-place updated result (x_remote_cpu is now the modified tensor)
            result_tensor, shm = load_shared_memory_tensor(task_timestamps[task_id]['x_shm_name'], (100, 100), torch.float32)
            print(f"Received Task {task_id}, result shape: {result_tensor.shape}, first element: {result_tensor[0, 0]}")

            # Cleanup: Unlink ONLY in the main process after last use
            shm.close()
            shm.unlink()  # Unlink only after main process has read it

        # Stop the worker
        parent_task_pipe.send(None)
        parent_task_pipe.close()
        worker.join()

    finally:
        print("[Main] Cleaning up shared memory objects")
        for shm_tuple in shared_memories.values():
            for shm in shm_tuple:
                try:
                    shm.close()
                    shm.unlink()  # Ensure no leaks
                except FileNotFoundError:
                    pass  # If already unlinked, ignore

    # Print timing results
    for task_id, timestamps in task_timestamps.items():
        print(f"Task {task_id}: Sent at {timestamps['task_sent']}, "
              f"Started at {timestamps['task_start']}, Ended at {timestamps['task_end']}, "
              f"Received at {timestamps['task_received']}, "
              f"Total round-trip latency: {timestamps['task_received'] - timestamps['task_sent']:.4f}s"
              f"Sent latency: {timestamps['task_start'] - timestamps['task_sent']:.4f}s "
                f"Compute latency: {timestamps['task_end'] - timestamps['task_start']:.4f}s "
                f"Recv latency: {timestamps['task_received'] - timestamps['task_end']:.4f}s ")

if __name__ == "__main__":
    main()