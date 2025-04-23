import torch
import torch.multiprocessing as mp
import time
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from dataclasses import dataclass

@dataclass
class Task:
    task_id: int
    layer_id: int
    shm_name: str
    shape: tuple
    dtype: torch.dtype

def create_shared_memory_tensor(tensor):
    """Create shared memory and store a tensor in it."""
    shm = SharedMemory(create=True, size=tensor.numel() * tensor.element_size())
    np_array = np.ndarray(tensor.shape, dtype=np.float32, buffer=shm.buf)
    np_array[:] = tensor.numpy()  # Copy tensor data into shared memory
    return shm

def load_shared_memory_tensor(shm_name, shape, dtype):
    """Load a tensor from shared memory."""
    shm = SharedMemory(name=shm_name)
    np_array = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
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

            # Load data from shared memory (in-place modification)
            x_remote_cpu, shm = load_shared_memory_tensor(task.shm_name, task.shape, task.dtype)

            # Perform CPU computation (modify tensor in-place)
            x_remote_cpu.mul_(2)  # In-place update

            task_end = time.time()
            print(f"[Worker] Task {task.task_id} completed at {task_end}")

            # Notify the main process that computation is done
            task_pipe.send((task.task_id, task_start, task_end))

            # Cleanup shared memory reference
            shm.close()  # Don't unlink yet, main process still needs it

    finally:
        print("[Worker] Cleaning up before exit")
        task_pipe.close()

def main():
    num_tasks = 10
    parent_task_pipe, child_task_pipe = mp.Pipe()

    worker = mp.Process(target=cpu_offload_worker, args=(child_task_pipe,))
    worker.start()

    task_timestamps = {}
    shared_memories = {}

    try:
        # Send tasks using shared memory
        for i in range(num_tasks):
            tensor = torch.rand(100, 100)
            shm = create_shared_memory_tensor(tensor)
            
            task = Task(i, 0, shm.name, tensor.shape, tensor.dtype)
            task_sent = time.time()
            parent_task_pipe.send(task)
            task_timestamps[i] = {'task_sent': task_sent, 'shm_name': shm.name}
            shared_memories[i] = shm  # Store reference for cleanup

        # Receive results
        for _ in range(num_tasks):
            task_id, task_start, task_end = parent_task_pipe.recv()
            task_received = time.time()
            task_timestamps[task_id].update({
                'task_start': task_start,
                'task_end': task_end,
                'task_received': task_received
            })

            # Load and verify result (which is already modified in-place)
            result_tensor, shm = load_shared_memory_tensor(task_timestamps[task_id]['shm_name'], (100, 100), torch.float32)
            print(f"Received Task {task_id}, result shape: {result_tensor.shape}, first element: {result_tensor[0, 0]}")

            # Cleanup shared memory
            shm.close()
            shm.unlink()  # Now safe to unlink after main process reads it

    finally:
        print("[Main] Cleaning up shared memory objects")
        for shm in shared_memories.values():
            shm.close()
            # shm.unlink()  # Ensure no leaks

        # Stop the worker process
        parent_task_pipe.send(None)
        parent_task_pipe.close()
        worker.join()

    # Print timing results
    for task_id, timestamps in task_timestamps.items():
        print(f"Task {task_id}: Sent at {timestamps['task_sent']}, "
              f"Started at {timestamps['task_start']}, Ended at {timestamps['task_end']}, "
              f"Received at {timestamps['task_received']}, "
              f"Total round-trip latency: {timestamps['task_received'] - timestamps['task_sent']:.4f}s")

if __name__ == "__main__":
    main()
