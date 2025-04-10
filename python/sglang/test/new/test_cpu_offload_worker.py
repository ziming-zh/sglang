import torch
import torch.multiprocessing as mp
import time
from dataclasses import dataclass

@dataclass
class Task:
    task_id: int
    layer_id: int
    x_remote_cpu: torch.Tensor
    topk_weights_remote_cpu: torch.Tensor
    topk_ids_remote_cpu: torch.Tensor

@dataclass
class TaskResult:
    task_id: int
    layer_id: int
    cpu_result: torch.Tensor

def fused_experts_cpu_impl(hidden_states):
    """Mock CPU computation function to simulate workload."""
    time.sleep(0.01)  # Simulate some computation delay
    return hidden_states * 2  # Dummy computation

def cpu_offload_worker(task_queue, result_queue_list):
    """Worker function to handle CPU offloading."""
    while True:
        task = task_queue.get()
        if task is None:
            break  # Stop the worker when None is received

        task_start = time.time()
        print(f"[Offload Worker] Task {task.task_id} started at {task_start}")

        # Perform CPU computation
        cpu_result = fused_experts_cpu_impl(task.x_remote_cpu).share_memory_()

        task_end = time.time()
        print(f"[Offload Worker] Task {task.task_id} completed at {task_end}")

        # Store the result
        result_queue = result_queue_list[task.layer_id]
        if not result_queue.full():
            task_result = TaskResult(task.task_id, task.layer_id, cpu_result)
            result_queue.put((task_result, task_start, task_end))
        else:
            print(f"Result queue full, dropping task {task.task_id}")

def main():

    num_tasks = 10
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    result_queue_list = {0: result_queue}

    worker = mp.Process(target=cpu_offload_worker, args=(task_queue, result_queue_list))
    worker.start()

    task_timestamps = {}

    # Send tasks to the worker and record timestamps
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            layer_id=0,
            x_remote_cpu=torch.rand(10).share_memory_(),
            topk_weights_remote_cpu=torch.rand(10).share_memory_(),
            topk_ids_remote_cpu=torch.randint(0, 10, (1,), dtype=torch.int64).share_memory_()
        )
        task_sent = time.time()
        task_queue.put(task)
        task_sent_complete = time.time()
        task_timestamps[i] = {'task_sent': task_sent, 'task_sent_complete': task_sent_complete}

    # Receive results and record timestamps
    for _ in range(num_tasks):
        task_result, task_start, task_end = result_queue.get()
        task_received = time.time()
        task_timestamps[task_result.task_id].update({
            'task_start': task_start,
            'task_end': task_end,
            'task_received': task_received
        })

    # Stop the worker
    task_queue.put(None)
    worker.join()

    # Print results
    print("\nTask Timing Results:")
    for task_id, timestamps in task_timestamps.items():
        print(f"Task {task_id}: Sent at {timestamps['task_sent']}, "
              f"Sent complete at {timestamps['task_sent_complete']}, "
              f"Started at {timestamps['task_start']}, Ended at {timestamps['task_end']}, "
              f"Received at {timestamps['task_received']}, "
              f"Total round-trip latency: {timestamps['task_received'] - timestamps['task_sent']:.4f}s")

if __name__ == "__main__":
    main()
