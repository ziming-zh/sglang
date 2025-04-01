from typing import List, Tuple
import torch
import time
import statistics
import threading
import random


class BaseTokenToKVPool:
    def __init__(self, size, dtype, device):
        self.store_dtype = dtype
        self.device = device
        self.size = size


class MHATokenToKVPool(BaseTokenToKVPool):
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        super().__init__(size, dtype, device)

        self.k_buffer = [
            torch.empty(
                (size + 1, head_num, head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.empty(
                (size + 1, head_num, head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]
        

def swap_inactive_requests(
    pool1: "MHATokenToKVPool",
    pool2: "MHATokenToKVPool",
    layer_id: int,
    stride_num: int,
    start_locs_1: List[int],
    end_loc_1: int,
    is_active_1: List[bool],
    start_locs_2: List[int],
    end_loc_2: int,
    is_active_2: List[bool],
) -> Tuple[int, int, List[int], List[int]]:
    """
    Swaps inactive requests between two KV pools in one shot and places them at the back.
    Returns updated end_loc_1, end_loc_2, start_locs_1, start_locs_2
    """
    def collect_spans(start_locs, is_active, end_loc):
        spans = []
        for i, active in enumerate(is_active):
            if not active:
                start = start_locs[i]
                end = start_locs[i + 1] if i + 1 < len(start_locs) else end_loc
                spans.append((i, start, end))
        return spans

    k1_list, v1_list = pool1.k_buffer[layer_id:layer_id + stride_num], pool1.v_buffer[layer_id:layer_id + stride_num]
    k2_list, v2_list = pool2.k_buffer[layer_id:layer_id + stride_num], pool2.v_buffer[layer_id:layer_id + stride_num]


    spans_1 = collect_spans(start_locs_1, is_active_1, end_loc_1)
    spans_2 = collect_spans(start_locs_2, is_active_2, end_loc_2)

    # Flatten and concatenate all inactive chunks
    def flatten_spans(tensor, spans):
        return torch.cat([tensor[start:end] for _, start, end in spans], dim=0)
    for k1, v1, k2, v2 in zip(k1_list, v1_list, k2_list, v2_list):
        dev1, dev2 = k1.device, k2.device
        k1_chunks = flatten_spans(k1, spans_1)
        v1_chunks = flatten_spans(v1, spans_1)
        k2_chunks = flatten_spans(k2, spans_2)
        v2_chunks = flatten_spans(v2, spans_2)
        print(f"Inactive chunks: {len(k1_chunks)}, {len(k2_chunks)}")

        # Swap chunks
        k1_new = k2_chunks.to(dev1, non_blocking=True)
        v1_new = v2_chunks.to(dev1, non_blocking=True)
        k2_new = k1_chunks.to(dev2, non_blocking=True)
        v2_new = v1_chunks.to(dev2, non_blocking=True)

        # Append to tail
        new_start_locs_1 = start_locs_1.copy()
        new_start_locs_2 = start_locs_2.copy()
        curr1 = end_loc_1 - len(k1_chunks)
        curr2 = end_loc_2 - len(k2_chunks)
        
        print(f"k1 shape: {k1.shape}, k2 shape: {k2.shape}")
        print(f"v1 shape: {v1.shape}, v2 shape: {v2.shape}")
        print(f"curr1: {curr1}, curr2: {curr2}")

        if k1_new.numel() > 0:
            k1[curr1:curr1 + k1_new.shape[0]].copy_(k1_new)
            v1[curr1:curr1 + v1_new.shape[0]].copy_(v1_new)
            # new_start_locs_1.extend(range(curr1, curr1 + k1_new.shape[0], 200))
            # curr1 += k1_new.shape[0]

        if k2_new.numel() > 0:
            k2[curr2:curr2 + k2_new.shape[0]].copy_(k2_new)
            v2[curr2:curr2 + v2_new.shape[0]].copy_(v2_new)
            # new_start_locs_2.extend(range(curr2, curr2 + k2_new.shape[0], 200))
            # curr2 += k2_new.shape[0]

    return curr1, curr2, new_start_locs_1, new_start_locs_2


def simulate_request_metadata(size: int, avg_len: int = 200, num_requests: int = 100):
    """
    Simulates a list of start_locs, an end_loc, and a parallel list of is_active bools.
    Total number of tokens = `size`.
    """
    start_locs = [0]
    is_active = []
    curr = 0
    for _ in range(num_requests):
        if curr >= size:
            break
        length = avg_len + random.randint(-10, 10)
        start_locs.append(min(curr + length, size))
        is_active.append(random.random() < 0.5)  # ~50% active
        curr = start_locs[-1]
    return start_locs[:-1], start_locs[-1], is_active

def inactive_swap_wrapper(pool1, pool2, layer_id, size, stride_num):
    start_locs_1, end_loc_1, is_active_1 = simulate_request_metadata(size)
    start_locs_2, end_loc_2, is_active_2 = simulate_request_metadata(size)

    swap_inactive_requests(
        pool1, pool2, layer_id, stride_num,
        start_locs_1, end_loc_1, is_active_1,
        start_locs_2, end_loc_2, is_active_2
    )

def parallel_inactive_exchange(source_pools, target_pools, layer_id, size, stride_num):
    threads = []
    for i in range(len(source_pools)):
        t = threading.Thread(
            target=inactive_swap_wrapper,
            args=(source_pools[i], target_pools[i], layer_id, size, stride_num)
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def run_inactive_swap_test(
    layer_num=32,
    layer_id=0,
    head_num=4,
    head_dim=128,
    size=29725,
    dtype=torch.bfloat16,
    num_gpus=8,
    stride_num=8,
):

    print(f"Allocating source KV pools on cuda:0 to cuda{num_gpus//2-1}...")
    source_pools = [
        MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, f"cuda:{i}")
        for i in range(num_gpus // 2)
    ]

    print(f"Allocating target KV pools on cuda:{num_gpus//2} to cuda:{num_gpus-1}...")
    target_pools = [
        MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, f"cuda:{i + num_gpus // 2}")
        for i in range(num_gpus // 2)
    ]

    # Estimate roughly 50% of size * 2 directions * K and V
    total_entries = size // 2 * 2 * head_num * head_dim * 2 * (num_gpus // 2) * stride_num
    total_gb = total_entries * torch.tensor([], dtype=dtype).element_size() / 1e9

    times = []

    print("\nStarting inactive swap benchmark (11 runs, discard first)...")
    for i in range(11):
        torch.cuda.synchronize()
        start_time = time.time()

        parallel_inactive_exchange(source_pools, target_pools, layer_id, size, stride_num)

        torch.cuda.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)

        tag = "(warm-up)" if i == 0 else ""
        print(f"[Run {i+1}] Inactive Transfer Time: {elapsed:.6f} s {tag}")

    usable_times = times[1:]
    avg_time = statistics.mean(usable_times)
    min_time = min(usable_times)
    max_time = max(usable_times)
    stddev_time = statistics.stdev(usable_times)
    avg_bandwidth = total_gb / avg_time

    print(f"\n=== Inactive Swap Benchmark Summary (10 runs, excluding warm-up) ===")
    print(f"Transferred (estimated): {total_gb:.2f} GB")
    print(f"Average Time: {avg_time:.6f} s")
    print(f"Min Time:     {min_time:.6f} s")
    print(f"Max Time:     {max_time:.6f} s")
    print(f"Std Dev:      {stddev_time:.6f} s")
    print(f"Average Bandwidth: {avg_bandwidth:.2f} GB/s")

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    run_inactive_swap_test(num_gpus=num_gpus)