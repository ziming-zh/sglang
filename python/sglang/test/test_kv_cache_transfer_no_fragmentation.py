from typing import List, Tuple
import torch
import time
import statistics
import threading
import random


class BaseTokenToKVPool:
    """A memory pool that maps a token location to its kv cache data."""

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: str,
    ):
        self.size = size
        self.dtype = dtype
        if dtype == torch.float8_e5m2:
            # NOTE: Store as torch.uint8 because Tensor index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.free_slots = None
        self.is_not_in_free_group = True
        self.free_group = []
        self.clear()

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int):
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index.to(self.device, non_blocking=True)

    def free(self, free_index: torch.Tensor):
        if self.is_not_in_free_group:
            self.free_slots = torch.concat((self.free_slots, free_index.cpu()))
        else:
            self.free_group.append(free_index)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.concat(self.free_group))

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_slots = torch.arange(1, self.size + 1, dtype=torch.int32)
        self.is_in_free_group = False
        self.free_group = []


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
        
def swap_inactive_requests_token_indices(
    pool1: "MHATokenToKVPool",
    pool2: "MHATokenToKVPool",
    layer_id: int,
    stride_num: int,
    request_to_tokens_1: List[List[int]],
    is_active_1: List[bool],
    request_to_tokens_2: List[List[int]],
    is_active_2: List[bool],
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Swaps inactive requests between two KV pools and returns new request_to_tokens_* mappings.
    """

    def collect_inactive(request_to_tokens, is_active):
        return [(i, token_ids) for i, (token_ids, active) in enumerate(zip(request_to_tokens, is_active)) if not active]

    # Get inactive requests and their token indices
    spans_1 = collect_inactive(request_to_tokens_1, is_active_1)
    spans_2 = collect_inactive(request_to_tokens_2, is_active_2)

    # Flatten all token indices to be swapped
    flat_tokens_1 = torch.tensor([idx for _, toks in spans_1 for idx in toks], dtype=torch.int32)
    flat_tokens_2 = torch.tensor([idx for _, toks in spans_2 for idx in toks], dtype=torch.int32)

    # Allocate new locations
    new_locs_1 = pool1.alloc(len(flat_tokens_2))
    new_locs_2 = pool2.alloc(len(flat_tokens_1))

    if new_locs_1 is None or new_locs_2 is None:
        raise RuntimeError("Not enough memory to perform token swap.")

    # Free old locations
    if len(flat_tokens_1) > 0:
        pool1.free(flat_tokens_1)
    if len(flat_tokens_2) > 0:
        pool2.free(flat_tokens_2)

    # Copy actual KV data
    for k1, v1, k2, v2 in zip(
        pool1.k_buffer[layer_id:layer_id + stride_num],
        pool1.v_buffer[layer_id:layer_id + stride_num],
        pool2.k_buffer[layer_id:layer_id + stride_num],
        pool2.v_buffer[layer_id:layer_id + stride_num]
    ):
        k1[new_locs_1] = k2[flat_tokens_2.to(k2.device)].to(k1.device)
        v1[new_locs_1] = v2[flat_tokens_2.to(v2.device)].to(v1.device)
        k2[new_locs_2] = k1[flat_tokens_1.to(k1.device)].to(k2.device)
        v2[new_locs_2] = v1[flat_tokens_1.to(v1.device)].to(v2.device)

    # Rebuild request-to-token-index mapping
    def rebuild_mapping(old_mapping, spans, flat_locs):
        new_mapping = old_mapping.copy()
        offset = 0
        for req_id, toks in spans:
            length = len(toks)
            new_mapping[req_id] = flat_locs[offset:offset + length].tolist()
            offset += length
        return new_mapping

    new_request_to_tokens_1 = rebuild_mapping(request_to_tokens_1, spans_1, new_locs_1)
    new_request_to_tokens_2 = rebuild_mapping(request_to_tokens_2, spans_2, new_locs_2)

    return new_request_to_tokens_1, new_request_to_tokens_2


def simulate_request_to_tokens(size: int, avg_len: int = 200, num_requests: int = 50):
    """
    Simulates request-to-token mapping using dynamic alloc. 
    Returns:
        request_to_tokens: List[List[int]]
        is_active: List[bool]
    """
    request_to_tokens = []
    is_active = []
    curr = 1  # Start from 1 to avoid reserved padded slot
    for _ in range(num_requests):
        if curr >= size:
            break
        length = avg_len + random.randint(-10, 10)
        tokens = list(range(curr, min(curr + length, size)))
        request_to_tokens.append(tokens)
        is_active.append(random.random() < 0.5)
        curr += length
    return request_to_tokens, is_active

def inactive_swap_wrapper_token_index(pool1, pool2, layer_id, size, stride_num):
    request_to_tokens_1, is_active_1 = simulate_request_to_tokens(size)
    request_to_tokens_2, is_active_2 = simulate_request_to_tokens(size)

    swap_inactive_requests_token_indices(
        pool1, pool2, layer_id, stride_num,
        request_to_tokens_1, is_active_1,
        request_to_tokens_2, is_active_2
    )


def parallel_inactive_exchange_token_index(source_pools, target_pools, layer_id, size, stride_num):
    threads = []
    for i in range(len(source_pools)):
        t = threading.Thread(
            target=inactive_swap_wrapper_token_index,
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

        parallel_inactive_exchange_token_index(source_pools, target_pools, layer_id, size, stride_num)

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