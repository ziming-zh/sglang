import torch
import time
import statistics
import threading

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

    def exchange_kv_buffer_async(
        self,
        peer_kv_pool: "MHATokenToKVPool",
        start_layer_idx: int,
        end_layer_idx: int,
    ):
        if len(self.k_buffer) != len(peer_kv_pool.k_buffer):
            raise ValueError("Mismatch in layer count between KV pools.")
        if self.k_buffer[0].shape[1:] != peer_kv_pool.k_buffer[0].shape[1:]:
            raise ValueError("Head number and dimension mismatch.")
        if self.store_dtype != peer_kv_pool.store_dtype:
            print(f"Warning: dtype mismatch ({self.store_dtype} vs {peer_kv_pool.store_dtype})")

        self_device = self.k_buffer[0].device
        peer_device = peer_kv_pool.k_buffer[0].device

        is_self_cpu = self_device.type == "cpu"
        is_peer_cpu = peer_device.type == "cpu"

        use_streams = not (is_self_cpu or is_peer_cpu)

        streams = []
        if use_streams:
            for _ in range(start_layer_idx, end_layer_idx):
                streams.append((
                    torch.cuda.Stream(device=self_device),
                    torch.cuda.Stream(device=peer_device),
                ))
        else:
            streams = [(None, None)] * (end_layer_idx - start_layer_idx)

        for layer_id, (stream_self, stream_peer) in zip(range(start_layer_idx, end_layer_idx), streams):
            self_k = self.k_buffer[layer_id]
            self_v = self.v_buffer[layer_id]
            peer_k = peer_kv_pool.k_buffer[layer_id]
            peer_v = peer_kv_pool.v_buffer[layer_id]

            if use_streams:
                # GPU ↔ GPU: async transfers with streams
                with torch.cuda.device(self_device), torch.cuda.stream(stream_self):
                    tmp_k_peer = peer_k.to(self_device, non_blocking=True)
                    tmp_v_peer = peer_v.to(self_device, non_blocking=True)
                    self.k_buffer[layer_id].copy_(tmp_k_peer)
                    self.v_buffer[layer_id].copy_(tmp_v_peer)

                with torch.cuda.device(peer_device), torch.cuda.stream(stream_peer):
                    tmp_k_self = self_k.to(peer_device, non_blocking=True)
                    tmp_v_self = self_v.to(peer_device, non_blocking=True)
                    peer_k.copy_(tmp_k_self)
                    peer_v.copy_(tmp_v_self)

            else:
                # At least one side is CPU → do CPU-staged transfer with pinned memory
                dtype = self.store_dtype
                shape_k = self_k.shape
                shape_v = self_v.shape

                pinned_k1 = torch.empty(shape_k, dtype=dtype, device="cpu", pin_memory=True)
                pinned_v1 = torch.empty(shape_v, dtype=dtype, device="cpu", pin_memory=True)
                pinned_k2 = torch.empty(shape_k, dtype=dtype, device="cpu", pin_memory=True)
                pinned_v2 = torch.empty(shape_v, dtype=dtype, device="cpu", pin_memory=True)

                # Device → pinned
                if not is_peer_cpu:
                    pinned_k1.copy_(peer_k.to("cpu", non_blocking=True))
                    pinned_v1.copy_(peer_v.to("cpu", non_blocking=True))
                else:
                    pinned_k1.copy_(peer_k)
                    pinned_v1.copy_(peer_v)

                if not is_self_cpu:
                    pinned_k2.copy_(self_k.to("cpu", non_blocking=True))
                    pinned_v2.copy_(self_v.to("cpu", non_blocking=True))
                else:
                    pinned_k2.copy_(self_k)
                    pinned_v2.copy_(self_v)

                # pinned → device
                if not is_self_cpu:
                    self.k_buffer[layer_id] = pinned_k1.to(self_device, non_blocking=True)
                    self.v_buffer[layer_id] = pinned_v1.to(self_device, non_blocking=True)
                else:
                    self.k_buffer[layer_id] = pinned_k1
                    self.v_buffer[layer_id] = pinned_v1

                if not is_peer_cpu:
                    peer_kv_pool.k_buffer[layer_id] = pinned_k2.to(peer_device, non_blocking=True)
                    peer_kv_pool.v_buffer[layer_id] = pinned_v2.to(peer_device, non_blocking=True)
                else:
                    peer_kv_pool.k_buffer[layer_id] = pinned_k2
                    peer_kv_pool.v_buffer[layer_id] = pinned_v2

        # Synchronize only for GPU ↔ GPU
        if use_streams:
            for s1, s2 in streams:
                s1.synchronize()
                s2.synchronize()

        print(f"[KV SWAP] Swapped layers [{start_layer_idx}, {end_layer_idx}) between {self_device} and {peer_device}")


def run_test(
    layer_num = 32,
    stride_num = 8,
    head_num = 2,
    head_dim = 128,
    size = 29725,
    dtype = torch.bfloat16,
    num_gpus = 8,
):
    device_a = "cuda:0"
    device_b = f"cuda:{num_gpus//2}"

    print("Allocating KV pools...")
    kv_pool_a = MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, device_a)
    kv_pool_b = MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, device_b)

    total_elements = 2 * layer_num * (size + 1) * head_num * head_dim * 2
    total_gb = total_elements * torch.tensor([], dtype=dtype).element_size() * stride_num / layer_num / 1e9

    times = []

    print("\nStarting benchmark (11 runs, discard first)...")
    for i in range(11):
        torch.cuda.synchronize()
        start_time = time.time()

        kv_pool_a.exchange_kv_buffer_async(kv_pool_b, 0, stride_num)

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        tag = "(warm-up)" if i == 0 else ""
        print(f"[Run {i+1}] Transfer Time: {elapsed:.6f} s {tag}")

    # Drop the first timing
    usable_times = times[1:]
    avg_time = statistics.mean(usable_times)
    min_time = min(usable_times)
    max_time = max(usable_times)
    stddev_time = statistics.stdev(usable_times)
    avg_bandwidth = total_gb / avg_time

    print(f"\n=== Benchmark Summary (10 runs, excluding warm-up) ===")
    print(f"Transferred: {total_gb:.2f} GB")
    print(f"Average Time: {avg_time:.6f} s")
    print(f"Min Time:     {min_time:.6f} s")
    print(f"Max Time:     {max_time:.6f} s")
    print(f"Std Dev:      {stddev_time:.6f} s")
    print(f"Average Bandwidth: {avg_bandwidth:.2f} GB/s")


def parallel_exchange(source_pools, target_pools, stride_num):
    threads = []

    for i in range(len(source_pools)):
        t = threading.Thread(
            target=source_pools[i].exchange_kv_buffer_async,
            args=(target_pools[i], 0, stride_num),
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


def run_parallel_test(
    layer_num = 32,
    stride_num = 4,
    head_num = 2,
    head_dim = 128,
    size = 29725,
    dtype = torch.bfloat16,
    num_gpus = 8,
):

    print(f"Allocating source KV pools on cuda:0 to cuda{num_gpus//2-1}:...")
    source_pools = [
        MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, f"cuda:{i}")
        for i in range(num_gpus//2)
    ]

    print(f"Allocating target KV pools on cuda:{num_gpus//2}...")
    target_pools = [
        MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, f"cuda:{i+num_gpus//2}")
        for i in range(num_gpus//2)
    ]

    total_elements = (
        (num_gpus//2) * 2 * stride_num * (size + 1) * head_num * head_dim * 2  # num_gpus//2 exchanges, K+V, 2 directions
    )
    total_gb = total_elements * torch.tensor([], dtype=dtype).element_size() / 1e9

    times = []

    print("\nStarting parallel benchmark (11 runs, discard first)...")
    for i in range(11):
        torch.cuda.synchronize()
        start_time = time.time()

        parallel_exchange(source_pools, target_pools, stride_num)

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        tag = "(warm-up)" if i == 0 else ""
        print(f"[Run {i+1}] Total Transfer Time: {elapsed:.6f} s {tag}")

    usable_times = times[1:]
    avg_time = statistics.mean(usable_times)
    min_time = min(usable_times)
    max_time = max(usable_times)
    stddev_time = statistics.stdev(usable_times)
    avg_bandwidth = total_gb / avg_time

    print(f"\n=== Parallel Benchmark Summary (10 runs, excluding warm-up) ===")
    print(f"Transferred: {total_gb:.2f} GB (total, num_gpus//2 exchanges)")
    print(f"Average Time: {avg_time:.6f} s")
    print(f"Min Time:     {min_time:.6f} s")
    print(f"Max Time:     {max_time:.6f} s")
    print(f"Std Dev:      {stddev_time:.6f} s")
    print(f"Average Bandwidth: {avg_bandwidth:.2f} GB/s")
    
def run_parallel_test_cpu(
    layer_num = 32,
    stride_num = 4,
    head_num = 2,
    head_dim = 128,
    size = 29725,
    dtype = torch.bfloat16,
    num_gpus = 8,
):

    print("Allocating source KV pools on cuda:0 to cuda:3...")
    source_pools = [
        MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, f"cuda:{i}")
        for i in range(num_gpus//2)
    ]

    print("Allocating target KV pools on cpu...")
    target_pools = [
        MHATokenToKVPool(size, dtype, head_num, head_dim, layer_num, "cpu")
        for _ in range(num_gpus//2)
    ]

    total_elements = (
        num_gpus//2 * 2 * stride_num * (size + 1) * head_num * head_dim * 2  # num_gpus//2 exchanges, K+V, 2 directions
    )
    total_gb = total_elements * torch.tensor([], dtype=dtype).element_size() / 1e9

    times = []

    print("\nStarting parallel benchmark (11 runs, discard first)...")
    for i in range(11):
        torch.cuda.synchronize()
        start_time = time.time()

        parallel_exchange(source_pools, target_pools, stride_num)

        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        tag = "(warm-up)" if i == 0 else ""
        print(f"[Run {i+1}] Total Transfer Time: {elapsed:.6f} s {tag}")

    usable_times = times[1:]
    avg_time = statistics.mean(usable_times)
    min_time = min(usable_times)
    max_time = max(usable_times)
    stddev_time = statistics.stdev(usable_times)
    avg_bandwidth = total_gb / avg_time

    print(f"\n=== Parallel Benchmark Summary (10 runs, excluding warm-up) ===")
    print(f"Transferred: {total_gb:.2f} GB (total, num_gpus//2 exchanges)")
    print(f"Average Time: {avg_time:.6f} s")
    print(f"Min Time:     {min_time:.6f} s")
    print(f"Max Time:     {max_time:.6f} s")
    print(f"Std Dev:      {stddev_time:.6f} s")
    print(f"Average Bandwidth: {avg_bandwidth:.2f} GB/s")
    

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    # run_test(num_gpus=num_gpus)
    run_parallel_test(num_gpus=num_gpus)
    # run_parallel_test_cpu(num_gpus=num_gpus)