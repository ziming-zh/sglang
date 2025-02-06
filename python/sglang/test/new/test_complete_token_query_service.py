import multiprocessing as mp
from multiprocessing import shared_memory, Manager, Lock

class CompleteTokenQueryService:
    def __init__(self, tp_rank_range, num_layers, manager):
        self.tp_rank_range = tp_rank_range
        self.lock = manager.Lock()  # Manager Lock for atomic access
        self.compute_cnt = manager.dict()  # Shared dictionary for token computation counts

        # Dictionary-based cache: cache_key[layer_id] and cache_value[layer_id]
        self.cache_key = manager.dict({layer_id: f"layer_{layer_id}" for layer_id in range(num_layers)})
        self.cache_value = manager.dict({layer_id: [] for layer_id in range(num_layers)})

    def update_token(self, token):
        """Atomically update the computation count for a specific token."""
        with self.lock:
            if token in self.compute_cnt:
                self.compute_cnt[token] += 1
            else:
                self.compute_cnt[token] = 1
            print(f"Token {token} updated: {self.compute_cnt[token]}")

    def query(self, round_id, layer_id):
        """Query completed tokens for a specific layer_id key."""
        key = f"{round_id}.{layer_id}"

        with self.lock:
            # Cache hit: Return stored result
            if self.cache_key.get(layer_id) == key:
                return list(self.cache_value[layer_id])  # Convert to list for safety

            # Cache miss: Compute the new value based on compute_cnt
            finished_tokens = [token for token, count in self.compute_cnt.items() if count == self.tp_rank_range]

            # Update cache entry for the layer
            self.cache_key[layer_id] = key
            self.cache_value[layer_id] = finished_tokens  # Store finished tokens for this layer

            return finished_tokens


def run_scheduler_process(server_args, port_args, gpu_id, tp_rank_range, shared_service, writer):
    """Function to be executed in a separate process."""
    for tp_rank in range(tp_rank_range):
        # Simulate token updates
        shared_service.update_token(f"token_{tp_rank}")
    import time; time.sleep(1)
    # Query using (round_id, layer_id)
    result = shared_service.query(1, 1)  # Example query
    writer.send(result)  # Send the result back via Pipe
    writer.close()

if __name__ == "__main__":
    tp_rank_range = 3  # Example range
    manager = mp.Manager()
    num_layers = 2  # Example number of layers
    service = CompleteTokenQueryService(tp_rank_range, num_layers, manager)

    # Launch scheduler processes
    tp_size_per_node = 4  # Example TP size
    base_gpu_id = 0  # Example base GPU ID
    server_args, port_args = None, None  # Placeholder for arguments

    scheduler_procs = []
    scheduler_pipe_readers = []

    for tp_rank in range(tp_rank_range):
        reader, writer = mp.Pipe(duplex=False)
        gpu_id = base_gpu_id + tp_rank % tp_size_per_node
        proc = mp.Process(
            target=run_scheduler_process,
            args=(server_args, port_args, gpu_id, tp_rank_range, service, writer),
        )
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    # Collect results
    for reader in scheduler_pipe_readers:
        print("Result:", reader.recv())

    # Wait for all processes to finish
    for proc in scheduler_procs:
        proc.join()
