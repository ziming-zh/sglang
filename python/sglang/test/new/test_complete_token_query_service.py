import multiprocessing as mp
from multiprocessing import shared_memory, Manager, Lock
import time

class CompleteTokenQueryService:
    def __init__(self, tp_rank_range, manager, num_layers=32):
        self.tp_rank_range = tp_rank_range
        self.locks = {layer_id: manager.Lock() for layer_id in range(num_layers)}

        # Dictionary-based cache: cache_key[layer_id] and cache_value[layer_id]
        self.cache_key = manager.dict({layer_id: f"layer_{layer_id}" for layer_id in range(num_layers)})
        self.cache_value = manager.dict({layer_id: [] for layer_id in range(num_layers)})

        # Each layer has its own compute_cnt dictionary
        self.compute_cnt = manager.dict({layer_id: manager.dict() for layer_id in range(num_layers)})

    def update_token(self, token, layer_id):
        """Atomically update the computation count for a specific token in the given layer."""
        
        update_start = time.time()
        with self.locks[layer_id]:
            layer_compute_cnt = self.compute_cnt[layer_id]  # Access specific layer compute count

            if token in layer_compute_cnt:
                layer_compute_cnt[token] += 1
            else:
                layer_compute_cnt[token] = 1
            # print(f"Token {token} updated in Layer {layer_id}: {layer_compute_cnt[token]}")
        update_end = time.time()
        print(f"Update time: {update_end - update_start}")
        
    def query(self, round_id, layer_id):
        """Query completed tokens for a specific layer_id key."""
        key = f"{round_id}.{layer_id}"

        with self.locks[layer_id]:
            # Cache hit: Return stored result
            if self.cache_key.get(layer_id) == key:
                # print(f"Cache hit for layer {layer_id}: {self.cache_value[layer_id]}")
                return list(self.cache_value[layer_id])  # Convert to list for safety

            finished_tokens = []
            layer_compute_cnt = self.compute_cnt[layer_id]  # Access layer-specific compute count

            for token, count in list(layer_compute_cnt.items()):  # Use list() to avoid runtime error
                if count == self.tp_rank_range:
                    print(f"self.tp_rank_range: {self.tp_rank_range}")
                    finished_tokens.append(token)
                    # print(f"Token {token} finished in Layer {layer_id}: {count}")
                    # Delete the token from the compute count
                    del layer_compute_cnt[token]

            # if len(finished_tokens) > 0:
            # print(f"Cache miss for layer {layer_id}: {finished_tokens}")

            # Update cache entry for the layer
            self.cache_key[layer_id] = key
            self.cache_value[layer_id] = finished_tokens  # Store finished tokens for this layer

            return finished_tokens



def run_scheduler_process(server_args, port_args, gpu_id, tp_rank_range, shared_service, writer):
    """Function to be executed in a separate process."""
    
    # Record start time for update_token operations
    update_start = time.perf_counter()
    
    for tp_rank in range(tp_rank_range):
        # Simulate token updates
        shared_service.update_token(tp_rank, 1)
    
    update_end = time.perf_counter()  # Record end time for update_token operations

    time.sleep(1)  # Simulating delay

    # Record start time for query operation
    query_start = time.perf_counter()

    # Query using (round_id, layer_id)
    result = shared_service.query(1, 1)  # Example query
    
    query_end = time.perf_counter()  # Record end time for query operation

    # Calculate time taken
    update_duration = update_end - update_start
    query_duration = query_end - query_start

    # Send result and timings back via Pipe
    writer.send({"result": result, "update_time": update_duration, "query_time": query_duration})
    writer.close()

if __name__ == "__main__":
    tp_rank_range = 4  # Example range
    manager = mp.Manager()
    num_layers = 2  # Example number of layers
    service = CompleteTokenQueryService(tp_rank_range, manager, num_layers)

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
