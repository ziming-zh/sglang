import multiprocessing as mp
import time

class CompleteTokenQueryService:
    def __init__(self, tp_rank_range, num_layers=32, cache_window=5):
        self.tp_rank_range = tp_rank_range
        self.num_layers = num_layers
        self.cache_window = cache_window

        # Locks per layer
        self.locks = [mp.Lock() for _ in range(num_layers)]

        # Shared memory for token computation counts
        self.compute_cnt = [mp.Array('i', 200, lock=False) for _ in range(num_layers)]  # Supports up to 100 tokens/layer

        # Shared cache
        self.cache_key = mp.Array('i', num_layers, lock=False)
        self.cache_value = [[mp.Array('i', 200, lock=False) for _ in range(cache_window)] for _ in range(num_layers)]  # Up to 100 cached tokens/layer

    def update_token(self, token, layer_id):
        """Atomically update the computation count for a specific token in the given layer."""
        with self.locks[layer_id]:
            self.compute_cnt[layer_id][token] += 1

    def query(self, round_id, layer_id):
        """Query completed tokens for a specific layer_id key."""

        with self.locks[layer_id]:
            if round_id <= self.cache_key[layer_id] and round_id > self.cache_key[layer_id] - self.cache_window:
                return [token for token in self.cache_value[layer_id][round_id % self.cache_window] if token != 0]

            finished_tokens = []
            for token in range(200):  # Check all tokens
                if self.compute_cnt[layer_id][token] == self.tp_rank_range:
                    finished_tokens.append(token)
                    self.compute_cnt[layer_id][token] = 0  # Reset count

            self.cache_key[layer_id] = round_id
            for i, token in enumerate(finished_tokens):
                self.cache_value[layer_id][round_id % self.cache_window][i] = token

            return finished_tokens


def run_scheduler_process(tp_rank_range, shared_service, writer):
    """Function to be executed in a separate process."""
    update_start = time.perf_counter()

    for tp_rank in range(tp_rank_range):
        shared_service.update_token(tp_rank+1, 1)

    update_end = time.perf_counter()
    time.sleep(1)  # Simulate delay

    query_start = time.perf_counter()
    result = shared_service.query(1, 1)  # Example query
    query_end = time.perf_counter()

    writer.send({"result": result, "update_time": update_end - update_start, "query_time": query_end - query_start})
    writer.close()


if __name__ == "__main__":
    tp_rank_range = 4  # Example range
    num_layers = 2  # Example number of layers
    service = CompleteTokenQueryService(tp_rank_range, num_layers)

    scheduler_procs = []
    scheduler_pipe_readers = []

    for tp_rank in range(tp_rank_range):
        reader, writer = mp.Pipe(duplex=False)
        proc = mp.Process(target=run_scheduler_process, args=(tp_rank_range, service, writer))
        proc.start()
        scheduler_procs.append(proc)
        scheduler_pipe_readers.append(reader)

    for reader in scheduler_pipe_readers:
        print("Result:", reader.recv())

    for proc in scheduler_procs:
        proc.join()
