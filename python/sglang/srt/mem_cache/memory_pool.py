"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Memory pool.

SGLang has two levels of memory pool.
ReqToTokenPool maps a a request to its token locations.
BaseTokenToKVPool maps a token location to its KV cache data.
"""

import logging
from typing import List, Tuple, Union

import torch

from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)


class ReqToTokenPool:
    """A memory pool that maps a request to its token locations."""

    def __init__(self, size: int, max_context_len: int, device: str, use_records: bool):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))
        self.write_records = []
        self.use_records = use_records

        if self.use_records:
            self.write = self.write_with_records
        else:
            self.write = self.write_without_records

    def write(self, indices, values):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, need_size: int) -> List[int]:
        if need_size > len(self.free_slots):
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    def free(self, free_index: Union[int, List[int]]):
        if isinstance(free_index, (int,)):
            self.free_slots.append(free_index)
        else:
            self.free_slots.extend(free_index)

    def clear(self):
        self.free_slots = list(range(self.size))
        self.write_records = []

    def write_without_records(self, indices, values):
        self.req_to_token[indices] = values

    def write_with_records(self, indices, values):
        self.req_to_token[indices] = values
        self.write_records.append((indices, values))

    def get_write_records(self):
        ret = self.write_records
        self.write_records = []
        return ret

    def apply_write_records(self, write_records: List[Tuple]):
        for indices, values in write_records:
            self.req_to_token[indices] = values
            
    # def combine(self, other: "ReqToTokenPool") -> None:
    #     """
    #     Combines another ReqToTokenPool into the current one (in-place modification).

    #     Args:
    #         other (ReqToTokenPool): Another ReqToTokenPool instance.
    #     """
    #     assert isinstance(other, ReqToTokenPool), "Other pool must be an instance of ReqToTokenPool."
    #     assert self.max_context_len == other.max_context_len, "max_context_len must match."
    #     assert self.device == other.device, "Both pools must be on the same device."

    #     # Compute new total size
    #     old_size = self.size
    #     new_size = self.size + other.size

    #     # Resize req_to_token tensor to accommodate new data
    #     new_req_to_token = torch.zeros(
    #         (new_size, self.max_context_len), dtype=torch.int32, device=self.device
    #     )
    #     new_req_to_token[:old_size] = self.req_to_token  # Copy old data
    #     new_req_to_token[old_size:] = other.req_to_token  # Copy new data

    #     # Update self attributes in place
    #     self.size = new_size
    #     self.req_to_token = new_req_to_token  # Replace with the resized tensor

    #     # Merge free slots with adjusted indices
    #     self.free_slots.extend(i + old_size for i in other.free_slots)

    #     # Merge write records if enabled
    #     if self.use_records:
    #         self.write_records.extend((indices + old_size, values) for indices, values in other.write_records)


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

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        raise NotImplementedError()
    
    def migrate_kv_buffer(self, source_kv_pool: "BaseTokenToKVPool") -> None:
        raise NotImplementedError()


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

        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
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

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # print("set_kv_buffer")
        # print(f"loc: {loc}")
        # print(f"cache_k's shape: {cache_k.shape}")
        # print(f"cache_v's shape: {cache_v.shape}")
        # print(f"[Before] k_buffer's shape: {self.k_buffer[layer.layer_id].shape}")
        # print(f"[Before] v_buffer's shape: {self.v_buffer[layer.layer_id].shape}")
        layer_id = layer.layer_id
        
        # Ensure loc is within bounds
        max_loc = self.k_buffer[layer_id].shape[0]
        assert torch.all((loc >= 0) & (loc < max_loc)), f"Error: [Layer {layer.layer_id}] loc index {loc} is out of bounds (valid range: 0 to {max_loc - 1})."

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v

    def migrate_kv_buffer(self, source_kv_pool: "MHATokenToKVPool"):
        """
        Migrate the KV buffer from another MHATokenToKVPool to the current one.

        Args:
            source_kv_pool (MHATokenToKVPool): The source KV pool to migrate from.

        Raises:
            ValueError: If the source and target KV pools are incompatible.
        """
        # Validate pool compatibility
        if len(self.k_buffer) != len(source_kv_pool.k_buffer):
            raise ValueError("Mismatch in layer count between KV pools.")
        if self.k_buffer[0].shape[1:] != source_kv_pool.k_buffer[0].shape[1:]:
            raise ValueError("Head number and head dimension mismatch between KV pools.")
        if self.store_dtype != source_kv_pool.store_dtype:
            print(f"Warning: store_dtype mismatch. Converting from {source_kv_pool.store_dtype} to {self.store_dtype}")

        # Migrate KV cache
        for layer_id in range(len(self.k_buffer)):
            # Convert dtype if necessary
            source_k = source_kv_pool.k_buffer[layer_id]
            source_v = source_kv_pool.v_buffer[layer_id]

            if source_k.dtype != self.store_dtype:
                source_k = source_k.to(self.store_dtype)
                source_v = source_v.to(self.store_dtype)

            # Copy data
            self.k_buffer[layer_id].copy_(source_k)
            self.v_buffer[layer_id].copy_(source_v)

        print("KV cache migration completed successfully.")

# This compiled version is slower in the unit test
# python3 -m unittest test_bench_serving.TestBenchServing.test_offline_throughput_non_stream_small_batch_size
@torch.compile(dynamic=True)
def copy_two_array(loc, dst_1, src_1, dst_2, src_2, dtype, store_dtype):
    dst_1[loc] = src_1.to(dtype).view(store_dtype)
    dst_2[loc] = src_2.to(dtype).view(store_dtype)


class MLATokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
    ):
        super().__init__(size, dtype, device)

        self.kv_lora_rank = kv_lora_rank
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.kv_buffer = [
            torch.empty(
                (size + 1, 1, kv_lora_rank + qk_rope_head_dim),
                dtype=self.store_dtype,
                device=device,
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id].view(self.dtype)
        return self.kv_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.kv_buffer[layer_id][..., : self.kv_lora_rank].view(self.dtype)
        return self.kv_buffer[layer_id][..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        
        # Ensure loc is within bounds
        max_loc = self.kv_buffer[layer_id].shape[0]
        assert torch.all((loc >= 0) & (loc < max_loc)), f"Error: loc index {loc} is out of bounds (valid range: 0 to {max_loc - 1})."
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
        else:
            self.kv_buffer[layer_id][loc] = cache_k


class DoubleSparseTokenToKVPool(BaseTokenToKVPool):

    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        heavy_channel_num: int,
    ):
        super().__init__(size, dtype, device)

        # [size, head_num, head_dim] for each layer
        self.k_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device=device)
            for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.empty((size + 1, head_num, head_dim), dtype=dtype, device=device)
            for _ in range(layer_num)
        ]

        # [size, head_num, heavy_channel_num] for each layer
        self.label_buffer = [
            torch.empty(
                (size + 1, head_num, heavy_channel_num), dtype=dtype, device=device
            )
            for _ in range(layer_num)
        ]

    def get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id]

    def get_label_buffer(self, layer_id: int):
        return self.label_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cache_label: torch.Tensor,
    ):
        # NOTE(Andy): ignore the dtype check
        layer_id = layer.layer_id
        self.k_buffer[layer_id][loc] = cache_k
        self.v_buffer[layer_id][loc] = cache_v
        self.label_buffer[layer_id][loc] = cache_label
