# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Store information about a forward batch.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

if TYPE_CHECKING:
    from sglang.srt.layers.attention import AttentionBackend
    from sglang.srt.managers.schedule_batch import ImageInputs, ModelWorkerBatch
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers wil be IDLE if no sequence are allocated.
    IDLE = auto()

    # A dummy first batch to start the pipeline for overlap scheduler.
    # It is now used for triggering the sampling_info_done event for the first prefill batch.
    DUMMY_FIRST = auto()

    def is_prefill(self):
        return self == ForwardMode.PREFILL

    def is_extend(self):
        return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED

    def is_idle(self):
        return self == ForwardMode.IDLE

    def is_dummy_first(self):
        return self == ForwardMode.DUMMY_FIRST

@dataclass
class ForwardBatch:
    """Store all inputs of a forward pass."""

    # The forward mode
    forward_mode: ForwardMode
    # The batch size
    batch_size: int
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool
    out_cache_loc: torch.Tensor

    # The sum of all sequence lengths
    seq_lens_sum: int

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_prefix_lens: Optional[torch.Tensor] = None
    extend_start_loc: Optional[torch.Tensor] = None
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None

    # For multimodal
    image_inputs: Optional[List[ImageInputs]] = None

    # Encoder-decoder
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # For LoRA
    lora_paths: Optional[List[str]] = None

    # For input embeddings
    input_embeds: Optional[torch.tensor] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Attention backend
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    attn_backend: AttentionBackend = None

    # For Qwen2-VL
    mrope_positions: torch.Tensor = None

    # For DP attention
    global_num_tokens: Optional[List[int]] = None
    gathered_buffer: Optional[torch.Tensor] = None
    can_run_dp_cuda_graph: bool = False
    
    # Decoding: rid_list
    rid_list: Optional[List[str]] = None
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def split(self, mask: torch.Tensor) -> Tuple["ForwardBatch", "ForwardBatch"]:
        """
        Splits the ForwardBatch into local and remote batches based on a binary mask.
        
        Args:
            mask (torch.Tensor): A binary tensor where 1 indicates local and 0 indicates remote.

        Returns:
            Tuple[ForwardBatch, ForwardBatch]: The local and remote batches.
        """
        assert mask.shape == self.input_ids.shape, "Mask shape must match input_ids shape"

        remote_mask = mask.bool()
        local_mask = ~remote_mask

        def split_tensor(tensor: Optional[torch.Tensor]):
            if tensor is None:
                return None, None
            local_tensor = tensor[local_mask].clone() if local_mask.any() else None
            remote_tensor = tensor[remote_mask].clone() if remote_mask.any() else None
            return local_tensor, remote_tensor

        def split_list(lst: Optional[List]):
            if lst is None:
                return None, None
            assert len(lst) == len(mask), "List length must match mask length"
            return ([lst[i] for i in range(len(lst)) if not mask[i].item()] if lst is not None else None,
                    [lst[i] for i in range(len(lst)) if mask[i].item()] if lst is not None else None)


        
        local_batch = self
        remote_batch = ForwardBatch()
        
        local_batch.forward_mode = self.forward_mode
        remote_batch.forward_mode = self.forward_mode

        # Split core tensors
        local_batch.input_ids, remote_batch.input_ids = split_tensor(self.input_ids)
        
        local_batch.positions, remote_batch.positions = split_tensor(self.positions)
        local_batch.out_cache_loc, remote_batch.out_cache_loc = split_tensor(self.out_cache_loc)
        # print("[SPLIT] local_batch.out_cache_loc: ", local_batch.out_cache_loc)
        # print("[SPLIT] remote_batch.out_cache_loc: ", remote_batch.out_cache_loc)

        
        local_batch.seq_lens, remote_batch.seq_lens = split_tensor(self.seq_lens)
        # print("[SPLIT] local_batch.seq_lens: ", local_batch.seq_lens)
        # print("[SPLIT] remote_batch.seq_lens: ", remote_batch.seq_lens)
        local_batch.seq_lens_sum = local_batch.seq_lens.sum().item() if local_batch.seq_lens is not None else 0
        remote_batch.seq_lens_sum = remote_batch.seq_lens.sum().item() if remote_batch.seq_lens is not None else 0
        # Split other metadata
        # local_batch.req_pool_indices = self.req_pool_indices if self.req_pool_indices is not None else None
        # remote_batch.req_pool_indices = self.req_pool_indices if self.req_pool_indices is not None else None
        # print(f"[SPLIT] self.req_pool_indices: {self.req_pool_indices}")
        local_batch.req_pool_indices, remote_batch.req_pool_indices = split_tensor(self.req_pool_indices)
        # print(f"[SPLIT] local_batch.req_pool_indices: {local_batch.req_pool_indices}")
        # print(f"[SPLIT] remote_batch.req_pool_indices: {remote_batch.req_pool_indices}")
        # local_batch.image_inputs, remote_batch.image_inputs = split_list(self.image_inputs)
        # local_batch.lora_paths, remote_batch.lora_paths = split_list(self.lora_paths)
        

        # Compute new batch sizes
        local_batch.batch_size = local_batch.input_ids.shape[0] if local_batch.input_ids is not None else 0
        remote_batch.batch_size = remote_batch.input_ids.shape[0] if remote_batch.input_ids is not None else 0
                
        # split rid_list
        local_batch.rid_list, remote_batch.rid_list = split_list(self.rid_list)
        # print(f"[SPLIT] local_batch.rid_list: {local_batch.rid_list}")
        # print(f"[SPLIT] remote_batch.rid_list: {remote_batch.rid_list}")
        # keep_indices = mask.nonzero(as_tuple=True)[0]
        # keep_indices_list = keep_indices.tolist()
        # remove_indices = (~mask).nonzero(as_tuple=True)[0]
        # remove_indices_list = remove_indices.tolist()
        # local_batch.sampling_info = self.sampling_info.filter_batch(keep_indices_list, keep_indices)
        # remote_batch.sampling_info = self.sampling_info.filter_batch(remove_indices_list, remove_indices)

        return local_batch, remote_batch

    def combine(self, fb_list: list["ForwardBatch"]) -> None:
        """
        Combines a list of ForwardBatch instances into the current one (in-place modification).
        Uses pre-allocated memory for tensors to avoid fragmentation and improve efficiency.

        Args:
            fb_list (list[ForwardBatch]): The list of ForwardBatches to combine.
        """
        if len(fb_list) == 0:
            return

        # Helper function to check if a tensor is valid
        def is_valid_tensor(t):
            return t is not None and t.numel() > 0

        # Helper function to check if a sequence length tensor is valid
        def is_valid_seq_len(t):
            return t is not None and (t.numel() > 1 or (t.numel() > 0 and t.item() != 0))

        # Collect tensors and lists from all batches
        input_ids_list = [self.input_ids] if is_valid_tensor(self.input_ids) else []
        seq_lens_list = [self.seq_lens] if is_valid_seq_len(self.seq_lens) else []
        positions_list = [self.positions] if is_valid_tensor(self.positions) else []
        out_cache_loc_list = [self.out_cache_loc] if is_valid_tensor(self.out_cache_loc) else []
        req_pool_indices_list = [self.req_pool_indices] if is_valid_tensor(self.req_pool_indices) else []

        # Collect tensors from other batches
        for other in fb_list:
            if is_valid_tensor(other.input_ids):
                input_ids_list.append(other.input_ids)
            if is_valid_seq_len(other.seq_lens):
                seq_lens_list.append(other.seq_lens)
            if is_valid_tensor(other.positions):
                positions_list.append(other.positions)
            if is_valid_tensor(other.out_cache_loc):
                out_cache_loc_list.append(other.out_cache_loc)
            if is_valid_tensor(other.req_pool_indices):
                req_pool_indices_list.append(other.req_pool_indices)

        # Pre-allocate memory and combine tensors
        def combine_tensors(tensor_list):
            if not tensor_list:
                return None

            # Calculate total size
            total_size = 0
            for tensor in tensor_list:
                if tensor.numel() == 1:
                    total_size += 1
                else:
                    total_size += tensor.numel()

            # Pre-allocate memory
            combined_tensor = torch.empty((total_size,), dtype=tensor_list[0].dtype, device=tensor_list[0].device)

            # Fill the pre-allocated tensor
            index = 0
            for tensor in tensor_list:
                if tensor.numel() == 1:
                    combined_tensor[index] = tensor.item()  # Extract the single element
                    index += 1
                else:
                    combined_tensor[index:index + tensor.numel()] = tensor  # Copy all elements
                    index += tensor.numel()

            return combined_tensor

        # Combine each tensor list
        self.input_ids = combine_tensors(input_ids_list)
        self.seq_lens = combine_tensors(seq_lens_list)
        self.positions = combine_tensors(positions_list)
        self.out_cache_loc = combine_tensors(out_cache_loc_list)
        self.req_pool_indices = combine_tensors(req_pool_indices_list)

        # Handle non-tensor attributes
        self.image_inputs = self.image_inputs if self.image_inputs is not None else []
        self.lora_paths = self.lora_paths if self.lora_paths is not None else []
        self.rid_list = self.rid_list if self.rid_list is not None else []

        for other in fb_list:
            if other.image_inputs is not None:
                self.image_inputs.extend(other.image_inputs)
            if other.lora_paths is not None:
                self.lora_paths.extend(other.lora_paths)
            if other.rid_list is not None:
                self.rid_list.extend(other.rid_list)

            self.seq_lens_sum += other.seq_lens_sum
            self.batch_size += other.batch_size

            # Update sampling info
            # self.sampling_info.merge_batch(other.sampling_info)
            
    
    def compute_mrope_positions(
        self, model_runner: ModelRunner, batch: ModelWorkerBatch
    ):
        device = model_runner.device
        hf_config = model_runner.model_config.hf_config
        mrope_positions_list = [None] * self.seq_lens.shape[0]
        if self.forward_mode.is_decode():
            for i, _ in enumerate(mrope_positions_list):
                mrope_position_delta = (
                    0
                    if batch.image_inputs[i] is None
                    else batch.image_inputs[i].mrope_position_delta
                )
                mrope_positions_list[i] = MRotaryEmbedding.get_next_input_positions(
                    mrope_position_delta,
                    int(self.seq_lens[i]) - 1,
                    int(self.seq_lens[i]),
                )
        elif self.forward_mode.is_extend():
            extend_start_loc_cpu = self.extend_start_loc.cpu().numpy()
            for i, image_inputs in enumerate(batch.image_inputs):
                extend_start_loc, extend_seq_len, extend_prefix_len = (
                    extend_start_loc_cpu[i],
                    batch.extend_seq_lens[i],
                    batch.extend_prefix_lens[i],
                )
                if image_inputs is None:
                    # text only
                    mrope_positions = [
                        [
                            pos
                            for pos in range(
                                extend_prefix_len, extend_prefix_len + extend_seq_len
                            )
                        ]
                    ] * 3
                else:
                    # TODO: current qwen2-vl do not support radix cache since mrope position calculation
                    mrope_positions, mrope_position_delta = (
                        MRotaryEmbedding.get_input_positions(
                            input_tokens=self.input_ids[
                                extend_start_loc : extend_start_loc + extend_seq_len
                            ],
                            image_grid_thw=image_inputs.image_grid_thws,
                            vision_start_token_id=hf_config.vision_start_token_id,
                            spatial_merge_size=hf_config.vision_config.spatial_merge_size,
                            context_len=0,
                        )
                    )
                    batch.image_inputs[i].mrope_position_delta = mrope_position_delta
                mrope_positions_list[i] = mrope_positions

        self.mrope_positions = torch.concat(
            [torch.tensor(pos, device=device) for pos in mrope_positions_list],
            axis=1,
        )
        self.mrope_positions = self.mrope_positions.to(torch.int64)

    @classmethod
    def init_new(
        cls,
        batch: ModelWorkerBatch,
        model_runner: ModelRunner,
    ):

        device = model_runner.device
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=len(batch.seq_lens),
            input_ids=batch.input_ids,
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            image_inputs=batch.image_inputs,
            encoder_cached=batch.encoder_cached,
            encoder_lens=batch.encoder_lens,
            encoder_lens_cpu=batch.encoder_lens_cpu,
            encoder_out_cache_loc=batch.encoder_out_cache_loc,
            seq_lens_sum=batch.seq_lens_sum,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
            global_num_tokens=batch.global_num_tokens,
            can_run_dp_cuda_graph=batch.can_run_dp_cuda_graph,
            lora_paths=batch.lora_paths,
            sampling_info=batch.sampling_info,
            input_embeds=batch.input_embeds,
            rid_list=batch.rid_list,
        )

        if ret.global_num_tokens is not None:
            max_len = max(ret.global_num_tokens)
            ret.gathered_buffer = torch.zeros(
                (max_len * model_runner.tp_size, model_runner.model_config.hidden_size),
                dtype=model_runner.dtype,
                device=device,
            )

        if ret.forward_mode.is_idle():
            return ret

        # Init position information
        if not ret.forward_mode.is_decode():
            ret.extend_seq_lens = torch.tensor(
                batch.extend_seq_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            ret.extend_prefix_lens = torch.tensor(
                batch.extend_prefix_lens, dtype=torch.int32
            ).to(device, non_blocking=True)
            if model_runner.server_args.attention_backend != "torch_native":
                ret.extend_num_tokens = batch.extend_num_tokens
                ret.positions, ret.extend_start_loc = compute_position_triton(
                    ret.extend_prefix_lens, ret.extend_seq_lens, ret.extend_num_tokens
                )
            else:
                ret.positions, ret.extend_start_loc = compute_position_torch(
                    ret.extend_prefix_lens, ret.extend_seq_lens
                )
            ret.extend_prefix_lens_cpu = batch.extend_prefix_lens
            ret.extend_seq_lens_cpu = batch.extend_seq_lens
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens

        if model_runner.model_is_mrope:
            ret.compute_mrope_positions(model_runner, batch)

        # Init attention information
        ret.req_to_token_pool = model_runner.req_to_token_pool
        ret.token_to_kv_pool = model_runner.token_to_kv_pool
        ret.attn_backend = model_runner.attn_backend

        # Init lora information
        if model_runner.server_args.lora_paths is not None:
            model_runner.lora_manager.prepare_lora_batch(ret)

        return ret


def compute_position_triton(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor, extend_seq_lens_sum
):
    """Compute positions. It is a fused version of `compute_position_torch`."""
    batch_size = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    extend_start_loc = torch.empty(
        batch_size, dtype=torch.int32, device=extend_seq_lens.device
    )

    # Launch kernel
    compute_position_kernel[(batch_size,)](
        positions,
        extend_start_loc,
        extend_prefix_lens,
        extend_seq_lens,
    )

    return positions, extend_start_loc


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0)

    prefix_len = tl.load(extend_prefix_lens + pid)
    seq_len = tl.load(extend_seq_lens + pid)

    # TODO: optimize this?
    cumsum_start = 0
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


def compute_position_torch(
    extend_prefix_lens: torch.Tensor, extend_seq_lens: torch.Tensor
):
    positions = torch.concat(
        [
            torch.arange(
                prefix_len, prefix_len + extend_len, device=extend_prefix_lens.device
            )
            for prefix_len, extend_len in zip(extend_prefix_lens, extend_seq_lens)
        ],
        axis=0,
    )
    extend_start_loc = torch.zeros_like(extend_seq_lens)
    extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
    return positions.to(torch.int64), extend_start_loc
