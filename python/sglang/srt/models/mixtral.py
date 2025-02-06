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

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/mixtral.py#L1
"""Inference-only Mixtral model."""

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import MixtralConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.rotary_embedding import get_rope

from python.sglang.srt.utils import CompleteTokenQueryService
from sglang.srt.layers.ep_moe.layer import EPMoE
from sglang.srt.layers.fused_moe_triton import FusedMoE
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader


class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        layer_id: int = 0,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        self.experts = MoEImpl(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            renormalize=True,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=f"{prefix}.experts",
            layer_id=layer_id,
        )
        self.swap_experts=True
       
    def forward(self, hidden_states: torch.Tensor, is_decode_mode: bool, residual: torch.Tensor, forward_batch: ForwardBatch, complete_token_manager: Optional[CompleteTokenQueryService] = None):
        if(self.swap_experts==True):
            print("[Testing]Swapping experts")
            import time
            start = time.time()
            self.experts.quant_method.update_experts(self.experts, 1, "remove", available_experts=self.experts.available_experts)
            self.experts.quant_method.update_experts(self.experts, 1, "load", torch.randn(2 * self.experts.intermediate_size_per_partition, self.hidden_size), torch.randn(self.hidden_size, self.experts.intermediate_size_per_partition), available_experts=self.experts.available_experts)
            end = time.time()
            print(f"[Testing]Swapping experts took {end-start} seconds")
            self.swap_experts=False
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        
        ### TODO: Add seperate logic here
        ### Tokens with local experts found should be kept (token_hit)
        ### Tokens without local experts found should be routed to other MoE instance (token_miss / sent to the dispatch_buffer)
        ### assert that there should be no miss during the prefilling phase
        
        available_experts = self.experts.available_experts
        
        # print(f"[MIXTRAL MoE]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        
        final_hidden_states, residual, forward_batch = self.experts(hidden_states, router_logits, is_decode_mode=is_decode_mode, residual=residual, forward_batch=forward_batch, complete_token_manager=complete_token_manager)
        # print(f"[MIXTRAL MoE]Forward batch out_cache_loc after expert forwarding: {forward_batch.out_cache_loc}")
        
        # print(f"[MIXTRAL before all-reduce]Final hidden states shape: {final_hidden_states.shape}")
        
        if self.tp_size > 1:
            if is_decode_mode:
                print(f"[Testing]Final hidden states shape: {final_hidden_states.shape}")
                print(f"[Testing]Forward batch is_local_toks: {forward_batch.is_local_toks}")
                final_hidden_states[forward_batch.is_local_toks] = tensor_model_parallel_all_reduce(final_hidden_states[forward_batch.is_local_toks]) # This all-reduce is causing problem, and we only need to do the all-to-all for 
            else:
                final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        torch.cuda.synchronize()
        # print(f"[MIXTRAL after all-reduce]Final hidden states shape: {final_hidden_states.shape}")
        # print(f"[MIXTRAL after all-reduce]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        ### TODO: Add collect logic here
        ### Collect the tokens from other MoE instance (that will sent back from the collect_buffer)
        ### assert that there should be nothing to collect during the prefilling phase
        
        
        return final_hidden_states.view(-1, *orig_shape[1:]), residual, forward_batch


class MixtralAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # print(f"[MIXTRAL Attention]Forward batch: {forward_batch}")
        qkv, _ = self.qkv_proj(hidden_states)
        # print(f"[MIXTRAL qkv_proj]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        print(f"[MIXTRAL qkv split]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        # print("Position storage after clone:", positions.storage().data_ptr())
        # check the size of q, k before and after rotary_emb
        # print(f"[MIXTRAL Attention]q shape before rotary_emb: {q.shape}")
        # print(f"[MIXTRAL Attention]k shape before rotary_emb: {k.shape}")
        
        # q, k = self.rotary_emb(positions, q, k)
        
        # print(f"[MIXTRAL Attention]q shape after rotary_emb: {q.shape}")
        # print(f"[MIXTRAL Attention]k shape after rotary_emb: {k.shape}")
        # print("Query storage:", q.storage().data_ptr())
        # print("Key storage:", k.storage().data_ptr())
        print(f"[MIXTRAL Attention]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.block_sparse_moe = MixtralMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.block_sparse_moe",
            layer_id=layer_id,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        complete_token_manager: Optional[CompleteTokenQueryService] = None,
    ):
        # Self Attention
        # quit if hidden_states and residual are empty
        # print(f"[MIXTRAL layer {self.layer_id}]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        if hidden_states.numel() == 0 and residual.numel() == 0:
            # print(f"[Mixtral layer {self.layer_id}]Both hidden states and residual are empty")
            assert False, "Both hidden states and residual are empty"
        
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # print(f"[Testing]Residual shape: {residual.shape}")
            # print(f"[Testing]Hidden states shape: {hidden_states.shape}")
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # print(f"[MIXTRAL layer {self.layer_id}]Forward batch out_cache_loc before self_attn: {forward_batch.out_cache_loc}")
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        
        ### TODO: Add speculation logit here
        # hidden_states = self.speculation_logit(hidden_states)
        
        is_decode_mode = forward_batch.forward_mode.is_decode()
        # print(f"[MIXTRAL layer {self.layer_id}]Forward batch out_cache_loc before post attention: {forward_batch.out_cache_loc}")

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # print(f"[MIXTRAL layer {self.layer_id}]Hidden states shape before moe: {hidden_states.shape}, device: {hidden_states.device}")
        # print(f"[MIXTRAL layer {self.layer_id}]Residual shape before moe: {residual.shape}, device: {residual.device}")
        # print(f"[MIXTRAL layer {self.layer_id}]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        
        hidden_states, residual, forward_batch = self.block_sparse_moe(hidden_states, is_decode_mode=is_decode_mode, residual=residual, forward_batch=forward_batch, complete_token_manager=complete_token_manager)
        # print(f"[MIXTRAL layer {self.layer_id}]Hidden states shape after moe: {hidden_states.shape}, device: {hidden_states.device}")
        # print(f"[MIXTRAL layer {self.layer_id}]Residual shape after moe: {residual.shape}, device: {residual.device}")
        # print(f"[MIXTRAL layer {self.layer_id}]Forward batch out_cache_loc: {forward_batch.out_cache_loc}")
        
        return hidden_states, residual, forward_batch


class MixtralModel(nn.Module):
    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                MixtralDecoderLayer(
                    config, i, quant_config=quant_config, prefix=f"{prefix}.layers"
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        complete_token_manager: Optional[CompleteTokenQueryService] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual, forward_batch = layer(
                positions, hidden_states, forward_batch, residual, complete_token_manager=complete_token_manager
            )
        if hidden_states.numel() > 0:
            hidden_states, _ = self.norm(hidden_states, residual)
            
        print(f"[Forward Finished]Hidden states shape: {hidden_states.shape}")
        return hidden_states


class MixtralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = MixtralModel(config, quant_config=quant_config, prefix="model")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        complete_token_manager: Optional[CompleteTokenQueryService] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds, complete_token_manager)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        MoEImpl = EPMoE if global_server_args_dict["enable_ep_moe"] else FusedMoE
        expert_params_mapping = MoEImpl.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.num_local_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip loading kv_scale from ckpts towards new design.
                    if name.endswith(".kv_scale") and name not in params_dict:
                        continue
                    if name is None:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = MixtralForCausalLM
