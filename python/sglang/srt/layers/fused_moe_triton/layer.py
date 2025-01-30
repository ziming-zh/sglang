# Adapted from https://github.com/vllm-project/vllm/blob/a6221a144af772fd1a68fe7e627935dc53e81738/vllm/model_executor/layers/fused_moe/layer.py

from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.custom_op import CustomOp

from python.sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.custom_op_util import register_custom_op
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import set_weight_attrs

if torch.cuda.is_available() or torch.hip.is_available():
    from sglang.srt.layers.fused_moe_triton.fused_moe import fused_experts
else:
    fused_experts = None  # type: ignore

import logging

logger = logging.getLogger(__name__)


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"


class FusedMoEMethodBase(QuantizeMethodBase):

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
    ) -> torch.Tensor:
        raise NotImplementedError


@register_custom_op("sglang_unquantized_fused_moe")
class UnquantizedFusedMoEMethod(FusedMoEMethodBase, CustomOp):
    """MoE method without quantization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_buffer = {}  # Dictionary to store async results on CPU
        self.gpu_buffer = {}  # Dictionary to store async results on GPU
        self.stream_cpu = torch.cuda.Stream()
        self.stream_gpu = torch.cuda.Stream()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        available_experts: Optional[List[bool]] = None,  # New argument for availability
        **extra_weight_attrs,
    ):
        # Default to all experts being available if not specified
        if available_experts is None:
            available_experts = [True] * num_experts

        # Ensure `available_experts` matches the number of experts
        assert len(available_experts) == num_experts, "Mismatch in number of experts"

        # Create weights only for available experts
        w13_weight_data = []
        w2_weight_data = []

        for expert_id, is_available in enumerate(available_experts):
            if is_available:
                # Initialize weights for the current expert
                w13_weight_data.append(
                    torch.empty(
                        2 * intermediate_size, hidden_size, dtype=params_dtype
                    )
                )
                w2_weight_data.append(
                    torch.empty(
                        hidden_size, intermediate_size, dtype=params_dtype
                    )
                )
            else:
                # Skip initializing weights for pruned experts
                w13_weight_data.append(None)
                w2_weight_data.append(None)

        # Stack weights for available experts
        w13_weight = torch.nn.Parameter(
            torch.stack([w for w in w13_weight_data if w is not None]),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.stack([w for w in w2_weight_data if w is not None]),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)
        
        self.w13_cpu = w13_weight.cpu()  # Simulate CPU expert weights
        self.w2_cpu = w2_weight.cpu()  # Simulate CPU expert weights
    
    def update_experts(
        self,
        layer: torch.nn.Module,
        expert_id: int,
        operation: str,
        w13_tensor: Optional[torch.Tensor] = None,
        w2_tensor: Optional[torch.Tensor] = None,
        available_experts: Optional[List[bool]] = None,
    ):
        """
        Dynamically add or remove experts in the layer using available_experts for rebuilding.

        Args:
            layer (torch.nn.Module): The module containing expert weights.
            expert_id (int): The ID of the expert to load or remove.
            operation (str): Either "load" or "remove".
            w13_tensor (torch.Tensor, optional): Tensor for w13_weight of the expert. Required for "load".
            w2_tensor (torch.Tensor, optional): Tensor for w2_weight of the expert. Required for "load".
            available_experts (List[bool], optional): A list indicating availability of experts.
        """
        assert operation in {"load", "remove"}, "Operation must be 'load' or 'remove'."
        if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
            raise AttributeError(
                "The layer does not have 'w13_weight' or 'w2_weight'. "
                "Make sure to initialize weights using 'create_weights' first."
            )

        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight

        # Ensure available_experts matches the number of experts
        if available_experts is None:
            raise ValueError("available_experts list must be provided.")
        current_experts = len(available_experts)
        assert available_experts.count(True) == len(w13_weight), f"Mismatch in number of available experts and weights length. Available experts: {available_experts}, w13_weight length: {len(w13_weight)}"

        if operation == "load":
            assert w13_tensor is not None and w2_tensor is not None, \
                "For 'load' operation, w13_tensor and w2_tensor must be provided."
            assert w13_tensor.size() == (2 * layer.intermediate_size_per_partition, layer.hidden_size), \
                "w13_tensor size mismatch."
            assert w2_tensor.size() == (layer.hidden_size, layer.intermediate_size_per_partition), \
                "w2_tensor size mismatch."
                
            # make sure tensor are on the same device and the same dtype as the weights
            w13_tensor = w13_tensor.to(dtype=w13_weight.dtype, device=w13_weight.device)
            w2_tensor = w2_tensor.to(dtype=w2_weight.dtype, device=w2_weight.device)
            
            in_place_update = False
            if available_experts[expert_id]:
                in_place_update = True
            
            # Update the available_experts list and weights
            if expert_id >= current_experts:
                # Extend the available_experts list if expert_id is beyond current range
                available_experts.extend([False] * (expert_id - current_experts + 1))
            available_experts[expert_id] = True

            # Update weights dynamically
            if in_place_update:
                w13_weight_data = [w13_tensor if idx == expert_id else w for idx, w in enumerate(w13_weight)]
                w2_weight_data = [w2_tensor if idx == expert_id else w for idx, w in enumerate(w2_weight)]
            else:
                # do insert
                w13_weight_data = [w for idx, w in enumerate(w13_weight)]
                w13_weight_data.insert(expert_id, w13_tensor)
                w2_weight_data = [w for idx, w in enumerate(w2_weight)]
                w2_weight_data.insert(expert_id, w2_tensor)

        elif operation == "remove":
            assert expert_id < current_experts, f"Expert {expert_id} does not exist."
            assert available_experts[expert_id], f"Expert {expert_id} is already removed."
            
            # Mark the expert as unavailable
            available_experts[expert_id] = False

            # Set weights to None for the removed expert
            w13_weight_data = [None if idx == expert_id else w for idx, w in enumerate(w13_weight)]
            w2_weight_data = [None if idx == expert_id else w for idx, w in enumerate(w2_weight)]

        # Rebuild the weights based on available_experts
        w13_weight = torch.nn.Parameter(
            torch.stack([w for w, available in zip(w13_weight_data, available_experts) if available]),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.stack([w for w, available in zip(w2_weight_data, available_experts) if available]),
            requires_grad=False,
        )

        # Update layer parameters
        layer.w13_weight = w13_weight
        layer.w2_weight = w2_weight
        layer.register_parameter("w13_weight", layer.w13_weight)
        layer.register_parameter("w2_weight", layer.w2_weight)


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        is_decode_mode: bool,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        return self.forward(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            is_decode_mode=is_decode_mode,
            residual=residual,
            forward_batch=forward_batch,
        )

    def forward_cuda(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        is_decode_mode: bool,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        topk_weights, topk_ids, is_remote_toks = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            is_decode_mode=is_decode_mode,
        )

        x_remote = x[is_remote_toks]
        x_local = x[~is_remote_toks]
        print(f"x_remote: {x_remote.shape}, x_local: {x_local.shape}")

        topk_weights_remote = topk_weights[is_remote_toks]
        topk_ids_remote = topk_ids[is_remote_toks]

        topk_weights_local = topk_weights[~is_remote_toks]
        topk_ids_local = topk_ids[~is_remote_toks]
        
        forward_batch_local, forward_batch_remote = forward_batch.split(is_remote_toks)
        
        # print(f"forward_batch_local: {forward_batch_local}, forward_batch_remote: {forward_batch_remote}")
        
        if residual is not None:
            residual_remote = residual[is_remote_toks]
            residual_local = residual[~is_remote_toks]
        else:
            residual_remote = None
            residual_local = None

        # Buffer x_remote to avoid small CPU offloads
        if not hasattr(self, "remote_forward_batch") or self.remote_forward_batch is None:
            self.remote_buffer = [(x_remote, topk_weights_remote, topk_ids_remote, residual_remote)]
            self.remote_forward_batch = forward_batch_remote

        elif x_remote.numel() > 0:
            self.remote_buffer.append((x_remote, topk_weights_remote, topk_ids_remote, residual_remote))
            self.remote_forward_batch = self.remote_forward_batch.combine(forward_batch_remote)

        # Only offload when buffer size is 20 or more
        if len(self.remote_buffer) >= 1:
            print(f"Offloading {len(self.remote_buffer)} remote tokens to CPU")
            with torch.cuda.stream(self.stream_cpu):
                # Concatenate buffered tensors before offloading
                x_remote_cpu = torch.cat([item[0] for item in self.remote_buffer], dim=0).to("cpu", non_blocking=True)
                topk_weights_remote_cpu = torch.cat([item[1] for item in self.remote_buffer], dim=0)
                topk_ids_remote_cpu = torch.cat([item[2] for item in self.remote_buffer], dim=0)
                if residual_remote is not None:
                    residual_remote_cpu = torch.cat([item[3] for item in self.remote_buffer], dim=0)
                else:
                    residual_remote_cpu = None

                w13_cpu = self.w13_cpu
                w2_cpu = self.w2_cpu
                
                # print(f"x_remote_cpu: {x_remote_cpu.shape}, topk_weights_remote_cpu: {topk_weights_remote_cpu.shape}, topk_ids_remote_cpu: {topk_ids_remote_cpu.shape}")

                # Process in CPU
                cpu_result = fused_experts_cpu_impl(
                    hidden_states=x_remote_cpu,
                    w13=w13_cpu,
                    w2=w2_cpu,
                    topk_weights=topk_weights_remote_cpu,
                    topk_ids=topk_ids_remote_cpu,
                )

                # Store CPU results mapped to the original x_remote data pointers
                self.cpu_buffer[x_remote_cpu.data_ptr()] = cpu_result, residual_remote_cpu, forward_batch_remote

            # Clear buffer after offloading
            self.remote_buffer.clear()
            # remove the self.remote_forward_batch attribute
            self.remote_forward_batch=None
            

        # Check for completed CPU computations and move back to GPU
        fetched_cpu_results = []
        fetched_residuals = []
        fetched_forward_batch = []
        for key, cpu_result in list(self.cpu_buffer.items()):
            with torch.cuda.stream(self.stream_gpu):
                gpu_result = cpu_result[0].to("cuda", non_blocking=True)
                residual_gpu = cpu_result[1].to("cuda", non_blocking=True)
                forward_batch_remote = cpu_result[2]
                fetched_cpu_results.append(gpu_result)
                fetched_residuals.append(residual_gpu)

                if not isinstance(forward_batch_remote, ForwardBatch):
                    fetched_forward_batch = forward_batch_remote
                else:
                    fetched_forward_batch.combine(forward_batch_remote)
                del self.cpu_buffer[key]

        if fetched_cpu_results:
            x_local = torch.cat([x_local] + fetched_cpu_results, dim=0)
            if residual_local is not None:
                residual_local = torch.cat([residual_local] + fetched_residuals, dim=0)
            forward_batch_local.combine(fetched_forward_batch)

        return fused_experts(
            hidden_states=x_local,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights_local,
            topk_ids=topk_ids_local,
            inplace=True,
        ), residual_local, forward_batch_local

    def forward_cpu(self, *args, **kwargs):
        raise NotImplementedError("The CPU backend currently does not support MoE.")

    def forward_tpu(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("The TPU backend currently does not support MoE.")

    forward_native = forward_cuda


def fused_experts_cpu_impl(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
):
    """
    CPU-based implementation of Mixture of Experts (MoE) calculation.
    
    Args:
        hidden_states (torch.Tensor): Input token embeddings [num_tokens, hidden_size].
        w13 (torch.Tensor): Expert weight matrix for the first layer [E, N, hidden_size].
        w2 (torch.Tensor): Expert weight matrix for the second layer [E, N, output_size].
        topk_weights (torch.Tensor): Top-k routing weights [num_tokens, k].
        topk_ids (torch.Tensor): Top-k expert IDs for each token [num_tokens, k].
        inplace (bool): If True, modify the hidden_states tensor in place.
    
    Returns:
        torch.Tensor: Output token embeddings after the MoE computation.
    """
    # Check dimensions
    num_tokens, hidden_size = hidden_states.shape
    E, N, _ = w13.shape
    assert hidden_states.shape[1] == w13.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "Top-k shape mismatch"
    
    k = topk_ids.shape[1]  # Number of selected experts per token
    output_size = hidden_size  # Output size is same as hidden size
    
    # Initialize output
    out_hidden_states = hidden_states if inplace else torch.zeros(
        (num_tokens, output_size), device="cpu", dtype=hidden_states.dtype
    )
    
    # Iterate over tokens to compute MoE outputs
    for token_idx in range(num_tokens):
        token_embedding = hidden_states[token_idx]  # [hidden_size]
        token_output = torch.zeros(output_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for expert_rank in range(k):
            expert_id = topk_ids[token_idx, expert_rank].item()%4  # temp: only 4 experts available
            expert_weight = topk_weights[token_idx, expert_rank].item()
            
            # Fetch expert weights from w13 (merged w1 and w3)
            w1_expert, w3_expert = torch.chunk(w13[expert_id], 2, dim=0)  # [N, hidden_size] each
            w2_expert = w2[expert_id]  # [N, output_size]

            # print(f"w1_expert: {w1_expert.shape}, w3_expert: {w3_expert.shape}")
            # print(f"w2_expert: {w2_expert.shape}")

            # Compute expert output using GLU mechanism
            intermediate1 = torch.matmul(token_embedding, w1_expert.T)  # [N]
            intermediate2 = torch.matmul(token_embedding, w3_expert.T)  # [N]
            
            # print(f"intermediate1: {intermediate1.shape}, intermediate2: {intermediate2.shape}")

            # Apply SiLU activation and gate using w3
            activated = torch.nn.functional.silu(intermediate1) * intermediate2  
            
            # print(f"activated: {activated.shape}")

            # Compute final expert output
            expert_output = torch.matmul(activated, w2_expert.T)  # [output_size]
            
            # print(f"expert_output: {expert_output.shape}")
            
            # print(f"token_output: {token_output.shape}")
            
            # print(f"expert_weight: {expert_weight}")

            # Weighted sum across experts
            token_output += expert_output * expert_weight

        
        # Store the token's output
        out_hidden_states[token_idx] = token_output
    
    return out_hidden_states

class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        available_experts: Optional[List[bool]] = None,
    ):
        super().__init__()
        # self.available_experts = available_experts or [True] * num_experts
        self.available_experts = available_experts or [True] * 4 + [False] * (num_experts - 4) # temp: only 4 experts available
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.hidden_size = hidden_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod()
            )
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        assert self.quant_method is not None

        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
            available_experts=self.available_experts,
        )


    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):
        # Load grouped weight scales for group quantization
        # or model weights
        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.tensor,
        tp_rank: int,
    ):

        if shard_id == "w2":
            self._load_w2(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        
        # skip loading if the weight is not available
        if not self.available_experts[expert_id]:
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        loaded_weight = (
            loaded_weight.t().contiguous()
            if (
                self.quant_method.__class__.__name__
                == "CompressedTensorsWNA16MoEMethod"
            )
            else loaded_weight
        )

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]
        tp_rank = get_tensor_model_parallel_rank()

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = ~shard_dim

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return

        # Case weight scales and zero_points
        if "scale" in weight_name or "zero" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.GROUP.value:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
            return
    def swap_experts(
        self,
        src_gpu: int,
        dst_gpu: int,
        expert_id: int,
    ):
        """
        Swap expert weights between GPUs using NCCL.

        Args:
            src_gpu (int): Source GPU ID.
            dst_gpu (int): Destination GPU ID.
            expert_id (int): ID of the expert to swap.
        """
        # Ensure NCCL is initialized
        if not dist.is_initialized():
            raise RuntimeError("Distributed environment not initialized.")

        # Get the expert weights
        expert_weights_src = {
            "w13": self.w13_weight[expert_id].to(src_gpu),
            "w2": self.w2_weight[expert_id].to(src_gpu),
        }

        # Create placeholder tensors on the destination GPU
        expert_weights_dst = {
            "w13": torch.empty_like(expert_weights_src["w13"], device=dst_gpu),
            "w2": torch.empty_like(expert_weights_src["w2"], device=dst_gpu),
        }

        # Use NCCL to transfer weights
        dist.broadcast(expert_weights_src["w13"], src=src_gpu, group=dist.group.WORLD)
        dist.broadcast(expert_weights_src["w2"], src=src_gpu, group=dist.group.WORLD)

        # Update weights on the destination GPU
        self.w13_weight[expert_id].data = expert_weights_dst["w13"]
        self.w2_weight[expert_id].data = expert_weights_dst["w2"]

        # Update availability status
        self.available_experts[expert_id] = True
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        is_decode_mode: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ):
        from sglang.srt.layers.fused_moe_triton.fused_moe import (
            fused_topk,
            grouped_topk,
        )

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        elif custom_routing_function is None:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                is_decode_mode=is_decode_mode,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
        # Adjust for pruned experts
        topk_ids_adjusted = []
        is_remote = [False] * len(topk_ids)
        
        for token_idx, token_topk_ids in enumerate(topk_ids):
            for expert_id in token_topk_ids:
                if not self.available_experts[expert_id]:
                    # print(f"[WARNING] Token {token_idx} has pruned expert {expert_id}.")
                    is_remote[token_idx] = True
            
            # adjusted_ids = [
            #     expert_id for expert_id in token_topk_ids if self.available_experts[expert_id]
            # ]
            # # If no available expert, fallback to the first available expert
            # if len(adjusted_ids) < top_k:
            #     print(f"[WARNING] Token {token_idx} has less than {top_k} available experts.")
            #     adjusted_ids += [
            #         idx for idx, available in enumerate(self.available_experts) if available
            #     ][: top_k - len(adjusted_ids)]
            # topk_ids_adjusted.append(adjusted_ids)

        # topk_ids = torch.tensor(topk_ids_adjusted, device=router_logits.device)
        
        is_remote = torch.tensor(is_remote, device=router_logits.device)
        # print(f"topk_ids: {topk_ids}")
        return topk_weights, topk_ids, is_remote

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, is_decode_mode: bool, residual: torch.Tensor, forward_batch: ForwardBatch):
        assert self.quant_method is not None
        
        exchanged_hidden_states = hidden_states
        
        # ### Try 1: All-to-all communication (route to other GPUs and route it back)

        # # Create CUDA streams for asynchronous operations
        # stream = torch.cuda.Stream(device=hidden_states.device)

        # # Define the portion of rows to exchange (e.g., first 50% of rows)
        # num_rows = hidden_states.size(0)
        # rows_to_exchange = num_rows // 2  # Adjust the portion as needed

        # # Split `hidden_states` into parts to send and keep
        # hidden_to_send = hidden_states[:rows_to_exchange]
        # hidden_to_keep = hidden_states[rows_to_exchange:]

        # # Prepare a tensor to store received rows from the other GPU
        # received_hidden = torch.zeros_like(hidden_to_send, device=hidden_states.device)

        # # Use torch.distributed.all_to_all to exchange rows between GPUs
        # with torch.cuda.stream(stream):
        #     # Perform the all-to-all communication
        #     torch.distributed.all_to_all_single(
        #         received_hidden,
        #         hidden_to_send
        #     )
            
        

        # # Synchronize the stream to ensure communication is complete
        # stream.synchronize()
        
        # Recombine the received rows with the remaining `hidden_states`
        # exchanged_hidden_states = torch.cat([received_hidden, hidden_to_keep], dim=0)
        
        # ### Try 2: Compute those on CPU asynchrously and move them back to GPU once it's done
        
        
        
        
        # # Move experts dynamically
        
        # very_large_tensor_to_send = torch.randn(80000000//1000, device=hidden_states.device)
        # very_large_tensor_to_receive = torch.zeros_like(very_large_tensor_to_send)
        # with torch.cuda.stream(stream):
        #     torch.distributed.all_to_all_single(
        #         very_large_tensor_to_receive,
        #         very_large_tensor_to_send
        #     )
        # stream.synchronize()
        



        # Perform the quantized matrix multiply on the exchanged hidden states
        final_hidden_states, residual, forward_batch = self.quant_method.apply(
            layer=self,
            x=exchanged_hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            is_decode_mode=is_decode_mode,
            residual=residual,
            forward_batch=forward_batch,
        )

        # Perform a tensor parallel all-reduce if necessary
        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states, residual, forward_batch


    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def _load_fp8_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        if "input_scale" in weight_name:
            if (
                param_data[expert_id] != 1
                and (param_data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}"
                )
            param_data[expert_id] = loaded_weight
        # Weight scales
        elif "weight_scale" in weight_name:
            # If we are in merged column case (gate_up_proj)
            if shard_id in ("w1", "w3"):
                # We have to keep the weight scales of w1 and w3 because
                # we need to re-quantize w1/w3 weights after weight loading.
                idx = 0 if shard_id == "w1" else 1
                param_data[expert_id][idx] = loaded_weight
            # If we are in the row parallel case (down_proj)
            else:
                param_data[expert_id] = loaded_weight
