from dataclasses import dataclass
import logging
from multiprocessing.shared_memory import SharedMemory
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module
from vllm import _custom_ops as ops
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

from python.sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.layers.custom_op_util import register_custom_op
from sglang.srt.layers.ep_moe.kernels import (
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_triton_kernel,
)
from sglang.srt.layers.fused_moe_triton.fused_moe import fused_topk, grouped_topk
from sglang.srt.layers.fused_moe_triton.layer import FusedMoEMethodBase
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import is_hip, set_weight_attrs

logger = logging.getLogger(__name__)

class TaskCounter:
    def __init__(self):
        self.task_count = 1

    def get_task_id(self):
        self.task_count += 1
        if self.task_count > 999:
            self.task_count = 1
        return self.task_count
    
task_counter = TaskCounter()


@dataclass
class Task:
    task_id: int
    layer_id: int
    x_shm_name: str
    x_shape: tuple
    x_dtype: torch.dtype
    topk_weights_shm_name: str
    topk_weights_shape: tuple
    topk_weights_dtype: torch.dtype
    topk_ids_shm_name: str
    topk_ids_shape: tuple
    topk_ids_dtype: torch.dtype
    
@dataclass
class TaskResult:
    task_id: int
    layer_id: int
    cpu_result_name: str
    cpu_result_shape: tuple
    cpu_result_dtype: torch.dtype

TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
    torch.bfloat16: np.float32  # Store bfloat16 as float32
}

def create_shared_memory_tensor(tensor):
    """Create shared memory and store a tensor in it."""
    # Convert bfloat16 to float32 before storing
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    
    shm = SharedMemory(create=True, size=tensor.numel() * tensor.element_size())
    np_array = np.ndarray(tensor.shape, dtype=TORCH_TO_NUMPY_DTYPE[tensor.dtype], buffer=shm.buf)
    np_array[:] = tensor.numpy()  # Copy tensor data into shared memory
    return shm

def load_shared_memory_tensor(shm_name, shape, dtype):
    """Load a tensor from shared memory."""
    np_dtype = TORCH_TO_NUMPY_DTYPE[dtype]  # Convert torch dtype to numpy dtype
    shm = SharedMemory(name=shm_name)
    np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

    # Convert float32 back to bfloat16 if needed
    if dtype == torch.bfloat16:
        return torch.tensor(np_array, dtype=torch.float32).to(torch.bfloat16), shm
    return torch.tensor(np_array, dtype=dtype), shm


def cpu_offload_worker(task_pipe, complete_token_manager, w13_cpu, w2_cpu):
    """
    Worker function that runs in a separate process to handle CPU offloading.
    """
    while True:
        # try:
        # print(f"Worker process waiting for task")
        task = task_pipe.recv()
        # print(f"Worker process received task {task.task_id}")
        if task is None:
            # print("Received stop signal, exiting worker process")
            break  # Stop the worker when None is received
        start_time = time.time()
        print(f"[Offload Worker] task {task.task_id} received at time {time.time()}")
        
        # Load all tensors from shared memory
        x_remote_cpu, x_shm = load_shared_memory_tensor(task.x_shm_name, task.x_shape, task.x_dtype)
        topk_weights, topk_weights_shm = load_shared_memory_tensor(task.topk_weights_shm_name, task.topk_weights_shape, task.topk_weights_dtype)
        topk_ids, topk_ids_shm = load_shared_memory_tensor(task.topk_ids_shm_name, task.topk_ids_shape, task.topk_ids_dtype)
        
        # Perform CPU computation
        cpu_result = fused_experts_cpu_impl(
            hidden_states=x_remote_cpu,
            w13=w13_cpu,
            w2=w2_cpu,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        
        cpu_result_shm = create_shared_memory_tensor(cpu_result)

        # if result_queue.full():
        #     print(f"Result queue is full, failed to put task {task.task_id}")
        #     pass
        # else:
        #     task_result = TaskResult(task.task_id, task.layer_id, cpu_result)
        #     result_queue.put(task_result)
        
        task_result = TaskResult(task.task_id, task.layer_id, cpu_result_shm.name, cpu_result.shape, cpu_result.dtype)
        task_pipe.send(task_result)
        
        # Cleanup shared memory references
        x_shm.close()
        topk_weights_shm.close()
        topk_ids_shm.close()
        
        
        # except Exception as e:
        #     print(f"Error in CPU offloading worker: {e}")
        end_time = time.time()
        print(f"[Offload Worker] task {task.task_id} completed from {start_time} to {end_time} in {end_time-start_time} seconds")

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
    
    # # Iterate over tokens to compute MoE outputs
    # for token_idx in range(num_tokens):
    #     token_embedding = hidden_states[token_idx]  # [hidden_size]
    #     token_output = torch.zeros(output_size, device=hidden_states.device, dtype=hidden_states.dtype)
        
    #     for expert_rank in range(k):
    #         expert_id = topk_ids[token_idx, expert_rank].item()%4  # temp: only 4 experts available
    #         expert_weight = topk_weights[token_idx, expert_rank].item()
            
    #         # Fetch expert weights from w13 (merged w1 and w3)
    #         w1_expert, w3_expert = torch.chunk(w13[expert_id], 2, dim=0)  # [N, hidden_size] each
    #         w2_expert = w2[expert_id]  # [N, output_size]

    #         # print(f"w1_expert: {w1_expert.shape}, w3_expert: {w3_expert.shape}")
    #         # print(f"w2_expert: {w2_expert.shape}")

    #         # Compute expert output using GLU mechanism
    #         intermediate1 = torch.matmul(token_embedding, w1_expert.T)  # [N]
    #         intermediate2 = torch.matmul(token_embedding, w3_expert.T)  # [N]
            
    #         # print(f"intermediate1: {intermediate1.shape}, intermediate2: {intermediate2.shape}")

    #         # Apply SiLU activation and gate using w3
    #         activated = torch.nn.functional.silu(intermediate1) * intermediate2  
            
    #         # print(f"activated: {activated.shape}")

    #         # Compute final expert output
    #         expert_output = torch.matmul(activated, w2_expert.T)  # [output_size]
            
    #         # print(f"expert_output: {expert_output.shape}")
            
    #         # print(f"token_output: {token_output.shape}")
            
    #         # print(f"expert_weight: {expert_weight}")

    #         # Weighted sum across experts
    #         token_output += expert_output * expert_weight

        
    #     # Store the token's output
    #     out_hidden_states[token_idx] = token_output
    
    return out_hidden_states

class GroupedGemmRunner(torch.nn.Module):
    flashinfer_gemm_warpper = None

    def __init__(self, device, use_flashinfer: bool = False):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        if self.use_flashinfer and GroupedGemmRunner.flashinfer_gemm_warpper is None:
            GroupedGemmRunner._init_flashinfer_wrapper(device)

    @classmethod
    def _init_flashinfer_wrapper(cls, device):
        from flashinfer import SegmentGEMMWrapper

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        cls.flashinfer_gemm_warpper = SegmentGEMMWrapper(workspace_buffer)

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
    ):
        if self.use_flashinfer:
            # TODO: flashinfer
            assert False
            assert GroupedGemmRunner.flashinfer_gemm_warpper is not None
            c = GroupedGemmRunner.flashinfer_gemm_warpper.run(
                x=a,
                weights=b,
                batch_size=batch_size,
                weight_column_major=weight_column_major,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
            )
        else:
            assert weight_column_major == True
            c = grouped_gemm_triton(
                a,
                b,
                c,
                batch_size,
                weight_column_major,
                seg_indptr,
                weight_indices,
                use_fp8_w8a8,
                scale_a,
                scale_b,
            )
        return c


class EPMoE(torch.nn.Module):
    """
    MoE Expert Parallel Impl


    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        layer_id: Optional[int] = None,
        available_experts: Optional[List[bool]] = None,
    ):
        super().__init__()
        pruned_top_k = 4
        # self.available_experts = available_experts or [True] * num_experts
        self.available_experts = available_experts or [True] * pruned_top_k + [False] * (num_experts - pruned_top_k) # temp: only 4 experts available
        self.num_available_experts = sum(self.available_experts)
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = get_tensor_model_parallel_rank()

        self.layer_id = layer_id
        self.num_experts = num_experts
        assert self.num_experts % self.tp_size == 0
        self.num_experts_per_partition = self.num_available_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedEPMoEMethod()
            self.use_fp8_w8a8 = False
            self.activation_scheme = None
        else:
            self.quant_method: Optional[QuantizeMethodBase] = Fp8EPMoEMethod(
                quant_config
            )
            self.use_fp8_w8a8 = True
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme

        self.quant_method.create_weights(
            layer=self,
            num_experts_per_partition=self.num_experts_per_partition,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )
        self.round_id = 0
        # set up CUDA streams for async offloading
        self.stream_cpu = torch.cuda.Stream()
        self.stream_gpu = torch.cuda.Stream()
        

        self.grouped_gemm_runner = None

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor, is_decode_mode: bool, forward_batch: ForwardBatch, residual: Optional[torch.Tensor] = None, parent_task_pipe=None, task_metadata=None, cpu_result_buffer=None):
        assert self.quant_method is not None
        self.round_id += 1
        self.task_metadata = task_metadata
        self.cpu_buffer = cpu_result_buffer
        self.parent_task_pipe = parent_task_pipe

        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device, use_flashinfer=False  # TODO: use flashinfer
            )
            
        forward_cuda_start = time.time()

        topk_weights, topk_ids, is_remote_toks = self.select_experts(
            hidden_states,
            router_logits,
            self.top_k,
            self.renormalize,
            is_decode_mode,
            self.topk_group,
            self.num_expert_group,
        )
        x_remote = hidden_states[is_remote_toks]
        x_local = hidden_states[~is_remote_toks]
        num_seqs = hidden_states.shape[0]
        
        print(f"[Layer {self.layer_id} SPLIT] x_remote: {x_remote.shape}, x_local: {x_local.shape}, cuda {x_local.device}")

        topk_weights_remote = topk_weights[is_remote_toks]
        topk_ids_remote = topk_ids[is_remote_toks]

        topk_weights_local = topk_weights[~is_remote_toks]
        topk_ids_local = topk_ids[~is_remote_toks]
        if is_decode_mode:
            forward_batch_local, forward_batch_remote = forward_batch.split(is_remote_toks)
        else:
            forward_batch_local = forward_batch
            forward_batch_remote = None
        
                    
        if residual is not None:
            residual_remote = residual[is_remote_toks]
            residual_local = residual[~is_remote_toks]
        else:
            residual_remote = None
            residual_local = None

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids_local, self.num_experts
        )
        
        if is_decode_mode:
            # split
            split_start = time.time()
            
            # Buffer x_remote to avoid small CPU offloads
            if not hasattr(self, "remote_forward_batch") or self.remote_forward_batch is None:
                self.remote_buffer = []
                self.remote_forward_batch_list = []

            if x_remote.numel() > 0:
                # print(f"Combining {x_remote.numel()} remote tokens, cuda {x_remote.device}")
                # self.remote_forward_batch = self.remote_forward_batch.combine(forward_batch_remote)
                self.remote_forward_batch_list.append(forward_batch_remote)
                self.remote_buffer.append((x_remote, topk_weights_remote, topk_ids_remote, residual_remote))
            # Only offload when buffer size is 20 or more
            # Offload when buffer size reaches threshold
            if len(self.remote_buffer) >= 1:
                # print(f"Offloading {len(self.remote_buffer)} remote tokens to CPU, cuda {x_remote.device}")
                
                # generate a unique task ID
                # task_id = random.randint(1, 999)
                task_id = task_counter.get_task_id()

                print(f"[Layer {self.layer_id}] Offloading task {task_id} to CPU at {time.time()}")

                with torch.cuda.stream(self.stream_cpu):
                    x_remote_cpu = torch.cat([item[0] for item in self.remote_buffer], dim=0).to("cpu")
                    self.remote_forward_batch = forward_batch_remote.combine(self.remote_forward_batch_list[:-1])
                    # print(f"offload task {task_id} dispatched at time {time.time()}, cuda {x_remote_cpu.device}")

                    topk_weights_remote_cpu = torch.cat([item[1] for item in self.remote_buffer], dim=0).to("cpu")
                    topk_ids_remote_cpu = torch.cat([item[2] for item in self.remote_buffer], dim=0).to("cpu")

                    if residual_remote is not None:
                        residual_remote_cpu = torch.cat([item[3] for item in self.remote_buffer], dim=0)
                    else:
                        residual_remote_cpu = None
                        
                    # Store in lookup dictionary instead of passing to task_queue
                    self.task_metadata[self.layer_id][task_id] = (residual_remote_cpu, forward_batch_remote)
                    
                    # Create shared memory for x_remote_cpu, topk_weights_remote_cpu, topk_ids_remote_cpu
                    x_shm = create_shared_memory_tensor(x_remote_cpu)
                    topk_weights_shm = create_shared_memory_tensor(topk_weights_remote_cpu)
                    topk_ids_shm = create_shared_memory_tensor(topk_ids_remote_cpu)
                    
                    # Send task to worker process
                    task = Task(
                        task_id=task_id,
                        layer_id=self.layer_id,
                        x_shm_name=x_shm.name,
                        x_shape=x_remote_cpu.shape,
                        x_dtype=x_remote_cpu.dtype,
                        topk_weights_shm_name=topk_weights_shm.name,
                        topk_weights_shape=topk_weights_remote_cpu.shape,
                        topk_weights_dtype=topk_weights_remote_cpu.dtype,
                        topk_ids_shm_name=topk_ids_shm.name,
                        topk_ids_shape=topk_ids_remote_cpu.shape,
                        topk_ids_dtype=topk_ids_remote_cpu.dtype,
                    )
                    parent_task_pipe.send(task)
                    print(f"Task {task_id} sent to worker process at {time.time()}, cuda {x_remote_cpu.device}")
                # Clear the buffer
                self.remote_buffer = []
                
                self.remote_forward_batch = None
            split_end = time.time()
            print(f"[Layer {self.layer_id}] Splitting remote tokens took {split_end - split_start} seconds")
    
        hidden_states = x_local
        topk_weights = topk_weights_local
        topk_ids = topk_ids_local
        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=self.fp8_dtype if self.use_fp8_w8a8 else hidden_states.dtype,
        )
        if self.activation_scheme == "dynamic":
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_experts_per_partition)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max

        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            self.w13_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
        )

        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )
        # GroupGemm-0
        gateup_output = torch.empty(
            gateup_input.shape[0],
            self.w13_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=self.w13_weight,
            c=gateup_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=self.w13_weight_scale,
        )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=self.fp8_dtype if self.use_fp8_w8a8 else hidden_states.dtype,
        )
        if self.w2_input_scale is None:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states.device,
            )
        silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
            gateup_output,
            down_input,
            gateup_output.shape[1],
            reorder_topk_ids,
            self.w2_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            BLOCK_SIZE=512,
        )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=self.w2_weight_scale,
        )

        # PostReorder
        output = torch.empty_like(hidden_states)
        post_reorder_triton_kernel[(hidden_states.size(0),)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.size(1),
            BLOCK_SIZE=512,
        )
        x_local = output
        
        if is_decode_mode:

            # retrieve results from the worker process
            
            # print("[FORWARD_CUDA] local input shape", x_local.shape, x_local.device)
            
            # retrieval
            # retrieval_start = time.time()
            
            # Check for completed CPU computations and move back to GPU
            fetched_cpu_results = []
            fetched_residuals = []
            fetched_forward_batch = []
            device = x_local.device
            self.retrieve_results()
            # query_and_retrieve_start = time.time()
            finished_tasks = self.complete_token_manager.query(self.round_id, self.layer_id)

            # print(f"[Layer {self.layer_id} TP-RANK {get_tensor_model_parallel_rank()}] retrieved results at {time.time()}, finished tasks: {finished_tasks}", flush=True)
            self.retrieve_results()
            # query_and_retrieve_end = time.time()
            # print(f"[Layer {self.layer_id}] Query and retrieve from {query_and_retrieve_start} to {query_and_retrieve_end} in {query_and_retrieve_end-query_and_retrieve_start} seconds")
            while not x_local.numel() and len(finished_tasks)<int(num_seqs*0.8)+1:
                time.sleep(0.01)
                self.round_id += 1
                self.retrieve_results()
                # stall here if x_local is empty and no remote tasks are finished
                finished_tasks.extend(self.complete_token_manager.query(self.round_id, self.layer_id))
                print(f"[Layer {self.layer_id} TP-RANK {get_tensor_model_parallel_rank()}] retrieved results at {time.time()}, round: {self.round_id}, finished tasks: {finished_tasks}")
                self.retrieve_results()
            
            # retrieval_end = time.time()
            # print(f"[Layer {self.layer_id}] Retrieved results from {retrieval_start} to {retrieval_end} in {retrieval_end-retrieval_start} seconds")
            # combination
            # combination_start = time.time()
                

            for key in finished_tasks:
                # if key not in self.cpu_buffer:
                #     self.unfinished_tasks.append(key)
                #     # print(f"[Layer {self.layer_id}] Task {key} not found in CPU buffer, keys in buffer: {self.cpu_buffer.keys()}, keys in finished_tasks: {finished_tasks}", flush=True)
                #     continue
                # try:
                #     cpu_result = self.cpu_buffer[key]
                # except KeyError as e:
                #     assert False, (f"[Time {time.time()}] Task {key} not found in CPU buffer, keys in buffer: {self.cpu_buffer.keys()}, keys in finished_tasks: {finished_tasks}")
                # print(f"[Task {key}] finished at {time.time()}")
                # with torch.cuda.stream(self.stream_gpu):
                cpu_result = self.cpu_buffer[self.layer_id][key]
                gpu_result = cpu_result[0].to(device)
                residual_gpu = cpu_result[1].to(device) if cpu_result[1] is not None else None
                forward_batch_remote = cpu_result[2]
                fetched_cpu_results.append(gpu_result)
                fetched_residuals.append(residual_gpu)
                fetched_forward_batch.append(forward_batch_remote)
                # print(f"[Combine] local input shape {x_local.shape} combined with {gpu_result.shape}")
                del self.cpu_buffer[self.layer_id][key]
            

            if len(fetched_cpu_results) > 0:
                print(f"[Layer {self.layer_id}] Combined {len(fetched_cpu_results)} remote results at {time.time()}")
                # print(f"fetched_gpu_results: {fetched_cpu_results}")
                x_local = torch.cat([x_local] + fetched_cpu_results, dim=0)
                if residual_local is not None:
                    residual_local = torch.cat([residual_local] + fetched_residuals, dim=0)
                # print(f"[Combine before] fetched_batch_local.out_cache_loc: {forward_batch_local.out_cache_loc}")
                forward_batch_local.combine(fetched_forward_batch)
                # print(f"[Combine after] fetched_batch_local.out_cache_loc: {forward_batch_local.out_cache_loc}")
                # synchronize streams
                torch.cuda.synchronize()
                    
            # print(f"[FORWARD_CUDA] local input shape after combine", x_local.shape)
            # print(f"[FORWARD_CUDA] local forward_batch", forward_batch_local.out_cache_loc) 
            
            # forward_batch_local.is_local_toks = (~is_remote_toks).nonzero(as_tuple=True)[0]
            # print(f"[FORWARD_CUDA] local forward_batch.is_local_toks", forward_batch_local.is_local_toks)
            # combination_end = time.time()
            # print(f"[TP-RANK {get_tensor_model_parallel_rank()}] Combination time: {combination_end-combination_start} seconds")
        # forward_cuda_end = time.time()
        # print(f"[Layer {self.layer_id}] Forward CUDA from {forward_cuda_start} to {forward_cuda_end} in {forward_cuda_end-forward_cuda_start} seconds")
        return x_local, residual_local, forward_batch_local

    def retrieve_results(self):
        """
        Retrieve results from the worker process.
        Should be called periodically to check for completed tasks.
        """
        # if self.result_queue.empty():
        #     print(f"[layer {self.layer_id}] Result queue is empty at {time.time()}")
        # while not self.result_queue.empty():
        # try:
        #     task_result = self.result_queue.get(block=block, timeout=timeout)
        # except Exception:
        #     return
        
        if self.parent_task_pipe.poll():
            task_result = self.parent_task_pipe.recv()
            
            # print(f"[Layer {self.layer_id}] Task {task_result.task_id} layer {task_result.layer_id} retrieved at {time.time()}")
        else:
            return
            
        
        cpu_result, cpu_result_shm = load_shared_memory_tensor(task_result.cpu_result_name, task_result.cpu_result_shape, task_result.cpu_result_dtype)
        
        if task_result is not None:
            self.complete_token_manager.update_token(task_result.task_id, task_result.layer_id)
            # print(f"Offload task {task_result.task_id} completed at time {time.time()}", flush=True)
            
            residual_remote_cpu, forward_batch_remote = self.task_metadata[task_result.layer_id][task_result.task_id]
            self.cpu_buffer[task_result.layer_id][task_result.task_id] = cpu_result, residual_remote_cpu, forward_batch_remote
            # remove the task metadata
            del self.task_metadata[task_result.layer_id][task_result.task_id]
            # print(f"[layer {self.layer_id}] Task {task_id} retrieved at {time.time()}")
            
        # cleanup shared memory
        cpu_result_shm.close()
        # print(f"[Layer {self.layer_id}] Task {task_result.task_id} retrieved and cleaned up at {time.time()}")
        
    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        is_decode_mode: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
    ):
        if self.use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                is_decode_mode = is_decode_mode,
            )
        else:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                is_decode_mode = is_decode_mode,
            )
        is_remote = [False] * len(topk_ids)
        
        for token_idx, token_topk_ids in enumerate(topk_ids):
            for expert_id in token_topk_ids:
                if not self.available_experts[expert_id]:
                    # print(f"[WARNING] Token {token_idx} has pruned expert {expert_id}.")
                    is_remote[token_idx] = True
        is_remote = torch.tensor(is_remote, device=router_logits.device)
        return topk_weights, topk_ids.to(torch.int32), is_remote

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

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if expert_id < self.start_expert_id or expert_id > self.end_expert_id:
            return
        expert_id = expert_id - self.start_expert_id

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Special case for fp8 scales.
        if "scale" in weight_name:
            self._load_fp8_scale(
                param.data, loaded_weight, weight_name, shard_id, expert_id
            )
            return

        expert_data = param.data[expert_id]
        if shard_id == "w2":
            param.data[expert_id] = loaded_weight
        elif shard_id == "w1":
            param.data[expert_id][: self.intermediate_size, :] = loaded_weight
        elif shard_id == "w3":
            param.data[expert_id][self.intermediate_size :, :] = loaded_weight
        else:
            raise ValueError(f"Expected shard_id w1,w2 or w3 but got {shard_id}")

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


@register_custom_op("sglang_unquantized_ep_moe")
class UnquantizedEPMoEMethod(FusedMoEMethodBase, CustomOp):
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        ones_tensor = torch.ones(num_experts_per_partition, dtype=torch.float32)
        w13_input_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Fp8EPMoEMethod(Fp8MoEMethod):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        # Allocate 2 scales for w1 and w3 respectively.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts_per_partition, 2, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts_per_partition, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update({"quant_method": "tensor"})
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If rocm, use float8_e4m3fnuz as dtype
            fp8_dtype = torch.float8_e4m3fnuz if is_hip() else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts_per_partition,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )

            for expert in range(layer.num_experts_per_partition):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError
