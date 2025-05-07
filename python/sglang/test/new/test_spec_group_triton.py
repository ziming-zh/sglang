import torch
import triton
import triton.language as tl
import time

@triton.jit
def specgroup_kernel(
    token_expert_preds_ptr,  # (N, L, K)
    group_expert_labels_ptr, # (G, L, L2)
    output_token_to_group_ptr, # (N,)
    N, G, L, K, L2,
    stride_n_l, stride_n_k,
    stride_g_l, stride_g_l2,
):
    pid = tl.program_id(0)  # each program handles one token
    if pid >= N:
        return

    best_score = tl.zeros((), dtype=tl.int32)
    best_group = tl.zeros((), dtype=tl.int32)

    # Loop over groups
    for g in range(G):
        score = tl.zeros((), dtype=tl.int32)

        # For each layer
        for l in range(L):
            for k in range(K):
                token_expert_idx = pid * stride_n_l + l * stride_n_k + k
                token_expert = tl.load(token_expert_preds_ptr + token_expert_idx)

                for l2 in range(L2):
                    group_expert_idx = g * stride_g_l + l * stride_g_l2 + l2
                    group_expert = tl.load(group_expert_labels_ptr + group_expert_idx)

                    score += tl.where(token_expert == group_expert, 1, 0)

        # Update best group
        if score > best_score:
            best_score = score
            best_group = g

    # Write result
    tl.store(output_token_to_group_ptr + pid, best_group)

# -------- Wrapper Function --------

def fast_partition_triton(token_expert_preds, group_expert_labels):
    """
    Args:
        token_expert_preds: (N, L, K) LongTensor
        group_expert_labels: (G, L, L2) LongTensor
    Returns:
        token_to_group: (N,) LongTensor
    """
    N, L, K = token_expert_preds.shape
    G, _, L2 = group_expert_labels.shape

    device = token_expert_preds.device
    assert device.type == 'cuda', "Triton kernel requires CUDA"

    # Output buffer
    token_to_group = torch.empty((N,), dtype=torch.long, device=device)

    # Launch Triton
    grid = lambda meta: (N,)

    specgroup_kernel[grid](
        token_expert_preds_ptr=token_expert_preds,
        group_expert_labels_ptr=group_expert_labels,
        output_token_to_group_ptr=token_to_group,
        N=N, G=G, L=L, K=K, L2=L2,
        stride_n_l=token_expert_preds.stride(1),
        stride_n_k=token_expert_preds.stride(2),
        stride_g_l=group_expert_labels.stride(1),
        stride_g_l2=group_expert_labels.stride(2),
    )

    return token_to_group

def rebalance_tokens(
    token_to_group: torch.Tensor,
    token_expert_preds: torch.Tensor,
    group_expert_labels: torch.Tensor,
    max_tokens_per_group: int,
):
    """
    Args:
        token_to_group: (N,) LongTensor - initial assignment after Triton
        token_expert_preds: (N, L, K) LongTensor - token predictions
        group_expert_labels: (G, L, L2) LongTensor - group expert labels
        max_tokens_per_group: int
    Returns:
        token_to_group_rebalanced: (N,) LongTensor - after load balancing
    """

    N, L, K = token_expert_preds.shape
    G, _, L2 = group_expert_labels.shape
    device = token_to_group.device

    counts = torch.bincount(token_to_group, minlength=G)
    overload = counts > max_tokens_per_group

    if not overload.any():
        return token_to_group  # already balanced

    print(f"⚡ Rebalancing {overload.sum().item()} overloaded groups...")

    # Precompute overlap scores for second-best assignment
    token_exp = token_expert_preds[:, None, :, :, None].expand(N, G, L, K, L2)
    group_exp = group_expert_labels[None, :, :, None, :].expand(N, G, L, K, L2)

    match = (token_exp == group_exp)
    match = match.any(dim=-1).float()  # (N, G, L, K)
    overlap_scores = match.sum(dim=-1).sum(dim=-1)  # (N, G)

    rebalanced_token_to_group = token_to_group.clone()

    for g in overload.nonzero(as_tuple=False).flatten():
        mask = (rebalanced_token_to_group == g)
        num_to_reassign = (counts[g] - max_tokens_per_group).item()

        if num_to_reassign == 0:
            continue

        group_scores = overlap_scores[mask, g]  # (tokens_in_g,)
        worst_indices = group_scores.argsort()[:num_to_reassign]

        candidates = overlap_scores[mask][worst_indices]
        candidates[:, g] = -1  # Mask out current group

        second_best = candidates.argmax(dim=-1)

        idx_in_global = mask.nonzero(as_tuple=False).flatten()[worst_indices]
        rebalanced_token_to_group[idx_in_global] = second_best

    return rebalanced_token_to_group


def test_fast_partition_triton():
    torch.manual_seed(42)

    # Config
    N = 300
    G = 8
    L = 8
    K = 4
    L2 = 4
    num_experts = 64
    max_tokens_per_group = (N + G - 1) // G
    device = "cuda" if torch.cuda.is_available() else "cpu"

    token_expert_preds = torch.randint(0, num_experts, (N, L, K), device=device)
    group_expert_labels = torch.randint(0, num_experts, (G, L, L2), device=device)

    # Warmup
    fast_partition_triton(token_expert_preds, group_expert_labels)
    rebalance_tokens(
        torch.zeros(N, dtype=torch.long, device=device),
        token_expert_preds,
        group_expert_labels,
        max_tokens_per_group
    )

    # Timing
    torch.cuda.synchronize()
    start = time.time()
    token_to_group = fast_partition_triton(token_expert_preds, group_expert_labels)
    # token_to_group = rebalance_tokens(token_to_group, token_expert_preds, group_expert_labels, max_tokens_per_group)
    torch.cuda.synchronize()
    end = time.time()

    runtime_ms = (end - start) * 1000

    print(f"[SpecGroup Triton] Partitioning runtime: {runtime_ms:.3f} ms")
    print(f"[SpecGroup Triton] Token to group assignment shape: {token_to_group.shape}")
    
    # Check
    assert token_to_group.shape == (N,)
    assert ((token_to_group >= 0) & (token_to_group < G)).all()

    counts = torch.bincount(token_to_group, minlength=G)
    print(f"[SpecGroup Multi-layer] Token counts per group: {counts.tolist()}")
    print(f"[SpecGroup Multi-layer] Partitioning runtime: {runtime_ms:.3f} ms")

    # Slack: allow up to +1 token due to integer rounding
    assert (counts <= max_tokens_per_group + 1).all()

if __name__ == "__main__":
    test_fast_partition_triton()