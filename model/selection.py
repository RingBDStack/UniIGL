import os
import torch
import torch.nn.functional as F
import random   
import numpy as np

def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


def information_select(
    logits_det: torch.Tensor,
    logits_mc: torch.Tensor,
    lambda_value: float,
    alpha: float,
    num: int,
    train_idx,
    labeled_nodes,
    class_balance_threshold: float = 0.05
):
    """
    Calculate information gain and select node pairs.

    Parameters:
    - logits_det: (N, C) First deterministic logits
    - logits_mc : (T, N, C) MC-Dropout logits
    - lambda_value: Mixup mixing coefficient
    - alpha: Second term weight coefficient
    - num: Number of node pairs to select
    - train_idx: Training set node indices
    - labeled_nodes: List of nodes grouped by class
    - class_balance_threshold: Threshold for determining if classes are balanced
    """
    train_idx = set(int(n) for n in train_idx)

    # Calculate class sizes and check balance
    class_sizes = [len(nodes) for nodes in labeled_nodes]
    max_size, min_size = max(class_sizes), min(class_sizes)
    class_balanced = (max_size - min_size) / max_size < class_balance_threshold

    # Sort classes
    sorted_cls = sorted(range(len(class_sizes)), key=lambda k: class_sizes[k])
    split = len(sorted_cls) // 2
    least_cls = sorted_cls[:split]
    most_cls  = sorted_cls[split:]
    # If classes are balanced, use all classes
    if class_balanced:
        least_cls = most_cls = sorted_cls

    # 收集节点
    nodes_least = [n for c in least_cls for n in labeled_nodes[c] if n in train_idx]
    nodes_most  = [n for c in most_cls  for n in labeled_nodes[c] if n in train_idx]
    if not nodes_least:
        nodes_least = list(train_idx)
    if not nodes_most:
        nodes_most  = list(train_idx)

    # 采样候选对
    sampled = set()
    while len(sampled) < num * 5:
        i = random.choice(nodes_least)
        j = random.choice(nodes_most)
        if i != j:
            sampled.add((i, j))

    # 预计算 softmax
    y_det = torch.softmax(logits_det, dim=-1)
    y_mc  = torch.softmax(logits_mc, dim=-1)

    def ent(p: torch.Tensor):
        p = p.clamp_min(1e-10)
        return -(p * p.log()).sum(dim=-1)

    # 计算信息增益
    ig_list = []
    for i, j in sampled:
        H1 = ent(lambda_value * y_det[i] + (1 - lambda_value) * y_det[j])
        mix_mc = lambda_value * y_mc[:, i, :] + (1 - lambda_value) * y_mc[:, j, :]
        H2 = ent(mix_mc).mean()
        IG = (H1 - alpha * H2).item()
        ig_list.append(((i, j), IG))

    # 按信息增益降序选取
    ig_list.sort(key=lambda x: x[1], reverse=True)
    return [(int(pair[0]), int(pair[1])) for pair, _ in ig_list[:num]]