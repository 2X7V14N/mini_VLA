"""V1.3 三色抓放的组合加权与语言配对损失。"""

from itertools import combinations

import torch


# 顺序必须匹配 COMBOS：
# red->red, red->blue, red->yellow,
# blue->red, blue->blue, blue->yellow,
# yellow->red, yellow->blue, yellow->yellow
COMBO_WEIGHTS = (
    0.05, 0.05, 0.16,
    0.05, 0.05, 0.16,
    0.16, 0.16, 0.16,
)


def weighted_action_loss(prediction, target, combo_index, sample_mask=None):
    """对每条样本的动作 MSE 按颜色组合加权。"""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction 与 target 形状不同："
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim != 3:
        raise ValueError("prediction 和 target 必须是 (B, chunk, action_dim)")
    if combo_index.ndim != 1 or combo_index.shape[0] != prediction.shape[0]:
        raise ValueError("combo_index 必须是形状为 (B,) 的一维张量")

    combo_index = combo_index.to(device=prediction.device, dtype=torch.long)
    if torch.any(combo_index < 0) or torch.any(combo_index >= 9):
        raise ValueError("combo_index 的取值必须在 0..8")

    table = prediction.new_tensor(COMBO_WEIGHTS)
    weights = table[combo_index]
    if sample_mask is not None:
        if sample_mask.ndim != 1 or sample_mask.shape[0] != prediction.shape[0]:
            raise ValueError("sample_mask 必须是形状为 (B,) 的一维张量")
        weights = weights * sample_mask.to(
            device=prediction.device, dtype=prediction.dtype
        )
    if not torch.any(weights > 0):
        raise ValueError("至少需要一条有效动作标签")
    per_sample_mse = (prediction - target).square().mean(dim=(1, 2))
    return (per_sample_mse * weights).sum() / weights.sum()


def _weighted_pair_mse(
    pred_delta, target_delta, endpoint_pairs, valid_pair_mask=None
):
    """计算一类颜色变化的加权差分损失。"""
    if pred_delta.shape != target_delta.shape or pred_delta.ndim != 4:
        raise ValueError(
            "pair tensors 必须形状相同，且为 (batch, pairs, chunk, action_dim)"
        )
    if pred_delta.size(1) != len(endpoint_pairs):
        raise ValueError("差分数量与 endpoint_pairs 数量不一致")

    pair_mse = (pred_delta - target_delta).square().mean(dim=(2, 3))
    table = pred_delta.new_tensor(COMBO_WEIGHTS)
    pair_weights = torch.stack([
        0.5 * (table[left] + table[right])
        for left, right in endpoint_pairs
    ])
    pair_weights = pair_weights.unsqueeze(0).expand_as(pair_mse)
    if valid_pair_mask is not None:
        if valid_pair_mask.shape != pair_mse.shape:
            raise ValueError("valid_pair_mask 形状必须是 (batch, pairs)")
        pair_weights = pair_weights * valid_pair_mask.to(
            device=pair_mse.device, dtype=pair_mse.dtype
        )
    if not torch.any(pair_weights > 0):
        return pair_mse.new_zeros(())
    return (pair_mse * pair_weights).sum() / pair_weights.sum()


def compute_loss_v13(prediction, target, valid_mask=None, pair_weight=1.0):
    """计算 V1.3 三色抓放总损失。

    prediction/target: (batch_ticks, 9, chunk_size, action_dim)
    九条标签的顺序必须是三种方块颜色为外层、三种区域颜色为内层。
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction 与 target 形状必须相同")
    if prediction.ndim != 4 or prediction.size(1) != 9:
        raise ValueError(
            "prediction 和 target 必须是 (batch_ticks, 9, chunk, action_dim)"
        )

    batch_ticks, _, chunk_size, action_dim = prediction.shape
    if valid_mask is None:
        valid_mask = torch.ones(
            batch_ticks, 9, device=prediction.device, dtype=torch.bool
        )
    if valid_mask.shape != (batch_ticks, 9):
        raise ValueError("valid_mask 必须是形状为 (batch_ticks, 9)")
    valid_mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    if not torch.all(valid_mask.any(dim=1)):
        raise ValueError("每个 tick 至少需要一条有效标签")

    combo_index = torch.arange(9, device=prediction.device)
    combo_index = combo_index.unsqueeze(0).expand(batch_ticks, 9).reshape(-1)

    base = weighted_action_loss(
        prediction.reshape(-1, chunk_size, action_dim),
        target.reshape(-1, chunk_size, action_dim),
        combo_index,
        sample_mask=valid_mask.reshape(-1),
    )

    # 网格坐标：行是方块颜色，列是放置区颜色。
    pred_grid = prediction.reshape(batch_ticks, 3, 3, chunk_size, action_dim)
    target_grid = target.reshape(batch_ticks, 3, 3, chunk_size, action_dim)
    valid_grid = valid_mask.reshape(batch_ticks, 3, 3)

    # 方块颜色差分：固定放置区，比较红/蓝/黄三种方块。
    block_pairs = list(combinations(range(3), 2))
    pred_block_deltas = []
    target_block_deltas = []
    block_endpoints = []
    block_valid_pairs = []

    for zone in range(3):
        for block_a, block_b in block_pairs:
            pred_block_deltas.append(
                pred_grid[:, block_a, zone] - pred_grid[:, block_b, zone]
            )
            target_block_deltas.append(
                target_grid[:, block_a, zone] - target_grid[:, block_b, zone]
            )
            block_endpoints.append((block_a * 3 + zone, block_b * 3 + zone))
            block_valid_pairs.append(
                valid_grid[:, block_a, zone] & valid_grid[:, block_b, zone]
            )

    pred_block_deltas = torch.stack(pred_block_deltas, dim=1)
    target_block_deltas = torch.stack(target_block_deltas, dim=1)
    block_valid_pairs = torch.stack(block_valid_pairs, dim=1)
    block_pair = _weighted_pair_mse(
        pred_block_deltas,
        target_block_deltas,
        block_endpoints,
        valid_pair_mask=block_valid_pairs,
    )

    # 放置区颜色差分：固定方块颜色，比较红/蓝/黄三个放置区。
    pred_zone_deltas = []
    target_zone_deltas = []
    zone_endpoints = []
    zone_valid_pairs = []

    for block in range(3):
        for zone_a, zone_b in block_pairs:
            pred_zone_deltas.append(
                pred_grid[:, block, zone_a] - pred_grid[:, block, zone_b]
            )
            target_zone_deltas.append(
                target_grid[:, block, zone_a] - target_grid[:, block, zone_b]
            )
            zone_endpoints.append((block * 3 + zone_a, block * 3 + zone_b))
            zone_valid_pairs.append(
                valid_grid[:, block, zone_a] & valid_grid[:, block, zone_b]
            )

    pred_zone_deltas = torch.stack(pred_zone_deltas, dim=1)
    target_zone_deltas = torch.stack(target_zone_deltas, dim=1)
    zone_valid_pairs = torch.stack(zone_valid_pairs, dim=1)
    zone_pair = _weighted_pair_mse(
        pred_zone_deltas,
        target_zone_deltas,
        zone_endpoints,
        valid_pair_mask=zone_valid_pairs,
    )

    pair = 0.5 * (block_pair + zone_pair)
    total = base + pair_weight * pair
    return total, base, block_pair, zone_pair
