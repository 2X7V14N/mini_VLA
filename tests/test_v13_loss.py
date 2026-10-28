import pytest
import torch

from mini_vla.training_v13 import (
    COMBO_WEIGHTS,
    weighted_action_loss,
    compute_loss_v13,
)


def test_v13_combo_weights_sum_to_80_20():
    assert sum(COMBO_WEIGHTS[index] for index in (0, 1, 3, 4)) == pytest.approx(0.20)
    assert sum(COMBO_WEIGHTS[index] for index in (2, 5, 6, 7, 8)) == pytest.approx(0.80)


def test_v13_weighted_action_loss_uses_combo_weights():
    # 每条样本的每个元素都相同，因此该样本的 MSE 就等于指定误差值。
    errors = torch.arange(9, dtype=torch.float32)
    target = torch.sqrt(errors).view(9, 1, 1).expand(9, 4, 3)
    prediction = torch.zeros_like(target)
    combo_index = torch.arange(9)

    loss = weighted_action_loss(prediction, target, combo_index)

    expected = sum(
        float(COMBO_WEIGHTS[i]) * float(errors[i]) for i in range(9)
    )
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_v13_weighted_action_loss_rejects_invalid_combo_index():
    prediction = torch.zeros(1, 4, 3)
    target = torch.zeros_like(prediction)

    with pytest.raises(ValueError, match="0..8"):
        weighted_action_loss(
            prediction,
            target,
            torch.tensor([9]),
        )

from mini_vla.training_v13 import compute_loss_v13


def test_v13_pair_losses_are_zero_for_exact_predictions():
    target = torch.randn(2, 9, 4, 3)
    total, base, block_pair, zone_pair = compute_loss_v13(target, target)

    assert total.item() == pytest.approx(0.0, abs=1e-7)
    assert base.item() == pytest.approx(0.0, abs=1e-7)
    assert block_pair.item() == pytest.approx(0.0, abs=1e-7)
    assert zone_pair.item() == pytest.approx(0.0, abs=1e-7)


def test_v13_pair_losses_detect_both_instruction_roles():
    # 每个组合有两个动作值：
    # dx 只由方块颜色决定，dy 只由放置区颜色决定。
    target = torch.zeros(1, 9, 1, 2)
    for block in range(3):
        for zone in range(3):
            index = block * 3 + zone
            target[0, index, 0, 0] = float(block)
            target[0, index, 0, 1] = float(zone)

    prediction = torch.zeros_like(target)
    _, _, block_pair, zone_pair = compute_loss_v13(prediction, target)

    assert block_pair.item() > 0
    assert zone_pair.item() > 0


def test_v13_masked_labels_do_not_contribute_to_base_or_pair_loss():
    target = torch.zeros(1, 9, 1, 1)
    prediction = target.clone()
    # 第 2 条为真实任务；其他八条即使预测得很差也必须被忽略。
    prediction[0, [0, 1, 3, 4, 5, 6, 7, 8]] = 100.0
    valid_mask = torch.zeros(1, 9, dtype=torch.bool)
    valid_mask[0, 2] = True

    total, base, block_pair, zone_pair = compute_loss_v13(
        prediction, target, valid_mask=valid_mask
    )

    assert total.item() == pytest.approx(0.0, abs=1e-7)
    assert base.item() == pytest.approx(0.0, abs=1e-7)
    assert block_pair.item() == pytest.approx(0.0, abs=1e-7)
    assert zone_pair.item() == pytest.approx(0.0, abs=1e-7)


def test_v13_masked_loss_requires_at_least_one_valid_label_per_tick():
    prediction = torch.zeros(1, 9, 1, 1)
    target = torch.zeros_like(prediction)

    with pytest.raises(ValueError, match="至少需要一条有效标签"):
        compute_loss_v13(
            prediction,
            target,
            valid_mask=torch.zeros(1, 9, dtype=torch.bool),
        )
