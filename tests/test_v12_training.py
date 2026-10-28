import numpy as np
import pytest
import torch

from scripts.v1_2.train_v12 import V12BatchSampler, compute_loss_v12
from mini_vla.data.grasp_place import CLOSE_ENTER_TOL, GraspPlaceEnv, Scene
from mini_vla.models.mini_vla import LanguageConditionedMiniVLA
from mini_vla.models.text import SimpleCharTokenizer


def test_v12_sampler_keeps_each_four_way_counterfactual_group_intact():
    sampler = V12BatchSampler(n_samples=40, ticks_per_batch=3, seed=13)
    batches = list(sampler)

    assert all(len(batch) % 4 == 0 for batch in batches)
    flattened_groups = []
    for batch in batches:
        for offset in range(0, len(batch), 4):
            group = batch[offset:offset + 4]
            assert group[1] == group[0] + 1
            assert group[2] == group[0] + 2
            assert group[3] == group[0] + 3
            flattened_groups.append(group[0] // 4)
    assert sorted(flattened_groups) == list(range(10))


def test_v12_pair_loss_supervises_block_and_zone_differences_separately():
    target = torch.zeros(4, 1, 3)
    target[1, 0, 0] = 1.0  # block color changes dx
    target[2, 0, 1] = 2.0  # zone color changes dy
    target[3, 0, :2] = torch.tensor([1.0, 2.0])
    prediction = torch.zeros_like(target, requires_grad=True)

    total, base, block_pair, zone_pair = compute_loss_v12(
        prediction, target, torch.nn.MSELoss(), pair_weight=1.0)
    total.backward()

    assert base.item() > 0
    assert block_pair.item() == pytest.approx(1.0 / 3.0)
    assert zone_pair.item() == pytest.approx(4.0 / 3.0)
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum().item() > 0


def test_v12_sampler_rejects_incomplete_tick_groups():
    with pytest.raises(ValueError, match="4 的倍数"):
        V12BatchSampler(n_samples=10, ticks_per_batch=2)


def test_v12_model_extracts_block_and_zone_color_as_distinct_roles():
    vocab = list("把红蓝色方块放到放置区")
    tokenizer = SimpleCharTokenizer(vocab, max_length=16)
    model = LanguageConditionedMiniVLA(
        vocab_size=tokenizer.vocab_size, max_length=16,
        state_dim=3, action_dim=3, chunk_size=4,
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
    )
    prompts = [
        "把红色方块放到蓝色放置区",
        "把蓝色方块放到红色放置区",
        "把蓝色方块放到蓝色放置区",
    ]
    tokens = tokenizer.batch_encode(prompts)
    block, zone = model._color_roles(tokens)

    assert block.tolist() == [0, 1, 1]
    assert zone.tolist() == [1, 0, 1]


def test_v12_close_margin_is_explicit_and_v11_default_is_unchanged():
    scene = Scene(
        red=np.array([0.20, 0.0]), blue=np.array([-0.20, 0.0]),
        robot_start=np.array([0.0, 0.0]),
        zones={"red": np.array([0.6, 0.6]),
               "blue": np.array([-0.6, 0.6])},
    )
    # V1.1 compatibility: no explicit threshold keeps REACH_TOL=0.10.
    legacy = GraspPlaceEnv(scene, "red")
    legacy.step(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    assert legacy.phase == "CLOSE"

    # V1.2: at 0.10 the robot is still approaching; it enters CLOSE only
    # after reaching the explicit 0.06 safety margin.
    v12 = GraspPlaceEnv(scene, "red", zone_color="red",
                        close_enter_tol=CLOSE_ENTER_TOL)
    v12.step(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    assert v12.phase == "MOVE_TO_OBJECT"
    v12.step(np.array([0.4, 0.0, 1.0], dtype=np.float32))
    assert v12.phase == "CLOSE"


@pytest.mark.parametrize("size,vision_tokens", [(64, 16), (128, 64)])
def test_v12_model_supports_64_and_128_pixel_images(size, vision_tokens):
    vocab = list("把红蓝色方块放到放置区")
    tokenizer = SimpleCharTokenizer(vocab, max_length=16)
    model = LanguageConditionedMiniVLA(
        vocab_size=tokenizer.vocab_size, max_length=16,
        state_dim=3, action_dim=3, chunk_size=4,
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
    )
    image = torch.zeros(2, 3, size, size)
    tokens = tokenizer.batch_encode([
        "把红色方块放到蓝色放置区",
        "把蓝色方块放到红色放置区",
    ])
    state = torch.zeros(2, 3)

    output = model(image, tokens, state)

    assert output.shape == (2, 4, 3)
    assert model.vision_encoder(image).shape == (2, vision_tokens, 128)
