import torch

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color
from mini_vla.models.mini_vla_v15a_dynamic import (
    LanguageConditionedMiniVLA3ColorDynamic,
)
from mini_vla.models.text import SimpleCharTokenizer


def test_dynamic_model_preserves_v13_action_shape_and_aux_heads():
    prompt = "把红色方块放到蓝色放置区"
    tok = SimpleCharTokenizer(sorted(set(prompt + "黄")), max_length=32)
    base = LanguageConditionedMiniVLA3Color(
        vocab_size=tok.vocab_size, max_length=32, state_dim=4, action_dim=3,
        chunk_size=4, red_token_id=tok.stoi["红"], blue_token_id=tok.stoi["蓝"],
        yellow_token_id=tok.stoi["黄"],
    )
    checkpoint = {
        "model_state": base.state_dict(), "vocab": sorted(set(prompt + "黄")),
        "max_length": 32, "state_dim": 4, "action_dim": 3, "chunk_size": 4,
    }
    model = LanguageConditionedMiniVLA3ColorDynamic.from_v13_checkpoint(
        checkpoint, tok, history=4
    )
    images = torch.zeros(2, 4, 3, 64, 64)
    tokens = tok.batch_encode([prompt, prompt])
    state = torch.zeros(2, 4)
    with torch.inference_mode():
        output = model(images, tokens, state, return_aux=True)
    assert output["action"].shape == (2, 4, 3)
    assert output["zone_position"].shape == (2, 2)
    assert output["zone_velocity"].shape == (2, 2)


def test_dynamic_model_matches_v13_action_at_zero_temporal_gate():
    """Migration must not alter actions before temporal learning starts."""
    prompt = "把红色方块放到蓝色放置区"
    tok = SimpleCharTokenizer(sorted(set(prompt + "黄")), max_length=32)
    base = LanguageConditionedMiniVLA3Color(
        vocab_size=tok.vocab_size, max_length=32, state_dim=4, action_dim=3,
        chunk_size=4, red_token_id=tok.stoi["红"], blue_token_id=tok.stoi["蓝"],
        yellow_token_id=tok.stoi["黄"],
    ).eval()
    checkpoint = {
        "model_state": base.state_dict(), "vocab": sorted(set(prompt + "黄")),
        "max_length": 32, "state_dim": 4, "action_dim": 3, "chunk_size": 4,
    }
    dynamic = LanguageConditionedMiniVLA3ColorDynamic.from_v13_checkpoint(
        checkpoint, tok, history=4
    ).eval()
    images = torch.randn(2, 4, 3, 64, 64)
    tokens = tok.batch_encode([prompt, prompt])
    state = torch.randn(2, 4)
    with torch.inference_mode():
        expected = base(images[:, -1], tokens, state)
        actual = dynamic(images, tokens, state)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
