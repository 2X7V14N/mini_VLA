import torch

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color
from mini_vla.models.mini_vla_v15c_dynamic import LanguageConditionedMiniVLA3ColorDynamicBoth
from mini_vla.models.text import SimpleCharTokenizer


def test_v15c_zero_gate_preserves_v13_action_path():
    prompt = "把红色方块放到蓝色放置区"
    tokenizer = SimpleCharTokenizer(sorted(set(prompt + "黄")), max_length=32)
    base = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size, max_length=32, state_dim=4, action_dim=3,
        chunk_size=4, red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    ).eval()
    checkpoint = {"model_state": base.state_dict(), "vocab": sorted(set(prompt + "黄")),
                  "max_length": 32, "state_dim": 4, "action_dim": 3, "chunk_size": 4}
    dynamic = LanguageConditionedMiniVLA3ColorDynamicBoth.from_v13_checkpoint(checkpoint, tokenizer, history=4).eval()
    image = torch.randn(2, 3, 64, 64)
    tokens = tokenizer.batch_encode([prompt, prompt])
    state = torch.randn(2, 4)
    with torch.inference_mode():
        expected = base(image, tokens, state)
        actual = dynamic(image.unsqueeze(1).repeat(1, 4, 1, 1, 1), tokens, state)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
