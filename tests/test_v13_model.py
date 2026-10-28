import itertools

import torch

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color
from mini_vla.models.text import SimpleCharTokenizer


COLORS = ("红", "蓝", "黄")
COLOR_IDS = {"红": 0, "蓝": 1, "黄": 2}


def test_v13_extracts_all_nine_block_zone_color_pairs_and_runs_forward():
    prompts = [
        f"把{block}色方块放到{zone}色放置区"
        for block, zone in itertools.product(COLORS, repeat=2)
    ]

    vocabulary = sorted(set("".join(prompts)))
    tokenizer = SimpleCharTokenizer(vocabulary, max_length=16)

    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=16,
        state_dim=4,
        action_dim=3,
        chunk_size=4,
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    model.eval()

    tokens = tokenizer.batch_encode(prompts)
    block_ids, zone_ids = model._color_roles(tokens)

    expected = list(itertools.product(range(3), repeat=2))
    actual = list(zip(block_ids.tolist(), zone_ids.tolist()))
    assert actual == expected

    images = torch.zeros(len(prompts), 3, 64, 64)
    states = torch.zeros(len(prompts), 4)

    with torch.inference_mode():
        actions = model(images, tokens, states)

    assert actions.shape == (9, 4, 3)
    assert model.block_color_embed.num_embeddings == 3
    assert model.zone_color_embed.num_embeddings == 3
