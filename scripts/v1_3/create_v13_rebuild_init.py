"""Create a fresh V1.3 state-4 initialization from the preserved dataset.

This does not load any learned weights.  It extracts the character vocabulary
from the existing V1.3 train split, constructs the exact V1.3 three-color
model shape, and writes a reproducible random initialization checkpoint for
``train_v13.py``.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color  # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v1.3_state4_three_color_1000_200_200")
    parser.add_argument("--out", default="checkpoints/v1.3_init.pt")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()

    data_root = ROOT / args.data
    train_files = sorted((data_root / "train").glob("ep_*.npz"))
    if not train_files:
        raise FileNotFoundError(f"找不到 V1.3 train episode：{data_root / 'train'}")
    vocab_chars = set()
    instruction_count = 0
    for path in train_files:
        with np.load(path, allow_pickle=False) as data:
            if "instruction" not in data:
                raise ValueError(f"{path.name}: 缺少 instruction")
            for instruction in data["instruction"].astype(str).tolist():
                vocab_chars.update(instruction)
                instruction_count += 1
    required = set("红蓝黄")
    if not required.issubset(vocab_chars):
        raise ValueError(f"训练数据词表缺少颜色字符：{sorted(required - vocab_chars)}")
    vocabulary = sorted(vocab_chars)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = SimpleCharTokenizer(vocabulary, max_length=args.max_length)
    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        state_dim=4,
        action_dim=3,
        chunk_size=4,
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    output = ROOT / args.out
    if output.exists():
        raise FileExistsError(f"输出文件已存在：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocabulary,
        "model_type": "LanguageConditionedMiniVLA3Color",
        "red_token_id": tokenizer.stoi["红"],
        "blue_token_id": tokenizer.stoi["蓝"],
        "yellow_token_id": tokenizer.stoi["黄"],
        "max_length": args.max_length,
        "state_dim": 4,
        "action_dim": 3,
        "chunk_size": 4,
        "image_size": 64,
        "best_val": float("inf"),
        "seed": args.seed,
        "init_from": "random_rebuild_from_dataset_vocab",
        "source_data": str(data_root),
        "instruction_count": instruction_count,
    }, output)
    print(f"created random V1.3 init: {output}")
    print(f"vocab={len(vocabulary)} max_length={args.max_length} "
          f"state_dim=4 action_dim=3 chunk=4 seed={args.seed}")
    print("learned parameters loaded: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
