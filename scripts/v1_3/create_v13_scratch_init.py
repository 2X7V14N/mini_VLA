"""Create a genuinely random V1.3 state-4 initialization checkpoint.

This file copies only the tokenizer/model-shape metadata from an existing V1.3
checkpoint.  It never loads or copies its learned parameters.  The resulting
checkpoint is intended for a from-scratch comparison run with train_v13.py.

Example:
    python scripts/v1_3/create_v13_scratch_init.py \
        --metadata-ckpt checkpoints/v1.3_state4_init.pt \
        --out checkpoints/v1.3_state4_scratch_init.pt \
        --seed 20261106
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.models.mini_vla_v13 import (  # noqa: E402
    LanguageConditionedMiniVLA3Color,
)
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-ckpt",
        default="checkpoints/v1.3_state4_init.pt",
        help="仅读取词表和结构元数据，不读取其中的模型参数",
    )
    parser.add_argument(
        "--out",
        default="checkpoints/v1.3_state4_scratch_init.pt",
    )
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()

    metadata_path = ROOT / args.metadata_ckpt
    output_path = ROOT / args.out
    if not metadata_path.is_file():
        raise FileNotFoundError(f"找不到元数据 checkpoint：{metadata_path}")
    if output_path.exists():
        raise FileExistsError(
            f"输出文件已存在，为避免覆盖而停止：{output_path}"
        )

    metadata = torch.load(
        metadata_path,
        map_location="cpu",
        weights_only=False,
    )
    if metadata.get("model_type") != "LanguageConditionedMiniVLA3Color":
        raise ValueError("元数据 checkpoint 不是 V1.3 三色模型")
    if int(metadata.get("state_dim", 0)) != 4:
        raise ValueError("元数据 checkpoint 不是四维状态模型")

    # The vocabulary is metadata, not learned knowledge.  All RNGs are seeded
    # so this independent baseline can be recreated exactly.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = SimpleCharTokenizer(
        metadata["vocab"], max_length=int(metadata["max_length"])
    )
    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=int(metadata["max_length"]),
        state_dim=4,
        action_dim=int(metadata["action_dim"]),
        chunk_size=int(metadata["chunk_size"]),
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )

    result = {
        "model_state": model.state_dict(),
        "vocab": metadata["vocab"],
        "model_type": "LanguageConditionedMiniVLA3Color",
        "red_token_id": tokenizer.stoi["红"],
        "blue_token_id": tokenizer.stoi["蓝"],
        "yellow_token_id": tokenizer.stoi["黄"],
        "max_length": int(metadata["max_length"]),
        "state_dim": 4,
        "action_dim": int(metadata["action_dim"]),
        "chunk_size": int(metadata["chunk_size"]),
        "image_size": int(metadata.get("image_size", 64)),
        "best_val": float("inf"),
        "seed": args.seed,
        "init_from": "random_scratch",
        "migration_note": (
            "V1.3 四维模型随机初始化；只复用了词表和结构元数据，"
            "没有加载任何 V1.2/V1.3 学习参数。"
        ),
        "state_spec": ["x", "y", "holding", "gripper_open"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)

    print(f"已创建从零训练初始化 checkpoint：{output_path}")
    print(f"随机种子：{args.seed}")
    print("已加载：词表、模型结构元数据")
    print("未加载：任何已有 checkpoint 的模型参数")
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
