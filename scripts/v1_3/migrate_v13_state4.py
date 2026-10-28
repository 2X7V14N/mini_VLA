"""将三维状态 V1.3 checkpoint 迁移为四维状态 checkpoint。

新状态顺序为 [x, y, holding, gripper_open]。前三维状态编码权重原样继承，
新增的 gripper_open 输入列初始化为零，保证迁移初始行为不依赖未训练的新特征。
"""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color
from mini_vla.models.text import SimpleCharTokenizer


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="checkpoints/v1.3_from_v12_init.pt",
    )
    parser.add_argument("--out", default="checkpoints/v1.3_from_v12_state4_init.pt")
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.is_file():
        raise FileNotFoundError(f"找不到源 checkpoint：{src}")
    if out.exists():
        raise FileExistsError(f"输出 checkpoint 已存在：{out}")

    old = torch.load(src, map_location="cpu", weights_only=False)
    if old.get("model_type") != "LanguageConditionedMiniVLA3Color":
        raise ValueError("源 checkpoint 不是 V1.3 三色模型")
    if int(old.get("state_dim", -1)) != 3:
        raise ValueError("源 checkpoint 不是三维状态模型")

    tokenizer = SimpleCharTokenizer(
        old["vocab"], max_length=int(old["max_length"])
    )
    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=int(old["max_length"]),
        state_dim=4,
        action_dim=int(old["action_dim"]),
        chunk_size=int(old["chunk_size"]),
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    new_state = model.state_dict()
    old_state = old["model_state"]

    special = "state_encoder.mlp.0.weight"
    for key, value in old_state.items():
        if key == special:
            continue
        if key not in new_state or new_state[key].shape != value.shape:
            raise ValueError(f"普通参数不兼容：{key}")
        new_state[key] = value.clone()

    old_input = old_state[special]
    new_input = new_state[special]
    if old_input.shape[1] != 3 or new_input.shape[1] != 4:
        raise ValueError(
            f"状态编码器输入形状异常：旧={tuple(old_input.shape)}，"
            f"新={tuple(new_input.shape)}"
        )
    new_input[:, :3] = old_input
    new_input[:, 3] = 0.0
    model.load_state_dict(new_state, strict=True)

    migrated = dict(old)
    migrated.update({
        "model_state": model.state_dict(),
        "state_dim": 4,
        "model_type": "LanguageConditionedMiniVLA3Color",
        "init_from": str(src),
        "state_spec": ["x", "y", "holding", "gripper_open"],
        "migration_note": (
            "V1.3 三维状态迁移为四维状态；前三维权重继承，"
            "gripper_open 输入列零初始化，需用四维状态数据训练。"
        ),
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, out)
    print(f"迁移完成：{src} -> {out}")
    print("状态：[x, y, holding, gripper_open]")
    print("已继承视觉、语言、融合、动作头和前三维状态权重")
    print("新增 gripper_open 权重列已置零，尚未训练")


if __name__ == "__main__":
    main()
