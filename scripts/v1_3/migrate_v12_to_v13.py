import argparse
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color
from mini_vla.models.text import SimpleCharTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="将 V1.2 双颜色 checkpoint 迁移为 V1.3 三颜色初始化 checkpoint"
    )
    parser.add_argument(
        "--src",
        default="checkpoints/v1.2_final.pt",
        help="V1.2 最终 checkpoint",
    )
    parser.add_argument(
        "--out",
        default="checkpoints/v1.3_init.pt",
        help="迁移后的初始化 checkpoint",
    )
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    if not src.is_file():
        raise FileNotFoundError(f"找不到 V1.2 checkpoint：{src}")
    if out.exists():
        raise FileExistsError(
            f"输出文件已存在，为避免覆盖而停止：{out}\n"
            "请为 --out 指定一个新文件名。"
        )

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if ckpt.get("model_type") != "LanguageConditionedMiniVLA":
        raise ValueError(f"不支持的模型类型：{ckpt.get('model_type')}")

    old_vocab = ckpt.get("vocab")
    old_state = ckpt.get("model_state")
    if not isinstance(old_vocab, list) or not isinstance(old_state, dict):
        raise ValueError("checkpoint 缺少有效的 vocab 或 model_state")

    # 延续 V1.2 的字符词表，只新增黄色字符。
    new_vocab = sorted(set(old_vocab) | {"黄"})
    max_length = int(ckpt["max_length"])
    tokenizer = SimpleCharTokenizer(new_vocab, max_length=max_length)

    for color in ("红", "蓝", "黄"):
        if color not in tokenizer.stoi:
            raise ValueError(f"新词表中缺少颜色字符：{color}")

    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=max_length,
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        chunk_size=int(ckpt["chunk_size"]),
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )

    new_state = model.state_dict()
    if set(old_state) != set(new_state):
        missing = sorted(set(new_state) - set(old_state))
        unexpected = sorted(set(old_state) - set(new_state))
        raise ValueError(
            "V1.2 和 V1.3 的参数名称不匹配，停止迁移。\n"
            f"新模型缺少于旧 checkpoint 的参数：{missing}\n"
            f"旧 checkpoint 多出的参数：{unexpected}"
        )

    token_key = "text_encoder.token_embed.weight"
    role_keys = ("block_color_embed.weight", "zone_color_embed.weight")

    # 除词嵌入矩阵和两张颜色角色表外，其余权重必须形状完全相同。
    special_keys = {token_key, *role_keys}
    for key in old_state:
        if key in special_keys:
            continue
        if old_state[key].shape != new_state[key].shape:
            raise ValueError(
                f"普通参数形状不一致：{key}: "
                f"旧={tuple(old_state[key].shape)}, "
                f"新={tuple(new_state[key].shape)}"
            )
        new_state[key] = old_state[key].clone()

    # 按字符匹配词嵌入行，避免新词表排序导致红、蓝权重错位。
    old_ids = {"<pad>": 0, "<unk>": 1}
    old_ids.update({char: index + 2 for index, char in enumerate(old_vocab)})
    new_ids = tokenizer.stoi

    old_token_weights = old_state[token_key]
    new_token_weights = new_state[token_key]

    if old_token_weights.shape[1] != new_token_weights.shape[1]:
        raise ValueError("新旧 token embedding 的维度不一致")

    for token, old_id in old_ids.items():
        if token in new_ids:
            new_token_weights[new_ids[token]].copy_(old_token_weights[old_id])

    # “黄”在 V1.2 中没有对应词向量，用红、蓝字符向量的均值初始化。
    # 这只是一个可训练的初始值，后续仍需通过 V1.3 数据学习。
    new_token_weights[new_ids["黄"]].copy_(
        (old_token_weights[old_ids["红"]]
         + old_token_weights[old_ids["蓝"]]) / 2
    )

    # 颜色角色编号固定为红=0、蓝=1、黄=2：
    # 复制 V1.2 的红蓝行；黄色行用两行均值初始化。
    for key in role_keys:
        old_weights = old_state[key]
        new_weights = new_state[key]

        if old_weights.shape[0] != 2 or new_weights.shape[0] != 3:
            raise ValueError(
                f"{key} 形状不符合预期：旧={tuple(old_weights.shape)}, "
                f"新={tuple(new_weights.shape)}"
            )

        new_weights[0].copy_(old_weights[0])  # 红
        new_weights[1].copy_(old_weights[1])  # 蓝
        new_weights[2].copy_(old_weights.mean(dim=0))  # 黄初始化

    model.load_state_dict(new_state, strict=True)

    # 保存到新路径；保留原 checkpoint，不做覆盖。
    out.parent.mkdir(parents=True, exist_ok=True)
    migrated = dict(ckpt)
    migrated.update({
        "model_state": model.state_dict(),
        "model_type": "LanguageConditionedMiniVLA3Color",
        "vocab": new_vocab,
        "red_token_id": tokenizer.stoi["红"],
        "blue_token_id": tokenizer.stoi["蓝"],
        "yellow_token_id": tokenizer.stoi["黄"],
        "init_from": str(src),
        "migration_note": (
            "V1.2 权重按字符映射继承；黄色 token 和颜色角色向量"
            "以红蓝均值初始化，需通过 V1.3 训练学习。"
        ),
    })
    torch.save(migrated, out)

    print(f"迁移完成：{src} -> {out}")
    print(f"旧词表字符数：{len(old_vocab)}；新词表字符数：{len(new_vocab)}")
    print("已继承：视觉、状态、融合、动作头、语言编码器等形状兼容参数")
    print("已按字符映射：已有 token embedding")
    print("已扩展：方块颜色和区域颜色嵌入从 2 行变为 3 行")
    print("新增黄色参数：用红蓝均值初始化，后续需要训练")
    print(f"模型类型：{migrated['model_type']}")


if __name__ == "__main__":
    main()
