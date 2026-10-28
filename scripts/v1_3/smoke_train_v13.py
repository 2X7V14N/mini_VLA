import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from mini_vla.data.dataset_v13 import (
    V13GroupedDataset,
    V13TickBatchSampler,
)
from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color
from mini_vla.models.text import SimpleCharTokenizer
from mini_vla.training_v13 import compute_loss_v13


def move_batch(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def forward_grouped(model, batch):
    """将一组 tick 展平为九条模型输入，再还原为九个预测。"""
    batch_ticks = batch["image"].shape[0]

    images = batch["image"].repeat_interleave(9, dim=0)
    token_ids = batch["token_ids"].reshape(batch_ticks * 9, -1)
    robot_state = batch["robot_state"].repeat_interleave(9, dim=0)

    prediction = model(images, token_ids, robot_state)
    return prediction.reshape(batch_ticks, 9, 4, 3)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    sums = torch.zeros(4, dtype=torch.float64)
    tick_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        batch_ticks = batch["image"].shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = forward_grouped(model, batch)
            losses = compute_loss_v13(
                prediction,
                batch["action_chunk"],
                valid_mask=batch["valid_mask"],
                pair_weight=1.0,
            )
            total, base, block_pair, zone_pair = losses

            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        sums += torch.tensor(
            [x.detach().item() for x in losses],
            dtype=torch.float64,
        ) * batch_ticks
        tick_count += batch_ticks

    if tick_count == 0:
        raise RuntimeError("数据集没有可用 tick")

    return (sums / tick_count).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v1.3_smoke_record1")
    parser.add_argument(
        "--init-ckpt",
        default="checkpoints/v1.3_three_color_init.pt",
    )
    parser.add_argument("--batch-ticks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    data_root = ROOT / args.data
    ckpt_path = ROOT / args.init_ckpt

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到初始化 checkpoint：{ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("model_type") != "LanguageConditionedMiniVLA3Color":
        raise ValueError(
            "初始化 checkpoint 不是迁移后的三色模型："
            f"{ckpt.get('model_type')}"
        )

    tokenizer = SimpleCharTokenizer(
        ckpt["vocab"],
        max_length=int(ckpt["max_length"]),
    )

    train_ds = V13GroupedDataset(data_root, "train", tokenizer)
    val_ds = V13GroupedDataset(data_root, "val", tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=V13TickBatchSampler(
            len(train_ds),
            ticks_per_batch=args.batch_ticks,
            seed=0,
        ),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=V13TickBatchSampler(
            len(val_ds),
            ticks_per_batch=args.batch_ticks,
            seed=1,
        ),
        num_workers=0,
    )

    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=int(ckpt["max_length"]),
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        chunk_size=int(ckpt["chunk_size"]),
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(
        f"train ticks={len(train_ds)}，val ticks={len(val_ds)}，"
        f"每批 {args.batch_ticks} ticks"
    )
    print("运行一轮训练和验证；此 smoke test 不保存 checkpoint。")

    train_losses = run_epoch(model, train_loader, device, optimizer)
    val_losses = run_epoch(model, val_loader, device)

    names = ("total", "base", "block_pair", "zone_pair")
    print("\n一轮结果")
    for name, train_value, val_value in zip(
        names, train_losses, val_losses
    ):
        print(f"{name:>10}: train={train_value:.6f}  val={val_value:.6f}")

    print("\n[通过] 前向、反向、优化器更新和验证流程均已运行。")
    print("这只是流程 smoke test，不表示模型已经训练完成或达标。")


if __name__ == "__main__":
    main()
