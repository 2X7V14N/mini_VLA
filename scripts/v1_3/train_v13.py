"""V1.3 三色抓放训练器。

每个 tick 有同一观测下的 9 种指令标签。
默认从 V1.2 迁移得到的三色初始化 checkpoint 开始训练。
只有验证集上的方块和放置区语言差分都通过门槛，才保存 checkpoint。
"""

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_vla.data.dataset_v13 import (
    V13BalancedMixtureBatchSampler,
    V13GroupedDataset,
    V13TickBatchSampler,
)
from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color
from mini_vla.models.text import SimpleCharTokenizer
from mini_vla.training_v13 import compute_loss_v13


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def forward_grouped(model, batch):
    """同一图像分别配九条指令，得到九种动作预测。"""
    batch_ticks = batch["image"].shape[0]

    images = batch["image"].repeat_interleave(9, dim=0)
    states = batch["robot_state"].repeat_interleave(9, dim=0)
    token_ids = batch["token_ids"].reshape(batch_ticks * 9, -1)

    prediction = model(images, token_ids, states)
    return prediction.reshape(batch_ticks, 9, 4, 3)


def distillation_action_loss(
    student_prediction,
    teacher_prediction,
    valid_mask,
    yellow_weight=0.1,
):
    """Keep red/blue actions near a frozen teacher while adapting yellow."""
    if student_prediction.shape != teacher_prediction.shape:
        raise ValueError("student 和 teacher 的动作形状必须一致")
    if valid_mask.shape != student_prediction.shape[:2]:
        raise ValueError("valid_mask 必须是 (batch_ticks, 9)")
    if not 0.0 <= yellow_weight <= 1.0:
        raise ValueError("yellow_weight 必须在 [0, 1] 内")

    route_weights = student_prediction.new_tensor(
        [1.0, 1.0, yellow_weight,
         1.0, 1.0, yellow_weight,
         yellow_weight, yellow_weight, yellow_weight]
    )
    weights = valid_mask.to(dtype=student_prediction.dtype) * route_weights
    per_label = F.smooth_l1_loss(
        student_prediction,
        teacher_prediction.detach(),
        reduction="none",
    ).mean(dim=(2, 3))
    if not torch.any(weights > 0):
        raise ValueError("蒸馏至少需要一条有效标签")
    return (per_label * weights).sum() / weights.sum()


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    uniform_active_labels=False,
    teacher=None,
    distill_weight=0.0,
    distill_yellow_weight=0.1,
):
    training = optimizer is not None
    model.train(training)

    # total、base、block_pair、zone_pair、distillation
    sums = torch.zeros(5, dtype=torch.float64)
    tick_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        batch_ticks = batch["image"].shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            prediction = forward_grouped(model, batch)
            if uniform_active_labels:
                valid = batch["valid_mask"].to(dtype=prediction.dtype)
                per_label = (
                    prediction - batch["action_chunk"]
                ).square().mean(dim=(2, 3))
                active_count = valid.sum()
                if not torch.all(batch["valid_mask"].sum(dim=1) == 1):
                    raise ValueError(
                        "uniform active-label training requires exactly one valid label per tick"
                    )
                base = (per_label * valid).sum() / active_count
                zero = base.new_zeros(())
                losses = (base, base, zero, zero)
            else:
                losses = compute_loss_v13(
                    prediction,
                    batch["action_chunk"],
                    valid_mask=batch["valid_mask"],
                    pair_weight=1.0,
                )
            distill = prediction.new_zeros(())
            if teacher is not None and distill_weight > 0.0:
                with torch.no_grad():
                    teacher_prediction = forward_grouped(teacher, batch)
                distill = distillation_action_loss(
                    prediction,
                    teacher_prediction,
                    batch["valid_mask"],
                    yellow_weight=distill_yellow_weight,
                )
            total = losses[0] + distill_weight * distill

            if training:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        sums += torch.tensor(
            [
                total.detach().item(),
                losses[1].detach().item(),
                losses[2].detach().item(),
                losses[3].detach().item(),
                distill.detach().item(),
            ],
            dtype=torch.float64,
        ) * batch_ticks
        tick_count += batch_ticks

    if tick_count == 0:
        raise RuntimeError("数据集没有可用 tick")

    return (sums / tick_count).tolist()


@torch.no_grad()
def pair_loss_floors(loader, device):
    """计算不使用模型预测时的差分损失基线。"""
    sums = torch.zeros(2, dtype=torch.float64)
    tick_count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["action_chunk"]
        zero_prediction = torch.zeros_like(target)

        _, _, block_pair, zone_pair = compute_loss_v13(
            zero_prediction,
            target,
            valid_mask=batch["valid_mask"],
            pair_weight=1.0,
        )

        n = target.shape[0]
        sums += torch.tensor(
            [block_pair.item(), zone_pair.item()],
            dtype=torch.float64,
        ) * n
        tick_count += n

    if tick_count == 0:
        raise RuntimeError("验证集没有可用 tick")

    return (sums / tick_count).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v1.3")
    parser.add_argument(
        "--init-ckpt",
        default="checkpoints/v1.3_init.pt",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-ticks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-pair-ratio", type=float, default=0.95)
    parser.add_argument(
        "--correction-data",
        action="store_true",
        help="纠错集只有实际指令的有效标签；按验证动作 loss 选 checkpoint，不使用九路差分门槛",
    )
    parser.add_argument(
        "--balanced-mixture",
        action="store_true",
        help="均衡抽取旧示范/纠错 × 阶段 × 九种有效指令",
    )
    parser.add_argument("--samples-per-bucket", type=int, default=32)
    parser.add_argument(
        "--open-correction-multiplier", type=int, default=1,
        help="额外抽样纠错数据中的 OPEN 阶段；2 表示每个颜色组合抽两倍",
    )
    parser.add_argument(
        "--yellow-combo-multiplier", type=int, default=1,
        help="额外抽样涉及黄色的五种组合；2 表示每个来源/阶段中抽两倍",
    )
    parser.add_argument(
        "--old-replay-multiplier", type=int, default=1,
        help="旧示范来源的抽样倍率；3 配合 correction=1 即为 75%%/25%%",
    )
    parser.add_argument(
        "--correction-multiplier", type=int, default=1,
        help="闭环纠错来源的抽样倍率；与 old-replay-multiplier 共同决定来源比例",
    )
    parser.add_argument(
        "--close-correction-multiplier", type=int, default=1,
        help="额外抽样纠错数据中的 CLOSE 阶段",
    )
    parser.add_argument(
        "--move-to-zone-correction-multiplier", type=int, default=1,
        help="额外抽样纠错数据中的 MOVE_TO_ZONE 阶段",
    )
    parser.add_argument(
        "--distill-teacher",
        help="冻结的 teacher checkpoint；启用连续动作知识蒸馏",
    )
    parser.add_argument(
        "--distill-weight", type=float, default=0.0,
        help="蒸馏损失权重；0 表示关闭",
    )
    parser.add_argument(
        "--distill-yellow-weight", type=float, default=0.1,
        help="黄色标签的蒸馏相对权重，避免锁定 teacher 的黄色错误",
    )
    parser.add_argument("--out", default="v1.3_base")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_ticks < 1:
        parser.error("--epochs 和 --batch-ticks 必须大于 0")
    if args.min_epochs < 1 or args.patience < 1:
        parser.error("--min-epochs 和 --patience 必须大于 0")
    if args.balanced_mixture and not args.correction_data:
        parser.error("--balanced-mixture 需要 --correction-data")
    if args.open_correction_multiplier < 1:
        parser.error("--open-correction-multiplier 必须至少为 1")
    if args.open_correction_multiplier != 1 and not args.balanced_mixture:
        parser.error("--open-correction-multiplier 需要 --balanced-mixture")
    if args.yellow_combo_multiplier < 1:
        parser.error("--yellow-combo-multiplier 必须至少为 1")
    if args.yellow_combo_multiplier != 1 and not args.balanced_mixture:
        parser.error("--yellow-combo-multiplier 需要 --balanced-mixture")
    if args.old_replay_multiplier < 1 or args.correction_multiplier < 1:
        parser.error("来源抽样倍率必须至少为 1")
    if args.close_correction_multiplier < 1 or args.move_to_zone_correction_multiplier < 1:
        parser.error("阶段 correction multiplier 必须至少为 1")
    if (
        (args.old_replay_multiplier != 1 or args.correction_multiplier != 1)
        and not args.balanced_mixture
    ):
        parser.error("来源抽样倍率需要 --balanced-mixture")
    if args.distill_weight < 0.0:
        parser.error("--distill-weight 不能小于 0")
    if args.distill_yellow_weight < 0.0 or args.distill_yellow_weight > 1.0:
        parser.error("--distill-yellow-weight 必须在 [0, 1] 内")
    if args.distill_weight > 0.0 and not args.distill_teacher:
        parser.error("启用蒸馏时必须提供 --distill-teacher")

    data_root = ROOT / args.data
    init_path = ROOT / args.init_ckpt
    checkpoint_path = ROOT / "checkpoints" / f"{args.out}.pt"

    if not init_path.is_file():
        raise FileNotFoundError(f"找不到初始化 checkpoint：{init_path}")
    if checkpoint_path.exists():
        raise FileExistsError(
            f"输出 checkpoint 已存在，为避免覆盖而停止：{checkpoint_path}\n"
            "请换一个 --out 名称。"
        )

    init_ckpt = torch.load(
        init_path,
        map_location="cpu",
        weights_only=False,
    )
    if init_ckpt.get("model_type") != "LanguageConditionedMiniVLA3Color":
        raise ValueError("初始化 checkpoint 不是 V1.3 三色模型")

    tokenizer = SimpleCharTokenizer(
        init_ckpt["vocab"],
        max_length=int(init_ckpt["max_length"]),
    )

    train_ds = V13GroupedDataset(data_root, "train", tokenizer)
    val_ds = V13GroupedDataset(data_root, "val", tokenizer)

    train_sampler = (
        V13BalancedMixtureBatchSampler(
            train_ds,
            ticks_per_batch=args.batch_ticks,
            samples_per_bucket=args.samples_per_bucket,
            seed=0,
            open_correction_multiplier=args.open_correction_multiplier,
            yellow_combo_multiplier=args.yellow_combo_multiplier,
            old_replay_multiplier=args.old_replay_multiplier,
            correction_multiplier=args.correction_multiplier,
            close_correction_multiplier=args.close_correction_multiplier,
            move_to_zone_correction_multiplier=args.move_to_zone_correction_multiplier,
        )
        if args.balanced_mixture
        else V13TickBatchSampler(
            len(train_ds),
            ticks_per_batch=args.batch_ticks,
            seed=0,
        )
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=0
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")
    if device.type == "cuda":
        print(f"GPU：{torch.cuda.get_device_name(0)}")

    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=int(init_ckpt["max_length"]),
        state_dim=int(init_ckpt["state_dim"]),
        action_dim=int(init_ckpt["action_dim"]),
        chunk_size=int(init_ckpt["chunk_size"]),
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(init_ckpt["model_state"], strict=True)
    model.to(device)

    teacher = None
    if args.distill_teacher:
        teacher_path = ROOT / args.distill_teacher
        if not teacher_path.is_file():
            raise FileNotFoundError(f"找不到 teacher checkpoint：{teacher_path}")
        teacher_ckpt = torch.load(
            teacher_path,
            map_location="cpu",
            weights_only=False,
        )
        if teacher_ckpt.get("model_type") != "LanguageConditionedMiniVLA3Color":
            raise ValueError("teacher checkpoint 不是 V1.3 三色模型")
        if teacher_ckpt.get("state_dim") != init_ckpt.get("state_dim"):
            raise ValueError("teacher 与 student 的状态维度不一致")
        teacher = LanguageConditionedMiniVLA3Color(
            vocab_size=tokenizer.vocab_size,
            max_length=int(init_ckpt["max_length"]),
            state_dim=int(init_ckpt["state_dim"]),
            action_dim=int(init_ckpt["action_dim"]),
            chunk_size=int(init_ckpt["chunk_size"]),
            red_token_id=tokenizer.stoi["红"],
            blue_token_id=tokenizer.stoi["蓝"],
            yellow_token_id=tokenizer.stoi["黄"],
        )
        teacher.load_state_dict(teacher_ckpt["model_state"], strict=True)
        teacher.to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    block_floor, zone_floor = pair_loss_floors(val_loader, device)
    print(
        f"train ticks={len(train_ds)}，val ticks={len(val_ds)}，"
        f"batch ticks={args.batch_ticks}"
    )
    print(
        "忽略语言的验证差分基线："
        f"block={block_floor:.6f}，zone={zone_floor:.6f}"
    )
    print(
        f"参数量：{sum(p.numel() for p in model.parameters()):,}；"
        f"训练上限 {args.epochs} 轮；lr={args.lr}"
    )
    if teacher is not None:
        print(
            f"知识蒸馏：teacher={args.distill_teacher}；"
            f"weight={args.distill_weight}；"
            f"yellow_weight={args.distill_yellow_weight}"
        )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer=optimizer,
            uniform_active_labels=args.correction_data,
            teacher=teacher,
            distill_weight=args.distill_weight,
            distill_yellow_weight=args.distill_yellow_weight,
        )
        val_metrics = run_epoch(
            model, val_loader, device,
            uniform_active_labels=args.correction_data,
            teacher=teacher,
            distill_weight=args.distill_weight,
            distill_yellow_weight=args.distill_yellow_weight,
        )

        train_total, train_base, train_block, train_zone, train_distill = train_metrics
        val_total, val_base, val_block, val_zone, val_distill = val_metrics

        values = train_metrics + val_metrics
        if not all(torch.isfinite(torch.tensor(value)) for value in values):
            print(f"[中止] 第 {epoch} 轮出现 NaN/Inf；不会保存 checkpoint。")
            break

        language_gate = args.correction_data or (
            val_block < args.min_pair_ratio * block_floor
            and val_zone < args.min_pair_ratio * zone_floor
        )

        if language_gate and val_total < best_val:
            best_val = val_total
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        elif best_state is not None:
            bad_epochs += 1

        print(
            f"epoch {epoch:>3}/{args.epochs} "
            f"train={train_total:.5f} "
            f"(base {train_base:.5f}, block {train_block:.5f}, "
            f"zone {train_zone:.5f}, distill {train_distill:.5f}) "
            f"val={val_total:.5f} "
            f"(base {val_base:.5f}, block {val_block:.5f}, "
            f"zone {val_zone:.5f}, distill {val_distill:.5f}) "
            f"{'纠错验证' if args.correction_data else ('语言门槛 OK' if language_gate else '语言门槛未达')} "
            f"{time.time() - started:.0f}s",
            flush=True,
        )

        if (
            epoch >= args.min_epochs
            and best_state is not None
            and bad_epochs >= args.patience
        ):
            print(
                f"验证 loss 连续 {args.patience} 轮未改善，"
                f"第 {epoch} 轮提前停止。"
            )
            break

    if best_state is None:
        print(
            "\n[未保存 checkpoint] 方块和放置区两路验证差分损失"
            "没有同时达到门槛。"
        )
        return 2

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(init_ckpt)
    result.update({
        "model_state": best_state,
        "model_type": "LanguageConditionedMiniVLA3Color",
        "vocab": init_ckpt["vocab"],
        "best_val": best_val,
        "train_data": str(data_root),
        "train_config": {
            "epochs": args.epochs,
            "batch_ticks": args.batch_ticks,
            "lr": args.lr,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
            "min_pair_ratio": args.min_pair_ratio,
            "correction_data": args.correction_data,
            "balanced_mixture": args.balanced_mixture,
            "samples_per_bucket": args.samples_per_bucket,
            "open_correction_multiplier": args.open_correction_multiplier,
            "yellow_combo_multiplier": args.yellow_combo_multiplier,
            "old_replay_multiplier": args.old_replay_multiplier,
            "correction_multiplier": args.correction_multiplier,
            "close_correction_multiplier": args.close_correction_multiplier,
            "move_to_zone_correction_multiplier": args.move_to_zone_correction_multiplier,
            "distill_teacher": args.distill_teacher,
            "distill_weight": args.distill_weight,
            "distill_yellow_weight": args.distill_yellow_weight,
            "init_checkpoint": str(init_path),
        },
    })
    torch.save(result, checkpoint_path)

    print(f"\n最佳验证 loss：{best_val:.6f}")
    print(f"已保存 checkpoint：{checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
