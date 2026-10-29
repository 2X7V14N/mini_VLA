"""Prepare/train V1.5a from a V1.3 checkpoint.

Use ``--check-only`` to validate the configuration and data without updating
weights.  The script deliberately refuses to start when the V1.3 checkpoint
is absent, because silently using random initialization would invalidate the
planned comparison.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import tomllib
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_dataset_v15a import (  # noqa: E402
    DynamicZoneDatasetV15a, V15aShuffleBatchSampler,
)
from mini_vla.models.mini_vla_v15a_dynamic import (  # noqa: E402
    LanguageConditionedMiniVLA3ColorDynamic,
)
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402


def load_config(path):
    with path.open("rb") as handle:
        return tomllib.load(handle)


def make_model(checkpoint, tokenizer, history):
    if checkpoint.get("model_type") == "LanguageConditionedMiniVLA3Color":
        return LanguageConditionedMiniVLA3ColorDynamic.from_v13_checkpoint(
            checkpoint, tokenizer, history=history
        )
    if checkpoint.get("model_type") == "LanguageConditionedMiniVLA3ColorDynamic":
        model = LanguageConditionedMiniVLA3ColorDynamic(
            vocab_size=tokenizer.vocab_size,
            max_length=int(checkpoint["max_length"]),
            state_dim=int(checkpoint["state_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            chunk_size=int(checkpoint["chunk_size"]),
            history=history,
            red_token_id=tokenizer.stoi["红"],
            blue_token_id=tokenizer.stoi["蓝"],
            yellow_token_id=tokenizer.stoi["黄"],
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        return model
    raise ValueError("初始化 checkpoint 既不是 V1.3，也不是 V1.5a 动态模型")


def loss_for_batch(model, batch, device, weights):
    images = batch["image_history"].to(device)
    tokens = batch["token_ids"].to(device)
    state = batch["robot_state"].to(device)
    output = model(images, tokens, state, return_aux=True)
    action_target = batch["action_chunk"].to(device)
    position_target = batch["zone_position"].to(device)
    velocity_target = batch["zone_velocity"].to(device) / 0.02
    action = F.smooth_l1_loss(output["action"], action_target)
    position = F.smooth_l1_loss(output["zone_position"], position_target)
    velocity = F.smooth_l1_loss(output["zone_velocity"], velocity_target)
    # A short-horizon consistency target: predicted zone position should move
    # by the predicted velocity over one tick (velocity labels are normalized).
    temporal_target = position_target + velocity_target * 0.02
    temporal_pred = output["zone_position"] + output["zone_velocity"] * 0.02
    temporal = F.smooth_l1_loss(temporal_pred, temporal_target)
    total = (weights["action"] * action + weights["position"] * position
             + weights["velocity"] * velocity
             + weights["temporal"] * temporal)
    return total, {"action": action, "position": position,
                   "velocity": velocity, "temporal": temporal}


def run_epoch(model, loader, device, weights, optimizer=None,
              grad_clip=0.5, max_skipped_batches=8):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("total", "action", "position", "velocity", "temporal")}
    count = 0
    skipped = 0
    for batch_index, batch in enumerate(loader, 1):
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            total, parts = loss_for_batch(model, batch, device, weights)
            if not torch.isfinite(total):
                bad_parts = [name for name, value in parts.items()
                             if not torch.isfinite(value)]
                if not training:
                    raise RuntimeError(
                        f"V1.5a 验证 loss 非有限，batch={batch_index} "
                        f"parts={bad_parts}"
                    )
                skipped += 1
                print(
                    f"[skip] batch={batch_index}: non-finite loss "
                    f"parts={bad_parts} ({skipped}/{max_skipped_batches})",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                if skipped > max_skipped_batches:
                    raise RuntimeError("V1.5a 非有限 loss batch 过多，已安全中止")
                continue
            if training:
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=False
                )
                if not torch.isfinite(grad_norm):
                    # Do not call optimizer.step(): doing so would write NaNs
                    # into Adam moments and subsequently poison the weights.
                    skipped += 1
                    print(
                        f"[skip] batch={batch_index}: non-finite gradient norm "
                        f"({skipped}/{max_skipped_batches})",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    if skipped > max_skipped_batches:
                        raise RuntimeError("V1.5a 非有限梯度 batch 过多，已安全中止")
                    continue
                optimizer.step()
        n = batch["image_history"].shape[0]
        totals["total"] += float(total.detach()) * n
        for key, value in parts.items():
            totals[key] += float(value.detach()) * n
        count += n
    if count == 0:
        raise RuntimeError("数据集为空")
    return {key: value / count for key, value in totals.items()}, skipped


def _optimizer_for(model, train_cfg):
    """Use conservative updates for transferred V1.3 weights.

    The temporal path and auxiliary heads start from random initialization and
    therefore receive a separate, higher learning rate.  This prevents early
    dynamic supervision from erasing V1.3 grasp/place behaviour.
    """
    policy_lr = float(train_cfg.get("policy_lr", train_cfg.get("lr", 1e-5)))
    aux_lr = float(train_cfg.get("auxiliary_lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    eps = float(train_cfg.get("adam_eps", 1e-6))
    auxiliary_prefixes = (
        "temporal_gru.", "temporal_proj.",
        "zone_position_head.", "zone_velocity_head.",
    )
    policy, auxiliary = [], []
    for name, parameter in model.named_parameters():
        is_auxiliary = name == "temporal_gate" or name.startswith(auxiliary_prefixes)
        (auxiliary if is_auxiliary else policy).append(parameter)
    return torch.optim.AdamW([
        {"params": policy, "lr": policy_lr},
        {"params": auxiliary, "lr": aux_lr},
    ], weight_decay=weight_decay, eps=eps), policy_lr, aux_lr


def train(config_path, check_only=False, resume=None, resume_epoch=0):
    cfg = load_config(config_path)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    data_root = ROOT / data_cfg["root"]
    init_path = ROOT / model_cfg["init_checkpoint"]
    output_path = ROOT / model_cfg["output_checkpoint"]
    history = int(model_cfg.get("history", 4))

    if not init_path.is_file():
        raise FileNotFoundError(
            f"找不到 V1.3 初始化 checkpoint：{init_path}\n"
            "请先恢复/生成 V1.3 checkpoint，再开始 V1.5a 正式训练。"
        )
    resume_path = ROOT / resume if resume else None
    source_path = resume_path or init_path
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"找不到恢复 checkpoint：{resume_path}")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    train_ds = DynamicZoneDatasetV15a(data_root, data_cfg.get("train_split", "train"), tokenizer, history)
    val_ds = DynamicZoneDatasetV15a(data_root, data_cfg.get("val_split", "val"), tokenizer, history)
    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_sampler=V15aShuffleBatchSampler(train_ds, batch_size, 0), num_workers=0)
    val_loader = DataLoader(val_ds, batch_sampler=V15aShuffleBatchSampler(val_ds, batch_size, 1), num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(checkpoint, tokenizer, history).to(device)
    weights = {
        "action": float(train_cfg.get("action_weight", 1.0)),
        "position": float(train_cfg.get("position_weight", 0.2)),
        "velocity": float(train_cfg.get("velocity_weight", 0.1)),
        "temporal": float(train_cfg.get("temporal_weight", 0.05)),
    }
    optimizer, policy_lr, aux_lr = _optimizer_for(model, train_cfg)
    grad_clip = float(train_cfg.get("grad_clip", 0.5))
    max_skipped_batches = int(train_cfg.get("max_skipped_batches", 8))
    if check_only:
        # Full preflight: validates one real batch, device placement, the
        # V1.3->V1.5a migration and all loss terms without an optimizer step.
        batch = next(iter(train_loader))
        with torch.inference_mode():
            total, parts = loss_for_batch(model, batch, device, weights)
        if not torch.isfinite(total):
            raise RuntimeError("V1.5a preflight loss 出现非有限值")
        print(f"[check-only] device={device}; init={source_path}")
        print(f"[check-only] data train={len(train_ds)} val={len(val_ds)}; "
              f"history={history}; batch_size={batch_size}")
        print(f"[check-only] lr policy={policy_lr:g} auxiliary={aux_lr:g}")
        print("[check-only] first-batch total/action/pos/vel/temp="
              f"{float(total):.6f}/{float(parts['action']):.6f}/"
              f"{float(parts['position']):.6f}/{float(parts['velocity']):.6f}/"
              f"{float(parts['temporal']):.6f}")
        print(f"[check-only] output={output_path}")
        return 0

    print(
        f"device={device}; initialization={source_path}; "
        "model_type=LanguageConditionedMiniVLA3ColorDynamic; "
        f"history={history}",
        flush=True,
    )
    print(
        f"samples: train={len(train_ds)}, validation={len(val_ds)}; "
        f"batch_size={batch_size}; epochs={train_cfg.get('epochs', 30)}; "
        f"patience={train_cfg.get('patience', 10)}",
        flush=True,
    )
    print(
        "loss weights: "
        f"action={weights['action']}, position={weights['position']}, "
        f"velocity={weights['velocity']}, temporal={weights['temporal']}; "
        f"lr policy={policy_lr:g}, auxiliary={aux_lr:g}; grad_clip={grad_clip:g}",
        flush=True,
    )
    print(f"output checkpoint={output_path}", flush=True)

    best = float(checkpoint.get("best_val", float("inf"))) if resume else float("inf")
    patience = int(train_cfg.get("patience", 10))
    stale = 0
    start_epoch = int(resume_epoch)
    if start_epoch < 0 or start_epoch >= int(train_cfg.get("epochs", 30)):
        raise ValueError("resume_epoch 必须在 [0, epochs) 内")
    if resume:
        print(f"resume={resume_path}; resume_epoch={start_epoch}; best_val={best:.6f}", flush=True)
    for epoch in range(start_epoch + 1, int(train_cfg.get("epochs", 30)) + 1):
        started = time.time()
        train_metrics, skipped_train = run_epoch(
            model, train_loader, device, weights, optimizer,
            grad_clip=grad_clip, max_skipped_batches=max_skipped_batches,
        )
        with torch.no_grad():
            val_metrics, skipped_val = run_epoch(
                model, val_loader, device, weights,
                grad_clip=grad_clip, max_skipped_batches=0,
            )
        print(
            f"epoch {epoch}/{train_cfg.get('epochs', 30)} "
            f"train(total/action/pos/vel/temp)="
            f"{train_metrics['total']:.5f}/{train_metrics['action']:.5f}/"
            f"{train_metrics['position']:.5f}/{train_metrics['velocity']:.5f}/"
            f"{train_metrics['temporal']:.5f} "
            f"val={val_metrics['total']:.5f}; skipped(train/val)="
            f"{skipped_train}/{skipped_val} ({time.time() - started:.1f}s)",
            flush=True,
        )
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            stale = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_type": "LanguageConditionedMiniVLA3ColorDynamic",
                "model_state": model.state_dict(),
                "vocab": checkpoint["vocab"],
                "max_length": checkpoint["max_length"],
                "state_dim": checkpoint["state_dim"],
                "action_dim": checkpoint["action_dim"],
                "chunk_size": checkpoint["chunk_size"],
                "history": history,
                "best_val": best,
                "epoch": epoch,
                "init_checkpoint": str(init_path),
                "training_config": {
                    "policy_lr": policy_lr,
                    "auxiliary_lr": aux_lr,
                    "weights": weights,
                    "batch_size": batch_size,
                    "grad_clip": grad_clip,
                },
            }, output_path)
        else:
            stale += 1
            if stale >= patience:
                break
    print(f"saved {output_path} best_val={best:.6f}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_5a/train.toml")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--resume",
        help="从已有 V1.5a checkpoint 恢复模型权重；不会恢复 Adam 状态",
    )
    parser.add_argument(
        "--resume-epoch", type=int, default=0,
        help="恢复 checkpoint 对应的已完成 epoch 数，例如从第 2 轮继续则传 2",
    )
    args = parser.parse_args()
    return train(
        ROOT / args.config,
        check_only=args.check_only,
        resume=args.resume,
        resume_epoch=args.resume_epoch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
