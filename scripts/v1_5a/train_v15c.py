"""Train V1.5c from V1.3 final on both-moving-object demonstrations."""

import argparse
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_dataset_v15c import DynamicBothDatasetV15c, V15cShuffleBatchSampler
from mini_vla.models.mini_vla_v15c_dynamic import LanguageConditionedMiniVLA3ColorDynamicBoth
from mini_vla.models.text import SimpleCharTokenizer


def loss_for_batch(model, batch, device, weights):
    out = model(batch["image_history"].to(device), batch["token_ids"].to(device), batch["robot_state"].to(device), return_aux=True)
    targets = {
        "action": batch["action_chunk"].to(device),
        "block_position": batch["block_position"].to(device),
        "block_velocity": batch["block_velocity"].to(device) / 0.02,
        "zone_position": batch["zone_position"].to(device),
        "zone_velocity": batch["zone_velocity"].to(device) / 0.02,
        "future_zone_position": batch["future_zone_position"].to(device),
    }
    parts = {name: F.smooth_l1_loss(out[name], target) for name, target in targets.items()}
    total = sum(weights[name] * value for name, value in parts.items())
    return total, parts


def run_epoch(model, loader, device, weights, optimizer=None, grad_clip=0.5):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, **{key: 0.0 for key in weights}}
    count, skipped = 0, 0
    for index, batch in enumerate(loader, 1):
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            total, parts = loss_for_batch(model, batch, device, weights)
            if not torch.isfinite(total):
                if not training: raise RuntimeError(f"验证 loss 非有限 batch={index}")
                skipped += 1; print(f"[skip] batch={index}: non-finite loss", flush=True); continue
            if training:
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
                if not torch.isfinite(norm):
                    skipped += 1; optimizer.zero_grad(set_to_none=True); print(f"[skip] batch={index}: non-finite gradient", flush=True); continue
                optimizer.step()
        n = batch["image_history"].shape[0]; count += n; totals["total"] += float(total.detach()) * n
        for key, value in parts.items(): totals[key] += float(value.detach()) * n
    if count == 0: raise RuntimeError("没有有效 batch")
    return {key: value / count for key, value in totals.items()}, skipped


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/v1_5a/train_c.toml"); parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(); cfg = tomllib.loads((ROOT / args.config).read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    init, output, history = ROOT / model_cfg["init_checkpoint"], ROOT / model_cfg["output_checkpoint"], int(model_cfg.get("history", 4))
    if not init.is_file(): raise FileNotFoundError(f"找不到 V1.3 checkpoint：{init}")
    checkpoint = torch.load(init, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3Color": raise ValueError("V1.5c 必须从 V1.3 checkpoint 初始化")
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    root = ROOT / data_cfg["root"]; future = int(model_cfg.get("future_ticks", 4)); batch_size = int(train_cfg.get("batch_size", 64))
    train_ds = DynamicBothDatasetV15c(root, "train", tokenizer, history, future); val_ds = DynamicBothDatasetV15c(root, "val", tokenizer, history, future)
    train_loader = DataLoader(train_ds, batch_sampler=V15cShuffleBatchSampler(train_ds, batch_size, 0), num_workers=0); val_loader = DataLoader(val_ds, batch_sampler=V15cShuffleBatchSampler(val_ds, batch_size, 1), num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = LanguageConditionedMiniVLA3ColorDynamicBoth.from_v13_checkpoint(checkpoint, tokenizer, history).to(device)
    weights = {"action": 1.0, "block_position": 0.15, "block_velocity": 0.10, "zone_position": 0.20, "zone_velocity": 0.10, "future_zone_position": 0.10}
    if args.check_only:
        with torch.inference_mode(): total, parts = loss_for_batch(model, next(iter(train_loader)), device, weights)
        if not torch.isfinite(total): raise RuntimeError("preflight loss 非有限")
        print(f"[check-only] device={device}; init={init}; train={len(train_ds)} val={len(val_ds)}; history={history}; future_ticks={future}")
        print(f"[check-only] total={float(total):.6f}; " + "; ".join(f"{k}={float(v):.6f}" for k,v in parts.items()))
        print(f"[check-only] output={output}"); return 0
    policy, auxiliary = [], []
    aux_prefixes = ("temporal_gru.", "temporal_proj.", "block_", "zone_", "future_zone_")
    for name, parameter in model.named_parameters(): (auxiliary if name == "temporal_gate" or name.startswith(aux_prefixes) else policy).append(parameter)
    optimizer = torch.optim.AdamW([{"params": policy, "lr": float(train_cfg["policy_lr"])}, {"params": auxiliary, "lr": float(train_cfg["auxiliary_lr"])}], weight_decay=float(train_cfg.get("weight_decay", 1e-4)), eps=1e-6)
    print(f"device={device}; initialization={init}; model=V1.5c; history={history}; future_ticks={future}", flush=True)
    print(f"samples: train={len(train_ds)} val={len(val_ds)}; batch={batch_size}; epochs={train_cfg['epochs']}", flush=True)
    print(f"weights={weights}; lr policy={train_cfg['policy_lr']} auxiliary={train_cfg['auxiliary_lr']}", flush=True)
    best, stale, patience = float("inf"), 0, int(train_cfg["patience"])
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        started = time.time(); train, skipped_train = run_epoch(model, train_loader, device, weights, optimizer); val, skipped_val = run_epoch(model, val_loader, device, weights)
        print(f"epoch {epoch}/{train_cfg['epochs']} train={train['total']:.5f} val={val['total']:.5f}; action={val['action']:.5f}; future={val['future_zone_position']:.5f}; skipped={skipped_train}/{skipped_val} ({time.time()-started:.1f}s)", flush=True)
        if val["total"] < best:
            best, stale = val["total"], 0; output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_type": "LanguageConditionedMiniVLA3ColorDynamicBoth", "model_state": model.state_dict(), "vocab": checkpoint["vocab"], "max_length": checkpoint["max_length"], "state_dim": checkpoint["state_dim"], "action_dim": checkpoint["action_dim"], "chunk_size": checkpoint["chunk_size"], "history": history, "future_ticks": future, "best_val": best, "epoch": epoch, "init_checkpoint": str(init)}, output)
        else:
            stale += 1
            if stale >= patience: break
    print(f"saved {output} best_val={best:.6f}")


if __name__ == "__main__": main()
