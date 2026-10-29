"""V1.5e: fine-tune V1.5c with closed-loop placement corrections."""

import argparse
import sys
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_dataset_v15c import DynamicBothDatasetV15c, V15cShuffleBatchSampler
from mini_vla.models.mini_vla_v15c_dynamic import LanguageConditionedMiniVLA3ColorDynamicBoth
from mini_vla.models.text import SimpleCharTokenizer


class TaggedDataset(Dataset):
    def __init__(self, dataset, correction):
        self.dataset, self.correction = dataset, bool(correction)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        result = self.dataset[index]
        result["is_correction"] = torch.tensor(self.correction, dtype=torch.bool)
        return result


class EpisodeMixtureBatchSampler(Sampler):
    """Mix source batches while retaining contiguous episode access patterns."""
    def __init__(self, base, correction, batch_size, correction_ratio, seed=0):
        if not 0.0 < correction_ratio < 1.0:
            raise ValueError("correction_ratio 必须在 (0, 1) 内")
        self.base, self.correction = base, correction
        self.batch_size, self.ratio, self.seed, self.epoch = int(batch_size), float(correction_ratio), int(seed), 0
        if self.batch_size < 1 or len(correction.episodes) == 0:
            raise ValueError("batch_size 必须为正且纠偏数据不能为空")

    def __len__(self):
        return (len(self.base) + self.batch_size - 1) // self.batch_size

    @staticmethod
    def _stream(dataset, offset, rng):
        while True:
            for episode_id in rng.permutation(len(dataset.episodes)):
                start, end = dataset.episode_ranges[episode_id]
                yield from range(offset + start, offset + end)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        base_stream = self._stream(self.base, 0, rng)
        correction_stream = self._stream(self.correction, len(self.base), rng)
        for _ in range(len(self)):
            stream = correction_stream if rng.random() < self.ratio else base_stream
            yield [next(stream) for _ in range(self.batch_size)]


def make_model(checkpoint, tokenizer):
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3ColorDynamicBoth":
        raise ValueError("V1.5e 必须从 V1.5c checkpoint 初始化")
    model = LanguageConditionedMiniVLA3ColorDynamicBoth(
        vocab_size=tokenizer.vocab_size, max_length=int(checkpoint["max_length"]),
        state_dim=int(checkpoint["state_dim"]), action_dim=int(checkpoint["action_dim"]),
        chunk_size=int(checkpoint["chunk_size"]), history=int(checkpoint["history"]),
        red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"], yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model


def loss_for_batch(model, batch, device, weights, correction_action_multiplier=1.0):
    out = model(batch["image_history"].to(device), batch["token_ids"].to(device), batch["robot_state"].to(device), return_aux=True)
    action_per_item = F.smooth_l1_loss(out["action"], batch["action_chunk"].to(device), reduction="none").flatten(1).mean(dim=1)
    is_correction = batch["is_correction"].to(device)
    action_scale = torch.where(is_correction, correction_action_multiplier, 1.0)
    parts = {
        "action": (action_per_item * action_scale).sum() / action_scale.sum().clamp_min(1.0),
        "block_position": F.smooth_l1_loss(out["block_position"], batch["block_position"].to(device)),
        "block_velocity": F.smooth_l1_loss(out["block_velocity"], batch["block_velocity"].to(device) / 0.02),
        "zone_position": F.smooth_l1_loss(out["zone_position"], batch["zone_position"].to(device)),
        "zone_velocity": F.smooth_l1_loss(out["zone_velocity"], batch["zone_velocity"].to(device) / 0.02),
        "future_zone_position": F.smooth_l1_loss(out["future_zone_position"], batch["future_zone_position"].to(device)),
    }
    return sum(weights[name] * value for name, value in parts.items()), parts


def run_epoch(model, loader, device, weights, optimizer=None, grad_clip=0.4, max_skipped_batches=8, correction_action_multiplier=1.0):
    training = optimizer is not None
    model.train(training)
    totals, count, skipped = {"total": 0.0, **{name: 0.0 for name in weights}}, 0, 0
    for batch_index, batch in enumerate(loader, 1):
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            total, parts = loss_for_batch(model, batch, device, weights, correction_action_multiplier)
            if not torch.isfinite(total):
                bad = [name for name, value in parts.items() if not torch.isfinite(value)]
                if not training:
                    raise RuntimeError(f"验证 loss 非有限 batch={batch_index}: {bad}")
                skipped += 1; print(f"[skip] batch={batch_index}: non-finite {bad} ({skipped}/{max_skipped_batches})", flush=True)
                if skipped > max_skipped_batches:
                    raise RuntimeError("非有限 batch 过多，已安全中止")
                continue
            if training:
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=False)
                if not torch.isfinite(norm):
                    skipped += 1; optimizer.zero_grad(set_to_none=True)
                    if skipped > max_skipped_batches:
                        raise RuntimeError("非有限梯度过多，已安全中止")
                    continue
                optimizer.step()
        n = len(batch["is_correction"]); count += n; totals["total"] += float(total.detach()) * n
        for name, value in parts.items():
            totals[name] += float(value.detach()) * n
    if count == 0:
        raise RuntimeError("没有有效 batch")
    return {name: value / count for name, value in totals.items()}, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_5a/train_e.toml")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = tomllib.loads((ROOT / args.config).read_text(encoding="utf-8"))
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["training"]
    init, output = ROOT / model_cfg["init_checkpoint"], ROOT / model_cfg["output_checkpoint"]
    checkpoint = torch.load(init, map_location="cpu", weights_only=False)
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    history, future = int(checkpoint["history"]), int(checkpoint["future_ticks"])
    base_root, correction_root = ROOT / data_cfg["base_root"], ROOT / data_cfg["correction_root"]
    print(f"indexing base train: {base_root / 'train'}", flush=True)
    base_train = DynamicBothDatasetV15c(base_root, "train", tokenizer, history, future, cache_episodes=int(train_cfg["cache_episodes"]))
    print(f"indexing correction train: {correction_root / 'train'}", flush=True)
    correction_train = DynamicBothDatasetV15c(correction_root, "train", tokenizer, history, future, cache_episodes=int(train_cfg["cache_episodes"]))
    print(f"indexing independent validation: {base_root / 'val'}", flush=True)
    val = DynamicBothDatasetV15c(base_root, "val", tokenizer, history, future, cache_episodes=int(train_cfg["cache_episodes"]))
    batch_size, workers = int(train_cfg["batch_size"]), int(train_cfg["loader_workers"])
    mixed = ConcatDataset([TaggedDataset(base_train, False), TaggedDataset(correction_train, True)])
    mixed_sampler = EpisodeMixtureBatchSampler(base_train, correction_train, batch_size, float(train_cfg["correction_ratio"]), seed=0)
    loader_options = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers:
        loader_options.update({"persistent_workers": True, "prefetch_factor": int(train_cfg["prefetch_factor"])})
    train_loader = DataLoader(mixed, batch_sampler=mixed_sampler, **loader_options)
    val_loader = DataLoader(TaggedDataset(val, False), batch_sampler=V15cShuffleBatchSampler(val, batch_size, seed=1), **loader_options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(checkpoint, tokenizer).to(device)
    weights = {name: float(train_cfg[f"{name}_weight"]) for name in ("action", "block_position", "block_velocity", "zone_position", "zone_velocity", "future_zone_position")}
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if not (n == "temporal_gate" or n.startswith(("temporal_gru.", "temporal_proj.", "block_", "zone_", "future_zone_")))], "lr": float(train_cfg["policy_lr"])},
        {"params": [p for n, p in model.named_parameters() if n == "temporal_gate" or n.startswith(("temporal_gru.", "temporal_proj.", "block_", "zone_", "future_zone_"))], "lr": float(train_cfg["auxiliary_lr"])},
    ], weight_decay=float(train_cfg["weight_decay"]), eps=1e-6)
    if args.check_only:
        with torch.inference_mode():
            total, parts = loss_for_batch(model, next(iter(train_loader)), device, weights, float(train_cfg["correction_action_multiplier"]))
        print(f"[check-only] device={device}; base={len(base_train)} correction={len(correction_train)} val={len(val)}")
        print(f"[check-only] total={float(total):.6f}; " + "; ".join(f"{k}={float(v):.6f}" for k, v in parts.items()))
        return 0
    print(f"device={device}; initialization={init}; model=V1.5e; output={output}", flush=True)
    print(f"samples: base={len(base_train)} correction={len(correction_train)} val={len(val)}; correction_ratio={train_cfg['correction_ratio']}; epochs={train_cfg['epochs']}", flush=True)
    print(f"weights={weights}; correction_action_multiplier={train_cfg['correction_action_multiplier']}; lr={train_cfg['policy_lr']}/{train_cfg['auxiliary_lr']}; workers={workers}", flush=True)
    best, stale = float("inf"), 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        started = time.time()
        train, skipped_train = run_epoch(model, train_loader, device, weights, optimizer, float(train_cfg["grad_clip"]), int(train_cfg["max_skipped_batches"]), float(train_cfg["correction_action_multiplier"]))
        validation, skipped_val = run_epoch(model, val_loader, device, weights)
        print(f"epoch {epoch}/{train_cfg['epochs']} train={train['total']:.5f} val={validation['total']:.5f}; action={validation['action']:.5f}; future={validation['future_zone_position']:.5f}; skipped={skipped_train}/{skipped_val} ({time.time()-started:.1f}s)", flush=True)
        if validation["total"] < best:
            best, stale = validation["total"], 0; output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_type": "LanguageConditionedMiniVLA3ColorDynamicBoth", "stage": "v1.5e", "model_state": model.state_dict(), "vocab": checkpoint["vocab"], "max_length": checkpoint["max_length"], "state_dim": checkpoint["state_dim"], "action_dim": checkpoint["action_dim"], "chunk_size": checkpoint["chunk_size"], "history": history, "future_ticks": future, "best_val": best, "epoch": epoch, "init_checkpoint": str(init)}, output)
        else:
            stale += 1
            if stale >= int(train_cfg["patience"]):
                print("validation patience reached; early stopping", flush=True); break
    print(f"saved {output} best_val={best:.6f}")


if __name__ == "__main__":
    main()
