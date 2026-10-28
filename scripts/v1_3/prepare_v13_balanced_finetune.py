"""Prepare valid-label replay plus phase/combo-balanced policy corrections."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_expert_v13 import COMBOS, instruction_v13  # noqa: E402

COLORS = ("red", "blue", "yellow")
PHASES = ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")


def make_record(image, state, prompts, actions, valid, phase, source):
    return {
        "image": np.asarray(image).copy(),
        "state": np.asarray(state, dtype=np.float32).copy(),
        "prompts": np.asarray(prompts).astype(str),
        "actions": np.asarray(actions, dtype=np.float32).copy(),
        "valid": np.asarray(valid, dtype=np.bool_).copy(),
        "phase": str(phase),
        "source": int(source),
    }


def sample_old_split(root, split, per_cell, rng):
    reservoirs = defaultdict(list)
    seen = Counter()
    for path in sorted((root / split).glob("ep_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if not bool(data["success"]):
                raise ValueError(f"old expert episode failed: {path}")
            n = len(data["instruction"])
            if n == 0 or n % 9:
                raise ValueError(f"invalid grouped episode: {path}")
            ticks = n // 9
            images = data["image"].reshape(ticks, 9, 64, 64, 3)
            states = data["robot_state"].reshape(ticks, 9, 4)
            prompts = data["instruction"].reshape(ticks, 9).astype(str)
            actions = data["action_chunk"].reshape(ticks, 9, 4, 3)
            phases = data["phase"].reshape(ticks, 9).astype(str)
            block, zone = str(data["exec_block"]), str(data["exec_zone"])
            active = COMBOS.index((block, zone))
            expected_prompt = instruction_v13(block, zone, int(data["template_index"][0]))
            if not np.all(prompts[:, active] == expected_prompt):
                raise ValueError(f"active instruction does not match rollout in {path.name}")
            if not np.all(images == images[:, :1]):
                raise ValueError(f"grouped image mismatch in {path.name}")
            if not np.allclose(states, states[:, :1]):
                raise ValueError(f"grouped state mismatch in {path.name}")

            for tick in range(ticks):
                phase = str(phases[tick, 0])
                if phase not in PHASES:
                    raise ValueError(f"unknown phase {phase} in {path.name}")
                target = np.zeros_like(actions[tick])
                target[active] = actions[tick, active]
                mask = np.zeros(9, dtype=np.bool_)
                mask[active] = True
                item = make_record(
                    images[tick, 0], states[tick, 0], prompts[tick], target,
                    mask, phase, 0,
                )
                key = (phase, active)
                seen[key] += 1
                bucket = reservoirs[key]
                if len(bucket) < per_cell:
                    bucket.append(item)
                else:
                    replacement = int(rng.integers(seen[key]))
                    if replacement < per_cell:
                        bucket[replacement] = item

    missing = [
        (phase, combo) for phase in PHASES for combo in range(9)
        if len(reservoirs[(phase, combo)]) < per_cell
    ]
    if missing:
        raise ValueError(f"old replay does not cover requested phase/combo cells: {missing}")
    return [item for phase in PHASES for combo in range(9)
            for item in reservoirs[(phase, combo)]], dict(seen)


def load_correction_split(root, split):
    records = []
    for path in sorted((root / split).glob("ep_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if not bool(data["success"]):
                raise ValueError(f"expert correction failed: {path}")
            ticks = len(data["instruction"]) // 9
            images = data["image"].reshape(ticks, 9, 64, 64, 3)
            states = data["robot_state"].reshape(ticks, 9, 4)
            prompts = data["instruction"].reshape(ticks, 9).astype(str)
            actions = data["action_chunk"].reshape(ticks, 9, 4, 3)
            masks = data["valid_combo_mask"].reshape(ticks, 9).astype(bool)
            phases = data["phase"].reshape(ticks, 9).astype(str)
            if not np.all(images == images[:, :1]) or not np.allclose(states, states[:, :1]):
                raise ValueError(f"correction paired observation mismatch in {path.name}")
            if not np.all(masks.sum(axis=1) == 1):
                raise ValueError(f"correction tick must have one active instruction: {path.name}")
            if not np.array_equal(masks.argmax(axis=1), np.full(ticks, COMBOS.index((str(data['exec_block']), str(data['exec_zone']))))):
                raise ValueError(f"correction mask conflicts with executed command: {path.name}")
            for tick in range(ticks):
                records.append(make_record(
                    images[tick, 0], states[tick, 0], prompts[tick], actions[tick],
                    masks[tick], phases[tick, 0], 1,
                ))
    if not records:
        raise FileNotFoundError(f"no correction episodes in {root / split}")
    return records


def pack_records(records, split_dir, ticks_per_file=32):
    split_dir.mkdir(parents=True, exist_ok=True)
    for file_index, start in enumerate(range(0, len(records), ticks_per_file)):
        batch = records[start:start + ticks_per_file]
        n = len(batch)
        np.savez_compressed(
            split_dir / f"ep_{file_index:05d}.npz",
            image=np.repeat(np.stack([r["image"] for r in batch]), 9, axis=0),
            instruction=np.stack([r["prompts"] for r in batch]).reshape(-1),
            robot_state=np.repeat(np.stack([r["state"] for r in batch]), 9, axis=0),
            action_chunk=np.concatenate([r["actions"] for r in batch], axis=0),
            valid_combo_mask=np.concatenate([r["valid"] for r in batch]),
            phase=np.repeat(np.asarray([r["phase"] for r in batch]), 9),
            balance_source=np.repeat(np.asarray([r["source"] for r in batch], dtype=np.int8), 9),
            combo_index=np.tile(np.arange(9, dtype=np.int32), n),
            pair_id=np.repeat(np.arange(n, dtype=np.int32), 9),
        )


def summarize(records):
    counts = Counter((r["source"], r["phase"], int(r["valid"].argmax())) for r in records)
    return {
        "old_replay_ticks": sum(r["source"] == 0 for r in records),
        "correction_ticks": sum(r["source"] == 1 for r in records),
        "phase_combo_counts": {
            f"{'old' if source == 0 else 'correction'}:{phase}:{block}->{zone}": counts[(source, phase, index)]
            for source in (0, 1) for phase in PHASES for index, (block, zone) in enumerate(COMBOS)
            if counts[(source, phase, index)]
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data", default="data/v1.3_state4_three_color_1000_200_200")
    parser.add_argument("--corrections", default="data/v1.3_state4_lenient_corrections_round1_seed20261031")
    parser.add_argument("--out", default="data/v1.3_state4_round1_replay_corrections")
    parser.add_argument("--replay-per-cell-train", type=int, default=32)
    parser.add_argument("--replay-per-cell-val", type=int, default=20)
    parser.add_argument("--seed", type=int, default=932026)
    args = parser.parse_args()
    out = ROOT / args.out
    if out.exists() and any(out.rglob("*.npz")):
        raise FileExistsError(f"output contains npz files: {out}")

    rng = np.random.default_rng(args.seed)
    train_replay, train_source_counts = sample_old_split(
        ROOT / args.old_data, "train", args.replay_per_cell_train, rng
    )
    val_replay, val_source_counts = sample_old_split(
        ROOT / args.old_data, "val", args.replay_per_cell_val, rng
    )
    train_corrections = load_correction_split(ROOT / args.corrections, "train")
    val_corrections = load_correction_split(ROOT / args.corrections, "val")

    train = train_replay + train_corrections
    val = val_replay + val_corrections
    rng.shuffle(train)
    rng.shuffle(val)
    pack_records(train, out / "train")
    pack_records(val, out / "val")

    manifest = {
        "old_data": args.old_data,
        "corrections": args.corrections,
        "seed": args.seed,
        "label_rule": "exactly one active instruction matching the physical rollout command",
        "train": summarize(train),
        "val": summarize(val),
        "train_sampler": (
            "old/correction source-phase-combo buckets; old requires all 36 cells, "
            "correction uses only cells present in real failure recoveries"
        ),
        "train_sampler_samples_per_bucket": 32,
        "train_sampler_missing_correction_cells": [
            f"{phase}:{index}"
            for phase in PHASES
            for index, _ in enumerate(COMBOS)
            if not any(
                r["source"] == 1
                and r["phase"] == phase
                and int(r["valid"].argmax()) == index
                for r in train
            )
        ],
        "old_replay_source_available_by_phase_combo": {
            f"{phase}:{index}": count
            for (phase, index), count in train_source_counts.items()
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"输出：{out}")
    print(f"train ticks={len(train)} (old {len(train_replay)}, correction {len(train_corrections)})")
    print(f"val ticks={len(val)} (old {len(val_replay)}, correction {len(val_corrections)})")
    print("旧示范标签：每个状态仅保留实际 rollout 指令；纠错标签：每个状态仅保留当前指令。")
    print(f"训练抽样：72 个 source/phase/combo bucket，每 bucket 每轮 32 tick。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
