"""Verify one-command labels and complete stratification buckets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dataset_v13 import PHASE_IDS, V13BalancedMixtureBatchSampler, V13GroupedDataset  # noqa: E402
from mini_vla.data.grasp_place_expert_v13 import COMBOS, instruction_v13  # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v1.3_balanced_replay_corrections_seed8132026")
    parser.add_argument("--ckpt", default="checkpoints/v1.3_final.pt")
    args = parser.parse_args()
    root = ROOT / args.data
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ckpt = __import__("torch").load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    tokenizer = SimpleCharTokenizer(ckpt["vocab"], max_length=int(ckpt["max_length"]))

    for split in ("train", "val"):
        dataset = V13GroupedDataset(root, split, tokenizer)
        if any(bucket < 0 for bucket in dataset.balance_buckets):
            raise ValueError(f"{split}: missing balance metadata")
        count_cells = {}
        for path in sorted((root / split).glob("ep_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                ticks = len(data["instruction"]) // 9
                masks = data["valid_combo_mask"].reshape(ticks, 9).astype(bool)
                if not np.all(masks.sum(axis=1) == 1):
                    raise ValueError(f"{split}/{path.name}: each tick must have one active label")
                active = masks.argmax(axis=1)
                phases = data["phase"].reshape(ticks, 9)[:, 0].astype(str)
                sources = data["balance_source"].reshape(ticks, 9)[:, 0].astype(int)
                prompts = data["instruction"].reshape(ticks, 9).astype(str)
                combo_ids = data["combo_index"].reshape(ticks, 9)
                for tick, (combo, phase, source) in enumerate(zip(active, phases, sources)):
                    if not np.array_equal(combo_ids[tick], np.arange(9)):
                        raise ValueError(f"{split}/{path.name}: combo index mismatch")
                    expected = [
                        instruction_v13(*COMBOS[int(combo)], template)
                        for template in range(3)
                    ]
                    if prompts[tick, int(combo)] not in expected:
                        raise ValueError(f"{split}/{path.name}: active prompt does not match active target")
                    if phase not in PHASE_IDS or source not in (0, 1):
                        raise ValueError(f"{split}/{path.name}: invalid phase/source")
                    key = int(source) * 36 + PHASE_IDS[phase] * 9 + int(combo)
                    count_cells[key] = count_cells.get(key, 0) + 1
        if split == "train":
            old_cells = sum(key < 36 for key in count_cells)
            correction_cells = sum(key >= 36 for key in count_cells)
            if old_cells != 36:
                raise ValueError(
                    f"train split old replay has only {old_cells}/36 source/phase/combo groups"
                )
            if correction_cells == 0:
                raise ValueError("train split 没有任何纠错 source/phase/combo group")
            print(
                f"train 分层：旧示范 {old_cells}/36；纠错 {correction_cells}/36 "
                "（纠错缺失格子按真实失败分布跳过）"
            )
        print(f"{split}: {len(dataset)} ticks; active labels only; buckets={len(count_cells)}")

        if split == "train":
            sampler = V13BalancedMixtureBatchSampler(
                dataset, ticks_per_batch=8, samples_per_bucket=32, seed=0
            )
            draws = [index for batch in sampler for index in batch]
            expected_draws = len(sampler) * 8
            if len(draws) != expected_draws:
                raise ValueError("balanced sampler epoch has wrong length")
            print(f"train sampler: {len(draws)} draws; sparse correction buckets supported")

    print(
        f"manifest train: old={manifest['train']['old_replay_ticks']} "
        f"correction={manifest['train']['correction_ticks']}"
    )
    print("[通过] 标签、阶段/组合分层和混合采样检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
