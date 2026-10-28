"""Audit one-command V1.3 expert recovery data before fine-tuning."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_expert_v13 import COMBOS, instruction_v13  # noqa: E402


def check_episode(path):
    errors = []
    with np.load(path, allow_pickle=False) as data:
        n = len(data["instruction"])
        if not n or n % 9:
            return [f"{path.name}: sample count must be positive and divisible by 9"], 0
        ticks = n // 9
        shapes = {
            "image": (n, 64, 64, 3),
            "robot_state": (n, 4),
            "action_chunk": (n, 4, 3),
            "valid_combo_mask": (n,),
            "combo_index": (n,),
            "pair_id": (n,),
        }
        for key, expected in shapes.items():
            if data[key].shape != expected:
                errors.append(f"{path.name}: {key} shape {data[key].shape}, expected {expected}")
        if not bool(data["success"]):
            errors.append(f"{path.name}: expert continuation failed ({data['failure']})")

        masks = data["valid_combo_mask"].reshape(ticks, 9).astype(bool)
        combo_ids = data["combo_index"].reshape(ticks, 9)
        if not np.array_equal(combo_ids, np.tile(np.arange(9), (ticks, 1))):
            errors.append(f"{path.name}: combo order is invalid")
        if not np.all(masks.sum(axis=1) == 1):
            errors.append(f"{path.name}: each tick must supervise exactly one active command")
        if not np.all(data["image"].reshape(ticks, 9, 64, 64, 3) == data["image"].reshape(ticks, 9, 64, 64, 3)[:, :1]):
            errors.append(f"{path.name}: grouped images differ within a tick")
        if not np.allclose(data["robot_state"].reshape(ticks, 9, 4), data["robot_state"].reshape(ticks, 9, 4)[:, :1]):
            errors.append(f"{path.name}: grouped states differ within a tick")

        expected_index = COMBOS.index((str(data["exec_block"]), str(data["exec_zone"])))
        if not np.all(masks[:, expected_index]):
            errors.append(f"{path.name}: active mask does not match executed instruction")
        expected_prompts = [
            instruction_v13(block, zone, int(data["template_index"][0]))
            for block, zone in COMBOS
        ]
        prompts = data["instruction"].reshape(ticks, 9)
        if not np.all(prompts == np.asarray(expected_prompts)[None, :]):
            errors.append(f"{path.name}: prompt order or text does not match combo indices")
    return errors, ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    root = ROOT / args.data
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "held-out official test split not loaded" not in manifest.get("scene_source", ""):
        raise ValueError("manifest does not prove independent generated scenes")

    failed = False
    total_ticks = {"train": 0, "val": 0}
    episode_counts = {}
    signatures = {"train": set(), "val": set()}
    for split in ("train", "val"):
        files = sorted((root / split).glob("ep_*.npz"))
        episode_counts[split] = len(files)
        if not files:
            print(f"[错误] {split} split has no recoverable episodes")
            failed = True
        for path in files:
            errors, ticks = check_episode(path)
            total_ticks[split] += ticks
            with np.load(path, allow_pickle=False) as data:
                keys = ["object_red", "object_blue", "object_yellow", "robot_start", "zone_red", "zone_blue", "zone_yellow"]
                signature = tuple(np.asarray(data[key]).round(10).tobytes() for key in keys)
                signatures[split].add(signature)
            if errors:
                failed = True
                for error in errors:
                    print(f"[错误] {error}")

    overlap = signatures["train"] & signatures["val"]
    if overlap:
        failed = True
        print(f"[错误] train/val share {len(overlap)} scenes")
    if episode_counts != manifest.get("correction_episodes"):
        failed = True
        print("[错误] episode file counts do not match manifest")

    print(f"新闭环场景：{manifest['scenes']}，seed={manifest['seed']}")
    print(f"模型基线成功率：{manifest['model_success_rate']:.1%} ({manifest['failure_count']} failures)")
    print(f"专家接续：train={episode_counts['train']} episodes/{total_ticks['train']} ticks; val={episode_counts['val']} episodes/{total_ticks['val']} ticks")
    print(f"专家无法从采样状态恢复：{manifest['expert_recovery_failures']}")
    if failed:
        print("[不通过] 纠错数据审计失败")
        return 1
    print("[通过] 失败轨迹纠错标签与实际指令/状态一致，train/val 场景隔离")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
