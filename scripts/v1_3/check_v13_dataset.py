import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

COLORS = ("red", "blue", "yellow")
EXPECTED_COMBOS = [
    (block, zone)
    for block in COLORS
    for zone in COLORS
]


def scene_signature(data):
    """用三种方块、三个区域和机器人起点的位置标识场景。"""
    keys = [
        "object_red", "object_blue", "object_yellow",
        "zone_red", "zone_blue", "zone_yellow",
        "robot_start",
    ]
    return tuple(
        np.asarray(data[key], dtype=np.float64).round(10).tobytes()
        for key in keys
    )


def check_episode(path):
    errors = []

    with np.load(path, allow_pickle=False) as data:
        n = len(data["instruction"])

        required_shapes = {
            "image": (n, 64, 64, 3),
            "robot_state": (n, 4),
            "action_chunk": (n, 4, 3),
            "valid_combo_mask": (n,),
            "object_position": (n, 2),
            "zone_position": (n, 2),
            "pair_id": (n,),
            "combo_index": (n,),
            "target_color": (n,),
            "zone_color": (n,),
        }

        for key, expected_shape in required_shapes.items():
            actual_shape = data[key].shape
            if actual_shape != expected_shape:
                errors.append(
                    f"{key} 形状错误：期望 {expected_shape}，实际 {actual_shape}"
                )

        if n == 0 or n % 9 != 0:
            errors.append(f"样本数 {n} 不是 9 的正整数倍")
        else:
            for start in range(0, n, 9):
                end = start + 9
                group = slice(start, end)

                pair_ids = data["pair_id"][group]
                combo_ids = data["combo_index"][group].tolist()

                if len(set(pair_ids.tolist())) != 1:
                    errors.append(f"样本 {start}:{end} 的 pair_id 不一致")

                if combo_ids != list(range(9)):
                    errors.append(
                        f"样本 {start}:{end} 的 combo_index 错误：{combo_ids}"
                    )

                first_image = data["image"][start]
                first_state = data["robot_state"][start]

                if not np.all(data["image"][group] == first_image):
                    errors.append(f"样本 {start}:{end} 的图像不一致")

                if not np.allclose(data["robot_state"][group], first_state):
                    errors.append(f"样本 {start}:{end} 的机器人状态不一致")

                valid_mask = data["valid_combo_mask"][group].astype(bool)
                if not np.all(valid_mask):
                    errors.append(
                        f"样本 {start}:{end} 未保留第一次 V1.3 的九组合监督规则"
                    )

                actual_combos = list(zip(
                    data["target_color"][group].astype(str).tolist(),
                    data["zone_color"][group].astype(str).tolist(),
                ))
                if actual_combos != EXPECTED_COMBOS:
                    errors.append(
                        f"样本 {start}:{end} 的颜色组合顺序或内容错误"
                    )

        if not bool(data["success"]):
            errors.append(f"专家 episode 失败：{data['failure']}")

        signature = scene_signature(data)

    return errors, n, signature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v1.3_smoke_record1")
    args = parser.parse_args()

    data_root = ROOT / args.data
    print(f"检查目录：{data_root}")

    all_signatures = {}
    total_episodes = 0
    total_samples = 0
    failed = False

    for split in ("train", "val", "test"):
        split_dir = data_root / split
        episodes = sorted(split_dir.glob("ep_*.npz"))

        if not episodes:
            print(f"{split}: 没找到 ep_*.npz")
            failed = True
            continue

        split_signatures = set()
        split_samples = 0

        for episode in episodes:
            errors, n, signature = check_episode(episode)
            split_samples += n

            if errors:
                failed = True
                for error in errors:
                    print(f"[错误] {episode.name}: {error}")

            split_signatures.add(signature)

        all_signatures[split] = split_signatures
        total_episodes += len(episodes)
        total_samples += split_samples

        print(
            f"{split:5} episodes={len(episodes):4} "
            f"samples={split_samples:6} "
            f"ticks={split_samples // 9:5}"
        )

    split_names = list(all_signatures)
    for i, first in enumerate(split_names):
        for second in split_names[i + 1:]:
            overlap = all_signatures[first] & all_signatures[second]
            if overlap:
                failed = True
                print(
                    f"[错误] {first}/{second} 有 {len(overlap)} 个重复场景"
                )

    print(f"\n总 episodes={total_episodes} samples={total_samples}")

    if failed:
        print("[不通过] 数据检查发现问题，请先修复。")
        return 1

    print(
        "[通过] 形状、四维状态、九种组合、有效标签掩码、配对观测、"
        "专家结果和 split 隔离均正常"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
