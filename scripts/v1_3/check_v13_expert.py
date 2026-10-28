import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_expert_v13 import (  # noqa: E402
    COMBOS,
    run_expert_episode,
    sample_scene_v13,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    totals = {combo: {"success": 0, "ticks": []} for combo in COMBOS}
    attempts = []

    start = time.time()
    for index in range(args.scenes):
        scene = sample_scene_v13(rng)
        attempts.append(1)  # 先记录场景采样成功；下步再补精确尝试数。

        for block, zone in COMBOS:
            result = run_expert_episode(scene, block, zone)
            slot = totals[(block, zone)]
            slot["success"] += int(result["success"])
            slot["ticks"].append(result["ticks"])

        if (index + 1) % 5 == 0 or index + 1 == args.scenes:
            done = (index + 1) * len(COMBOS)
            success = sum(v["success"] for v in totals.values())
            print(
                f"{index + 1}/{args.scenes} 场景，"
                f"专家成功率 {success / done:.1%}",
                flush=True,
            )

    total = args.scenes * len(COMBOS)
    successes = sum(v["success"] for v in totals.values())

    print("\n各组合结果")
    for (block, zone), values in totals.items():
        count = args.scenes
        mean_ticks = float(np.mean(values["ticks"]))
        print(
            f"{block:>6} -> {zone:<6} "
            f"{values['success']:>4}/{count} "
            f"{values['success'] / count:>6.1%} "
            f"平均 tick={mean_ticks:.1f}"
        )

    print(
        f"\n总计：{successes}/{total} = {successes / total:.2%}，"
        f"耗时 {time.time() - start:.1f}s"
    )
    if successes != total:
        print("[不通过] 有专家失败；先查环境、容差和采样约束。")
        return 1

    print("[通过] 测试范围内 9 种组合全部成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
