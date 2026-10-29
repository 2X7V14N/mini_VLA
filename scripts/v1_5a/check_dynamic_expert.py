"""V1.5a 专家验收：20 个场景 × 9 种颜色组合。

    python scripts/v1_5a/check_dynamic_expert.py --scenes 20

每条 episode 中，指令指定的放置区以 0.01–0.02 world unit/tick 匀速移动，
到边界后按轴反弹。方块和其他放置区静止。

验收：专家成功率 >= 99%，无 WRONG_ZONE，无区域瞬移。
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_dynamic_expert_v15 import (   # noqa: E402
    expert_action_v15,
)
from mini_vla.data.grasp_place_dynamic_v15 import (          # noqa: E402
    MAX_ZONE_SPEED, MIN_ZONE_SPEED, ZONE_MOTION_BOUND,
    DynamicZoneEnvV15,
)
from mini_vla.data.grasp_place_expert_v13 import (           # noqa: E402
    COMBOS, sample_scene_v13,
)

FAIL_NAMES = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
              "ZONE_NOT_REACHED", "DROP_OUTSIDE", "WRONG_ZONE", "MAX_TICKS")


def direction_bin(angle):
    return int(((angle % (2.0 * np.pi)) // (np.pi / 4.0))) % 8


def speed_bin(speed):
    edges = (0.010, 0.0125, 0.015, 0.0175, 0.0201)
    for index in range(len(edges) - 1):
        if edges[index] <= speed < edges[index + 1]:
            return index
    return len(edges) - 2


def run_one(scene, block, zone, velocity, max_ticks):
    env = DynamicZoneEnvV15(
        scene, block, zone,
        moving_zone=zone,
        zone_velocity=velocity,
    )

    teleport_bad = 0
    while not env.done and env.ticks < max_ticks:
        old_position = env.zones[zone].copy()
        old_velocity = env.zone_velocities[zone].copy()

        env.step(expert_action_v15(env, block))

        if not env.done:
            delta = env.zones[zone] - old_position
            if (np.linalg.norm(delta)
                    > np.linalg.norm(old_velocity) + 1e-9):
                teleport_bad += 1
            if np.any(np.abs(env.zones[zone])
                      > ZONE_MOTION_BOUND + 1e-9):
                teleport_bad += 1

    info = env.info()
    obj = info["object"]
    return {
        "block": block,
        "zone": zone,
        "speed": float(np.linalg.norm(velocity)),
        "angle": float(np.arctan2(velocity[1], velocity[0])),
        "success": bool(info["success"]),
        "failure": info["failure"],
        "ticks": info["ticks"],
        "in_target": env.in_target_zone(obj),
        "in_any": env.in_any_zone(obj),
        "teleport_bad": teleport_bad,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--speed-min", type=float, default=MIN_ZONE_SPEED)
    parser.add_argument("--speed-max", type=float, default=MAX_ZONE_SPEED)
    parser.add_argument("--max-ticks", type=int, default=1500)
    args = parser.parse_args()

    if not (0.0 < args.speed_min <= args.speed_max):
        parser.error("需要 0 < speed-min <= speed-max")

    rng = np.random.default_rng(args.seed)
    rows = []
    total = args.scenes * len(COMBOS)

    for scene_index in range(args.scenes):
        scene = sample_scene_v13(rng)
        for block, zone in COMBOS:
            speed = float(rng.uniform(args.speed_min, args.speed_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            velocity = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            ) * speed
            rows.append(run_one(scene, block, zone, velocity,
                                args.max_ticks))
        done = (scene_index + 1) * len(COMBOS)
        rate = sum(row["success"] for row in rows) / len(rows)
        print(f"{done}/{total} 条，累计成功率 {rate:.1%}", flush=True)

    successes = sum(row["success"] for row in rows)
    failures = Counter(
        row["failure"] or "UNKNOWN" for row in rows if not row["success"]
    )
    teleport = sum(row["teleport_bad"] for row in rows)

    print("\n" + "=" * 70)
    print(f"场景 {args.scenes} × 组合 {len(COMBOS)} = {total} 条")
    print(f"专家成功 {successes}/{total} = {successes / total:.2%}")
    print(f"平均 tick {np.mean([row['ticks'] for row in rows]):.1f}")
    print(f"区域瞬移/越界检查违规：{teleport}")

    print("\n按方块→区域组合：")
    for block, zone in COMBOS:
        subset = [row for row in rows
                  if row["block"] == block and row["zone"] == zone]
        rate = sum(row["success"] for row in subset) / len(subset)
        print(f"  {block}->{zone:<7}{rate:>8.1%}")

    print("\n按速度区间：")
    speed_groups = defaultdict(list)
    for row in rows:
        speed_groups[speed_bin(row["speed"])].append(row)
    speed_edges = ("0.010-0.0125", "0.0125-0.015",
                   "0.015-0.0175", "0.0175-0.020")
    for index in sorted(speed_groups):
        subset = speed_groups[index]
        rate = sum(row["success"] for row in subset) / len(subset)
        print(f"  {speed_edges[index]:<16}{rate:>8.1%}  n={len(subset)}")

    print("\n按运动方向（8 个 45° 区间）：")
    direction_groups = defaultdict(list)
    for row in rows:
        direction_groups[direction_bin(row["angle"])].append(row)
    for index in sorted(direction_groups):
        subset = direction_groups[index]
        rate = sum(row["success"] for row in subset) / len(subset)
        lo = index * 45
        print(f"  {lo:>3}-{lo + 45:<3}°{rate:>10.1%}  n={len(subset)}")

    print("\n失败构成：")
    if failures:
        for name in FAIL_NAMES:
            if failures.get(name):
                print(f"  {name:<20}{failures[name]:>5}")
    else:
        print("  无失败")

    passed = (successes / total >= 0.99 and not failures.get("WRONG_ZONE")
              and teleport == 0)
    print("\n" + "=" * 70)
    print("[通过] V1.5a 专家验收" if passed else "[未通过] V1.5a 专家验收")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
