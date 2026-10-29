"""V1.5a 小规模数据录制：每条 (场景, 方块, 放置区) 组合独立 rollout。

每个场景 × 9 种指令 = 9 条 episode。每条 episode 中只有该指令指定的
放置区以 0.01–0.02 world unit/tick 移动，其他区域和方块静止。

产物：
    {out}/{train,val,test}/ep_{scene:04d}_{combo:02d}.npz
    {out}/summary.json

先跑小规模：
    python scripts/v1_5a/record_dynamic_data.py --train 20 --val 5 --test 5
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_dynamic_expert_v15 import (  # noqa: E402
    expert_action_v15,
)
from mini_vla.data.grasp_place_dynamic_v15 import (         # noqa: E402
    MAX_ZONE_SPEED, MIN_ZONE_SPEED, DynamicZoneEnvV15,
)
from mini_vla.data.grasp_place_expert_v13 import (          # noqa: E402
    COLORS, COMBOS, instruction_v13, sample_scene_v13,
)
from mini_vla.data.dynamic_render_v15a import render_dynamic_v15a  # noqa: E402


CHUNK_SIZE = 4


def plan_chunk(env, block, chunk_size=CHUNK_SIZE):
    """Generate a short expert chunk without changing the recorded env."""
    sim = copy.deepcopy(env)
    actions = []
    for _ in range(chunk_size):
        if sim.done:
            actions.append(np.zeros(3, dtype=np.float32))
            continue
        action = expert_action_v15(sim, block)
        actions.append(action.copy())
        sim.step(action)
    return np.asarray(actions, dtype=np.float32)


def record_episode(scene, block, zone, velocity, template_index,
                   max_ticks):
    env = DynamicZoneEnvV15(
        scene, block, zone,
        moving_zone=zone,
        zone_velocity=velocity,
    )
    rows = []

    while not env.done and env.ticks < max_ticks:
        rows.append({
            "image": render_dynamic_v15a(env),
            "instruction": instruction_v13(block, zone, template_index),
            "robot_state": env.observation()["robot_state"].copy(),
            # Preserve the V1.3 action-head contract (chunk=4). Closed-loop
            # execution consumes only the first action of this chunk.
            "action_chunk": plan_chunk(env, block),
            "phase": env.phase,
            "object_positions": np.stack(
                [env.objects[color].copy() for color in COLORS]),
            "zone_positions": np.stack(
                [env.zones[color].copy() for color in COLORS]),
            "zone_velocities": np.stack(
                [env.zone_velocities[color].copy() for color in COLORS]),
            "holding": bool(env.holding),
            "gripper_open": float(env.gripper_open),
            "timestep": env.ticks,
        })
        env.step(expert_action_v15(env, block))

    info = env.info()
    success = bool(info["success"])
    failure = info["failure"]
    if not env.done:
        success = False
        failure = "MAX_TICKS"

    return rows, {
        "success": success,
        "failure": failure,
        "ticks": info["ticks"],
    }


def save_episode(path, rows, scene, block, zone, velocity,
                 template_index, combo_index, scene_id, info):
    np.savez_compressed(
        path,
        image=np.stack([row["image"] for row in rows]),
        instruction=np.asarray([row["instruction"] for row in rows]),
        robot_state=np.stack([row["robot_state"] for row in rows]),
        action_chunk=np.stack([row["action_chunk"] for row in rows]),
        phase=np.asarray([row["phase"] for row in rows]),
        object_positions=np.stack(
            [row["object_positions"] for row in rows]),
        zone_positions=np.stack(
            [row["zone_positions"] for row in rows]),
        zone_velocities=np.stack(
            [row["zone_velocities"] for row in rows]),
        holding=np.asarray([row["holding"] for row in rows], dtype=np.bool_),
        gripper_open=np.asarray(
            [row["gripper_open"] for row in rows], dtype=np.float32),
        timestep=np.asarray(
            [row["timestep"] for row in rows], dtype=np.int32),
        exec_block=np.asarray(block),
        exec_zone=np.asarray(zone),
        moving_zone=np.asarray(zone),
        combo_index=np.int32(combo_index),
        template_index=np.int32(template_index),
        scene_id=np.asarray(scene_id),
        colors=np.asarray(COLORS),
        objects=np.stack([scene.objects[color] for color in COLORS]),
        zones=np.stack([scene.zones[color] for color in COLORS]),
        robot_start=np.asarray(scene.robot_start, dtype=np.float64),
        initial_velocity=np.asarray(velocity, dtype=np.float64),
        success=np.bool_(info["success"]),
        failure=np.asarray(info["failure"] or ""),
        ticks=np.int32(info["ticks"]),
    )


def record_split(name, scenes, seed, out_root, started,
                 speed_min, speed_max, max_ticks):
    rng = np.random.default_rng(seed)
    split_dir = out_root / name
    split_dir.mkdir(parents=True, exist_ok=True)

    stats = {"scenes": scenes, "episodes": 0, "frames": 0,
             "success": 0, "failures": {}, "speeds": [], "angles": []}

    print(f"\n[{name}] {scenes} 个场景 × {len(COMBOS)} 组合 -> {split_dir}",
          flush=True)
    for scene_index in range(scenes):
        scene = sample_scene_v13(rng)
        scene_id = f"{name}_{scene_index:04d}"

        for combo_index, (block, zone) in enumerate(COMBOS):
            speed = float(rng.uniform(speed_min, speed_max))
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            velocity = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float64) * speed
            template_index = (scene_index + combo_index) % 3

            rows, info = record_episode(
                scene, block, zone, velocity, template_index, max_ticks)

            save_episode(
                split_dir / f"ep_{scene_index:04d}_{combo_index:02d}.npz",
                rows, scene, block, zone, velocity,
                template_index, combo_index, scene_id, info,
            )

            stats["episodes"] += 1
            stats["frames"] += len(rows)
            stats["success"] += int(info["success"])
            stats["speeds"].append(speed)
            stats["angles"].append(angle)
            if not info["success"]:
                key = info["failure"] or "UNKNOWN"
                stats["failures"][key] = stats["failures"].get(key, 0) + 1

        if (scene_index + 1) % 5 == 0 or scene_index + 1 == scenes:
            done = scene_index + 1
            print(
                f"  {done}/{scenes} 场景  "
                f"episode {stats['episodes']}  "
                f"专家成功率 {stats['success'] / stats['episodes']:.1%}  "
                f"帧 {stats['frames']}  "
                f"{time.time() - started:.1f}s",
                flush=True,
            )
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/v1.5a_smoke")
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=5)
    parser.add_argument("--test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=8150)
    parser.add_argument("--speed-min", type=float,
                        default=MIN_ZONE_SPEED)
    parser.add_argument("--speed-max", type=float,
                        default=MAX_ZONE_SPEED)
    parser.add_argument("--max-ticks", type=int, default=1500)
    args = parser.parse_args()

    if not (0.0 < args.speed_min <= args.speed_max):
        parser.error("需要 0 < speed-min <= speed-max")

    out_root = ROOT / args.out
    if out_root.exists() and any(out_root.rglob("*.npz")):
        print(f"[中止] {out_root} 已有 .npz，请换 --out。")
        return 1

    started = time.time()
    summary = {
        "seed": args.seed,
        "colors": list(COLORS),
        "combos": [f"{b}->{z}" for b, z in COMBOS],
        "speed_min": args.speed_min,
        "speed_max": args.speed_max,
        "splits": {},
    }

    for name, count, offset in (
            ("train", args.train, 0),
            ("val", args.val, 10_000),
            ("test", args.test, 20_000)):
        summary["splits"][name] = record_split(
            name, count, args.seed + offset, out_root, started,
            args.speed_min, args.speed_max, args.max_ticks)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    for name, stats in summary["splits"].items():
        rate = stats["success"] / max(1, stats["episodes"])
        print(f"{name:<6} 场景 {stats['scenes']:>4}  "
              f"episode {stats['episodes']:>4}  "
              f"帧 {stats['frames']:>6}  "
              f"专家成功率 {rate:.1%}  失败 {stats['failures']}")
    total_episodes = sum(v["episodes"] for v in summary["splits"].values())
    total_success = sum(v["success"] for v in summary["splits"].values())
    total_frames = sum(v["frames"] for v in summary["splits"].values())
    print(f"\n总计 episode {total_episodes}，帧 {total_frames}，"
          f"成功率 {total_success / max(1, total_episodes):.1%}")
    print(f"数据目录：{out_root}")
    print(f"摘要：{summary_path}")

    ok = total_success == total_episodes
    print(f"\n{'[通过]' if ok else '[不通过]'} 专家成功率必须 100%")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
