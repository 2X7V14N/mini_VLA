"""Record V1.5c expert episodes: commanded block and zone both move."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_render_v15a import render_dynamic_v15a
from mini_vla.data.grasp_place_dynamic_expert_v15c import expert_action_v15c
from mini_vla.data.grasp_place_dynamic_v15c import DynamicBothEnvV15
from mini_vla.data.grasp_place_expert_v13 import COLORS, COMBOS, instruction_v13, sample_scene_v13


def velocity(rng, low, high):
    speed, angle = rng.uniform(low, high), rng.uniform(0.0, 2.0 * np.pi)
    return speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


def plan_chunk(env, block, chunk_size=4):
    sim, actions = copy.deepcopy(env), []
    for _ in range(chunk_size):
        actions.append(np.zeros(3, dtype=np.float32) if sim.done else expert_action_v15c(sim, block))
        if not sim.done:
            sim.step(actions[-1])
    return np.asarray(actions, dtype=np.float32)


def record_episode(scene, block, zone, block_velocity, zone_velocity, template, max_ticks):
    env = DynamicBothEnvV15(
        scene, block, zone, moving_zone=zone,
        block_velocity=block_velocity, zone_velocity=zone_velocity,
    )
    rows = []
    while not env.done and env.ticks < max_ticks:
        rows.append({
            "image": render_dynamic_v15a(env),
            "instruction": instruction_v13(block, zone, template),
            "robot_state": env.observation()["robot_state"].copy(),
            "action_chunk": plan_chunk(env, block),
            "object_positions": np.stack([env.objects[c].copy() for c in COLORS]),
            "object_velocities": np.stack([env.block_velocity[c].copy() for c in COLORS]),
            "zone_positions": np.stack([env.zones[c].copy() for c in COLORS]),
            "zone_velocities": np.stack([env.zone_velocities[c].copy() for c in COLORS]),
            "phase": env.phase,
            "holding": bool(env.holding),
            "gripper_open": float(env.gripper_open),
        })
        env.step(expert_action_v15c(env, block))
    if not env.done:
        env._fail("MAX_TICKS")
    return rows, env.info()


def save_episode(path, rows, scene, block, zone, block_velocity, zone_velocity, template, combo, scene_id, info):
    np.savez_compressed(
        path,
        image=np.stack([r["image"] for r in rows]),
        instruction=np.asarray([r["instruction"] for r in rows]),
        robot_state=np.stack([r["robot_state"] for r in rows]),
        action_chunk=np.stack([r["action_chunk"] for r in rows]),
        object_positions=np.stack([r["object_positions"] for r in rows]),
        object_velocities=np.stack([r["object_velocities"] for r in rows]),
        zone_positions=np.stack([r["zone_positions"] for r in rows]),
        zone_velocities=np.stack([r["zone_velocities"] for r in rows]),
        phase=np.asarray([r["phase"] for r in rows]),
        holding=np.asarray([r["holding"] for r in rows], dtype=np.bool_),
        gripper_open=np.asarray([r["gripper_open"] for r in rows], dtype=np.float32),
        exec_block=np.asarray(block), exec_zone=np.asarray(zone), combo_index=np.int32(combo),
        template_index=np.int32(template), scene_id=np.asarray(scene_id), colors=np.asarray(COLORS),
        objects=np.stack([scene.objects[c] for c in COLORS]), zones=np.stack([scene.zones[c] for c in COLORS]),
        robot_start=np.asarray(scene.robot_start), initial_block_velocity=block_velocity,
        initial_zone_velocity=zone_velocity, success=np.bool_(info["success"]),
        failure=np.asarray(info["failure"] or ""), ticks=np.int32(info["ticks"]),
    )


def record_split(name, scenes, seed, root, low, high, max_ticks):
    rng, directory = np.random.default_rng(seed), root / name
    directory.mkdir(parents=True, exist_ok=True)
    stats = {"scenes": scenes, "episodes": 0, "frames": 0, "success": 0, "failures": {}}
    for scene_index in range(scenes):
        scene = sample_scene_v13(rng)
        for combo, (block, zone) in enumerate(COMBOS):
            block_v, zone_v = velocity(rng, low, high), velocity(rng, low, high)
            rows, info = record_episode(scene, block, zone, block_v, zone_v, (scene_index + combo) % 3, max_ticks)
            save_episode(directory / f"ep_{scene_index:04d}_{combo:02d}.npz", rows, scene, block, zone, block_v, zone_v, (scene_index + combo) % 3, combo, f"{name}_{scene_index:04d}", info)
            stats["episodes"] += 1; stats["frames"] += len(rows); stats["success"] += int(info["success"])
            if not info["success"]:
                key = info["failure"] or "UNKNOWN"; stats["failures"][key] = stats["failures"].get(key, 0) + 1
        if (scene_index + 1) % 25 == 0 or scene_index + 1 == scenes:
            print(f"[{name}] {scene_index + 1}/{scenes}; expert={stats['success']}/{stats['episodes']}; frames={stats['frames']}", flush=True)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/v1.5c")
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8151)
    parser.add_argument("--speed-min", type=float, default=0.01)
    parser.add_argument("--speed-max", type=float, default=0.02)
    parser.add_argument("--max-ticks", type=int, default=600)
    args = parser.parse_args()
    if not (0 < args.speed_min <= args.speed_max):
        parser.error("需要 0 < speed-min <= speed-max")
    root = ROOT / args.out
    if root.exists() and any(root.rglob("*.npz")):
        raise FileExistsError(f"为避免覆盖，数据目录已有 episode：{root}")
    summary = {"version": "v1.5c", "seed": args.seed, "speed_min": args.speed_min, "speed_max": args.speed_max, "splits": {}}
    for name, count, offset in (("train", args.train, 0), ("val", args.val, 10_000), ("test", args.test, 20_000)):
        summary["splits"][name] = record_split(name, count, args.seed + offset, root, args.speed_min, args.speed_max, args.max_ticks)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(x["episodes"] for x in summary["splits"].values()); success = sum(x["success"] for x in summary["splits"].values())
    print(f"expert total={success}/{total} ({success / max(1,total):.1%}); data={root}")
    return 0 if success == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
