"""录制 V1.3 三色抓放数据。

每个物理 rollout 的每个 tick 记录同一观测下的 9 种指令动作标签，
    沿用第一次 V1.3 的全组合监督规则，并记录四维机器人状态：
    x、y、holding、gripper_open。
正式训练时再通过 loss 权重实现新组合 80%、红蓝复习组合 20%。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_expert_v13 import (  # noqa: E402
    COMBOS,
    V13_FINE_TOL,
    expert_action_v13,
    instruction_v13,
    sample_scene_v13,
    valid_combo_mask_v13,
)
from mini_vla.data.grasp_place_render_v13 import render_v13  # noqa: E402
from mini_vla.data.grasp_place_v13 import GraspPlaceEnvV13  # noqa: E402

CHUNK_SIZE = 4
MAX_TICKS = 1500


def plan_chunk(env, block_color, zone_color):
    """从当前物理状态出发，推演指定指令接下来的专家动作。"""
    import copy

    sim = copy.deepcopy(env)

    # 空手时可以切换目标方块；已抓住时，手里的物体不能被反事实切换。
    if not sim.holding:
        sim.color = block_color

    sim.zone_color = zone_color
    sim.target_zone = sim.zones[zone_color]

    actions = []
    for _ in range(CHUNK_SIZE):
        if sim.done:
            actions.append(np.zeros(3, dtype=np.float32))
            continue

        action = expert_action_v13(sim, block_color)
        actions.append(action.copy())
        sim.step(action)

    return np.asarray(actions, dtype=np.float32)


def record_episode(scene, exec_block, exec_zone, template_index):
    """执行一条专家轨迹，并在每个 tick 记录九种指令标签。"""
    env = GraspPlaceEnvV13(scene, exec_block, exec_zone)
    rows = []

    while not env.done and env.ticks < MAX_TICKS:
        image = render_v13(env)
        robot_state = env.observation()["robot_state"]
        valid_mask = valid_combo_mask_v13(
            env.holding, exec_block, exec_zone
        )

        for combo_index, (block_color, zone_color) in enumerate(COMBOS):
            rows.append({
                "image": image.copy(),
                "instruction": instruction_v13(
                    block_color, zone_color, template_index
                ),
                "robot_state": robot_state.copy(),
                "action_chunk": plan_chunk(env, block_color, zone_color),
                "valid_combo_mask": valid_mask[combo_index],
                "phase": env.phase,
                "target_color": block_color,
                "zone_color": zone_color,
                "object_position": env.objects[block_color].copy(),
                "zone_position": env.zones[zone_color].copy(),
                "pair_id": env.ticks,
                "combo_index": combo_index,
                "timestep": env.ticks,
                "template_index": template_index,
            })

        action = expert_action_v13(env, exec_block)
        env.step(action)

    if not env.done:
        env._fail("MAX_TICKS")

    return rows, env.info()


def save_episode(path, rows, scene, exec_block, exec_zone, info):
    np.savez_compressed(
        path,
        image=np.stack([row["image"] for row in rows]),
        instruction=np.asarray([row["instruction"] for row in rows]),
        robot_state=np.stack([row["robot_state"] for row in rows]),
        action_chunk=np.stack([row["action_chunk"] for row in rows]),
        valid_combo_mask=np.asarray(
            [row["valid_combo_mask"] for row in rows], dtype=np.bool_
        ),
        phase=np.asarray([row["phase"] for row in rows]),
        target_color=np.asarray([row["target_color"] for row in rows]),
        zone_color=np.asarray([row["zone_color"] for row in rows]),
        object_position=np.stack([row["object_position"] for row in rows]),
        zone_position=np.stack([row["zone_position"] for row in rows]),
        pair_id=np.asarray([row["pair_id"] for row in rows], dtype=np.int32),
        combo_index=np.asarray(
            [row["combo_index"] for row in rows], dtype=np.int32
        ),
        timestep=np.asarray([row["timestep"] for row in rows], dtype=np.int32),
        template_index=np.asarray(
            [row["template_index"] for row in rows], dtype=np.int32
        ),
        exec_block=np.asarray(exec_block),
        exec_zone=np.asarray(exec_zone),
        object_red=scene.objects["red"],
        object_blue=scene.objects["blue"],
        object_yellow=scene.objects["yellow"],
        robot_start=scene.robot_start,
        zone_red=scene.zones["red"],
        zone_blue=scene.zones["blue"],
        zone_yellow=scene.zones["yellow"],
        success=np.bool_(info["success"]),
        failure=np.asarray(info["failure"] or ""),
        ticks=np.int32(info["ticks"]),
    )


def record_split(name, count, seed, output_root, started):
    rng = np.random.default_rng(seed)
    split_dir = output_root / name
    split_dir.mkdir(parents=True, exist_ok=True)

    stats = {"scenes": count, "success": 0, "samples": 0, "failures": {}}

    print(f"\n[{name}] {count} 个场景 -> {split_dir}", flush=True)

    for index in range(count):
        scene = sample_scene_v13(rng)

        # 轮换实际执行组合，避免物理 rollout 只覆盖少数组合。
        exec_block, exec_zone = COMBOS[index % len(COMBOS)]
        template_index = index % 3

        rows, info = record_episode(
            scene, exec_block, exec_zone, template_index
        )
        save_episode(
            split_dir / f"ep_{index:04d}.npz",
            rows, scene, exec_block, exec_zone, info,
        )

        stats["samples"] += len(rows)
        stats["success"] += int(info["success"])
        if not info["success"]:
            reason = info["failure"] or "UNKNOWN"
            stats["failures"][reason] = stats["failures"].get(reason, 0) + 1

        if (index + 1) % 5 == 0 or index + 1 == count:
            rate = stats["success"] / (index + 1)
            print(
                f"  {index + 1}/{count} 场景"
                f"  专家成功率 {rate:.1%}"
                f"  样本 {stats['samples']}"
                f"  用时 {time.time() - started:.1f}s",
                flush=True,
            )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/v1.3_smoke")
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=5)
    parser.add_argument("--test", type=int, default=5)
    parser.add_argument("--seed", type=int, default=5150)
    args = parser.parse_args()

    output_root = ROOT / args.out
    if output_root.exists() and any(output_root.rglob("*.npz")):
        raise FileExistsError(
            f"{output_root} 已含有 .npz 文件，请换一个 --out 路径。"
        )

    started = time.time()
    summary = {"seed": args.seed, "splits": {}}

    for name, count, seed_offset in (
        ("train", args.train, 0),
        ("val", args.val, 10_000),
        ("test", args.test, 20_000),
    ):
        summary["splits"][name] = record_split(
            name, count, args.seed + seed_offset, output_root, started
        )

    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    total_scenes = sum(v["scenes"] for v in summary["splits"].values())
    total_samples = sum(v["samples"] for v in summary["splits"].values())
    total_success = sum(v["success"] for v in summary["splits"].values())

    print("\n数据录制汇总")
    for name, stats in summary["splits"].items():
        rate = stats["success"] / max(1, stats["scenes"])
        print(
            f"{name:5} 场景={stats['scenes']:>4} "
            f"样本={stats['samples']:>7} 专家成功率={rate:.1%} "
            f"失败={stats['failures']}"
        )

    print(
        f"总计：场景 {total_scenes}，样本 {total_samples}，"
        f"成功率 {total_success / max(1, total_scenes):.1%}"
    )
    print(f"数据目录：{output_root}")
    print(f"摘要文件：{summary_path}")

    return 0 if total_success == total_scenes else 1


if __name__ == "__main__":
    raise SystemExit(main())
