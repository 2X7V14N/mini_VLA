"""V1.5a 数据检查：动态放置区小规模数据。

    python scripts/v1_5a/check_dynamic_dataset.py --data data/v1.5a_smoke

检查：
    ① 每个场景恰好 9 条 episode，对应 9 种指令组合
    ② 指令、动作、状态、阶段形状正确
    ③ 只有指令指定的区域移动；其他区域静止
    ④ 区域逐 tick 连续运动、无瞬移、边界内
    ⑤ 非目标方块静止；目标方块只在持有时跟随机器人
    ⑥ 专家成功率 100%，无 WRONG_ZONE
    ⑦ split 按完整 scene 划分，场景不重叠
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_dynamic_v15 import (  # noqa: E402
    ZONE_MOTION_BOUND,
)
from mini_vla.data.grasp_place_expert_v13 import (   # noqa: E402
    COLORS, COMBOS, instruction_v13,
)

PASS, FAIL = "[OK]  ", "[FAIL]"
nc = nf = 0
PHASES = ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")


def check(name, cond, detail=""):
    global nc, nf
    nc += 1
    if not cond:
        nf += 1
    print(f"  {PASS if cond else FAIL} {name:<46}{detail}")


def scene_key(z):
    return tuple(np.round(np.concatenate([
        z["objects"].ravel(),
        z["zones"].ravel(),
        z["robot_start"],
    ]), 6).tolist())


def check_split(name, files, speed_min, speed_max):
    print(f"\n[A] {name}：{len(files)} 条 episode")
    errors = Counter()
    phases = Counter()
    speeds = []
    angles = []
    by_scene = defaultdict(list)
    scenes = set()
    total_frames = 0

    for path in files:
        with np.load(path, allow_pickle=False) as z:
            image = z["image"]
            state = z["robot_state"]
            action = z["action_chunk"]
            instruction = z["instruction"]
            holding = z["holding"]
            gripper = z["gripper_open"]
            object_pos = z["object_positions"]
            zone_pos = z["zone_positions"]
            zone_vel = z["zone_velocities"]
            colors = [str(c) for c in z["colors"]]
            block = str(z["exec_block"])
            zone = str(z["exec_zone"])
            moving = str(z["moving_zone"])
            combo_index = int(z["combo_index"])
            template_index = int(z["template_index"])
            scene_id = str(z["scene_id"])
            velocity0 = z["initial_velocity"]
            success = bool(z["success"])
            failure = str(z["failure"])
            ticks = int(z["ticks"])
            phase = z["phase"]

            frames = len(image)
            total_frames += frames
            if colors != list(COLORS):
                errors["colors"] += 1
                continue
            if image.shape[1:] != (64, 64, 3) or image.dtype != np.uint8:
                errors["image_shape"] += 1
            if state.shape != (frames, 4):
                errors["state_shape"] += 1
            if action.shape != (frames, 4, 3):
                errors["action_shape"] += 1
            if object_pos.shape != (frames, 3, 2):
                errors["object_shape"] += 1
            if zone_pos.shape != (frames, 3, 2):
                errors["zone_shape"] += 1
            if zone_vel.shape != (frames, 3, 2):
                errors["velocity_shape"] += 1
            if ticks != frames:
                errors["ticks"] += 1
            if not success:
                errors["expert_failure"] += 1
            if failure == "WRONG_ZONE":
                errors["wrong_zone"] += 1

            if moving != zone:
                errors["moving_zone"] += 1
            expected_combo = COMBOS[combo_index]
            if (block, zone) != expected_combo:
                errors["combo_index"] += 1
            expected_instruction = instruction_v13(
                block, zone, template_index)
            if not np.all(instruction == expected_instruction):
                errors["instruction"] += 1

            if not np.all(np.isin(action[:, 0, 2], (-1.0, 0.0, 1.0))):
                errors["dgrip"] += 1
            if not np.allclose(holding.astype(np.float32), state[:, 2],
                               atol=1e-6):
                errors["holding"] += 1
            if not np.allclose(gripper, state[:, 3], atol=1e-6):
                errors["gripper"] += 1

            moving_index = COLORS.index(zone)
            target_index = COLORS.index(block)
            moving_pos = zone_pos[:, moving_index]
            moving_vel = zone_vel[:, moving_index]

            if not np.allclose(moving_vel[0], velocity0, atol=1e-9):
                errors["initial_velocity"] += 1
            if np.any(np.abs(moving_pos)
                      > ZONE_MOTION_BOUND + 1e-9):
                errors["zone_bounds"] += 1
            speed = float(np.linalg.norm(velocity0))
            if not (speed_min - 1e-9 <= speed <= speed_max + 1e-9):
                errors["speed_range"] += 1
            if np.any(np.abs(np.linalg.norm(moving_vel, axis=1) - speed)
                      > 1e-9):
                errors["speed_drift"] += 1

            # 逐 tick 连续运动，不允许瞬移。
            for t in range(frames - 1):
                delta = moving_pos[t + 1] - moving_pos[t]
                if (np.linalg.norm(delta)
                        > np.linalg.norm(moving_vel[t]) + 1e-9):
                    errors["teleport"] += 1
                    break

            # 其他区域静止。
            for index, color in enumerate(COLORS):
                if index == moving_index:
                    continue
                if not np.allclose(
                        np.diff(zone_pos[:, index], axis=0), 0.0,
                        atol=1e-9):
                    errors["static_zone_moved"] += 1
                if np.any(np.abs(zone_vel[:, index]) > 1e-9):
                    errors["static_zone_velocity"] += 1

            # 非目标方块静止；目标方块只在持有时跟随机器人。
            robot_xy = state[:, :2].astype(np.float64)
            for index, color in enumerate(COLORS):
                if index == target_index:
                    continue
                if not np.allclose(
                        np.diff(object_pos[:, index], axis=0), 0.0,
                        atol=1e-9):
                    errors["static_object_moved"] += 1
            target_delta = np.diff(
                object_pos[:, target_index].astype(np.float64), axis=0)
            robot_delta = np.diff(robot_xy, axis=0)
            expected = np.where(
                holding[:-1, None], robot_delta, 0.0)
            if not np.allclose(target_delta, expected, atol=1e-6):
                errors["carry"] += 1

            for p in phase:
                phases[str(p)] += 1
            speeds.append(speed)
            angles.append(float(np.arctan2(velocity0[1], velocity0[0])))
            by_scene[scene_id].append((block, zone))
            scenes.add(scene_key(z))

    check("图像 64x64 uint8", errors["image_shape"] == 0,
          str(errors["image_shape"]))
    check("robot_state 为 (T,4)", errors["state_shape"] == 0,
          str(errors["state_shape"]))
    check("action_chunk 为 (T,4,3)", errors["action_shape"] == 0,
          str(errors["action_shape"]))
    check("物体/区域/速度数组形状正确",
          errors["object_shape"] + errors["zone_shape"]
          + errors["velocity_shape"] == 0)
    check("ticks == 帧数", errors["ticks"] == 0, str(errors["ticks"]))
    check("指令与组合/模板一致", errors["instruction"] == 0,
          str(errors["instruction"]))
    check("moving_zone == exec_zone", errors["moving_zone"] == 0,
          str(errors["moving_zone"]))
    check("combo_index 与颜色组合一致", errors["combo_index"] == 0,
          str(errors["combo_index"]))
    check("dgrip 取值在 {-1,0,+1}", errors["dgrip"] == 0,
          str(errors["dgrip"]))
    check("holding/gripper_open 与状态一致",
          errors["holding"] + errors["gripper"] == 0)
    check("只有指定区域移动",
          errors["static_zone_moved"] + errors["static_zone_velocity"] == 0,
          f"{errors['static_zone_moved']}/{errors['static_zone_velocity']}")
    check("区域逐 tick 无瞬移", errors["teleport"] == 0,
          str(errors["teleport"]))
    check("区域始终在运动边界内", errors["zone_bounds"] == 0,
          str(errors["zone_bounds"]))
    check("速度在 [min,max] 内且大小不变",
          errors["speed_range"] + errors["speed_drift"] == 0,
          f"{errors['speed_range']}/{errors['speed_drift']}")
    check("非目标方块静止", errors["static_object_moved"] == 0,
          str(errors["static_object_moved"]))
    check("目标方块只在持有时跟随机器人", errors["carry"] == 0,
          str(errors["carry"]))
    check("专家成功率 100%", errors["expert_failure"] == 0,
          str(errors["expert_failure"]))
    check("无 WRONG_ZONE", errors["wrong_zone"] == 0,
          str(errors["wrong_zone"]))

    bad_scenes = {sid: combos for sid, combos in by_scene.items()
                  if len(combos) != len(COMBOS)
                  or set(combos) != set(COMBOS)}
    check("每个场景恰好覆盖 9 种组合", not bad_scenes, str(len(bad_scenes)))

    print(f"       场景 {len(by_scene)}，episode {len(files)}，"
          f"帧 {total_frames}")
    print(f"       阶段分布 {dict(phases)}")
    return {
        "scenes": scenes,
        "phases": phases,
        "speeds": speeds,
        "angles": angles,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/v1.5a_smoke")
    parser.add_argument("--speed-min", type=float, default=0.01)
    parser.add_argument("--speed-max", type=float, default=0.02)
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        print(f"{root} 不存在")
        return 1

    files = {
        name: sorted((root / name).glob("ep_*.npz"))
        for name in ("train", "val", "test")
    }
    if not files["train"]:
        print(f"{root}/train 下没有 ep_*.npz")
        return 1

    result = {
        name: check_split(name, paths, args.speed_min, args.speed_max)
        for name, paths in files.items() if paths
    }

    scene_sets = [value["scenes"] for value in result.values()]
    overlap = sum(
        len(scene_sets[i] & scene_sets[j])
        for i in range(len(scene_sets))
        for j in range(i + 1, len(scene_sets))
    )
    check("split 间场景不重叠", overlap == 0, str(overlap))

    for name in ("train", "val"):
        if name in result:
            phases = result[name]["phases"]
            check(f"{name} 覆盖四个阶段",
                  all(phases.get(p, 0) > 0 for p in PHASES),
                  str({p: phases.get(p, 0) for p in PHASES}))

    all_speeds = [s for value in result.values() for s in value["speeds"]]
    all_angles = [a for value in result.values() for a in value["angles"]]
    print("\n[B] 速度 / 方向覆盖")
    print(f"  速度 min={min(all_speeds):.4f} max={max(all_speeds):.4f} "
          f"mean={np.mean(all_speeds):.4f}")
    angle_degrees = np.degrees(np.asarray(all_angles)) % 360.0
    bins = np.histogram(angle_degrees, bins=np.arange(0, 361, 45))[0]
    print(f"  方向 8 个 45° 区间计数: {bins.tolist()}")

    print("\n" + "=" * 62)
    print(f"{nc - nf}/{nc} 通过" if nf == 0 else f"有 {nf} 项失败")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
