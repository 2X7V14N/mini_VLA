"""V1.2 闭环纠偏数据录制。

这个脚本在训练/验证 rollout 中注入一次受控的机器人位置偏移，随后让
V1.2 专家从偏移后的状态继续纠正。标签由偏移后的状态重新计算，不能
直接复制原始专家动作。

默认用法（训练/验证带扰动，测试保持干净）：

    python scripts/v1_2/record_grasp_place_v12_corrections.py \
        --out data/v1.2_correction_128 --train 1000 --val 200 --test 200

新增样本仍然严格遵守 V1.2 的四条配对格式，现有 train_v12.py 可以直接
读取这个目录。附加字段不会影响 RobotDataset：
    corrected / perturb_stage / perturb_tick / perturb_delta

扰动是状态扰动，不是给图像加像素噪声。它模拟模型闭环执行后走偏，专家
在该状态上给出恢复动作。V1.2 图像统一为 128x128。测试 split 默认不注入扰动，
用于评估真实闭环。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place import (              # noqa: E402
    CLOSE_ENTER_TOL, GraspPlaceEnv,
)
from mini_vla.data.grasp_place_v12 import (  # noqa: E402
    COMBOS, V12_FINE_TOL, expert_action_v12, instruction_v12, plan_chunk_v12,
    sample_scene_v12,
)
from mini_vla.data.grasp_place_render import render_v12  # noqa: E402


PHASES = ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")
FAIL_CODES = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
              "ZONE_NOT_REACHED", "DROP_OUTSIDE", "WRONG_ZONE", "MAX_TICKS")


def inject_position_offset(env, delta):
    """Apply a physically consistent planar perturbation to the current state.

    When holding an object, move the object with the gripper so the perturbation
    does not manufacture a slip. The expert must still recover the displaced
    robot/object pair on subsequent ticks.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if delta.shape != (2,) or not np.all(np.isfinite(delta)):
        raise ValueError("delta 必须是有限的二维向量")
    env.robot = env.robot + delta
    if env.holding:
        env.objects[env.color] = env.objects[env.color] + delta


def sample_delta(rng, min_norm, max_norm):
    """Sample a uniformly oriented offset with a bounded magnitude."""
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    norm = float(rng.uniform(min_norm, max_norm))
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float64) * norm


def sample_safe_delta(env, rng, stage, min_norm, max_norm):
    """Sample a correction that does not manufacture an unrecoverable CLOSE.

    The V1.2 environment transitions to CLOSE at CLOSE_ENTER_TOL and keeps that phase
    while closing. Therefore a perturbation that moves the gripper outside
    CAPTURE_RADIUS would create a synthetic CLOSE_MISS before the expert has a
    chance to re-approach. For CLOSE, cap the perturbation so the displaced
    state remains capturable; other phases may use the requested range.
    """
    if stage != "CLOSE":
        return sample_delta(rng, min_norm, max_norm)

    from mini_vla.data.grasp_place import CAPTURE_RADIUS

    obj = env.target_object
    dist = float(np.linalg.norm(env.robot - obj))
    safe_max = CAPTURE_RADIUS - 1e-3 - dist
    if safe_max < min_norm:
        return None
    return sample_delta(rng, min_norm, min(max_norm, safe_max))


def choose_correction(rng, enabled, stages):
    if not enabled:
        return None, None
    stage = str(rng.choice(stages))
    return stage, None


def record_rollout(scene, exec_block, exec_zone, template_index, rng,
                   correction=True, correction_stages=PHASES,
                   min_offset=0.01, max_offset=0.06, chunk_size=4,
                   max_ticks=1500, image_size=128, fine_tol=V12_FINE_TOL):
    """Record one rollout and optionally inject one correction perturbation."""
    env = GraspPlaceEnv(scene, exec_block, zone_color=exec_zone,
                        close_enter_tol=CLOSE_ENTER_TOL)
    samples = []
    correction_stage, _ = choose_correction(
        rng, correction, tuple(correction_stages))
    corrected = False
    perturb_tick = -1
    perturb_delta = np.zeros(2, dtype=np.float64)
    t = 0

    while not env.done and t < max_ticks:
        # Inject before rendering/labeling, so the observation and the expert
        # action are both defined at the displaced state.
        if (correction_stage is not None and not corrected
                and env.phase == correction_stage
                and env.phase_ticks >= 1):
            candidate = sample_safe_delta(
                env, rng, correction_stage, min_offset, max_offset)
            if candidate is not None:
                perturb_delta = candidate
                inject_position_offset(env, perturb_delta)
                corrected = True
                perturb_tick = t

        image = render_v12(env, image_size=image_size)
        state = env.observation()["robot_state"]
        for block_color, zone_color in COMBOS:
            samples.append({
                "image": image.copy(),
                "instruction": instruction_v12(block_color, zone_color,
                                               template_index),
                "robot_state": state.copy(),
                "action_chunk": plan_chunk_v12(
                    env, block_color, zone_color, chunk_size,
                    fine_tol=fine_tol),
                "phase": env.phase,
                "target_color": block_color,
                "zone_color": zone_color,
                "object_position": env.objects[block_color].copy(),
                "zone_position": np.asarray(env.zones[zone_color]).copy(),
                "pair_id": t,
                "timestep": t,
                "template_index": template_index,
                "corrected": np.bool_(corrected),
                "perturb_stage": np.array(correction_stage or ""),
                "perturb_tick": np.int32(perturb_tick),
                "perturb_delta": perturb_delta.copy(),
            })

        env.step(expert_action_v12(env, exec_block, fine_tol=fine_tol))
        t += 1

    return samples, env.info(), {
        "corrected": corrected,
        "stage": correction_stage or "",
        "tick": perturb_tick,
        "delta": perturb_delta,
    }


def save_rollout(path, samples, exec_block, exec_zone, scene, info, meta):
    np.savez_compressed(
        path,
        image=np.stack([s["image"] for s in samples]),
        instruction=np.array([s["instruction"] for s in samples]),
        robot_state=np.stack([s["robot_state"] for s in samples]),
        action_chunk=np.stack([s["action_chunk"] for s in samples]),
        phase=np.array([s["phase"] for s in samples]),
        target_color=np.array([s["target_color"] for s in samples]),
        zone_color=np.array([s["zone_color"] for s in samples]),
        object_position=np.stack([s["object_position"] for s in samples]),
        zone_position=np.stack([s["zone_position"] for s in samples]),
        pair_id=np.array([s["pair_id"] for s in samples], dtype=np.int32),
        timestep=np.array([s["timestep"] for s in samples], dtype=np.int32),
        template_index=np.array([s["template_index"] for s in samples],
                                dtype=np.int32),
        corrected=np.array([s["corrected"] for s in samples], dtype=np.bool_),
        perturb_stage=np.array([s["perturb_stage"] for s in samples]),
        perturb_tick=np.array([s["perturb_tick"] for s in samples],
                              dtype=np.int32),
        perturb_delta=np.stack([s["perturb_delta"] for s in samples]),
        exec_block=np.array(exec_block),
        exec_zone=np.array(exec_zone),
        red=np.asarray(scene.red, dtype=np.float64),
        blue=np.asarray(scene.blue, dtype=np.float64),
        robot_start=np.asarray(scene.robot_start, dtype=np.float64),
        zones_red=np.asarray(scene.zones["red"], dtype=np.float64),
        zones_blue=np.asarray(scene.zones["blue"], dtype=np.float64),
        success=np.bool_(info["success"]),
        failure=np.array(info["failure"] or ""),
        ticks=np.int32(info["ticks"]),
        correction_stage=np.array(meta["stage"]),
        correction_tick=np.int32(meta["tick"]),
        correction_delta=np.asarray(meta["delta"], dtype=np.float64),
    )


def record_split(name, n_scenes, seed, out_root, t0, correction, stages,
                 min_offset, max_offset, template_offset, image_size,
                 fine_tol):
    rng = np.random.default_rng(seed)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"scenes": 0, "samples": 0, "corrected": 0,
             "success": 0, "failures": {}, "stages": {}}

    print(f"\n[{name}] {n_scenes} 个场景 -> {out_dir} "
          f"({'带扰动' if correction else '干净测试'})", flush=True)
    for i in range(n_scenes):
        scene = sample_scene_v12(rng)
        exec_block, exec_zone = COMBOS[i % len(COMBOS)]
        samples, info, meta = record_rollout(
            scene, exec_block, exec_zone, (i + template_offset) % 3, rng,
            correction=correction, correction_stages=stages,
            min_offset=min_offset, max_offset=max_offset,
            image_size=image_size, fine_tol=fine_tol)
        save_rollout(out_dir / f"ep_{i:04d}.npz", samples,
                     exec_block, exec_zone, scene, info, meta)

        stats["scenes"] += 1
        stats["samples"] += len(samples)
        stats["corrected"] += int(meta["corrected"])
        stats["success"] += int(info["success"])
        if not info["success"]:
            key = info["failure"] or "UNKNOWN"
            stats["failures"][key] = stats["failures"].get(key, 0) + 1
        if meta["stage"]:
            stats["stages"][meta["stage"]] = (
                stats["stages"].get(meta["stage"], 0) + 1)

        if (i + 1) % 25 == 0 or i + 1 == n_scenes:
            print(f"  {i + 1:>4}/{n_scenes} 纠偏 {stats['corrected']:>4} "
                  f"专家成功 {stats['success'] / stats['scenes']:.1%} "
                  f"样本 {stats['samples']:>7} {time.time() - t0:.0f}s",
                  flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1.2_correction")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=6150)
    ap.add_argument("--min-offset", type=float, default=0.01)
    ap.add_argument("--max-offset", type=float, default=0.06)
    ap.add_argument("--image-size", type=int, choices=(64, 128), default=128,
                    help="V1.2 渲染分辨率")
    ap.add_argument("--fine-tol", type=float, default=V12_FINE_TOL,
                    help="专家在目标区域内允许松手的中心误差")
    ap.add_argument("--test-corrections", action="store_true",
                    help="也给 test 注入扰动；默认测试集保持干净")
    ap.add_argument("--stages", nargs="+", choices=PHASES, default=list(PHASES),
                    help="允许注入扰动的阶段")
    args = ap.parse_args()
    if not (0.0 < args.min_offset <= args.max_offset):
        ap.error("需要满足 0 < min-offset <= max-offset")

    out_root = ROOT / args.out
    if out_root.exists() and any(out_root.rglob("*.npz")):
        print(f"[中止] {out_root} 已有 .npz，请换 --out。")
        return 1

    t0 = time.time()
    summary = {"seed": args.seed, "min_offset": args.min_offset,
               "max_offset": args.max_offset, "stages": args.stages,
               "image_size": args.image_size,
               "fine_tol": args.fine_tol,
               "close_enter_tol": CLOSE_ENTER_TOL,
               "splits": {}}
    for name, count, offset, is_test in (
            ("train", args.train, 0, False),
            ("val", args.val, 10000, False),
            ("test", args.test, 20000, not args.test_corrections)):
        summary["splits"][name] = record_split(
            name, count, args.seed + offset, out_root, t0,
            correction=not is_test, stages=args.stages,
            min_offset=args.min_offset, max_offset=args.max_offset,
            template_offset=offset, image_size=args.image_size,
            fine_tol=args.fine_tol)

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    for name, s in summary["splits"].items():
        print(f"{name:<8} 场景 {s['scenes']:>4} 纠偏 {s['corrected']:>4} "
              f"专家成功 {s['success'] / max(1, s['scenes']):.1%} "
              f"阶段 {s['stages']}")
    print(f"总用时 {time.time() - t0:.0f}s，summary.json 已写入")
    ok = all(s["success"] == s["scenes"]
             for s in summary["splits"].values())
    print(f"{'[通过]' if ok else '[不通过]'} 专家成功率必须 100%")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
