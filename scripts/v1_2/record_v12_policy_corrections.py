"""从 V1.2 模型自己的闭环轨迹采集纠偏数据。

这个脚本与随机位置扰动数据不同：模型在全新场景上执行，脚本把模型
实际走到的每个状态保存下来，再从同一个状态计算四种指令的专家动作块。
这样训练集覆盖的是模型真实会遇到的接近、闭合和搬运偏差。

输出目录会先复制基础 V1.2 数据，再追加新的 train/val rollout；test
保持基础数据原样。新场景使用独立随机种子，不会把当前测试场景带进训练。

示例：
    python scripts/v1_2/record_v12_policy_corrections.py \
        --ckpt checkpoints/v1.2_final.pt \
        --base-data data/v1.2_correction_64_close08_safe \
        --out data/v1.2_correction_64_close08_policy \
        --train 400 --val 100 --fine-tol 0.08
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mini_vla.data.grasp_place import CLOSE_ENTER_TOL, GraspPlaceEnv, Scene  # noqa: E402
from mini_vla.data.grasp_place_render import render  # noqa: E402
from mini_vla.data.grasp_place_v12 import (  # noqa: E402
    COMBOS,
    V12_FINE_TOL,
    expert_action_v12,
    instruction_v12,
    plan_chunk_v12,
    sample_scene_v12,
)
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402
from mini_vla.models.mini_vla import LanguageConditionedMiniVLA  # noqa: E402


PHASES = ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")


def load_policy(ckpt_path: Path, device: torch.device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    tok = SimpleCharTokenizer(ck["vocab"], max_length=ck["max_length"])
    if ck.get("model_type") != "LanguageConditionedMiniVLA":
        raise ValueError("纠偏采集需要 LanguageConditionedMiniVLA checkpoint")
    model = LanguageConditionedMiniVLA(
        vocab_size=tok.vocab_size,
        max_length=ck["max_length"],
        state_dim=ck["state_dim"],
        action_dim=ck["action_dim"],
        chunk_size=ck["chunk_size"],
        red_token_id=ck["red_token_id"],
        blue_token_id=ck["blue_token_id"],
    )
    model.load_state_dict(ck["model_state"])
    model.image_size = int(ck.get("image_size", 64))
    model.to(device).eval()
    return model, tok, ck


@torch.inference_mode()
def model_action(model, tok, device, env, block, zone, template_index):
    image = render(env, image_size=model.image_size)
    x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    x = x.to(device) / 255.0
    text = tok.encode(instruction_v12(block, zone, template_index))
    text = text.unsqueeze(0).to(device)
    state = torch.from_numpy(env.observation()["robot_state"])
    state = state.unsqueeze(0).to(device)
    # 闭环只执行第一步，和 eval_v12_closed_loop.py 完全一致。
    return model(x, text, state)[0, 0].detach().cpu().numpy().astype(np.float32), image


def scene_from_sample(scene):
    # 仅用于让类型/字段意图清晰；sample_scene_v12 已返回 Scene。
    return Scene(
        red=np.asarray(scene.red, dtype=np.float64),
        blue=np.asarray(scene.blue, dtype=np.float64),
        robot_start=np.asarray(scene.robot_start, dtype=np.float64),
        zones={k: np.asarray(v, dtype=np.float64)
               for k, v in scene.zones.items()},
    )


def record_rollout(scene, exec_block, exec_zone, template_index, model, tok,
                   device, chunk_size, fine_tol, max_ticks):
    """执行一条模型 rollout，并在每个模型状态上生成四路专家标签。"""
    env = GraspPlaceEnv(
        scene, exec_block, zone_color=exec_zone,
        close_enter_tol=CLOSE_ENTER_TOL,
    )
    samples = []
    t = 0
    while not env.done and t < max_ticks:
        action, image = model_action(
            model, tok, device, env, exec_block, exec_zone, template_index)
        state = env.observation()["robot_state"]
        for block_color, zone_color in COMBOS:
            samples.append({
                "image": image.copy(),
                "instruction": instruction_v12(
                    block_color, zone_color, template_index),
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
                "corrected": np.bool_(True),
                "perturb_stage": np.array(env.phase),
                "perturb_tick": np.int32(t),
                "perturb_delta": np.zeros(2, dtype=np.float64),
            })
        env.step(action)
        t += 1

    return samples, env.info(), t


def save_rollout(path, samples, exec_block, exec_zone, scene, info,
                 template_index, source="policy_closed_loop"):
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
        template_index=np.array(
            [s["template_index"] for s in samples], dtype=np.int32),
        corrected=np.ones(len(samples), dtype=np.bool_),
        perturb_stage=np.array([s["perturb_stage"] for s in samples]),
        perturb_tick=np.array([s["perturb_tick"] for s in samples], dtype=np.int32),
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
        correction_stage=np.array(source),
        correction_tick=np.int32(-1),
        correction_delta=np.zeros(2, dtype=np.float64),
    )


def copy_base_dataset(base_root: Path, out_root: Path):
    if out_root.exists() and any(out_root.rglob("*.npz")):
        raise FileExistsError(
            f"{out_root} 已有 .npz；请换一个 --out，避免覆盖已有实验")
    out_root.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        src = base_root / split
        dst = out_root / split
        dst.mkdir()
        for path in sorted(src.glob("ep_*.npz")):
            shutil.copy2(path, dst / path.name)
    return {
        split: (max(
            [int(p.stem.split("_")[-1]) for p in (out_root / split).glob("ep_*.npz")]
            or [-1]
        ) + 1)
        for split in ("train", "val", "test")
    }


def choose_command(index, rng, blue_probability):
    """优先采蓝色方块，同时保留两种放置区的覆盖。"""
    if rng.random() < blue_probability:
        block = "blue"
    else:
        block = "red"
    zone = "blue" if rng.random() < 0.5 else "red"
    return block, zone


def record_split(name, n_scenes, seed, starts, out_root, model, tok,
                 device, chunk_size, fine_tol, max_ticks, blue_probability):
    rng = np.random.default_rng(seed)
    stats = {"scenes": 0, "samples": 0, "success": 0, "failures": {},
             "phases": {}, "commands": {}}
    t0 = time.time()
    out_dir = out_root / name
    print(f"\n[{name}] 新场景 {n_scenes} 条（模型闭环纠偏）", flush=True)
    for i in range(n_scenes):
        scene = scene_from_sample(sample_scene_v12(rng))
        block, zone = choose_command(i, rng, blue_probability)
        tidx = int(rng.integers(0, 3))
        samples, info, ticks = record_rollout(
            scene, block, zone, tidx, model, tok, device, chunk_size,
            fine_tol, max_ticks)
        path = out_dir / f"ep_{starts[name] + i:04d}.npz"
        save_rollout(path, samples, block, zone, scene, info, tidx)
        stats["scenes"] += 1
        stats["samples"] += len(samples)
        stats["success"] += int(info["success"])
        stats["commands"][f"{block}->{zone}"] = (
            stats["commands"].get(f"{block}->{zone}", 0) + 1)
        if not info["success"]:
            code = info["failure"] or "MAX_TICKS"
            stats["failures"][code] = stats["failures"].get(code, 0) + 1
        # env.phase is terminal at the end; summarize stages from samples.
        for s in samples:
            stats["phases"][s["phase"]] = (
                stats["phases"].get(s["phase"], 0) + 1)
        if (i + 1) % 25 == 0 or i + 1 == n_scenes:
            print(
                f"  {i + 1:>4}/{n_scenes}  模型成功 {stats['success'] / stats['scenes']:.1%} "
                f"样本 {stats['samples']:>7} 失败 {stats['failures']} "
                f"{time.time() - t0:.0f}s", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base-data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=9031)
    ap.add_argument("--fine-tol", type=float, default=V12_FINE_TOL)
    ap.add_argument("--blue-probability", type=float, default=0.67)
    ap.add_argument("--max-ticks", type=int, default=250)
    args = ap.parse_args()
    if not (0.0 < args.fine_tol < 0.11):
        ap.error("fine-tol 必须在 (0, 0.11) 内")
    if not (0.0 <= args.blue_probability <= 1.0):
        ap.error("blue-probability 必须在 [0, 1] 内")

    base_root = ROOT / args.base_data
    out_root = ROOT / args.out
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load_policy(ROOT / args.ckpt, device)
    starts = copy_base_dataset(base_root, out_root)
    print(f"设备：{device}；模型图像：{model.image_size}x{model.image_size}")
    print(f"基础数据：{base_root}")
    print(f"合并输出：{out_root}")

    summary = {
        "source_checkpoint": str(args.ckpt),
        "base_data": str(args.base_data),
        "seed": args.seed,
        "fine_tol": args.fine_tol,
        "image_size": model.image_size,
        "blue_probability": args.blue_probability,
        "splits": {},
    }
    summary["splits"]["train"] = record_split(
        "train", args.train, args.seed, starts, out_root, model, tok, device,
        ck["chunk_size"], args.fine_tol, args.max_ticks, args.blue_probability)
    summary["splits"]["val"] = record_split(
        "val", args.val, args.seed + 10000, starts, out_root, model, tok, device,
        ck["chunk_size"], args.fine_tol, args.max_ticks, args.blue_probability)
    # test 目录只保留基础数据，防止新采集轨迹污染最终闭环评测。
    summary["splits"]["test"] = {
        "scenes_added": 0,
        "policy_rollout": False,
        "note": "copied from base-data unchanged",
    }
    (out_root / "policy_correction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[完成] 已追加模型闭环纠偏数据；test split 保持基础数据不变")
    for name in ("train", "val"):
        s = summary["splits"][name]
        print(f"{name:<5} 新场景 {s['scenes']}  新样本 {s['samples']} "
              f"模型成功 {s['success'] / max(1, s['scenes']):.1%} "
              f"失败 {s['failures']} 命令 {s['commands']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
