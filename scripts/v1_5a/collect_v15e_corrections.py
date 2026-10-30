"""Collect DAgger-style placement corrections from a V1.5 dynamic rollout.

Only train scenes are replayed. A correction starts from the policy's own
carried-object state immediately before a placement failure, then follows the
simulator oracle.  Thus the action target is defined on states the policy
actually visits, not only on ideal expert trajectories.
"""

import argparse
import copy
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_render_v15a import render_dynamic_v15a
from mini_vla.data.grasp_place_dynamic_expert_v15c import expert_action_v15c
from mini_vla.data.grasp_place_dynamic_v15c import DynamicBothEnvV15
from mini_vla.data.grasp_place_v13 import SceneV13
from mini_vla.models.mini_vla_v15c_dynamic import LanguageConditionedMiniVLA3ColorDynamicBoth
from mini_vla.models.text import SimpleCharTokenizer

COLORS = ("red", "blue", "yellow")


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3ColorDynamicBoth":
        raise ValueError("纠偏采集必须使用 V1.5 双动态 checkpoint")
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    model = LanguageConditionedMiniVLA3ColorDynamicBoth(
        vocab_size=tokenizer.vocab_size, max_length=int(checkpoint["max_length"]),
        state_dim=int(checkpoint["state_dim"]), action_dim=int(checkpoint["action_dim"]),
        chunk_size=int(checkpoint["chunk_size"]), history=int(checkpoint["history"]),
        red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"], yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), tokenizer, int(checkpoint["history"])


def metadata(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "scene": SceneV13(
                objects={color: data["objects"][i].astype(np.float64) for i, color in enumerate(COLORS)},
                robot_start=data["robot_start"].astype(np.float64),
                zones={color: data["zones"][i].astype(np.float64) for i, color in enumerate(COLORS)},
            ),
            "block": str(data["exec_block"]), "zone": str(data["exec_zone"]),
            "block_velocity": data["initial_block_velocity"].astype(np.float64),
            "zone_velocity": data["initial_zone_velocity"].astype(np.float64),
            "instruction": str(data["instruction"][0]), "scene_id": str(data["scene_id"]),
        }


def make_env(meta):
    return DynamicBothEnvV15(meta["scene"], meta["block"], meta["zone"], moving_zone=meta["zone"],
                             block_velocity=meta["block_velocity"], zone_velocity=meta["zone_velocity"])


def oracle_chunk(env, block, chunk_size=4):
    simulation, actions = copy.deepcopy(env), []
    for _ in range(chunk_size):
        action = np.zeros(3, dtype=np.float32) if simulation.done else expert_action_v15c(simulation, block)
        actions.append(action)
        if not simulation.done:
            simulation.step(action)
    return np.asarray(actions, dtype=np.float32)


def record_oracle_segment(start_env, meta, max_ticks):
    """Return an oracle rescue segment only if it finishes successfully."""
    env, rows = copy.deepcopy(start_env), []
    while not env.done and len(rows) < max_ticks:
        rows.append({
            "image": render_dynamic_v15a(env), "instruction": meta["instruction"],
            "robot_state": env.observation()["robot_state"].copy(),
            "action_chunk": oracle_chunk(env, meta["block"]),
            "object_positions": np.stack([env.objects[color].copy() for color in COLORS]),
            "object_velocities": np.stack([env.block_velocity[color].copy() for color in COLORS]),
            "zone_positions": np.stack([env.zones[color].copy() for color in COLORS]),
            "zone_velocities": np.stack([env.zone_velocities[color].copy() for color in COLORS]),
            "phase": env.phase, "holding": bool(env.holding), "gripper_open": float(env.gripper_open),
        })
        env.step(expert_action_v15c(env, meta["block"]))
    return rows, env.info()


@torch.inference_mode()
def rollout_policy(model, tokenizer, history, device, meta, max_ticks):
    env, frames = make_env(meta), []
    tokens = tokenizer.batch_encode([meta["instruction"]]).to(device)
    first_open, last_carried = None, None
    for _ in range(max_ticks):
        if env.done:
            break
        if env.holding and env.phase in ("MOVE_TO_ZONE", "OPEN"):
            last_carried = copy.deepcopy(env)
            if env.phase == "OPEN" and first_open is None:
                first_open = copy.deepcopy(env)
        frames.append(render_dynamic_v15a(env))
        recent = frames[-history:]
        if len(recent) < history:
            recent = [recent[0]] * (history - len(recent)) + recent
        image = torch.from_numpy(np.stack(recent)).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0
        state = torch.from_numpy(env.observation()["robot_state"]).unsqueeze(0).to(device)
        env.step(model(image, tokens, state)[0, 0].cpu().numpy())
    if not env.done:
        env._fail("MAX_TICKS")
    info = env.info()
    # Placement errors use the first OPEN state. An early slip has no OPEN
    # state, so use its last still-carried state instead.
    start = first_open if info["failure"] in ("DROP_OUTSIDE", "WRONG_ZONE") else last_carried
    return info, start


def save_segment(path, rows, meta, policy_info, oracle_info, correction_id):
    np.savez_compressed(
        path,
        image=np.stack([row["image"] for row in rows]),
        instruction=np.asarray([row["instruction"] for row in rows]),
        robot_state=np.stack([row["robot_state"] for row in rows]),
        action_chunk=np.stack([row["action_chunk"] for row in rows]),
        object_positions=np.stack([row["object_positions"] for row in rows]),
        object_velocities=np.stack([row["object_velocities"] for row in rows]),
        zone_positions=np.stack([row["zone_positions"] for row in rows]),
        zone_velocities=np.stack([row["zone_velocities"] for row in rows]),
        phase=np.asarray([row["phase"] for row in rows]),
        holding=np.asarray([row["holding"] for row in rows], dtype=np.bool_),
        gripper_open=np.asarray([row["gripper_open"] for row in rows], dtype=np.float32),
        exec_block=np.asarray(meta["block"]), exec_zone=np.asarray(meta["zone"]),
        scene_id=np.asarray(f"correction_{correction_id:05d}_{meta['scene_id']}"),
        source_failure=np.asarray(policy_info["failure"] or ""),
        # This file is a successful oracle rescue segment even though the
        # policy state that triggered it was a failure.
        success=np.bool_(oracle_info["success"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1_5a/collect_e.toml")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = tomllib.loads((ROOT / args.config).read_text(encoding="utf-8"))["collection"]
    checkpoint, source = ROOT / cfg["source_checkpoint"], ROOT / cfg["source_data"] / "train"
    output = ROOT / cfg["output_data"]
    if not checkpoint.is_file() or not source.is_dir():
        raise FileNotFoundError("找不到动态 checkpoint 或训练数据")
    paths = sorted(source.glob("ep_*.npz"))
    requested = int(cfg["scenes"]) * 9
    if requested > len(paths):
        raise ValueError(f"请求 {requested} 条 train episode，但仅有 {len(paths)} 条")
    rng = np.random.default_rng(int(cfg["seed"]))
    scene_paths = [paths[index:index + 9] for index in range(0, len(paths), 9)]
    selected = [item for group in rng.permutation(len(scene_paths))[:int(cfg["scenes"])] for item in scene_paths[group]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, history = load_model(checkpoint, device)
    if args.check_only:
        print(f"[check-only] device={device}; checkpoint={checkpoint}; source={source}; selected={len(selected)}; output={output}")
        return 0
    if output.exists() and any(output.rglob("*.npz")):
        raise FileExistsError(f"为避免覆盖，纠偏数据目录已有 episode：{output}")
    train_out = output / "train"; train_out.mkdir(parents=True, exist_ok=False)
    allowed = set(cfg["failure_types"])
    stats = {"rollouts": 0, "policy_failures": {}, "eligible": 0, "oracle_success": 0, "corrections": 0, "frames": 0}
    for index, path in enumerate(selected, 1):
        meta = metadata(path)
        policy_info, start = rollout_policy(model, tokenizer, history, device, meta, int(cfg["rollout_max_ticks"]))
        stats["rollouts"] += 1
        failure = policy_info["failure"] or "SUCCESS"
        stats["policy_failures"][failure] = stats["policy_failures"].get(failure, 0) + 1
        if failure in allowed and start is not None:
            stats["eligible"] += 1
            rows, oracle_info = record_oracle_segment(start, meta, int(cfg["oracle_max_ticks"]))
            if oracle_info["success"]:
                stats["oracle_success"] += 1
                save_segment(
                    train_out / f"ep_{stats['corrections']:05d}.npz",
                    rows, meta, policy_info, oracle_info, stats["corrections"],
                )
                stats["corrections"] += 1; stats["frames"] += len(rows)
        if index % 50 == 0 or index == len(selected):
            print(f"{index}/{len(selected)} rollouts; corrections={stats['corrections']}; oracle={stats['oracle_success']}/{stats['eligible']}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({"config": cfg, **stats}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved corrections={stats['corrections']}; frames={stats['frames']}; data={output}")
    if stats["corrections"] == 0:
        raise RuntimeError("未收集到可由专家恢复的纠偏轨迹")


if __name__ == "__main__":
    main()
