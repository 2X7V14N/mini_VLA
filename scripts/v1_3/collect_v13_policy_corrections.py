"""Run a fresh V1.3 closed-loop audit and build expert recovery data.

The existing held-out test split is never read by this script. Failure rollouts
are saved in full, while training labels supervise only the command that was
actually active in that physical state.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_expert_v13 import (  # noqa: E402
    COMBOS,
    expert_action_v13,
    instruction_v13,
    sample_scene_v13,
)
from mini_vla.data.grasp_place_render_v13 import render_v13  # noqa: E402
from mini_vla.data.grasp_place_v13 import GraspPlaceEnvV13  # noqa: E402
from mini_vla.data.v13_eval_standards import (  # noqa: E402
    OFFICIAL_V13,
)
from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color  # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402

COLORS = ("red", "blue", "yellow")
CHUNK_SIZE = 4
# Dense windows around the failure boundary are especially important for OPEN
# and DROP_OUTSIDE: the expert must see the last few pre-release states.
RECOVERY_OFFSETS = (1, 2, 4, 8, 16, 32)


def load_policy(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt.get("model_type") != "LanguageConditionedMiniVLA3Color":
        raise ValueError("checkpoint 不是 LanguageConditionedMiniVLA3Color")
    tokenizer = SimpleCharTokenizer(ckpt["vocab"], max_length=int(ckpt["max_length"]))
    model = LanguageConditionedMiniVLA3Color(
        vocab_size=tokenizer.vocab_size,
        max_length=int(ckpt["max_length"]),
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        chunk_size=int(ckpt["chunk_size"]),
        red_token_id=tokenizer.stoi["红"],
        blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model, tokenizer, ckpt


def scene_arrays(scene):
    out = {f"object_{color}": scene.objects[color] for color in COLORS}
    out.update({f"zone_{color}": scene.zones[color] for color in COLORS})
    out["robot_start"] = scene.robot_start
    return out


def snapshot(env):
    return {
        "robot": env.robot.copy(),
        "objects": {color: env.objects[color].copy() for color in COLORS},
        "holding": bool(env.holding),
        "gripper_open": float(env.gripper_open),
        "cmd": int(env.cmd),
        "phase": env.phase,
        "phase_ticks": int(env.phase_ticks),
        "ticks": int(env.ticks),
    }


def restore_env(scene, block, zone, saved, standard=OFFICIAL_V13):
    env = GraspPlaceEnvV13(
        scene,
        block,
        zone,
        close_enter_tol=standard.close_enter_tol,
        zone_tol=standard.zone_tol,
        phase_max_ticks=standard.phase_max_ticks,
    )
    env.robot = np.asarray(saved["robot"], dtype=np.float64).copy()
    env.objects = {
        color: np.asarray(saved["objects"][color], dtype=np.float64).copy()
        for color in COLORS
    }
    env.holding = bool(saved["holding"])
    env.gripper_open = float(saved["gripper_open"])
    env.cmd = int(saved["cmd"])
    env.phase = saved["phase"]
    env.phase_ticks = int(saved["phase_ticks"])
    env.ticks = int(saved["ticks"])
    env.failure = None
    env.done = False
    return env


def policy_action(model, tokenizer, device, env, block, zone, template_index):
    image = render_v13(env)
    prompt = instruction_v13(block, zone, template_index)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()[None].to(device) / 255.0
    token_ids = tokenizer.encode(prompt)[None].to(device)
    state = torch.from_numpy(env.observation()["robot_state"])[None].to(device)
    with torch.no_grad():
        action = model(image_tensor, token_ids, state)[0, 0]
    return action.detach().cpu().numpy().astype(np.float32), image, prompt


def run_policy_episode(model, tokenizer, device, scene, block, zone,
                       template_index, max_ticks, standard=OFFICIAL_V13):
    env = GraspPlaceEnvV13(
        scene,
        block,
        zone,
        close_enter_tol=standard.close_enter_tol,
        zone_tol=standard.zone_tol,
        phase_max_ticks=standard.phase_max_ticks,
    )
    trace = []
    while not env.done and env.ticks < max_ticks:
        saved = snapshot(env)
        action, image, prompt = policy_action(
            model, tokenizer, device, env, block, zone, template_index
        )
        trace.append({
            "image": image,
            "robot_state": env.observation()["robot_state"].copy(),
            "instruction": prompt,
            "action": action.copy(),
            "state": saved,
        })
        env.step(action)
    if not env.done:
        env._fail("MAX_TICKS")
    return env, trace


def expert_chunk(env, block):
    sim = copy.deepcopy(env)
    actions = []
    for _ in range(CHUNK_SIZE):
        if sim.done:
            actions.append(np.zeros(3, dtype=np.float32))
        else:
            action = expert_action_v13(sim, block)
            actions.append(action.copy())
            sim.step(action)
    return np.asarray(actions, dtype=np.float32)


def record_expert_recovery(scene, block, zone, template_index, start_state,
                           max_ticks, standard=OFFICIAL_V13):
    env = restore_env(scene, block, zone, start_state, standard=standard)
    rows = []
    combo_index = COMBOS.index((block, zone))
    while not env.done and len(rows) < max_ticks:
        image = render_v13(env)
        state = env.observation()["robot_state"]
        prompt_by_combo = [instruction_v13(b, z, template_index) for b, z in COMBOS]
        actions = np.zeros((9, CHUNK_SIZE, 3), dtype=np.float32)
        actions[combo_index] = expert_chunk(env, block)
        valid = np.zeros(9, dtype=np.bool_)
        valid[combo_index] = True
        rows.append((image, state.copy(), prompt_by_combo, actions, valid, env.phase))
        env.step(expert_action_v13(env, block))
    if not env.done:
        env._fail("MAX_TICKS")
    return rows, env.info()


def save_failure_trace(path, scene, block, zone, template_index, trace, info):
    np.savez_compressed(
        path,
        **scene_arrays(scene),
        image=np.stack([row["image"] for row in trace]),
        robot_state=np.stack([row["robot_state"] for row in trace]),
        instruction=np.asarray([row["instruction"] for row in trace]),
        model_action=np.stack([row["action"] for row in trace]),
        robot=np.stack([row["state"]["robot"] for row in trace]),
        object_positions=np.stack([
            np.stack([row["state"]["objects"][color] for color in COLORS])
            for row in trace
        ]),
        holding=np.asarray([row["state"]["holding"] for row in trace], dtype=np.bool_),
        gripper_open=np.asarray([row["state"]["gripper_open"] for row in trace], dtype=np.float32),
        gripper_cmd=np.asarray([row["state"]["cmd"] for row in trace], dtype=np.int8),
        phase=np.asarray([row["state"]["phase"] for row in trace]),
        phase_ticks=np.asarray([row["state"]["phase_ticks"] for row in trace], dtype=np.int32),
        ticks=np.asarray([row["state"]["ticks"] for row in trace], dtype=np.int32),
        block=np.asarray(block), zone=np.asarray(zone),
        template_index=np.int32(template_index),
        failure=np.asarray(info["failure"] or ""),
    )


def save_recovery_episode(path, rows, scene, block, zone, template_index, info):
    combo_ids = np.tile(np.arange(9, dtype=np.int32), len(rows))
    np.savez_compressed(
        path,
        image=np.repeat(np.stack([r[0] for r in rows]), 9, axis=0),
        instruction=np.asarray([p for r in rows for p in r[2]]),
        robot_state=np.repeat(np.stack([r[1] for r in rows]), 9, axis=0),
        action_chunk=np.concatenate([r[3] for r in rows], axis=0),
        valid_combo_mask=np.concatenate([r[4] for r in rows]),
        phase=np.repeat(np.asarray([r[5] for r in rows]), 9),
        target_color=np.tile(np.asarray([b for b, _ in COMBOS]), len(rows)),
        zone_color=np.tile(np.asarray([z for _, z in COMBOS]), len(rows)),
        object_position=np.repeat(
            np.stack([scene.objects[b] for b, _ in COMBOS])[None, 0], len(rows), axis=0
        ).reshape(-1, 2),
        zone_position=np.repeat(
            np.stack([scene.zones[z] for _, z in COMBOS])[None, 0], len(rows), axis=0
        ).reshape(-1, 2),
        pair_id=np.repeat(np.arange(len(rows), dtype=np.int32), 9),
        combo_index=combo_ids,
        timestep=np.repeat(np.arange(len(rows), dtype=np.int32), 9),
        template_index=np.full(len(rows) * 9, template_index, dtype=np.int32),
        **scene_arrays(scene),
        exec_block=np.asarray(block), exec_zone=np.asarray(zone),
        success=np.bool_(info["success"]),
        failure=np.asarray(info["failure"] or ""),
        ticks=np.int32(info["ticks"]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/v1.3_final.pt")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8132026)
    parser.add_argument(
        "--max-ticks", type=int,
        help="覆盖评测标准的总 tick 上限；不指定时使用该标准默认值",
    )
    parser.add_argument("--out", default="data/v1.3_policy_corrections_seed8132026")
    args = parser.parse_args()
    standard = OFFICIAL_V13
    max_ticks = standard.total_max_ticks if args.max_ticks is None else args.max_ticks
    if args.scenes < 2:
        parser.error("--scenes 至少为 2，才能分离纠错训练和验证场景")

    output = ROOT / args.out
    if output.exists() and any(output.rglob("*.npz")):
        raise FileExistsError(f"输出目录已包含 npz 文件：{output}")
    raw_dir = output / "failure_traces"
    train_dir = output / "train"
    val_dir = output / "val"
    for directory in (raw_dir, train_dir, val_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, ckpt = load_policy(ROOT / args.ckpt, device)
    rng = np.random.default_rng(args.seed)
    split_at = max(1, int(args.scenes * 0.8))
    if split_at == args.scenes:
        split_at -= 1

    manifest = {
        "checkpoint": args.ckpt,
        "seed": args.seed,
        "scene_source": "fresh generated scenes; held-out official test split not loaded",
        "checkpoint_best_val": ckpt.get("best_val"),
        "evaluation_standard": standard.as_dict(),
        "scenes": args.scenes,
        "split_at_scene": split_at,
        "rows": [],
        "correction_episodes": {"train": 0, "val": 0},
        "expert_recovery_failures": 0,
    }
    success_count = 0
    fail_index = 0
    episode_index = {"train": 0, "val": 0}

    print(f"设备：{device}；checkpoint：{args.ckpt}")
    print(f"新生成场景：{args.scenes}；种子：{args.seed}；闭环：{args.scenes * 9}")
    for scene_index in range(args.scenes):
        scene = sample_scene_v13(rng)
        template_index = scene_index % 3
        for block, zone in COMBOS:
            env, trace = run_policy_episode(
                model, tokenizer, device, scene, block, zone,
                template_index, max_ticks, standard=standard,
            )
            info = env.info()
            success_count += int(info["success"])
            row = {
                "scene_index": scene_index, "block": block, "zone": zone,
                "success": bool(info["success"]), "failure": info["failure"] or "",
                "ticks": int(info["ticks"]), "failure_phase": trace[-1]["state"]["phase"] if trace else "",
            }
            manifest["rows"].append(row)
            if info["success"]:
                continue

            trace_name = f"failure_{fail_index:05d}.npz"
            save_failure_trace(raw_dir / trace_name, scene, block, zone,
                               template_index, trace, info)
            row["trace_file"] = f"failure_traces/{trace_name}"
            fail_index += 1

            split = "train" if scene_index < split_at else "val"
            recovered = False
            for offset in RECOVERY_OFFSETS:
                if not trace:
                    break
                start_index = max(0, len(trace) - 1 - offset)
                start_state = trace[start_index]["state"]
                corrections, correction_info = record_expert_recovery(
                    scene, block, zone, template_index, start_state, max_ticks,
                    standard=standard,
                )
                if correction_info["success"]:
                    episode_name = f"ep_{episode_index[split]:05d}.npz"
                    save_recovery_episode(
                        (train_dir if split == "train" else val_dir) / episode_name,
                        corrections, scene, block, zone, template_index, correction_info,
                    )
                    row["recovery_file"] = f"{split}/{episode_name}"
                    row["recovery_start_trace_tick"] = int(start_state["ticks"])
                    row["recovery_ticks"] = len(corrections)
                    episode_index[split] += 1
                    manifest["correction_episodes"][split] += 1
                    recovered = True
                    break
            if not recovered:
                row["expert_recovery"] = "failed from tested pre-failure states; excluded from training"
                manifest["expert_recovery_failures"] += 1

        if (scene_index + 1) % 10 == 0 or scene_index + 1 == args.scenes:
            total = len(manifest["rows"])
            rate = success_count / total
            print(
                f"{scene_index + 1}/{args.scenes} 场景；"
                f"模型成功率 {rate:.1%}；失败轨迹 {fail_index}；"
                f"专家纠错 train/val={episode_index['train']}/{episode_index['val']}",
                flush=True,
            )

    manifest["model_success_rate"] = success_count / len(manifest["rows"])
    manifest["failure_count"] = fail_index
    manifest["manifest"] = "manifest.json"
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n新场景成功率：{success_count}/{len(manifest['rows'])} = {manifest['model_success_rate']:.1%}")
    print(f"失败轨迹：{raw_dir}")
    print(f"可用专家纠错：train={episode_index['train']} val={episode_index['val']}")
    print(f"清单：{output / 'manifest.json'}")
    return 0 if episode_index["train"] and episode_index["val"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
