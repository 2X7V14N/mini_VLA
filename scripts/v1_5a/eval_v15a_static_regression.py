"""Evaluate a V1.5a temporal model on the V1.3 official static benchmark."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_expert_v13 import COMBOS, instruction_v13, sample_scene_v13
from mini_vla.data.grasp_place_render_v13 import render_v13
from mini_vla.data.grasp_place_v13 import GraspPlaceEnvV13
from mini_vla.data.v13_eval_standards import OFFICIAL_V13
from mini_vla.models.mini_vla_v15a_dynamic import LanguageConditionedMiniVLA3ColorDynamic
from mini_vla.models.text import SimpleCharTokenizer


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3ColorDynamic":
        raise ValueError("checkpoint 不是 V1.5a 动态模型")
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    model = LanguageConditionedMiniVLA3ColorDynamic(
        vocab_size=tokenizer.vocab_size, max_length=int(checkpoint["max_length"]),
        state_dim=int(checkpoint["state_dim"]), action_dim=int(checkpoint["action_dim"]),
        chunk_size=int(checkpoint["chunk_size"]), history=int(checkpoint["history"]),
        red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), tokenizer, int(checkpoint["history"])


@torch.inference_mode()
def run_episode(model, tokenizer, history, device, scene, block, zone, template):
    env = GraspPlaceEnvV13(
        scene, block, zone, close_enter_tol=OFFICIAL_V13.close_enter_tol,
        zone_tol=OFFICIAL_V13.zone_tol, phase_max_ticks=OFFICIAL_V13.phase_max_ticks,
    )
    tokens = tokenizer.batch_encode([instruction_v13(block, zone, template)]).to(device)
    frames, phases = [], set()
    failure_stage = ""
    while not env.done and env.ticks < OFFICIAL_V13.total_max_ticks:
        frame = render_v13(env)
        frames.append(frame)
        recent = frames[-history:]
        if len(recent) < history:
            recent = [recent[0]] * (history - len(recent)) + recent
        image = torch.from_numpy(np.stack(recent)).permute(0, 3, 1, 2).float()
        image = (image / 255.0).unsqueeze(0).to(device)
        state = torch.from_numpy(env.observation()["robot_state"]).unsqueeze(0).to(device)
        phase_before = env.phase
        action = model(image, tokens, state)[0, 0].cpu().numpy()
        env.step(action)
        phases.add(env.phase)
        failure_stage = phase_before
    if not env.done:
        env._fail("MAX_TICKS")
    info = env.info()
    released = not info["holding"] and "OPEN" in phases
    point = np.asarray(info["object"])
    in_target = released and bool(np.all(np.abs(point - env.target_zone) <= OFFICIAL_V13.zone_half))
    return {
        "success": in_target, "environment_success": bool(info["success"]),
        "failure": info["failure"] or "", "failure_stage": failure_stage if not in_target else "",
        "block": block, "zone": zone, "ticks": int(info["ticks"]),
        "reached_target": "CLOSE" in phases,
        "captured": "MOVE_TO_ZONE" in phases or "OPEN" in phases,
        "reached_zone": "OPEN" in phases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/v1.5a.pt")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--scene-seed", type=int, default=20261105)
    parser.add_argument("--out", default="reports/v1.5a_static_regression.json")
    args = parser.parse_args()
    if args.scenes < 1:
        raise ValueError("scenes 必须为正数")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, history = load_model(ROOT / args.checkpoint, device)
    rng, rows = np.random.default_rng(args.scene_seed), []
    for index in range(args.scenes):
        scene = sample_scene_v13(rng)
        template = index % 3
        for block, zone in COMBOS:
            row = run_episode(model, tokenizer, history, device, scene, block, zone, template)
            row["scene_index"] = index
            rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == args.scenes:
            print(f"{index + 1}/{args.scenes}: success={sum(r['success'] for r in rows)}/{len(rows)}", flush=True)
    successes = sum(row["success"] for row in rows)
    failures = Counter(row["failure"] or "POSTHOC_MISS" for row in rows if not row["success"])
    print(f"static official: success={successes}/{len(rows)} ({successes / len(rows):.1%})")
    print(f"failures={dict(failures)}")
    print("pipeline: reached_target={:.1%}; captured={:.1%}; reached_zone={:.1%}".format(
        sum(r["reached_target"] for r in rows) / len(rows),
        sum(r["captured"] for r in rows) / len(rows),
        sum(r["reached_zone"] for r in rows) / len(rows),
    ))
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "checkpoint": args.checkpoint, "standard": OFFICIAL_V13.as_dict(),
        "scene_seed": args.scene_seed, "scenes": args.scenes, "trials": len(rows),
        "successes": successes, "success_rate": successes / len(rows), "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
