"""Closed-loop evaluation for a trained V1.5a dynamic-zone checkpoint."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_render_v15a import render_dynamic_v15a  # noqa: E402
from mini_vla.data.grasp_place_dynamic_v15 import DynamicZoneEnvV15  # noqa: E402
from mini_vla.data.grasp_place_v13 import SceneV13  # noqa: E402
from mini_vla.models.mini_vla_v15a_dynamic import (  # noqa: E402
    LanguageConditionedMiniVLA3ColorDynamic,
)
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402


def episode_meta(path):
    with np.load(path, allow_pickle=False) as z:
        colors = ("red", "blue", "yellow")
        return {
            "scene": SceneV13(
                objects={c: z["objects"][i].astype(np.float64) for i, c in enumerate(colors)},
                robot_start=z["robot_start"].astype(np.float64),
                zones={c: z["zones"][i].astype(np.float64) for i, c in enumerate(colors)},
            ),
            "block": str(z["exec_block"]),
            "zone": str(z["exec_zone"]),
            "velocity": z["initial_velocity"].astype(np.float64),
            "instruction": str(z["instruction"][0]),
        }


def run_episode(model, tokenizer, path, device, history, max_ticks=600):
    meta = episode_meta(path)
    env = DynamicZoneEnvV15(
        meta["scene"], meta["block"], meta["zone"],
        moving_zone=meta["zone"], zone_velocity=meta["velocity"],
    )
    tokens = tokenizer.batch_encode([meta["instruction"]]).to(device)
    frames = []
    phase_ticks = Counter()
    seen_phases = set()
    for _ in range(max_ticks):
        frames.append(render_dynamic_v15a(env))
        recent = frames[-history:]
        if len(recent) < history:
            recent = [recent[0]] * (history - len(recent)) + recent
        image = torch.from_numpy(np.stack(recent)).permute(0, 3, 1, 2).float()
        image = (image / 255.0).unsqueeze(0).to(device)
        state = torch.from_numpy(env.observation()["robot_state"]).unsqueeze(0).to(device)
        with torch.inference_mode():
            action = model(image, tokens, state)[0, 0].cpu().numpy()
        phase_before = env.phase
        env.step(action)
        phase_ticks[phase_before] += 1
        seen_phases.add(env.phase)
        if env.done:
            break
    if not env.done:
        env._fail("MAX_TICKS")
    info = env.info()
    return {
        "success": bool(info["success"]),
        "failure": info["failure"] or "",
        "ticks": int(info["ticks"]),
        "block": meta["block"],
        "zone": meta["zone"],
        "reached_target": "CLOSE" in seen_phases,
        "captured": "MOVE_TO_ZONE" in seen_phases or "OPEN" in seen_phases,
        "reached_zone": "OPEN" in seen_phases,
        "failure_stage": phase_before if info["failure"] else "",
        "phase_ticks": dict(phase_ticks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/v1.5a_round1")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--out", default="reports/v1.5a_dynamic_results.json")
    args = parser.parse_args()
    checkpoint_path = ROOT / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 V1.5a checkpoint：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3ColorDynamic":
        raise ValueError("checkpoint 不是 V1.5a 动态模型")
    paths = sorted((ROOT / args.data / "test").glob("ep_*.npz"))[: args.scenes * 9]
    if not paths:
        raise FileNotFoundError(f"测试集为空：{ROOT / args.data / 'test'}")
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    history = int(checkpoint.get("history", 4))
    model = LanguageConditionedMiniVLA3ColorDynamic(
        vocab_size=tokenizer.vocab_size, max_length=int(checkpoint["max_length"]),
        state_dim=int(checkpoint["state_dim"]), action_dim=int(checkpoint["action_dim"]),
        chunk_size=int(checkpoint["chunk_size"]), history=history,
        red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    outcomes = []
    for index, episode_path in enumerate(paths, 1):
        outcomes.append(
            run_episode(model, tokenizer, episode_path, device, history, args.max_ticks)
        )
        if index % 50 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} success="
                  f"{sum(item['success'] for item in outcomes)}/{index}", flush=True)
    failures = Counter(item["failure"] or "SUCCESS" for item in outcomes)
    successes = sum(item["success"] for item in outcomes)
    print(f"candidate: success={successes}/{len(outcomes)} "
          f"({successes / len(outcomes):.1%}) failures={dict(failures)}")
    print("stages: " + str(Counter(
        item["failure_stage"] for item in outcomes if not item["success"]
    )))
    print("pipeline: reached_target={:.1%}; captured={:.1%}; reached_zone={:.1%}".format(
        sum(item["reached_target"] for item in outcomes) / len(outcomes),
        sum(item["captured"] for item in outcomes) / len(outcomes),
        sum(item["reached_zone"] for item in outcomes) / len(outcomes),
    ))
    by_combo = defaultdict(list)
    for item in outcomes:
        by_combo[(item["block"], item["zone"])].append(item)
    print("combo:")
    for key in sorted(by_combo):
        rows = by_combo[key]
        count = sum(row["success"] for row in rows)
        print(f"  {key[0]}->{key[1]}: {count}/{len(rows)} ({count / len(rows):.1%})")
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "data": args.data, "scenes": args.scenes,
        "trials": len(outcomes), "successes": successes,
        "success_rate": successes / len(outcomes), "max_ticks": args.max_ticks,
        "rows": outcomes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
