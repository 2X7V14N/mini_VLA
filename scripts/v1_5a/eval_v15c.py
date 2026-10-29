"""Closed-loop V1.5c evaluation: target block and zone both move."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_render_v15a import render_dynamic_v15a
from mini_vla.data.grasp_place_dynamic_v15c import DynamicBothEnvV15
from mini_vla.data.grasp_place_v13 import SceneV13
from mini_vla.models.mini_vla_v15c_dynamic import LanguageConditionedMiniVLA3ColorDynamicBoth
from mini_vla.models.text import SimpleCharTokenizer


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3ColorDynamicBoth":
        raise ValueError("checkpoint 不是 V1.5c 双动态模型")
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    model = LanguageConditionedMiniVLA3ColorDynamicBoth(
        vocab_size=tokenizer.vocab_size, max_length=int(checkpoint["max_length"]),
        state_dim=int(checkpoint["state_dim"]), action_dim=int(checkpoint["action_dim"]),
        chunk_size=int(checkpoint["chunk_size"]), history=int(checkpoint["history"]),
        red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval(), tokenizer, int(checkpoint["history"])


def metadata(path):
    with np.load(path, allow_pickle=False) as data:
        colors = ("red", "blue", "yellow")
        return {
            "scene": SceneV13(
                objects={color: data["objects"][i].astype(np.float64) for i, color in enumerate(colors)},
                robot_start=data["robot_start"].astype(np.float64),
                zones={color: data["zones"][i].astype(np.float64) for i, color in enumerate(colors)},
            ),
            "block": str(data["exec_block"]), "zone": str(data["exec_zone"]),
            "block_velocity": data["initial_block_velocity"].astype(np.float64),
            "zone_velocity": data["initial_zone_velocity"].astype(np.float64),
            "instruction": str(data["instruction"][0]),
        }


@torch.inference_mode()
def run_episode(model, tokenizer, history, device, path, max_ticks):
    meta = metadata(path)
    env = DynamicBothEnvV15(
        meta["scene"], meta["block"], meta["zone"], moving_zone=meta["zone"],
        block_velocity=meta["block_velocity"], zone_velocity=meta["zone_velocity"],
    )
    tokens = tokenizer.batch_encode([meta["instruction"]]).to(device)
    frames, seen = [], set()
    failure_stage = ""
    for _ in range(max_ticks):
        if env.done:
            break
        frames.append(render_dynamic_v15a(env))
        recent = frames[-history:]
        if len(recent) < history:
            recent = [recent[0]] * (history - len(recent)) + recent
        image = torch.from_numpy(np.stack(recent)).permute(0, 3, 1, 2).float().unsqueeze(0).to(device) / 255.0
        state = torch.from_numpy(env.observation()["robot_state"]).unsqueeze(0).to(device)
        failure_stage = env.phase
        action = model(image, tokens, state)[0, 0].cpu().numpy()
        env.step(action)
        seen.add(env.phase)
    if not env.done:
        env._fail("MAX_TICKS")
    info = env.info()
    return {
        "success": bool(info["success"]), "failure": info["failure"] or "",
        "failure_stage": failure_stage if not info["success"] else "", "ticks": int(info["ticks"]),
        "block": meta["block"], "zone": meta["zone"],
        "reached_target": "CLOSE" in seen, "captured": "MOVE_TO_ZONE" in seen or "OPEN" in seen,
        "reached_zone": "OPEN" in seen,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/v1.5c.pt")
    parser.add_argument("--data", default="data/v1.5c")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--out", default="reports/v1.5c_results.json")
    args = parser.parse_args()
    paths = sorted((ROOT / args.data / "test").glob("ep_*.npz"))[:args.scenes * 9]
    if len(paths) != args.scenes * 9:
        raise ValueError(f"测试数据不足：需要 {args.scenes * 9} 条，实际 {len(paths)} 条")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, history = load_model(ROOT / args.checkpoint, device)
    rows = []
    for index, path in enumerate(paths, 1):
        rows.append(run_episode(model, tokenizer, history, device, path, args.max_ticks))
        if index % 50 == 0 or index == len(paths):
            print(f"{index}/{len(paths)} success={sum(row['success'] for row in rows)}/{index}", flush=True)
    success = sum(row["success"] for row in rows)
    print(f"v1.5c: success={success}/{len(rows)} ({success / len(rows):.1%})")
    print("failures=" + str(Counter(row["failure"] or "SUCCESS" for row in rows)))
    print("stages=" + str(Counter(row["failure_stage"] for row in rows if not row["success"])))
    print("pipeline: target={:.1%}; captured={:.1%}; zone={:.1%}".format(
        sum(row["reached_target"] for row in rows) / len(rows),
        sum(row["captured"] for row in rows) / len(rows),
        sum(row["reached_zone"] for row in rows) / len(rows),
    ))
    combos = defaultdict(list)
    for row in rows: combos[(row["block"], row["zone"])].append(row)
    print("combo:")
    for key in sorted(combos):
        group = combos[key]; correct = sum(row["success"] for row in group)
        print(f"  {key[0]}->{key[1]}: {correct}/{len(group)} ({correct / len(group):.1%})")
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "data": args.data, "scenes": args.scenes, "trials": len(rows), "successes": success, "success_rate": success / len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results saved: {out}")


if __name__ == "__main__":
    main()
