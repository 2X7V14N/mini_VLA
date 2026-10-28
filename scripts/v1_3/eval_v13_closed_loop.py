"""V1.3 三色模型闭环评估。

示例：
    python scripts/v1_3/eval_v13_closed_loop.py \
      --ckpt checkpoints/v1.3_final.pt \
      --data data/v1.3_three_color_1000_200_200 \
      --scenes 200 --ablation-scenes 50

每个测试场景运行 9 种方块色/区域色指令，并只执行模型输出动作块的
第一步（receding horizon）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_expert_v13 import (  # noqa: E402
    COMBOS,
    instruction_v13,
    sample_scene_v13,
)
from mini_vla.data.grasp_place_render_v13 import render_v13  # noqa: E402
from mini_vla.data.grasp_place_v13 import GraspPlaceEnvV13, SceneV13  # noqa: E402
from mini_vla.data.v13_eval_standards import (  # noqa: E402
    OFFICIAL_V13,
)
from mini_vla.models.mini_vla_v13 import LanguageConditionedMiniVLA3Color  # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("model_type") != "LanguageConditionedMiniVLA3Color":
        raise ValueError("checkpoint 不是 LanguageConditionedMiniVLA3Color")

    tokenizer = SimpleCharTokenizer(
        ckpt["vocab"], max_length=int(ckpt["max_length"])
    )
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


def load_scenes(test_dir, limit):
    scenes = []
    paths = sorted(Path(test_dir).glob("ep_*.npz"))[:limit]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            scene = SceneV13(
                objects={
                    color: data[f"object_{color}"].astype(np.float64)
                    for color in ("red", "blue", "yellow")
                },
                robot_start=data["robot_start"].astype(np.float64),
                zones={
                    color: data[f"zone_{color}"].astype(np.float64)
                    for color in ("red", "blue", "yellow")
                },
            )
            # 每个 episode 使用同一个语言模板；三个模板都在训练集中出现。
            template_index = int(np.asarray(data["template_index"]).reshape(-1)[0])
        scenes.append((scene, template_index))
    return scenes


@torch.no_grad()
def model_action(model, tokenizer, device, env, block, zone, template_index,
                 zero_image=False, override=None):
    image = render_v13(env)
    if zero_image:
        image = np.zeros_like(image)
    command_block, command_zone = override or (block, zone)
    instruction = instruction_v13(command_block, command_zone, template_index)
    image_tensor = (
        torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device)
        / 255.0
    )
    token_ids = tokenizer.encode(instruction).unsqueeze(0).to(device)
    state = torch.from_numpy(env.observation()["robot_state"])
    # 兼容旧版三维 checkpoint：新环境提供四维状态时，旧模型只接收前三维。
    expected_state_dim = model.state_encoder.mlp[0].in_features
    if expected_state_dim == 3:
        state = state[:3]
    elif expected_state_dim != state.numel():
        raise ValueError(
            f"checkpoint 状态维度 {expected_state_dim} 与环境状态维度 "
            f"{state.numel()} 不匹配"
        )
    state = state.unsqueeze(0).to(device)
    return model(image_tensor, token_ids, state)[0, 0].detach().cpu().numpy()


def random_action(*args, rng, **kwargs):
    return rng.uniform(-1.0, 1.0, 3).astype(np.float32)


def placement_success(point, target_zone, released_in_open, zone_half):
    """Score a released object without changing the rollout environment."""
    if not released_in_open:
        return False
    point = np.asarray(point, dtype=np.float64)
    target_zone = np.asarray(target_zone, dtype=np.float64)
    return bool(np.all(np.abs(point - target_zone) <= zone_half))


def run_episode(scene, block, zone, action_fn, template_index, max_ticks=1500,
                standard=OFFICIAL_V13, **kwargs):
    env = GraspPlaceEnvV13(
        scene,
        block,
        zone,
        close_enter_tol=standard.close_enter_tol,
        zone_tol=standard.zone_tol,
        phase_max_ticks=standard.phase_max_ticks,
    )
    phases = set()
    phase_ticks = {
        "MOVE_TO_OBJECT": 0,
        "CLOSE": 0,
        "MOVE_TO_ZONE": 0,
        "OPEN": 0,
    }
    last_phase = env.phase
    while not env.done and env.ticks < max_ticks:
        if action_fn is random_action:
            action = action_fn(None, None, None, None, **kwargs)
        else:
            action = action_fn(
                env, block, zone, template_index, **kwargs
            )
        prev = env.phase
        env.step(np.asarray(action, dtype=np.float32))
        phase_ticks[prev] += 1
        if env.phase != prev:
            phases.add(env.phase)
        last_phase = prev

    if not env.done:
        env._fail("MAX_TICKS")
    info = env.info()
    released_in_open = not info["holding"] and "OPEN" in phases
    final_object = np.asarray(info["object"], dtype=np.float64)
    in_target = placement_success(
        final_object,
        env.target_zone,
        released_in_open,
        standard.zone_half,
    )
    return {
        "block": block,
        "zone": zone,
        "success": in_target,
        "environment_success": bool(info["success"]),
        "environment_failure": info["failure"] or "",
        "failure": info["failure"] or "",
        "ticks": int(info["ticks"]),
        "reached_target": "CLOSE" in phases,
        "captured": "MOVE_TO_ZONE" in phases or "OPEN" in phases,
        "reached_zone": "OPEN" in phases,
        "in_target": in_target,
        "released_in_open": released_in_open,
        "final_object": final_object.tolist(),
        "failure_stage": last_phase if info["failure"] else "",
        "phase_ticks": phase_ticks,
    }


def summarize(title, rows):
    n = len(rows)
    print(f"\n{title}（{n} 次闭环）")
    if not n:
        return
    reached = sum(row["reached_target"] for row in rows) / n
    captured = sum(row["captured"] for row in rows) / n
    reached_zone = sum(row["reached_zone"] for row in rows) / n
    in_target = sum(row["in_target"] for row in rows) / n
    success = sum(row["success"] for row in rows) / n
    zone_rows = [row for row in rows if row["reached_zone"]]
    zone_correct = sum(row["in_target"] for row in zone_rows)
    zone_choice = zone_correct / len(zone_rows) if zone_rows else float("nan")
    if zone_rows:
        zone_choice_text = f"{zone_choice:7.1%} ({zone_correct}/{len(zone_rows)})"
        if len(zone_rows) < 20:
            zone_choice_text += "；样本太少"
    else:
        zone_choice_text = "    nan% (0/0)"
    print(f"  到达指定方块      {reached:7.1%}")
    print(f"  捕获目标          {captured:7.1%}")
    print(f"  到达放置区        {reached_zone:7.1%}")
    print(f"  落在指定放置区    {in_target:7.1%}")
    print(f"  完整成功          {success:7.1%}")
    print(f"  方块选择正确      {reached:7.1%}")
    print(f"  区域选择正确      {zone_choice_text}")
    failures = {}
    stages = {}
    for row in rows:
        if not row["success"]:
            failures[row["failure"]] = failures.get(row["failure"], 0) + 1
            stages[row["failure_stage"]] = stages.get(row["failure_stage"], 0) + 1
    print(f"  失败构成          {failures or '无失败'}")
    print(f"  失败阶段          {stages or '无失败'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/v1.3_final.pt")
    parser.add_argument("--data", default="data/v1.3_three_color_1000_200_200")
    parser.add_argument("--scenes", type=int, default=200)
    parser.add_argument("--ablation-scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--scene-seed", type=int,
        help="generate an independent deterministic scene set instead of reading --data/test",
    )
    parser.add_argument(
        "--max-ticks", type=int,
        help="覆盖评测标准的总 tick 上限；不指定时使用该标准的默认值",
    )
    parser.add_argument("--failure-log", default="reports/v1.3_three_color_failures.json")
    parser.add_argument("--results-log", help="save every scene/instruction outcome, including successes")
    args = parser.parse_args()
    standard = OFFICIAL_V13
    max_ticks = standard.total_max_ticks if args.max_ticks is None else args.max_ticks
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, ckpt = load_model(ROOT / args.ckpt, device)
    if args.scene_seed is None:
        scenes = load_scenes(ROOT / args.data / "test", args.scenes)
        scene_source = f"{args.data}/test"
    else:
        scene_rng = np.random.default_rng(args.scene_seed)
        scenes = [
            (sample_scene_v13(scene_rng), index % 3)
            for index in range(args.scenes)
        ]
        scene_source = f"generated(seed={args.scene_seed})"
    if not scenes:
        raise RuntimeError("test 集没有找到 ep_*.npz")

    print(f"设备：{device}；checkpoint：{args.ckpt}")
    print(f"best val：{ckpt.get('best_val', float('nan')):.5f}")
    print(
        f"评测标准：{standard.name}；进入闭合容差={standard.close_enter_tol:.3f}；"
        f"进入放置区容差={standard.zone_tol:.3f}；总上限={max_ticks} tick"
    )
    print(
        f"最终落点成功范围：边长 {2 * standard.zone_half:.3f} 的正方形；"
        "只用于结束后的评分，不改变场景图像和动作过程"
    )
    print(f"测试场景：{len(scenes)}；来源：{scene_source}；9 种组合；闭环次数：{9 * len(scenes)}")

    def policy(env, block, zone, tidx, **kwargs):
        return model_action(model, tokenizer, device, env, block, zone, tidx, **kwargs)

    rows = []
    for index, (scene, tidx) in enumerate(scenes):
        for block, zone in COMBOS:
            row = run_episode(
                scene, block, zone, policy, tidx, max_ticks,
                standard=standard,
            )
            row["scene_index"] = index
            row["trial_id"] = f"{index:04d}:{block}:{zone}"
            row["instruction"] = instruction_v13(block, zone, tidx)
            rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(scenes):
            rate = sum(row["success"] for row in rows) / len(rows)
            print(f"  {index + 1}/{len(scenes)}；累计成功率 {rate:.1%}", flush=True)

    summarize("模型", rows)

    environment_successes = sum(row["environment_success"] for row in rows)
    official_successes = sum(row["success"] for row in rows)
    reclassified = sum(
        row["success"] and not row["environment_success"] for row in rows
    )
    print(
        f"环境原始成功率：{environment_successes}/{len(rows)} "
        f"({environment_successes / len(rows):.1%})；"
        f"按正式落点范围新增计为成功：{reclassified}"
    )

    print("\n按指令组合：")
    for block, zone in COMBOS:
        subset = [row for row in rows if row["block"] == block and row["zone"] == zone]
        rate = sum(row["success"] for row in subset) / len(subset)
        line = (
            f"  {block:6} -> {zone:6} {rate:7.1%} "
            f"({sum(row['success'] for row in subset)}/{len(subset)})"
        )
        print(line)

    rng = np.random.default_rng(args.seed)
    random_rows = []
    for scene, tidx in scenes:
        for block, zone in COMBOS:
            random_rows.append(run_episode(
                scene, block, zone, random_action, tidx, max_ticks,
                standard=standard, rng=rng,
            ))
    summarize("随机策略", random_rows)

    ablation_scenes = scenes[:max(0, args.ablation_scenes)]
    if ablation_scenes:
        print(f"\n闭环消融：前 {len(ablation_scenes)} 个场景")
        for title, kwargs in (
            ("图像置零", {"zero_image": True}),
            ("方块指令打乱", {"override_role": "block"}),
            ("区域指令打乱", {"override_role": "zone"}),
        ):
            ab_rows = []
            for scene, tidx in ablation_scenes:
                for block, zone in COMBOS:
                    override = None
                    if kwargs.get("override_role") == "block":
                        wrong = next(color for color in ("red", "blue", "yellow") if color != block)
                        override = (wrong, zone)
                    elif kwargs.get("override_role") == "zone":
                        wrong = next(color for color in ("red", "blue", "yellow") if color != zone)
                        override = (block, wrong)
                    ab_rows.append(run_episode(
                        scene, block, zone, policy, tidx, max_ticks,
                        standard=standard,
                        zero_image=kwargs.get("zero_image", False),
                        override=override,
                    ))
            rate = sum(row["success"] for row in ab_rows) / len(ab_rows)
            print(f"  {title:<12} 成功率 {rate:.1%}")

    if args.failure_log:
        path = ROOT / args.failure_log
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "checkpoint": args.ckpt,
            "evaluation_standard": standard.as_dict(),
            "official_placement_zone_side_length": 2 * standard.zone_half,
            "scene_source": scene_source,
            "rows": [
            row for row in rows if not row["success"]
        ]}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n失败记录已保存：{path}")

    if args.results_log:
        path = ROOT / args.results_log
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "checkpoint": args.ckpt,
            "best_val": ckpt.get("best_val", float("nan")),
            "evaluation_standard": standard.as_dict(),
            "official_placement_zone_side_length": 2 * standard.zone_half,
            "environment_successes": sum(row["environment_success"] for row in rows),
            "environment_success_rate": (
                sum(row["environment_success"] for row in rows) / len(rows)
            ),
            "scene_source": scene_source,
            "scene_seed": args.scene_seed,
            "scenes": len(scenes),
            "trials": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_rate": sum(row["success"] for row in rows) / len(rows),
            "rows": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"逐条结果已保存：{path}")

    success = sum(row["success"] for row in rows) / len(rows)
    return 0 if success >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
