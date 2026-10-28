"""V1.2 闭环评测：多放置区 + 放置区随机化。

    python scripts/v1_2/eval_v12_closed_loop.py
    python scripts/v1_2/eval_v12_closed_loop.py --scenes 100
    python scripts/v1_2/eval_v12_closed_loop.py --ablation-scenes 50

test 集每个场景跑 4 种指令（方块色 × 区域色）。
模型每 tick 输出 action_chunk，只执行第 1 步（receding horizon）。

分层报告：
    到达方块 / 捕获 / 到达区域 / 落在指定区域 / 完整成功
    方块选择正确率 / 区域选择正确率 / 失败构成
    随机策略基线
可选闭环消融：图像置零、方块指令打乱、区域指令打乱。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place import (                            # noqa: E402
    CLOSE_ENTER_TOL, GraspPlaceEnv, Scene,
)
from mini_vla.data.grasp_place_render import render               # noqa: E402
from mini_vla.data.grasp_place_v12 import instruction_v12         # noqa: E402
from mini_vla.models.mini_vla import (                            # noqa: E402
    LanguageConditionedMiniVLA, MiniVLA,
)
from mini_vla.models.text import SimpleCharTokenizer              # noqa: E402

COMBS = (("red", "red"), ("blue", "red"), ("red", "blue"), ("blue", "blue"))
FAIL_CODES = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
              "ZONE_NOT_REACHED", "DROP_OUTSIDE", "WRONG_ZONE", "MAX_TICKS")


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    tok = SimpleCharTokenizer(ck["vocab"], max_length=ck["max_length"])
    kwargs = dict(vocab_size=tok.vocab_size, max_length=ck["max_length"],
                  state_dim=ck["state_dim"], action_dim=ck["action_dim"],
                  chunk_size=ck["chunk_size"])
    if ck.get("model_type") == "LanguageConditionedMiniVLA":
        model = LanguageConditionedMiniVLA(
            red_token_id=ck["red_token_id"],
            blue_token_id=ck["blue_token_id"], **kwargs)
    else:
        model = MiniVLA(**kwargs)
    model.load_state_dict(ck["model_state"])
    model.image_size = int(ck.get("image_size", 64))
    model.to(device).eval()
    return model, tok, ck


def load_scenes(split_dir, limit):
    scenes = []
    for f in sorted(Path(split_dir).glob("ep_*.npz"))[:limit]:
        z = np.load(f, allow_pickle=False)
        scenes.append((
            Scene(red=z["red"].astype(np.float64),
                  blue=z["blue"].astype(np.float64),
                  robot_start=z["robot_start"].astype(np.float64),
                  zones={"red": z["zones_red"].astype(np.float64),
                         "blue": z["zones_blue"].astype(np.float64)}),
            int(z["template_index"][0]),
        ))
    return scenes


def make_model_policy(model, tok, device):
    @torch.no_grad()
    def act(env, block, zone, tidx, zero_image=False, override=None):
        img = render(env, image_size=model.image_size)
        if zero_image:
            img = np.zeros_like(img)
        b, z = override if override is not None else (block, zone)
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        x = x.to(device) / 255.0
        t = tok.encode(instruction_v12(b, z, tidx)).unsqueeze(0).to(device)
        st = torch.from_numpy(env.observation()["robot_state"])
        st = st.unsqueeze(0).to(device)
        return model(x, t, st)[0, 0].cpu().numpy()
    return act


def make_random_policy(rng):
    def act(env, block, zone, tidx, **kwargs):
        return rng.uniform(-1.0, 1.0, 3).astype(np.float32)
    return act


def run_episode(scene, tidx, block, zone, act_fn,
                max_ticks=1500, **act_kwargs):
    env = GraspPlaceEnv(scene, block, zone_color=zone,
                        close_enter_tol=CLOSE_ENTER_TOL)
    phases = set()
    trace = []
    while not env.done and env.ticks < max_ticks:
        prev = env.phase
        action = np.asarray(
            act_fn(env, block, zone, tidx, **act_kwargs), dtype=np.float32)
        before_robot = env.robot.copy()
        before_object = env.target_object.copy()
        before_zone = env.target_zone.copy()
        env.step(action)
        trace.append({
            "tick": int(env.ticks - 1),
            "phase_before": str(prev),
            "phase_after": str(env.phase),
            "robot_before": before_robot.tolist(),
            "robot_after": env.robot.tolist(),
            "object_after": env.target_object.tolist(),
            "zone": before_zone.tolist(),
            "action": action.tolist(),
            "holding": bool(env.holding),
            "gripper_open": float(env.gripper_open),
        })
        if env.phase != prev:
            phases.add(env.phase)

    info = env.info()
    obj = info["object"]
    return {
        "block": block, "zone": zone, "tidx": tidx,
        "success": bool(info["success"]),
        "failure": info["failure"],
        "ticks": info["ticks"],
        # 环境只有在指令指定的方块附近才会进入 CLOSE
        "reached_target": "CLOSE" in phases,
        "captured": ("MOVE_TO_ZONE" in phases) or ("OPEN" in phases),
        "reached_zone": "OPEN" in phases,
        "in_target": env.in_target_zone(obj),
        "in_any": env.in_any_zone(obj),
        "failure_stage": (trace[-1]["phase_before"]
                           if info["failure"] and trace else ""),
        "trace": trace[-10:],
    }


def summarize(tag, rows):
    n = len(rows)
    if n == 0:
        print(f"\n{tag}：没有数据")
        return {}
    s = {
        "n": n,
        "reached_target": sum(r["reached_target"] for r in rows) / n,
        "captured": sum(r["captured"] for r in rows) / n,
        "reached_zone": sum(r["reached_zone"] for r in rows) / n,
        "in_target": sum(r["in_target"] for r in rows) / n,
        "in_any": sum(r["in_any"] for r in rows) / n,
        "success": sum(r["success"] for r in rows) / n,
    }
    zone_rows = [r for r in rows if r["reached_zone"]]
    s["zone_choice"] = (sum(r["in_target"] for r in zone_rows) / len(zone_rows)
                        if zone_rows else float("nan"))

    print(f"\n{tag}　（{n} 次闭环）")
    print(f"  到达指定方块      {s['reached_target']:>7.1%}")
    print(f"  捕获目标          {s['captured']:>7.1%}")
    print(f"  到达放置区        {s['reached_zone']:>7.1%}")
    print(f"  落在指定放置区    {s['in_target']:>7.1%}")
    print(f"  完整成功          {s['success']:>7.1%}")
    print(f"  方块选择正确      {s['reached_target']:>7.1%}")
    print(f"  区域选择正确      {s['zone_choice']:>7.1%}"
          f"  （到达区域后，落在指定区的比例）")

    fails = {}
    for r in rows:
        if not r["success"]:
            k = r["failure"] or "UNKNOWN"
            fails[k] = fails.get(k, 0) + 1
    print("  失败构成")
    if not fails:
        print("    没有失败")
    else:
        for k in FAIL_CODES:
            if fails.get(k):
                print(f"    {k:<18}{fails[k]:>4}")
    stages = {}
    for r in rows:
        if not r["success"]:
            stage = r["failure_stage"] or "UNKNOWN"
            stages[stage] = stages.get(stage, 0) + 1
    print(f"  失败阶段        {stages}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v1.2"))
    ap.add_argument("--ckpt",
                    default=str(ROOT / "checkpoints"
                               / "v1.2_role_conditioned_v3.pt"))
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--ablation-scenes", type=int, default=0,
                    help="额外跑闭环消融的场景数；0 表示不跑")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--max-ticks", type=int, default=1500)
    ap.add_argument("--failure-log", default="reports/v1.2_failure_traces.json",
                    help="保存失败阶段和最后 10 tick；设为空字符串可关闭")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, ck = load_model(args.ckpt, device)
    print(f"设备 {device}　checkpoint {args.ckpt}")
    print(f"best val {ck['best_val']:.5f}　"
          f"model_type {ck.get('model_type', 'MiniVLA')}")
    print(f"图像分辨率 {ck.get('image_size', 64)}x{ck.get('image_size', 64)}")

    scenes = load_scenes(Path(args.data) / "test", args.scenes)
    print(f"测试场景 {len(scenes)}（模型完全没见过）"
          f"　→ 每场景 4 种指令 = {4 * len(scenes)} 次闭环")

    policy = make_model_policy(model, tok, device)
    rng = np.random.default_rng(args.seed)
    random_policy = make_random_policy(rng)

    rows, rows_rand = [], []
    for i, (scene, tidx) in enumerate(scenes):
        for block, zone in COMBS:
            rows.append(run_episode(scene, tidx, block, zone, policy,
                                    max_ticks=args.max_ticks))
            rows_rand.append(run_episode(scene, tidx, block, zone, random_policy,
                                         max_ticks=args.max_ticks))
        if (i + 1) % 25 == 0:
            k = sum(r["success"] for r in rows) / len(rows)
            print(f"  {i + 1}/{len(scenes)}　模型累计成功率 {k:.1%}", flush=True)

    sm = summarize("模型", rows)
    sr = summarize("随机策略", rows_rand)

    if args.failure_log:
        log_path = ROOT / args.failure_log
        log_path.parent.mkdir(parents=True, exist_ok=True)
        failures = [r for r in rows if not r["success"]]
        # trace 已经转换成 Python list，可直接写 JSON。
        log_path.write_text(json.dumps({
            "checkpoint": str(args.ckpt),
            "seed": args.seed,
            "failures": failures,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n失败轨迹已保存：{log_path}（{len(failures)} 条）")

    print("\n按指令组合：")
    for block, zone in COMBS:
        sub = [r for r in rows if r["block"] == block and r["zone"] == zone]
        print(f"  {block}->{zone:<5} "
              f"{sum(r['success'] for r in sub):>4}/{len(sub):<4}"
              f" = {sum(r['success'] for r in sub) / len(sub):.1%}")

    # ---- 可选闭环消融 ----
    ab = {}
    if args.ablation_scenes > 0:
        ab_scenes = scenes[:args.ablation_scenes]
        print(f"\n闭环消融（前 {len(ab_scenes)} 个场景 × 4 指令）：")
        # 图像置零
        r_img = []
        for scene, tidx in ab_scenes:
            for block, zone in COMBS:
                r_img.append(run_episode(scene, tidx, block, zone, policy,
                                         max_ticks=args.max_ticks,
                                         zero_image=True))
        ab["图像置零"] = sum(r["success"] for r in r_img) / len(r_img)
        print(f"  图像置零       成功 {ab['图像置零']:>7.1%}")

        # 方块指令打乱：环境仍按原指令，模型收到另一个方块颜色的指令
        r_blk = []
        for scene, tidx in ab_scenes:
            for block, zone in COMBS:
                other = "blue" if block == "red" else "red"
                r_blk.append(run_episode(scene, tidx, block, zone, policy,
                                         max_ticks=args.max_ticks,
                                         override=(other, zone)))
        ab["方块指令打乱"] = sum(r["success"] for r in r_blk) / len(r_blk)
        print(f"  方块指令打乱   成功 {ab['方块指令打乱']:>7.1%}")

        # 区域指令打乱：环境仍按原指令，模型收到另一个区域颜色的指令
        r_zon = []
        for scene, tidx in ab_scenes:
            for block, zone in COMBS:
                other = "blue" if zone == "red" else "red"
                r_zon.append(run_episode(scene, tidx, block, zone, policy,
                                         max_ticks=args.max_ticks,
                                         override=(block, other)))
        ab["区域指令打乱"] = sum(r["success"] for r in r_zon) / len(r_zon)
        print(f"  区域指令打乱   成功 {ab['区域指令打乱']:>7.1%}")

    # ---- 验收 ----
    random_success = sr.get("success", 1.0)
    checks = [
        ("方块选择 >= 95%", sm["reached_target"] >= 0.95),
        ("区域选择 >= 95%", sm["zone_choice"] >= 0.95),
        ("完整成功 >= 90%", sm["success"] >= 0.90),
        ("显著优于随机", sm["success"] > 2 * random_success),
    ]
    if ab:
        for k, v in ab.items():
            checks.append((f"{k} 明显变差", v < 0.5 * sm["success"]))

    print("\n" + "=" * 62)
    for name, ok in checks:
        print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}")
    print(f"\n{'[通过]' if all(o for _, o in checks) else '[不通过]'} V1.2 闭环评测")
    return 0 if all(o for _, o in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
