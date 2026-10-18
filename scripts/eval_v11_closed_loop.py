"""V1.1 第 6 步：闭环评测。

    python scripts/eval_v11_closed_loop.py
    python scripts/eval_v11_closed_loop.py --scenes 50 --no-plot

用 test 集的场景（模型完全没见过），红蓝指令各跑一次 = 400 次闭环。
模型每 tick 输出 action_chunk，只执行第 1 步（receding horizon）。

分层报：
    到达目标 / 捕获 / 到达放置区 / 完整成功
    颜色选择正确率 / 已捕获样本的搬运成功率
    失败构成
外加随机策略基线，证明模型不是"乱走也能碰上"。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place import (      # noqa: E402
    GraspPlaceEnv, Scene, REACH_TOL,
)
from mini_vla.data.grasp_place_data import TEMPLATES     # noqa: E402
from mini_vla.data.grasp_place_render import render      # noqa: E402
from mini_vla.models.mini_vla import MiniVLA             # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer     # noqa: E402

FAIL_CODES = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
              "ZONE_NOT_REACHED", "DROP_OUTSIDE", "MAX_TICKS")
COLORS = ("red", "blue")


def load_scenes(split_dir):
    scenes = []
    for f in sorted(Path(split_dir).glob("ep_*.npz")):
        z = np.load(f, allow_pickle=False)
        scenes.append(Scene(red=z["red"], blue=z["blue"],
                            robot_start=z["robot_start"]))
    return scenes


def model_policy(model, tok, device, template_index=0):
    # ★ no_grad 要加在内层 act 上：它是在工厂函数【返回之后】才被调用的，
    #   加在外层只会覆盖"创建闭包"那一瞬间，管不到后来的推理。
    @torch.no_grad()
    def act(env, color):
        img = render(env)
        x = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tid = tok.encode(TEMPLATES[color][template_index]).unsqueeze(0)
        st = torch.from_numpy(env.observation()["robot_state"]).unsqueeze(0)
        out = model(x.to(device), tid.to(device), st.to(device))
        return out[0, 0].cpu().numpy()          # ★ 只取 chunk 的第一步
    return act


def random_policy(rng):
    return lambda env, color: rng.uniform(-1.0, 1.0, 3).astype(np.float32)


def run_episode(scene, color, act_fn, max_ticks=1500):
    env = GraspPlaceEnv(scene, color)
    phases = set()
    while not env.done and env.ticks < max_ticks:
        prev = env.phase
        env.step(act_fn(env, color))
        if env.phase != prev:
            phases.add(env.phase)

    info = env.info()
    # 到达目标 = 进入过 CLOSE（环境是在【指令颜色】那个物体附近才判到位的）
    reached = "CLOSE" in phases
    captured = ("MOVE_TO_ZONE" in phases) or ("OPEN" in phases)
    # 颜色选择正确：结束时离指令指定的物体比另一个近
    obj_t = env.objects[color]
    obj_o = env.objects["blue" if color == "red" else "red"]
    picked = float(np.linalg.norm(env.robot - obj_t)) < \
        float(np.linalg.norm(env.robot - obj_o)) or captured
    return {
        "color": color, "success": bool(info["success"]),
        "failure": info["failure"], "phase": info["phase"],
        "ticks": info["ticks"], "reached": reached,
        "captured": captured, "reached_zone": "OPEN" in phases,
        "picked": bool(picked),
    }


def summarize(tag, rows):
    n = len(rows)
    if n == 0:
        print(f"\n{tag}：没有数据")
        return {}
    s = {
        "n": n,
        "reached": sum(r["reached"] for r in rows) / n,
        "captured": sum(r["captured"] for r in rows) / n,
        "zone": sum(r["reached_zone"] for r in rows) / n,
        "success": sum(r["success"] for r in rows) / n,
        "picked": sum(r["picked"] for r in rows) / n,
    }
    cap = [r for r in rows if r["captured"]]
    s["carry_ok"] = (sum(r["success"] for r in cap) / len(cap)) if cap else float("nan")

    print(f"\n{tag}　（{n} 次闭环）")
    print(f"  到达目标        {s['reached']:>7.1%}")
    print(f"  捕获目标        {s['captured']:>7.1%}")
    print(f"  到达放置区      {s['zone']:>7.1%}")
    print(f"  完整成功        {s['success']:>7.1%}")
    print(f"  颜色选择正确    {s['picked']:>7.1%}")
    print(f"  已捕获→搬运成功 {s['carry_ok']:>7.1%}")

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
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v1.1"))
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "v1.1_grasp_place.pt"))
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    tok = SimpleCharTokenizer(ck["vocab"], max_length=ck["max_length"])
    model = MiniVLA(vocab_size=tok.vocab_size, max_length=ck["max_length"],
                    state_dim=ck["state_dim"], action_dim=ck["action_dim"],
                    chunk_size=ck["chunk_size"]).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    print(f"设备 {device}　checkpoint 最佳 val {ck['best_val']:.5f}")

    scenes = load_scenes(Path(args.data) / "test")[:args.scenes]
    print(f"测试场景 {len(scenes)}（模型完全没见过）"
          f"　→ 红蓝各一次 = {2 * len(scenes)} 次闭环")

    mp = model_policy(model, tok, device)
    rng = np.random.default_rng(0)
    rp = random_policy(rng)

    rows_m, rows_r = [], []
    for i, sc in enumerate(scenes):
        for color in COLORS:
            rows_m.append(run_episode(sc, color, mp))
            rows_r.append(run_episode(sc, color, rp))
        if (i + 1) % 25 == 0:
            k = sum(r["success"] for r in rows_m) / len(rows_m)
            print(f"  {i + 1}/{len(scenes)}　模型累计成功率 {k:.1%}", flush=True)

    sm = summarize("模型", rows_m)
    summarize("随机策略", rows_r)

    print("\n" + "=" * 60)
    for c in COLORS:
        v = sum(r["success"] for r in rows_m if r["color"] == c) / len(scenes)
        print(f"  {c:>4} 指令成功率 {v:>7.1%}　（门槛 85%）")

    checks = [
        ("红色 >= 85%", sum(r["success"] for r in rows_m if r["color"] == "red")
         / len(scenes) >= 0.85),
        ("蓝色 >= 85%", sum(r["success"] for r in rows_m if r["color"] == "blue")
         / len(scenes) >= 0.85),
        ("颜色选择 >= 95%", sm["picked"] >= 0.95),
        ("已捕获→搬运 >= 90%", sm["carry_ok"] >= 0.90),
        ("显著优于随机", sm["success"] > 2 * (sum(r["success"] for r in rows_r)
                                              / max(1, len(rows_r)))),
    ]
    print()
    for name, ok in checks:
        print(f"  {'[OK]  ' if ok else '[FAIL]'} {name}")
    print(f"\n{'[通过]' if all(o for _, o in checks) else '[不通过]'} 第 6 步闭环评测")
    return 0 if all(o for _, o in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
