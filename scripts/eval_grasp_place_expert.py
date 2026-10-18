"""V1.1 第 2 步：专家验收。

    python scripts/eval_grasp_place_expert.py
    python scripts/eval_grasp_place_expert.py --scenes 200 --sheet 12

跑 200 个随机起点场景 × 红蓝两条指令 = 400 条 episode，检查：

    ① 成功率 >= 99%
    ② 失败构成（每一类分别计数，不能只有一个总数）
    ③ 专家夹的是不是【指令指定的那个颜色】
    ④ 搬运是【逐 tick 增量跟随】，不是瞬移
    ⑤ 放置后目标离区域中心的距离

不接模型、不录数据、不覆盖任何 checkpoint。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_vla.data.grasp_place import (      # noqa: E402
    GraspPlaceEnv, sample_scene, CAPTURE_RADIUS, MAX_STEP,
    ZONE_CENTER, ZONE_HALF, ZONE_TOL,
)
from mini_vla.data.grasp_place_expert import expert_action   # noqa: E402

FAIL_CODES = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
              "ZONE_NOT_REACHED", "DROP_OUTSIDE", "MAX_TICKS")


def run_episode(scene, color, rng_seed):
    """跑一条 episode，返回这条 episode 的全部诊断量。"""
    env = GraspPlaceEnv(scene, color)
    hold_carry = []          # 搬运期间：每 tick 的 (目标位移, 夹爪位移)
    tick_to_target = tick_to_zone = -1
    picked_correct = None

    for _ in range(1500):
        if env.done:
            break

        prev_phase = env.phase
        prev_obj = env.target_object.copy()
        prev_robot = env.robot.copy()

        env.step(expert_action(env))

        if prev_phase == "MOVE_TO_OBJECT" and env.phase == "CLOSE":
            tick_to_target = env.ticks
            d_red = float(np.linalg.norm(env.robot - env.objects["red"]))
            d_blue = float(np.linalg.norm(env.robot - env.objects["blue"]))
            picked_correct = (("red" if d_red < d_blue else "blue") == color)

        if prev_phase == "MOVE_TO_ZONE" and env.phase == "OPEN":
            tick_to_zone = env.ticks

        if env.holding:
            hold_carry.append((float(np.linalg.norm(env.target_object - prev_obj)),
                               float(np.linalg.norm(env.robot - prev_robot))))

    info = env.info()
    obj = info["object"]
    return {
        "scene_seed": rng_seed,
        "color": color,
        "success": bool(info["success"]),
        "failure": info["failure"],
        "ticks": info["ticks"],
        "tick_to_target": tick_to_target,
        "tick_to_zone": tick_to_zone,
        "picked_correct": picked_correct,
        "carry_ticks": len(hold_carry),
        "max_step_obj": max((c[0] for c in hold_carry), default=0.0),
        "max_step_err": max((abs(c[0] - c[1]) for c in hold_carry), default=0.0),
        "zone_dist": float(np.linalg.norm(obj - ZONE_CENTER)),
        "obj": obj.copy(),
        "robot_start": scene.robot_start.copy(),
        "scene": scene,
    }


def save_sheet(rows, n_show):
    """把 N 条轨迹画成一张拼图，存成 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    pick = rows[:n_show]
    cols = min(4, len(pick))
    nrow = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(nrow, cols, figsize=(3.1 * cols, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, pick):
        sc = r["scene"]
        ax.add_patch(Rectangle(ZONE_CENTER - ZONE_HALF, 2 * ZONE_HALF,
                               2 * ZONE_HALF, facecolor="#ffe08a",
                               edgecolor="#c9a227", lw=1.0, alpha=0.6))
        ax.plot(*sc.red, "o", ms=10, color="#e02424")
        ax.plot(*sc.blue, "o", ms=10, color="#1d4ed8")
        ax.plot(*r["robot_start"], "o", ms=7, color="#16a34a", mec="white")
        ax.plot(*r["obj"], "x", ms=9, color="#111", mew=1.6)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_title(f"{r['color']}  {'OK' if r['success'] else r['failure']}",
                     fontsize=8, color="#15803d" if r["success"] else "#b91c1c")
        ax.tick_params(labelsize=6)

    for ax in axes[len(pick):]:
        ax.axis("off")

    out = Path(__file__).resolve().parent.parent / "docs" / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "v1.1_expert_trajectories.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\n轨迹拼图已存：{path.relative_to(Path(__file__).resolve().parent.parent)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sheet", type=int, default=0,
                    help="额外画 N 条轨迹拼图，存到 docs/experiments/")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.scenes):
        scene = sample_scene(rng)
        for color in ("red", "blue"):
            rows.append(run_episode(scene, color, args.seed + i))
        if (i + 1) % 50 == 0:
            print(f"  已跑 {i + 1}/{args.scenes} 个场景 ...", flush=True)

    n = len(rows)
    n_ok = sum(r["success"] for r in rows)
    rate = n_ok / n

    print("\n" + "=" * 68)
    print(f"场景 {args.scenes}　episode {n}（红 {n // 2} / 蓝 {n // 2}）")
    print(f"成功率 {rate:.2%}　（门槛 99%）")

    print("\n失败构成")
    fails = [r for r in rows if not r["success"]]
    if not fails:
        print("  没有失败")
    else:
        for code in FAIL_CODES:
            k = sum(1 for r in fails if r["failure"] == code)
            if k:
                print(f"  {code:<18}{k:>4}")
        print(f"  {'合计':<18}{len(fails):>4}")

    picked = [r["picked_correct"] for r in rows if r["picked_correct"] is not None]
    print(f"\n颜色选择正确：{sum(picked)}/{len(picked)}"
          f"（{sum(picked) / max(1, len(picked)):.1%}）")

    def stats(key, label, unit="tick"):
        v = np.array([r[key] for r in rows if r[key] >= 0])
        if len(v) == 0:
            print(f"  {label:<14}（无数据）")
            return
        print(f"  {label:<14}min {v.min():>5}　中位 {int(np.median(v)):>5}　"
              f"max {v.max():>5} {unit}")

    print("\n各阶段耗时")
    stats("tick_to_target", "到目标")
    stats("tick_to_zone", "到放置区")
    stats("ticks", "总计")

    max_obj = max(r["max_step_obj"] for r in rows)
    max_err = max(r["max_step_err"] for r in rows)
    print("\n搬运连续性")
    print(f"  目标单步最大位移   {max_obj:.4f}　（动作上限 {MAX_STEP:.2f}）")
    print(f"  目标位移 - 夹爪位移 最大误差 {max_err:.2e}")
    ok_carry = (max_obj <= MAX_STEP + 1e-9) and (max_err <= 1e-9)
    print(f"  {'[OK]' if ok_carry else '[FAIL]'} 搬运是逐 tick 增量跟随，不是瞬移")

    zd = np.array([r["zone_dist"] for r in rows])
    print("\n放置精度")
    print(f"  目标到区域中心：中位 {np.median(zd):.4f}　"
          f"最大 {zd.max():.4f}　（区域半宽 {ZONE_HALF:.2f}，容差 {ZONE_TOL:.2f}）")

    print("\n" + "=" * 68)
    verdict = rate >= 0.99 and ok_carry
    print(f"{'[通过]' if verdict else '[不通过]'} "
          f"成功率 {rate:.2%} / 门槛 99%"
          + ("" if ok_carry else "　搬运连续性不合格"))

    if args.sheet > 0:
        save_sheet(rows, args.sheet)

    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
