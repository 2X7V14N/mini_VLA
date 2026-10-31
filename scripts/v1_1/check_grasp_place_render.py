"""V1.1 第 3 步（上）：渲染检查。

    python scripts/v1_1/check_grasp_place_render.py

只回答一个问题：**64×64 的图里，模型能不能看见它该看见的东西？**

检查四件事：
    ① 红/蓝目标、放置区、机器人各自占多少像素（不能小到看不见）
    ② 被搬的目标在搬运阶段确实在动（否则模型没有"我夹住了"的视觉线索）
    ③ 四个阶段的图彼此不同（否则模型分不出阶段）
    ④ 生成的 PNG 拼图能直接看
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mini_vla.data.grasp_place import GraspPlaceEnv, Scene
from mini_vla.data.grasp_place_expert import expert_action
from mini_vla.data.grasp_place_render import render, RED, BLUE, GREEN, ZONE
from mini_vla.data.synthetic import _world_to_pixel

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "experiments" / "v1.1_render_phases.png"

SCENE = Scene(red=np.array([0.40, 0.20]),
              blue=np.array([-0.45, 0.30]),
              robot_start=np.array([0.00, 0.00]))

PHASES = ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")


def count_color(img, color):
    return int(np.all(img == np.array(color, dtype=np.uint8), axis=-1).sum())


def main():
    env = GraspPlaceEnv(SCENE, "red")
    frames, last = {}, None
    track = []                     # 每 tick: (phase, robot, red, blue)

    for _ in range(1500):
        if env.done:
            break
        if env.phase != last:
            frames.setdefault(env.phase, render(env).copy())
            last = env.phase
        track.append((env.phase, env.robot.copy(),
                      env.objects["red"].copy(), env.objects["blue"].copy()))
        env.step(expert_action(env))
    frames.setdefault(env.phase, render(env).copy())

    ok = True

    def centroid_px(p):
        r, c = _world_to_pixel(float(p[0]), float(p[1]))
        return np.array([r, c], dtype=float)

    print("① 各元素在 64×64 里占多少像素（各阶段第一帧）")
    print(f"{'阶段':<16}{'红':>6}{'蓝':>6}{'黄':>6}{'绿':>6}")
    for ph in PHASES:
        if ph not in frames:
            continue
        img = frames[ph]
        r_, b_, y_, g_ = (count_color(img, RED), count_color(img, BLUE),
                          count_color(img, ZONE), count_color(img, GREEN))
        print(f"{ph:<16}{r_:>6}{b_:>6}{y_:>6}{g_:>6}")
        if min(r_, b_, y_, g_) <= 0:
            ok = False
            print(f"  ★ {ph} 有元素完全看不见")
        elif min(r_, b_, y_, g_) < 8:
            print(f"  ⚠ {ph} 有元素少于 8 像素，模型可能学不到")

    # ★ 修正：必须比【搬运开始】和【搬运结束】。
    #   原来比的是 "CLOSE 第一帧" vs "MOVE_TO_ZONE 第一帧" ——
    #   闭合阶段夹爪不动、刚进搬运时目标也还没动，那两帧本来就应该一样。
    print("\n② 搬运阶段：被搬的目标到底动了没有")
    carry = [t for t in track if t[0] == "MOVE_TO_ZONE"]
    if len(carry) >= 2:
        p0 = centroid_px(carry[0][2])
        p1 = centroid_px(carry[-1][2])
        moved_px = float(np.linalg.norm(p1 - p0))
        print(f"  搬运开始 红点像素 {p0}　->　搬运结束 {p1}")
        print(f"  位移 {moved_px:.1f} 像素（搬运 {len(carry)} tick）")
        ok_carry = moved_px >= 3.0
        print(f"  {'[OK]' if ok_carry else '[FAIL]'} 目标确实被搬动"
              f"（>=3px，这是'夹住了'的视觉线索）")
        ok = ok and ok_carry
    else:
        print("  [FAIL] 没有搬运阶段的帧")
        ok = False

    # ★ 修正：不再要求"四个阶段的图两两不同"。
    #   CLOSE 和 MOVE_TO_ZONE 的第一帧本来就是同一张（闭合时什么都不动），
    #   这两个阶段模型靠观测里的 grip（0 vs 1）区分，不需要图像不同。
    #   真正该查的是：**图随时间在变**，否则模型看不到任何进展。
    print("\n③ 图像随时间变化（否则模型看不到进展）")
    for ph in ("MOVE_TO_OBJECT", "MOVE_TO_ZONE"):
        seg = [t for t in track if t[0] == ph]
        if len(seg) < 2:
            continue
        d = float(np.linalg.norm(centroid_px(seg[-1][1]) - centroid_px(seg[0][1])))
        good = d >= 3.0
        ok = ok and good
        print(f"  {ph:<16} 机器人像素位移 {d:>5.1f}　{'[OK]' if good else '[FAIL]'}")
    for ph in ("CLOSE", "OPEN"):
        n = sum(1 for t in track if t[0] == ph)
        print(f"  {ph:<16} {n} tick（按设计机器人不动，正常）")

    # ---- 存拼图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        OUT.parent.mkdir(parents=True, exist_ok=True)
        names = [p for p in PHASES if p in frames]
        n = len(names)
        fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4))
        axes = np.atleast_1d(axes)
        for ax, ph in zip(axes, names):
            ax.imshow(frames[ph], interpolation="nearest")
            ax.set_title(ph, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("V1.1 渲染：64×64 原图（最近邻放大）", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT, dpi=140)
        plt.close(fig)
        print(f"\n拼图已存：{OUT.relative_to(ROOT)}")
    except Exception as e:
        print(f"\n[提示] 拼图没存成：{e}")

    print("\n" + "=" * 60)
    print("[通过] 渲染可见性检查" if ok else "[不通过] 有元素看不见或阶段图重复")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
