"""V1.1 可视化：把状态机的四个阶段在二维画布上跑一遍。

    python scripts/v1_1/watch_grasp_place.py
    python scripts/v1_1/watch_grasp_place.py --color blue
    python scripts/v1_1/watch_grasp_place.py --seed 7 --speed 0.01
    python scripts/v1_1/watch_grasp_place.py --no-wait      # 演示完窗口自动关

画的东西：
    红点 / 蓝点   两个目标（被搬走的那个会跟着动）
    绿点          机器人（二维夹爪中心）
    绿色轨迹      机器人走过的路径
    深绿圆环      夹爪开合度（环大 = 张开，环小 = 闭合）
    黄色方块      固定放置区域
    标题          当前阶段 / 是否夹住 / 夹爪开度 / tick

★ 这里用的是【写死的驱动逻辑】，只为了让画面动起来。
  正式的专家策略（带完整的阶段超时和失败归因）是第 2 步才写的。
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mini_vla.data.grasp_place import (      # noqa: E402
    GraspPlaceEnv, Scene, sample_scene, MAX_STEP, ZONE_CENTER, ZONE_HALF,
)

import matplotlib.pyplot as plt                # noqa: E402
from matplotlib.patches import Circle, Rectangle   # noqa: E402


FIXED_SCENE = Scene(red=np.array([0.40, 0.20]),
                    blue=np.array([-0.45, 0.30]),
                    robot_start=np.array([0.00, 0.00]))


def set_point(artist, p):
    """画一个点。

    ★ 不能写 artist.set_data(*p)：matplotlib 3.7 起传标量会触发弃用警告。
    """
    artist.set_data([float(p[0])], [float(p[1])])


def drive(env):
    """写死的驱动：按当前阶段决定这一 tick 的动作。"""
    a = np.zeros(3, dtype=np.float32)
    if env.phase == "MOVE_TO_OBJECT":
        d = env.target_object - env.robot
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            a[:2] = d / n * min(1.0, n / MAX_STEP)
        a[2] = +1.0
    elif env.phase == "CLOSE":
        a[2] = -1.0
    elif env.phase == "MOVE_TO_ZONE":
        d = ZONE_CENTER - env.robot
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            a[:2] = d / n * min(1.0, n / MAX_STEP)
        a[2] = -1.0
    elif env.phase == "OPEN":
        a[2] = +1.0
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", choices=("red", "blue"), default="red")
    ap.add_argument("--seed", type=int, default=None,
                    help="给定种子就用随机场景；不给就用固定场景")
    ap.add_argument("--speed", type=float, default=0.02, help="每 tick 停多少秒")
    ap.add_argument("--no-wait", action="store_true", help="结束后不阻塞窗口")
    args = ap.parse_args()

    if args.seed is None:
        scene = FIXED_SCENE
    else:
        scene = sample_scene(np.random.default_rng(args.seed))

    env = GraspPlaceEnv(scene, args.color)
    target_color = ("#e02424" if args.color == "red" else "#1d4ed8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.ion()

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)
    ax.axhline(0, color="0.9", lw=0.8, zorder=0)
    ax.axvline(0, color="0.9", lw=0.8, zorder=0)
    ax.set_xlabel("x（归一化世界坐标）")
    ax.set_ylabel("y（归一化世界坐标）")

    # 放置区
    ax.add_patch(Rectangle(ZONE_CENTER - ZONE_HALF, 2 * ZONE_HALF, 2 * ZONE_HALF,
                           facecolor="#ffe08a", edgecolor="#c9a227",
                           lw=1.6, alpha=0.6, zorder=1))

    trail, = ax.plot([], [], "-", lw=1.3, color="#16a34a", alpha=0.45, zorder=2)
    red_dot, = ax.plot([], [], "o", ms=15, color="#e02424", zorder=3)
    blue_dot, = ax.plot([], [], "o", ms=15, color="#1d4ed8", zorder=3)
    halo, = ax.plot([], [], "o", ms=26, mfc="none", mec=target_color,
                    mew=1.2, ls=":", alpha=0.8, zorder=3)
    robot_dot, = ax.plot([], [], "o", ms=9, color="#16a34a",
                         mec="white", mew=1.0, zorder=6)
    grip_ring = Circle((0, 0), 0.06, fill=False, color="#065f46", lw=2.0, zorder=5)
    ax.add_patch(grip_ring)

    hist = []
    for _ in range(1200):
        if env.done:
            break
        hist.append(env.robot.copy())

        set_point(red_dot, env.objects["red"])
        set_point(blue_dot, env.objects["blue"])
        set_point(halo, env.objects["red" if args.color == "red" else "blue"])
        set_point(robot_dot, env.robot)
        grip_ring.center = (env.robot[0], env.robot[1])
        grip_ring.set_radius(0.03 + 0.09 * env.gripper_open)
        trail.set_data([p[0] for p in hist], [p[1] for p in hist])

        state = "夹住" if env.holding else "空手"
        ax.set_title(
            f"阶段 {env.phase}　｜　{state}　｜　"
            f"夹爪开度 {env.gripper_open:.2f}　｜　tick {env.ticks}",
            fontsize=11)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if args.speed > 0:
            time.sleep(args.speed)   # ★ 不能用 plt.pause：非 GUI 后端会卡死

        env.step(drive(env))

    # ---- 收尾：画最后一帧 + 结果 ----
    set_point(red_dot, env.objects["red"])
    set_point(blue_dot, env.objects["blue"])
    set_point(robot_dot, env.robot)
    grip_ring.center = (env.robot[0], env.robot[1])
    grip_ring.set_radius(0.03 + 0.09 * env.gripper_open)
    hist.append(env.robot.copy())
    trail.set_data([p[0] for p in hist], [p[1] for p in hist])

    ok = env.info()["success"]
    ax.set_title(
        f"{'成功' if ok else '失败'}：{env.phase}"
        + (f"　（{env.failure}）" if env.failure else "")
        + f"　｜　{tick_msg(env)}",
        fontsize=12, color="#15803d" if ok else "#b91c1c")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    print(f"结果: {env.phase}　成功={ok}　失败码={env.failure}　"
          f"tick={env.ticks}　目标最终位置={np.round(env.target_object, 3).tolist()}")

    if not args.no_wait:
        plt.ioff()
        plt.show(block=True)


def tick_msg(env):
    return f"共 {env.ticks} tick"


if __name__ == "__main__":
    main()
