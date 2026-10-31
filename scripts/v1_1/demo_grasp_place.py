"""V1.1 交互式二维抓取演示。

启动：
    python scripts/v1_1/demo_grasp_place.py

使用训练好的模型闭环：
    python scripts/v1_1/demo_grasp_place.py \
        --checkpoint checkpoints/v1.1_grasp_place.pt

每轮会随机生成一个合法场景，然后在终端输入“红色”或“蓝色”。
图形窗口显示当前二维场景、夹爪轨迹、目标轨迹、当前阶段和抓取状态。
按 Ctrl+C 或输入 q 退出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place import (  # noqa: E402
    ZONE_CENTER,
    ZONE_HALF,
    GraspPlaceEnv,
    sample_scene,
)
from mini_vla.data.grasp_place_data import TEMPLATES  # noqa: E402


def parse_color(text: str) -> str | None:
    """把中文或英文输入统一成环境使用的颜色名。"""
    text = text.strip().lower()
    if text in {"红", "红色", "red", "r"}:
        return "red"
    if text in {"蓝", "蓝色", "blue", "b"}:
        return "blue"
    return None


class ModelPolicy:
    """V1.1 checkpoint 的单步 receding-horizon 策略。"""

    def __init__(self, checkpoint: Path):
        import torch

        from mini_vla.models.mini_vla import MiniVLA
        from mini_vla.models.text import SimpleCharTokenizer

        self.torch = torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.device = device
        self.tokenizer = SimpleCharTokenizer(
            ckpt["vocab"], max_length=ckpt["max_length"]
        )
        self.model = MiniVLA(
            vocab_size=self.tokenizer.vocab_size,
            max_length=ckpt["max_length"],
            state_dim=ckpt.get("state_dim", 3),
            action_dim=ckpt.get("action_dim", 3),
            chunk_size=ckpt.get("chunk_size", 4),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"模型策略：{checkpoint}")
        print(f"设备：{device}")

    @staticmethod
    def instruction(color: str) -> str:
        return TEMPLATES[color][0]

    def action(self, env: GraspPlaceEnv, color: str) -> np.ndarray:
        from mini_vla.data.grasp_place_render import render

        image = render(env)
        img = self.torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        img = img.unsqueeze(0).to(self.device)
        tokens = self.tokenizer.encode(self.instruction(color)).unsqueeze(0)
        tokens = tokens.to(self.device)
        state = self.torch.from_numpy(env.observation()["robot_state"])
        state = state.unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            pred = self.model(img, tokens, state)
        return pred[0, 0].detach().cpu().numpy().astype(np.float32)


class SceneView:
    """Matplotlib 实时二维视图。"""

    def __init__(self, interval: float):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        self.plt = plt
        self.Rectangle = Rectangle
        plt.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "DejaVu Sans"
        ]
        plt.rcParams["axes.unicode_minus"] = False
        self.interval = interval
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7.2, 7.2))
        self.fig.canvas.manager.set_window_title("Mini-VLA V1.1 抓取演示")
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.25)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.zone = None
        self.robot_line, = self.ax.plot([], [], "-", color="#16a34a", lw=1.4,
                                        alpha=0.75, label="夹爪轨迹")
        self.object_line, = self.ax.plot([], [], "--", color="#111827", lw=1.1,
                                         alpha=0.75, label="被夹目标轨迹")
        self.robot_point, = self.ax.plot([], [], "o", ms=10, color="#16a34a",
                                         mec="white", mew=1.2, label="夹爪")
        self.red_point, = self.ax.plot([], [], "o", ms=12, color="#ef4444",
                                       mec="white", mew=1.0, label="红色目标")
        self.blue_point, = self.ax.plot([], [], "o", ms=12, color="#2563eb",
                                        mec="white", mew=1.0, label="蓝色目标")
        self.ax.legend(loc="lower left", fontsize=9)

    def reset(self, scene, color: str | None = None):
        self.ax.clear()
        self.ax.set_xlim(-1.0, 1.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.25)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.add_patch(self.Rectangle(
            ZONE_CENTER - ZONE_HALF,
            2 * ZONE_HALF,
            2 * ZONE_HALF,
            facecolor="#fde68a", edgecolor="#ca8a04", alpha=0.65,
            label="固定放置区",
        ))
        self.robot_line, = self.ax.plot([], [], "-", color="#16a34a", lw=1.4,
                                        alpha=0.75, label="夹爪轨迹")
        self.object_line, = self.ax.plot([], [], "--", color="#111827", lw=1.1,
                                         alpha=0.75, label="被夹目标轨迹")
        self.robot_point, = self.ax.plot(*scene.robot_start, "o", ms=10,
                                         color="#16a34a", mec="white", mew=1.2,
                                         label="夹爪")
        self.start_point, = self.ax.plot(
            *scene.robot_start, "*", ms=13, color="#15803d",
            mec="white", mew=0.8, label="起始位置"
        )
        self.zone_center, = self.ax.plot(
            *ZONE_CENTER, "+", ms=12, mew=2.0, color="#a16207",
            label="放置目标中心"
        )
        self.red_point, = self.ax.plot(*scene.red, "o", ms=12, color="#ef4444",
                                       mec="white", mew=1.0, label="红色目标")
        self.blue_point, = self.ax.plot(*scene.blue, "o", ms=12, color="#2563eb",
                                        mec="white", mew=1.0, label="蓝色目标")
        self.ax.legend(loc="lower left", fontsize=9)
        if color is None:
            self.ax.set_title("随机场景：请选择要抓取的目标")
        else:
            self.ax.set_title(f"模型指令：{color}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.01)

    def update(self, env: GraspPlaceEnv, robot_path, object_path):
        self.robot_line.set_data(
            np.asarray(robot_path)[:, 0], np.asarray(robot_path)[:, 1]
        )
        self.object_line.set_data(
            np.asarray(object_path)[:, 0], np.asarray(object_path)[:, 1]
        )
        self.robot_point.set_data([env.robot[0]], [env.robot[1]])
        self.red_point.set_data([env.objects["red"][0]], [env.objects["red"][1]])
        self.blue_point.set_data([env.objects["blue"][0]], [env.objects["blue"][1]])
        status = (
            f"阶段：{env.phase}    tick：{env.ticks}\n"
            f"holding：{int(env.holding)}    gripper_open：{env.gripper_open:.2f}"
        )
        self.ax.set_title(status)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(self.interval)

    def hold(self):
        self.plt.pause(0.2)


def run_episode(scene, color: str, policy: ModelPolicy, view: SceneView,
                max_ticks: int):
    env = GraspPlaceEnv(scene, color)
    view.reset(scene, color)
    robot_path = [env.robot.copy()]
    object_path = [env.objects[color].copy()]

    for _ in range(max_ticks):
        if env.done:
            break
        action = policy.action(env, color)
        env.step(action)
        robot_path.append(env.robot.copy())
        object_path.append(env.objects[color].copy())
        view.update(env, robot_path, object_path)

    info = env.info()
    print("\n" + "=" * 54)
    print(f"指令：{color}　策略：训练模型")
    print(f"结果：{'成功' if info['success'] else '失败'}")
    print(f"阶段：{info['phase']}　tick：{info['ticks']}　holding：{info['holding']}")
    if info["failure"]:
        print(f"失败原因：{info['failure']}")
    print(f"目标位置：({info['object'][0]:+.3f}, {info['object'][1]:+.3f})")
    print(f"放置区中心：({ZONE_CENTER[0]:+.3f}, {ZONE_CENTER[1]:+.3f})")
    print("=" * 54)


def main():
    ap = argparse.ArgumentParser(description="V1.1 训练模型二维抓取可视化")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "checkpoints" / "v1.1_grasp_place.pt")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子；默认每轮使用不可重复的随机场景")
    ap.add_argument("--interval", type=float, default=0.05,
                    help="每个 tick 的可视化暂停秒数")
    ap.add_argument("--max-ticks", type=int, default=600)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        ap.error(f"找不到 checkpoint：{args.checkpoint}")

    rng = np.random.default_rng(args.seed)
    policy = ModelPolicy(args.checkpoint)
    view = SceneView(args.interval)

    print("V1.1 二维抓取演示")
    print("输入：红色 / 蓝色，也可以输入 red / blue")
    print("输入 n 生成下一场景，输入 q 退出。")
    try:
        while True:
            scene = sample_scene(rng)
            print("\n已生成随机场景：")
            print(f"  红色目标 ({scene.red[0]:+.3f}, {scene.red[1]:+.3f})")
            print(f"  蓝色目标 ({scene.blue[0]:+.3f}, {scene.blue[1]:+.3f})")
            print(f"  夹爪起点 ({scene.robot_start[0]:+.3f}, {scene.robot_start[1]:+.3f})")
            # 先把未开始执行的随机场景画出来，输入颜色前也能看到全部点位。
            view.reset(scene)
            choice = input("请选择要抓取的目标 [红色/蓝色/q]：").strip().lower()
            if choice in {"q", "quit", "exit"}:
                break
            color = parse_color(choice)
            if color is None:
                print("无法识别，请输入 红色、蓝色、red 或 blue。")
                continue

            run_episode(scene, color, policy, view, args.max_ticks)
            input("按 Enter 生成下一场景，输入 q 退出：")
    except KeyboardInterrupt:
        print("\n已退出演示。")
    finally:
        view.plt.ioff()
        view.plt.close(view.fig)


if __name__ == "__main__":
    main()
