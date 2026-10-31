"""Interactive all-moving-scene demo for the final V1.5e Mini-VLA model.

The interactive mode creates the random scene *before* an instruction is
entered. Every block and every placement zone moves while the user types in
the window's text box. Press Enter to give the instruction and let the policy
control the robot in that same, already-moving scene.

This is intentionally a harder visualisation than the V1.5e benchmark: its
official data moves only the instructed block and instructed zone. The model
still receives only rendered images, the exact instruction text, and robot
state. Coordinates and velocities are never policy inputs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.dynamic_render_v15a import render_dynamic_v15a
from mini_vla.data.grasp_place import ZONE_HALF
from mini_vla.data.grasp_place_dynamic_v15 import (
    MAX_ZONE_SPEED,
    MIN_ZONE_SPEED,
    ZONE_MOTION_BOUND,
)
from mini_vla.data.grasp_place_expert_v13 import COLORS, INSTRUCTION_TEMPLATES, sample_scene_v13
from mini_vla.data.grasp_place_v13 import GraspPlaceEnvV13
from mini_vla.models.mini_vla_v15c_dynamic import LanguageConditionedMiniVLA3ColorDynamicBoth
from mini_vla.models.text import SimpleCharTokenizer


COLOR_CN_TO_EN = {"红": "red", "蓝": "blue", "黄": "yellow"}
COLOR_EN_TO_CN = {value: key for key, value in COLOR_CN_TO_EN.items()}
COLOR_HEX = {"red": "#ef4444", "blue": "#2563eb", "yellow": "#eab308"}
PHASE_CN = {
    "MOVE_TO_OBJECT": "前往方块", "CLOSE": "闭合夹爪",
    "MOVE_TO_ZONE": "搬运至放置区", "OPEN": "张开夹爪",
    "DONE": "完成", "FAILED": "失败",
}
FAILURE_CN = {
    "TARGET_NOT_REACHED": "未到达方块", "CLOSE_MISS": "闭合时未夹住方块",
    "OBJECT_SLIPPED": "搬运中物体滑落", "ZONE_NOT_REACHED": "未到达放置区",
    "WRONG_ZONE": "放入错误放置区", "DROP_OUTSIDE": "未放入放置区",
    "MAX_TICKS": "超过最大步数",
}


@dataclass(frozen=True)
class InstructionRoles:
    """Roles parsed only for simulator assignment and UI highlighting."""

    block: str
    zone: str
    in_distribution: bool


def parse_instruction_roles(instruction: str) -> InstructionRoles:
    """Extract two colour roles, while the policy receives the original text."""
    matches = [COLOR_CN_TO_EN[char] for char in instruction if char in COLOR_CN_TO_EN]
    if len(matches) != 2:
        raise ValueError("指令必须恰好包含两次颜色：红、蓝、黄（先方块、后放置区）。")
    templates = {
        template.format(block=COLOR_EN_TO_CN[matches[0]], zone=COLOR_EN_TO_CN[matches[1]])
        for template in INSTRUCTION_TEMPLATES
    }
    return InstructionRoles(matches[0], matches[1], instruction in templates)


def random_velocity(rng: np.random.Generator) -> np.ndarray:
    speed = rng.uniform(MIN_ZONE_SPEED, MAX_ZONE_SPEED)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    return speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


class AllMovingEnvV15(GraspPlaceEnvV13):
    """Demo environment: all blocks and zones move, not only task targets."""

    def __init__(self, scene, block_velocities: dict[str, np.ndarray], zone_velocities: dict[str, np.ndarray]):
        self.initial_block_velocities = self._validate_velocities(block_velocities, "block_velocities")
        self.initial_zone_velocities = self._validate_velocities(zone_velocities, "zone_velocities")
        self.task_assigned = False
        # Roles are replaced by assign_task before the first policy action.
        super().__init__(scene, "red", "red")

    @staticmethod
    def _validate_velocities(values: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
        if set(values) != set(COLORS):
            raise ValueError(f"{name} 必须为每种颜色提供速度")
        checked = {key: np.asarray(value, dtype=np.float64).copy() for key, value in values.items()}
        if any(value.shape != (2,) or not np.all(np.isfinite(value)) for value in checked.values()):
            raise ValueError(f"{name} 的每个速度必须是有限二维向量")
        return checked

    def reset(self):
        super().reset()
        self.zones = {key: np.asarray(value, dtype=np.float64).copy() for key, value in self.scene.zones.items()}
        self.target_zone = self.zones[self.zone_color]
        self.block_velocities = {key: value.copy() for key, value in self.initial_block_velocities.items()}
        self.zone_velocities = {key: value.copy() for key, value in self.initial_zone_velocities.items()}
        self.task_assigned = False
        return self.observation()

    @staticmethod
    def _move_and_reflect(position: np.ndarray, velocity: np.ndarray):
        position += velocity
        for axis in range(2):
            if position[axis] < -ZONE_MOTION_BOUND:
                position[axis] = -ZONE_MOTION_BOUND
                velocity[axis] = abs(velocity[axis])
            elif position[axis] > ZONE_MOTION_BOUND:
                position[axis] = ZONE_MOTION_BOUND
                velocity[axis] = -abs(velocity[axis])

    def _move_world(self):
        for color in COLORS:
            self._move_and_reflect(self.zones[color], self.zone_velocities[color])
        for color in COLORS:
            if self.holding and color == self.color:
                continue
            self._move_and_reflect(self.objects[color], self.block_velocities[color])
        self.target_zone = self.zones[self.zone_color]

    def advance_preview(self):
        """Move the world while waiting for GUI input, but not task time."""
        if not self.task_assigned and not self.done:
            self._move_world()

    def assign_task(self, roles: InstructionRoles):
        if self.task_assigned:
            raise RuntimeError("当前场景已经分配了任务")
        self.color, self.zone_color = roles.block, roles.zone
        self.target_zone = self.zones[self.zone_color]
        self.task_assigned = True

    def step(self, action):
        if not self.task_assigned:
            raise RuntimeError("请先指定指令，再执行模型动作")
        result = super().step(action)
        if not self.done:
            self._move_world()
        return result


def load_policy(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3ColorDynamicBoth":
        raise ValueError(f"checkpoint 不是 V1.5e 双动态模型：{checkpoint.get('model_type')!r}")
    tokenizer = SimpleCharTokenizer(checkpoint["vocab"], int(checkpoint["max_length"]))
    model = LanguageConditionedMiniVLA3ColorDynamicBoth(
        vocab_size=tokenizer.vocab_size, max_length=int(checkpoint["max_length"]),
        state_dim=int(checkpoint["state_dim"]), action_dim=int(checkpoint["action_dim"]),
        chunk_size=int(checkpoint["chunk_size"]), history=int(checkpoint["history"]),
        red_token_id=tokenizer.stoi["红"], blue_token_id=tokenizer.stoi["蓝"],
        yellow_token_id=tokenizer.stoi["黄"],
    ).to(device).eval()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, tokenizer, int(checkpoint["history"])


class PolicyRunner:
    """Makes a real receding-horizon model step from policy observations only."""

    def __init__(self, model, tokenizer, history: int, device: torch.device, instruction: str):
        self.model, self.tokenizer = model, tokenizer
        self.history, self.device = history, device
        self.tokens = tokenizer.batch_encode([instruction]).to(device)
        self.frames: list[np.ndarray] = []

    @torch.inference_mode()
    def action(self, env: AllMovingEnvV15) -> np.ndarray:
        self.frames.append(render_dynamic_v15a(env))
        recent = self.frames[-self.history:]
        if len(recent) < self.history:
            recent = [recent[0]] * (self.history - len(recent)) + recent
        image = torch.from_numpy(np.stack(recent)).permute(0, 3, 1, 2)
        image = image.float().unsqueeze(0).to(self.device).div_(255.0)
        state = torch.from_numpy(env.observation()["robot_state"]).unsqueeze(0).to(self.device)
        return self.model(image, self.tokens, state)[0, 0].detach().cpu().numpy().astype(np.float32)


class SceneView:
    """GUI that displays an all-moving scene and accepts a text instruction."""

    def __init__(self, interval: float):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button, TextBox

        self.plt, self.Rectangle = plt, Rectangle
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        self.interval = max(0.001, interval)
        self.fig, self.ax = plt.subplots(figsize=(9.2, 8.5))
        self.fig.subplots_adjust(bottom=0.20)
        self.fig.canvas.manager.set_window_title("Mini-VLA V1.5e：全场景动态演示")
        input_ax = self.fig.add_axes([0.12, 0.065, 0.48, 0.05])
        button_ax = self.fig.add_axes([0.63, 0.065, 0.12, 0.05])
        new_scene_ax = self.fig.add_axes([0.77, 0.065, 0.14, 0.05])
        self.instruction_box = TextBox(input_ax, "输入指令：", initial="把红色方块放到蓝色放置区")
        self.start_button = Button(button_ax, "执行")
        self.new_scene_button = Button(new_scene_ax, "新场景")
        self.submitted_instruction: str | None = None
        self.new_scene_requested = False
        self.instruction_box.on_submit(self._submit)
        self.start_button.on_clicked(lambda _event: self._submit(self.instruction_box.text))
        self.new_scene_button.on_clicked(lambda _event: self._request_new_scene())
        self.zone_patches: dict[str, object] = {}
        self.block_points: dict[str, object] = {}
        self.robot_path_line = self.block_path_line = self.robot_point = self.status = None

    def _submit(self, text: str):
        text = text.strip()
        if text:
            self.submitted_instruction = text

    def _request_new_scene(self):
        self.new_scene_requested = True

    def clear_requests(self):
        """Start a fresh scene without discarding the text box's last input."""
        self.submitted_instruction = None
        self.new_scene_requested = False

    def is_open(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def setup_scene(self, env: AllMovingEnvV15, roles: InstructionRoles | None = None, instruction: str | None = None):
        self.ax.clear()
        self.ax.set_xlim(-1.0, 1.0); self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_aspect("equal"); self.ax.grid(alpha=0.22)
        self.ax.set_xlabel("世界坐标 x"); self.ax.set_ylabel("世界坐标 y")
        self.zone_patches, self.block_points = {}, {}
        for color in COLORS:
            selected_zone = roles is not None and color == roles.zone
            selected_block = roles is not None and color == roles.block
            patch = self.Rectangle(
                env.zones[color] - ZONE_HALF, 2 * ZONE_HALF, 2 * ZONE_HALF,
                facecolor=COLOR_HEX[color], edgecolor="#111827" if selected_zone else COLOR_HEX[color],
                linewidth=3.0 if selected_zone else 1.4, linestyle="-" if selected_zone else "--", alpha=0.25,
                label=(f"目标放置区（{COLOR_EN_TO_CN[color]}）" if selected_zone else f"{COLOR_EN_TO_CN[color]}色放置区"),
            )
            self.ax.add_patch(patch); self.zone_patches[color] = patch
            point, = self.ax.plot(
                [env.objects[color][0]], [env.objects[color][1]], "o", ms=12, color=COLOR_HEX[color],
                mec="#111827" if selected_block else "white", mew=2.5 if selected_block else 1.0,
                label=(f"目标方块（{COLOR_EN_TO_CN[color]}）" if selected_block else f"{COLOR_EN_TO_CN[color]}色方块"),
            )
            self.block_points[color] = point
        self.robot_path_line, = self.ax.plot([], [], "-", color="#16a34a", lw=1.6, alpha=0.8, label="机器人轨迹")
        self.block_path_line, = self.ax.plot([], [], "--", color="#111827", lw=1.1, alpha=0.75, label="目标方块轨迹")
        self.robot_point, = self.ax.plot([env.robot[0]], [env.robot[1]], "o", ms=11, color="#16a34a", mec="white", mew=1.5, label="机器人")
        self.ax.legend(loc="lower left", fontsize=8, ncol=2)
        self.status = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va="top", ha="left", fontsize=10,
                                   bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#94a3b8"})
        if roles is None:
            self.ax.set_title("全场景已创建：三块方块和三个放置区正在运动\n在下方输入指令，按 Enter 或点击“执行”", fontsize=12)
        else:
            distribution = "训练内模板" if roles.in_distribution else "全场景动态 + 未见表达（分布外）"
            self.ax.set_title(f"V1.5e 执行：{instruction}\n{distribution}", fontsize=12)

    def update(self, env: AllMovingEnvV15, robot_path: list[np.ndarray], block_path: list[np.ndarray] | None, roles: InstructionRoles | None, last_action: np.ndarray | None):
        self.robot_path_line.set_data(np.asarray(robot_path)[:, 0], np.asarray(robot_path)[:, 1])
        if block_path:
            self.block_path_line.set_data(np.asarray(block_path)[:, 0], np.asarray(block_path)[:, 1])
        for color in COLORS:
            self.block_points[color].set_data([env.objects[color][0]], [env.objects[color][1]])
            self.zone_patches[color].set_xy(env.zones[color] - ZONE_HALF)
        self.robot_point.set_data([env.robot[0]], [env.robot[1]])
        if roles is None:
            status = "所有方块、所有放置区：运动中\n等待在窗口下方输入任务指令"
        else:
            action_text = "等待模型动作" if last_action is None else "[{:+.2f}, {:+.2f}, {:+.2f}]".format(*last_action)
            result = "执行中" if not env.done else ("成功" if env.info()["success"] else f"失败：{FAILURE_CN.get(env.failure, env.failure)}")
            status = (f"tick：{env.ticks}    阶段：{PHASE_CN.get(env.phase, env.phase)}\n"
                      f"夹持：{'是' if env.holding else '否'}    夹爪开度：{env.gripper_open:.2f}\n"
                      f"模型动作 [dx, dy, dgrip]：{action_text}\n结果：{result}")
            if env.done:
                status += "\n点击下方“新场景”继续"
        self.status.set_text(status)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events(); self.plt.pause(self.interval)


def build_environment(rng: np.random.Generator) -> AllMovingEnvV15:
    return AllMovingEnvV15(
        sample_scene_v13(rng),
        block_velocities={color: random_velocity(rng) for color in COLORS},
        zone_velocities={color: random_velocity(rng) for color in COLORS},
    )


def run_episode(env, runner: PolicyRunner, roles: InstructionRoles, max_ticks: int, view: SceneView | None = None):
    robot_path, block_path = [env.robot.copy()], [env.objects[roles.block].copy()]
    last_action = None
    if view is not None:
        view.update(env, robot_path, block_path, roles, last_action)
    while not env.done and env.ticks < max_ticks:
        last_action = runner.action(env)
        env.step(last_action)
        robot_path.append(env.robot.copy()); block_path.append(env.objects[roles.block].copy())
        if view is not None:
            view.update(env, robot_path, block_path, roles, last_action)
    if not env.done:
        env._fail("MAX_TICKS")
        if view is not None:
            view.update(env, robot_path, block_path, roles, last_action)
    return env.info()


def print_result(instruction: str, info: dict):
    result = "成功" if info["success"] else f"失败：{FAILURE_CN.get(info['failure'], info['failure'])}"
    print(f"\n指令：{instruction}\n结果：{result}；tick={info['ticks']}；最终阶段={info['phase']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="V1.5e 全方块、全放置区动态的可视化闭环演示")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "v1.5e.pt")
    parser.add_argument("--instruction", help="可选：跳过 GUI 输入、立即执行此原句")
    parser.add_argument("--seed", type=int, help="随机种子；相同种子可复现场景")
    parser.add_argument("--interval", type=float, default=0.05, help="预览与每 tick 的显示间隔（秒）")
    parser.add_argument("--max-ticks", type=int, default=600)
    parser.add_argument("--headless", action="store_true", help="不创建图形窗口，适合快速检查")
    parser.add_argument("--smoke", action="store_true", help="运行一次 5 tick 的无界面冒烟检查")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error(f"找不到 checkpoint：{args.checkpoint}")
    if args.max_ticks <= 0:
        parser.error("max-ticks 必须为正数")
    if args.smoke:
        args.headless, args.max_ticks = True, min(args.max_ticks, 5)
        args.instruction = args.instruction or "把红色方块放到蓝色放置区"
    if args.headless and not args.instruction:
        parser.error("--headless 需要同时提供 --instruction（或使用 --smoke）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, history = load_policy(args.checkpoint, device)
    rng = np.random.default_rng(args.seed)
    view = None if args.headless else SceneView(args.interval)
    print(f"设备：{device}；checkpoint：{args.checkpoint}；history={history}")
    print("交互模式已先创建全动态场景：三块方块、三个放置区都会运动。")
    print("注意：V1.5e 正式训练只让目标方块与目标区运动；本演示的全动态场景属于更难的分布外测试。")

    # Headless and --instruction are intentionally one-shot. The normal GUI
    # path below remains alive across any number of reset scenes.
    if view is None or args.instruction:
        instruction = args.instruction
        env = build_environment(rng)
        roles = parse_instruction_roles(instruction)
        env.assign_task(roles)
        if view is not None:
            view.setup_scene(env, roles, instruction)
        info = run_episode(env, PolicyRunner(model, tokenizer, history, device, instruction), roles, args.max_ticks, view)
        print_result(instruction, info)
        if view is not None:
            print("关闭可视化窗口后，程序退出。")
            view.plt.ioff(); view.plt.show()
        return 0

    # Interactive lifecycle: preview -> execute -> wait for New Scene -> preview.
    while view.is_open():
        env = build_environment(rng)
        view.clear_requests()
        view.setup_scene(env)
        robot_path = [env.robot.copy()]
        while view.is_open() and view.submitted_instruction is None and not view.new_scene_requested:
            env.advance_preview()
            view.update(env, robot_path, None, None, None)
        if not view.is_open():
            break
        if view.new_scene_requested:
            continue
        instruction = view.submitted_instruction
        try:
            roles = parse_instruction_roles(instruction)
        except ValueError as error:
            print(f"指令未执行：{error}")
            # Return to a fresh preview scene so invalid input cannot leave a
            # partially assigned task in the current world.
            continue
        env.assign_task(roles)
        view.setup_scene(env, roles, instruction)
        info = run_episode(env, PolicyRunner(model, tokenizer, history, device, instruction), roles, args.max_ticks, view)
        print_result(instruction, info)
        # Keep the completed scene visible. A button event ends this wait and
        # immediately creates a new full-dynamic scene at the top of the loop.
        while view.is_open() and not view.new_scene_requested:
            view.plt.pause(view.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
