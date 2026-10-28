"""V1.1 二维抓取并放置 —— 环境状态机（第 1 步：不接模型）。

    python scripts/test_grasp_place_env.py

★ 世界坐标沿用 V1：[-1, 1]²，机器人是一个点。
★ 这一版【不渲染图像】。图像在"录制数据"那一步才接进来，
  因为那一步才需要把画面存成 64×64。环境本身不需要看图。
"""

from dataclasses import dataclass

import numpy as np

# ===========================================================================
# 世界与几何参数（全部显式写在这里，不在脚本里散落魔法数字）
# ===========================================================================
TARGET_MARGIN   = 0.60      # 红/蓝目标的采样范围 [-0.6, 0.6]
ROBOT_MARGIN    = 0.80      # 机器人起点的采样范围
MIN_TARGET_SEP  = 0.40      # 红蓝目标最小间距
MIN_ROBOT_CLEAR = 0.20      # 起点离目标的最小距离

TARGET_RADIUS   = 0.08      # 目标半径（只影响外来的可视化/判定口径）
CAPTURE_RADIUS  = 0.10      # 目标中心离夹爪中心多近才能夹住
REACH_TOL       = 0.10      # 算"到达目标"
CLOSE_ENTER_TOL = 0.06      # V1.2 进入闭合阶段的安全距离（小于捕获半径）
ZONE_TOL        = 0.10      # 算"到达放置区"

ZONE_CENTER     = np.array([0.78, 0.78], dtype=np.float64)
ZONE_HALF       = 0.11      # 放置区是边长 0.22 的正方形
ZONE_CLEAR      = 0.32      # 目标中心离放置区中心的最小距离（防止重叠）

# ---- 动作 ----
MAX_STEP        = 0.10      # 每 tick 最大位移（沿用 V1 的 step_size）
GRIP_RATE       = 0.25      # 夹爪每 tick 的开合速度
GRIP_CLOSED     = 0.05      # 小于等于它算"闭到底"
GRIP_OPEN       = 0.95      # 大于等于它算"开到底"
DG_THRESH       = 0.50      # dgrip 解码阈值

# ---- 各阶段最大 tick ----
MAX_TICKS = {
    "MOVE_TO_OBJECT": 200,
    "CLOSE":           20,
    "MOVE_TO_ZONE":   300,
    "OPEN":            20,
}


# ===========================================================================
# 场景
# ===========================================================================
@dataclass
class Scene:
    red: np.ndarray
    blue: np.ndarray
    robot_start: np.ndarray
    # V1.2：{"red": xy, "blue": xy}。None 表示 V1.1 的那个固定放置区。
    zones: dict | None = None


def sample_scene(rng, max_tries=500):
    """采一个合法场景。所有约束都在这里检查，采样失败就重采。"""
    for _ in range(max_tries):
        red = rng.uniform(-TARGET_MARGIN, TARGET_MARGIN, 2)
        blue = rng.uniform(-TARGET_MARGIN, TARGET_MARGIN, 2)
        robot = rng.uniform(-ROBOT_MARGIN, ROBOT_MARGIN, 2)

        if np.linalg.norm(red - blue) < MIN_TARGET_SEP:
            continue
        if np.linalg.norm(robot - red) < MIN_ROBOT_CLEAR:
            continue
        if np.linalg.norm(robot - blue) < MIN_ROBOT_CLEAR:
            continue
        if np.linalg.norm(red - ZONE_CENTER) < ZONE_CLEAR:
            continue
        if np.linalg.norm(blue - ZONE_CENTER) < ZONE_CLEAR:
            continue
        return Scene(red=red.astype(np.float64),
                     blue=blue.astype(np.float64),
                     robot_start=robot.astype(np.float64))
    raise RuntimeError(f"采样失败：{max_tries} 次都没有找到合法场景")


# ===========================================================================
# 环境
# ===========================================================================
class GraspPlaceEnv:
    """二维抓取并放置环境。

    对外只暴露三件事：
        reset()              -> obs
        step(action)         -> (obs, info)
        observation()        -> {"robot_state": [x, y, grip]}

    action = [dx, dy, dgrip]，每一步的语义见 decode_grip()。
    """

    def __init__(self, scene, color, zone_color=None, close_enter_tol=None):
        if color not in ("red", "blue"):
            raise ValueError("color 必须是 'red' 或 'blue'")
        self.scene = scene
        self.color = color
        self.zone_color = zone_color
        # V1.1 保持原来的 REACH_TOL 行为；V1.2 显式传入更保守的阈值。
        self.close_enter_tol = (REACH_TOL if close_enter_tol is None
                                else float(close_enter_tol))
        if not (0.0 < self.close_enter_tol <= CAPTURE_RADIUS):
            raise ValueError(
                "close_enter_tol 必须在 (0, CAPTURE_RADIUS] 范围内")

        # ★ 向后兼容：zone_color=None 时就是 V1.1 的那个固定区，
        #   行为、渲染、判定全部不变。
        if zone_color is None:
            self.zones = {"target": np.asarray(ZONE_CENTER, dtype=np.float64)}
        else:
            if scene.zones is None:
                raise ValueError("场景没有 zones，不能指定 zone_color")
            self.zones = {k: np.asarray(v, dtype=np.float64)
                          for k, v in scene.zones.items()}
            if zone_color not in self.zones:
                raise ValueError(f"场景里没有 {zone_color} 放置区")
        self.target_zone = (self.zones["target"] if zone_color is None
                            else self.zones[zone_color])
        self.reset()

    # ---- 生命周期 ----
    def reset(self):
        self.robot = self.scene.robot_start.copy()
        self.objects = {"red": self.scene.red.copy(),
                        "blue": self.scene.blue.copy()}
        self.holding = False
        self.gripper_open = 1.0
        self.cmd = 0                 # -1 闭合 / 0 保持 / +1 张开
        self.phase = "MOVE_TO_OBJECT"
        self.ticks = 0
        self.phase_ticks = 0
        self.failure = None
        self.done = False
        return self.observation()

    # ---- 观测 ----
    @property
    def target_object(self):
        return self.objects[self.color]

    def observation(self):
        """★ grip 是【实际夹持状态】，不是夹爪命令。

        计划里写死了这条：闭合命令本身不能提前把 grip 改成 1。
        """
        return {"robot_state": np.array(
            [self.robot[0], self.robot[1], 1.0 if self.holding else 0.0],
            dtype=np.float32)}

    # ---- 动作解码（录制 / 回放 / 闭环必须共用这一份）----
    @staticmethod
    def decode_grip(dgrip):
        """连续 dgrip -> 离散命令。返回 -1 / 0 / +1。"""
        dgrip = float(dgrip)
        if dgrip <= -DG_THRESH:
            return -1
        if dgrip >= DG_THRESH:
            return +1
        return 0

    # ---- 单步 ----
    def step(self, action):
        if self.done:
            return self.observation(), self.info()

        dx, dy, dgrip = float(action[0]), float(action[1]), float(action[2])

        # ① 解码夹爪命令（0 = 保持，所以不能直接覆盖）
        cmd = self.decode_grip(dgrip)
        if cmd != 0:
            self.cmd = cmd

        # ② 移动：有最大步长，且记录【实际】走了多远
        # ★ 动作是【归一化位移】∈[-1,1]（沿用 V1 约定），
        #   这里乘回 MAX_STEP 才是世界位移。
        delta = np.array([dx, dy], dtype=np.float64) * MAX_STEP
        n = float(np.linalg.norm(delta))
        if n > MAX_STEP:                 # 兜底：模型输出可能略超 1
            delta *= MAX_STEP / n
        before = self.robot.copy()
        self.robot = self.robot + delta
        gripper_delta = self.robot - before

        # ③ 携带：目标只跟随夹爪的【实际位移增量】
        #    ★ 不能写成 object = gripper_pos + offset（那等于瞬移），
        #      也不能写成 object = ZONE_CENTER（那是伪造搬运）。
        if self.holding:
            self.objects[self.color] = self.objects[self.color] + gripper_delta

        # ④ 夹爪开合
        if self.cmd == -1:
            self.gripper_open = max(0.0, self.gripper_open - GRIP_RATE)
        elif self.cmd == +1:
            self.gripper_open = min(1.0, self.gripper_open + GRIP_RATE)

        # ⑤ 推进状态机
        self.ticks += 1
        self.phase_ticks += 1
        self._advance()

        return self.observation(), self.info()

    # ---- 状态机 ----
    def _advance(self):
        p = self.phase
        obj = self.target_object

        if p == "MOVE_TO_OBJECT":
            if np.linalg.norm(self.robot - obj) <= self.close_enter_tol:
                self._enter("CLOSE")
            elif self.phase_ticks >= MAX_TICKS[p]:
                self._fail("TARGET_NOT_REACHED")

        elif p == "CLOSE":
            if self.gripper_open <= GRIP_CLOSED:
                if np.linalg.norm(self.robot - obj) <= CAPTURE_RADIUS:
                    self.holding = True          # ★ 闭到底 + 目标在半径内
                    self._enter("MOVE_TO_ZONE")
                else:
                    self._fail("CLOSE_MISS")     # 闭到底了，但没夹到东西
            elif self.phase_ticks >= MAX_TICKS[p]:
                self._fail("MAX_TICKS")

        elif p == "MOVE_TO_ZONE":
            if self.gripper_open >= GRIP_OPEN:
                # 搬运途中松手 -> 目标掉在原地
                self.holding = False
                self._fail("OBJECT_SLIPPED")
            elif np.linalg.norm(self.robot - self.target_zone) <= ZONE_TOL:
                self._enter("OPEN")
            elif self.phase_ticks >= MAX_TICKS[p]:
                self._fail("ZONE_NOT_REACHED")

        elif p == "OPEN":
            if self.gripper_open >= GRIP_OPEN:
                self.holding = False
                obj = self.objects[self.color]
                if self.in_target_zone(obj):
                    self._enter("DONE")
                elif self.in_any_zone(obj):
                    self._fail("WRONG_ZONE")     # 放进了区域，但放错了那一个
                else:
                    self._fail("DROP_OUTSIDE")
            elif self.phase_ticks >= MAX_TICKS[p]:
                self._fail("MAX_TICKS")

    def _enter(self, phase):
        self.phase = phase
        self.phase_ticks = 0
        if phase in ("DONE", "FAILED"):
            self.done = True

    def _fail(self, reason):
        self.failure = reason
        self._enter("FAILED")

    # ---- 工具 ----
    @staticmethod
    def in_zone(point):
        """V1.1 的固定区判定（保留：现有脚本还在调）。"""
        return bool(np.all(np.abs(np.asarray(point) - ZONE_CENTER) <= ZONE_HALF))

    def in_target_zone(self, point):
        """点是否落在【指令指定的】那个放置区里。"""
        return bool(np.all(np.abs(np.asarray(point) - self.target_zone) <= ZONE_HALF))

    def in_any_zone(self, point):
        """点是否落在【任何一个】放置区里 —— 用来区分 WRONG_ZONE 和 DROP_OUTSIDE。"""
        p = np.asarray(point)
        return any(np.all(np.abs(p - c) <= ZONE_HALF) for c in self.zones.values())

    def info(self):
        return {
            "phase": self.phase,
            "holding": self.holding,
            "gripper_open": self.gripper_open,
            "robot": self.robot.copy(),
            "object": self.objects[self.color].copy(),
            "ticks": self.ticks,
            "failure": self.failure,
            "success": self.phase == "DONE",
            "gripper_cmd": self.cmd,
        }
