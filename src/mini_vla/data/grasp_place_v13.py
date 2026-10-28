"""V1.3 三色抓取放置环境；V1.2 环境保持不变。"""

from dataclasses import dataclass

import numpy as np

from mini_vla.data.grasp_place import (
    CAPTURE_RADIUS,
    CLOSE_ENTER_TOL,
    DG_THRESH,
    GRIP_CLOSED,
    GRIP_OPEN,
    GRIP_RATE,
    MAX_STEP,
    MAX_TICKS,
    ZONE_HALF,
    ZONE_TOL,
)

COLORS = ("red", "blue", "yellow")


@dataclass
class SceneV13:
    objects: dict[str, np.ndarray]
    robot_start: np.ndarray
    zones: dict[str, np.ndarray]


class GraspPlaceEnvV13:
    def __init__(
        self,
        scene: SceneV13,
        color: str,
        zone_color: str,
        close_enter_tol: float = CLOSE_ENTER_TOL,
        zone_tol: float = ZONE_TOL,
        phase_max_ticks: dict[str, int] | None = None,
    ):
        if color not in COLORS or zone_color not in COLORS:
            raise ValueError(f"color 和 zone_color 必须属于 {COLORS}")
        if set(scene.objects) != set(COLORS):
            raise ValueError(f"scene.objects 必须包含 {COLORS}")
        if set(scene.zones) != set(COLORS):
            raise ValueError(f"scene.zones 必须包含 {COLORS}")
        if not (0.0 < close_enter_tol <= CAPTURE_RADIUS):
            raise ValueError("close_enter_tol 必须在 (0, CAPTURE_RADIUS] 内")
        if not (0.0 < zone_tol <= ZONE_HALF):
            raise ValueError("zone_tol 必须在 (0, ZONE_HALF] 内")
        max_ticks = dict(MAX_TICKS if phase_max_ticks is None else phase_max_ticks)
        if set(max_ticks) != set(MAX_TICKS) or any(value <= 0 for value in max_ticks.values()):
            raise ValueError("phase_max_ticks 必须为四个阶段提供正整数上限")

        self.scene = scene
        self.color = color
        self.zone_color = zone_color
        self.close_enter_tol = float(close_enter_tol)
        self.zones = {
            key: np.asarray(value, dtype=np.float64)
            for key, value in scene.zones.items()
        }
        self.target_zone = self.zones[zone_color]
        self.zone_tol = float(zone_tol)
        self.phase_max_ticks = {key: int(value) for key, value in max_ticks.items()}
        self.reset()

    def reset(self):
        self.robot = np.asarray(self.scene.robot_start, dtype=np.float64).copy()
        self.objects = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in self.scene.objects.items()
        }
        self.holding = False
        self.gripper_open = 1.0
        self.cmd = 0
        self.phase = "MOVE_TO_OBJECT"
        self.phase_ticks = 0
        self.ticks = 0
        self.failure = None
        self.done = False
        return self.observation()

    @property
    def target_object(self):
        return self.objects[self.color]

    def observation(self):
        return {
            "robot_state": np.array(
                [self.robot[0], self.robot[1], float(self.holding), self.gripper_open],
                dtype=np.float32,
            )
        }

    @staticmethod
    def decode_grip(dgrip):
        if dgrip <= -DG_THRESH:
            return -1
        if dgrip >= DG_THRESH:
            return 1
        return 0

    def step(self, action):
        if self.done:
            return self.observation(), self.info()

        dx, dy, dgrip = map(float, action[:3])
        grip_cmd = self.decode_grip(dgrip)
        if grip_cmd:
            self.cmd = grip_cmd

        delta = np.array([dx, dy], dtype=np.float64) * MAX_STEP
        distance = float(np.linalg.norm(delta))
        if distance > MAX_STEP:
            delta *= MAX_STEP / distance

        old_robot = self.robot.copy()
        self.robot += delta
        if self.holding:
            self.objects[self.color] += self.robot - old_robot

        if self.cmd == -1:
            self.gripper_open = max(0.0, self.gripper_open - GRIP_RATE)
        elif self.cmd == 1:
            self.gripper_open = min(1.0, self.gripper_open + GRIP_RATE)

        self.ticks += 1
        self.phase_ticks += 1
        self._advance()
        return self.observation(), self.info()

    def _advance(self):
        if self.phase == "MOVE_TO_OBJECT":
            if np.linalg.norm(self.robot - self.target_object) <= self.close_enter_tol:
                self._enter("CLOSE")
            elif self.phase_ticks >= self.phase_max_ticks["MOVE_TO_OBJECT"]:
                self._fail("TARGET_NOT_REACHED")

        elif self.phase == "CLOSE":
            if self.gripper_open <= GRIP_CLOSED:
                if np.linalg.norm(self.robot - self.target_object) <= CAPTURE_RADIUS:
                    self.holding = True
                    self._enter("MOVE_TO_ZONE")
                else:
                    self._fail("CLOSE_MISS")
            elif self.phase_ticks >= self.phase_max_ticks["CLOSE"]:
                self._fail("MAX_TICKS")

        elif self.phase == "MOVE_TO_ZONE":
            if self.gripper_open >= GRIP_OPEN:
                self.holding = False
                self._fail("OBJECT_SLIPPED")
            elif np.linalg.norm(self.robot - self.target_zone) <= self.zone_tol:
                self._enter("OPEN")
            elif self.phase_ticks >= self.phase_max_ticks["MOVE_TO_ZONE"]:
                self._fail("ZONE_NOT_REACHED")

        elif self.phase == "OPEN":
            if self.gripper_open >= GRIP_OPEN:
                self.holding = False
                if self.in_target_zone(self.target_object):
                    self._enter("DONE")
                elif self.in_any_zone(self.target_object):
                    self._fail("WRONG_ZONE")
                else:
                    self._fail("DROP_OUTSIDE")
            elif self.phase_ticks >= self.phase_max_ticks["OPEN"]:
                self._fail("MAX_TICKS")

    def _enter(self, phase):
        self.phase = phase
        self.phase_ticks = 0
        if phase in ("DONE", "FAILED"):
            self.done = True

    def _fail(self, reason):
        self.failure = reason
        self._enter("FAILED")

    def in_target_zone(self, point):
        point = np.asarray(point)
        return bool(np.all(np.abs(point - self.target_zone) <= ZONE_HALF))

    def in_any_zone(self, point):
        point = np.asarray(point)
        return any(
            np.all(np.abs(point - center) <= ZONE_HALF)
            for center in self.zones.values()
        )

    def info(self):
        return {
            "phase": self.phase,
            "holding": self.holding,
            "gripper_open": self.gripper_open,
            "robot": self.robot.copy(),
            "object": self.target_object.copy(),
            "ticks": self.ticks,
            "failure": self.failure,
            "success": self.phase == "DONE",
            "gripper_cmd": self.cmd,
        }
