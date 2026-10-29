"""V1.5a：只有放置区移动，方块和其他放置区静止。

    from mini_vla.data.grasp_place_dynamic_v15 import DynamicZoneEnvV15
    env = DynamicZoneEnvV15(scene, "red", "blue",
                            moving_zone="blue",
                            zone_velocity=(0.012, 0.0))

tick 顺序：
    state_t -> image_t -> action_t -> 机器人/夹爪更新 -> 状态机推进
    -> 放置区移动 -> state_{t+1} -> image_{t+1}

放置区匀速直线运动，到边界后按轴反弹，不会瞬移。
"""

import numpy as np

from mini_vla.data.grasp_place_v13 import COLORS, GraspPlaceEnvV13

# 区域中心的运动边界。V1.3 的采样范围是 ±0.72；区域边缘最远
# 0.72 + ZONE_HALF = 0.83，仍在 [-1, 1] 画布内。
ZONE_MOTION_BOUND = 0.72
MIN_ZONE_SPEED = 0.01
MAX_ZONE_SPEED = 0.02


class DynamicZoneEnvV15(GraspPlaceEnvV13):
    """在 V1.3 环境中加入一个匀速移动的放置区。"""

    def __init__(self, scene, color, zone_color, moving_zone=None,
                 zone_velocity=(0.0, 0.0), **kwargs):
        if moving_zone is None:
            moving_zone = zone_color
        if moving_zone not in COLORS:
            raise ValueError(f"moving_zone 必须属于 {COLORS}")
        if moving_zone not in scene.zones:
            raise ValueError("场景中没有 moving_zone 对应的放置区")

        velocity = np.asarray(zone_velocity, dtype=np.float64)
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            raise ValueError("zone_velocity 必须是有限的二维向量")

        # 父类 __init__ 最后会调用 self.reset()，所以这两个字段必须先设置。
        self.moving_zone = moving_zone
        self.initial_velocity = velocity.copy()
        super().__init__(scene, color, zone_color, **kwargs)

    # ---- 生命周期 ----
    def reset(self):
        super().reset()
        # V1.3 的 reset 不恢复区域位置；动态环境必须恢复初始轨迹。
        self.zones = {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in self.scene.zones.items()
        }
        self.target_zone = self.zones[self.zone_color]
        self.zone_velocities = {
            key: np.zeros(2, dtype=np.float64) for key in self.zones
        }
        self.zone_velocities[self.moving_zone] = self.initial_velocity.copy()
        return self.observation()

    # ---- 区域运动 ----
    def _reflect_at_bounds(self, key):
        position = self.zones[key]
        velocity = self.zone_velocities[key]
        for axis in range(2):
            if position[axis] < -ZONE_MOTION_BOUND:
                position[axis] = -ZONE_MOTION_BOUND
                velocity[axis] = abs(velocity[axis])
            elif position[axis] > ZONE_MOTION_BOUND:
                position[axis] = ZONE_MOTION_BOUND
                velocity[axis] = -abs(velocity[axis])
            elif (position[axis] == -ZONE_MOTION_BOUND
                  and velocity[axis] < 0):
                velocity[axis] = -velocity[axis]
            elif (position[axis] == ZONE_MOTION_BOUND
                  and velocity[axis] > 0):
                velocity[axis] = -velocity[axis]

    def _move_zones(self):
        for key, velocity in self.zone_velocities.items():
            if not np.any(velocity):
                continue
            self.zones[key] += velocity
            self._reflect_at_bounds(key)

    # ---- 单步 ----
    def step(self, action):
        super().step(action)
        if not self.done:
            self._move_zones()
            self.target_zone = self.zones[self.zone_color]
        return self.observation(), self.info()

    def info(self):
        result = super().info()
        result["zone_positions"] = {
            key: value.copy() for key, value in self.zones.items()
        }
        result["zone_velocities"] = {
            key: value.copy() for key, value in self.zone_velocities.items()
        }
        result["moving_zone"] = self.moving_zone
        return result
