"""V1.5c environment: target block and target zone both move.

The target block moves with a constant velocity until it is captured.  Once
held, it follows the robot exactly as in the base environment.  The target
zone keeps moving for the whole episode and reflects at the workspace bounds.
"""

import numpy as np

from mini_vla.data.grasp_place_dynamic_v15 import (
    MAX_ZONE_SPEED, MIN_ZONE_SPEED, ZONE_MOTION_BOUND, DynamicZoneEnvV15,
)
from mini_vla.data.grasp_place_v13 import COLORS


class DynamicBothEnvV15(DynamicZoneEnvV15):
    """Both the commanded block and destination zone move continuously."""

    def __init__(self, scene, color, zone_color, moving_zone=None,
                 zone_velocity=(0.0, 0.0), block_velocity=(0.0, 0.0),
                 **kwargs):
        block_velocity = np.asarray(block_velocity, dtype=np.float64)
        if block_velocity.shape != (2,) or not np.all(np.isfinite(block_velocity)):
            raise ValueError("block_velocity 必须是有限的二维向量")
        self.initial_block_velocity = block_velocity.copy()
        super().__init__(
            scene, color, zone_color, moving_zone=moving_zone,
            zone_velocity=zone_velocity, **kwargs,
        )

    def reset(self):
        super().reset()
        self.block_velocity = {
            key: np.zeros(2, dtype=np.float64) for key in COLORS
        }
        self.block_velocity[self.color] = self.initial_block_velocity.copy()
        return self.observation()

    def _move_free_block(self):
        if self.holding or self.done:
            return
        self.objects[self.color] += self.block_velocity[self.color]
        position = self.objects[self.color]
        velocity = self.block_velocity[self.color]
        for axis in range(2):
            if position[axis] < -ZONE_MOTION_BOUND:
                position[axis] = -ZONE_MOTION_BOUND
                velocity[axis] = abs(velocity[axis])
            elif position[axis] > ZONE_MOTION_BOUND:
                position[axis] = ZONE_MOTION_BOUND
                velocity[axis] = -abs(velocity[axis])

    def step(self, action):
        result = super().step(action)
        # The parent step carries the block when held and advances the state
        # machine.  Free motion is applied only after that transition.
        self._move_free_block()
        return self.observation(), self.info()

    def info(self):
        result = super().info()
        result["block_velocities"] = {
            key: value.copy() for key, value in self.block_velocity.items()
        }
        result["moving_block"] = self.color
        return result
