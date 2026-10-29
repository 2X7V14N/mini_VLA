"""Oracle expert for V1.5c, using simulator-only positions and velocities."""

import numpy as np

from mini_vla.data.grasp_place import MAX_STEP, ZONE_HALF
from mini_vla.data.grasp_place_expert_v13 import expert_action_v13
from mini_vla.data.grasp_place_dynamic_expert_v15 import expert_action_v15


def expert_action_v15c(env, block_color, release_margin=0.04,
                        position_gain=0.5, intercept_ticks=2):
    """Track the moving block before capture and moving zone after capture."""
    if not env.holding:
        target = env.target_object + env.block_velocity[block_color] * intercept_ticks
        error = target - env.robot
        action = np.zeros(3, dtype=np.float32)
        distance = float(np.linalg.norm(error))
        if distance > MAX_STEP:
            error *= MAX_STEP / distance
        action[:2] = error / MAX_STEP
        action[2] = -1.0 if env.phase == "CLOSE" else +1.0
        return action

    # Keep the V1.5 moving-zone controller for the carried phase, but use the
    # current object/zone positions and zone velocity from V1.5c.
    zone = np.asarray(env.target_zone, dtype=np.float64)
    obj = np.asarray(env.target_object, dtype=np.float64)
    velocity = np.asarray(env.zone_velocities[env.zone_color], dtype=np.float64)
    desired = velocity + position_gain * (zone - obj)
    distance = float(np.linalg.norm(desired))
    if distance > MAX_STEP:
        desired *= MAX_STEP / distance
    action = np.zeros(3, dtype=np.float32)
    action[:2] = desired / MAX_STEP
    future_zone = zone + velocity * max(1, int(round((1.0 - env.gripper_open) / 0.25)))
    # Entering OPEN already means the robot reached the moving zone.  Keep
    # opening continuously; closing again would reset the gripper and can
    # deterministically hit the OPEN phase timeout.
    action[2] = +1.0 if env.phase == "OPEN" else -1.0
    return action
