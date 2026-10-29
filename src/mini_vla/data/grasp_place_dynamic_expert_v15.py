"""V1.5a 专家：接近静止方块 + 追踪移动放置区 + 择机释放。

专家可以读取仿真真实位置和速度；这些信息只用于生成动作/标签，
不能作为模型输入。
"""

import numpy as np

from mini_vla.data.grasp_place import MAX_STEP, ZONE_HALF
from mini_vla.data.grasp_place_expert_v13 import expert_action_v13


def expert_action_v15(env, block_color, release_margin=0.04,
                      position_gain=0.5):
    """返回 [dx, dy, dgrip]。

    空手阶段完全复用 V1.3 专家；持物阶段用
        desired = zone_velocity + position_gain * (zone - offset - robot)
    同时完成速度匹配和位置纠正，再按物体是否在区域内决定释放。
    """
    if not env.holding:
        return expert_action_v13(env, block_color)

    zone = np.asarray(env.target_zone, dtype=np.float64)
    velocity = np.asarray(
        env.zone_velocities.get(env.zone_color, np.zeros(2)),
        dtype=np.float64,
    )
    obj = env.target_object
    offset = obj - env.robot

    # The carried object follows the robot with a fixed offset.  Use the
    # object-to-zone error directly; subtracting the robot twice biases motion.
    error = zone - obj
    desired = velocity + position_gain * error
    distance = float(np.linalg.norm(desired))
    if distance > MAX_STEP:
        desired *= MAX_STEP / distance

    action = np.zeros(3, dtype=np.float32)
    action[:2] = desired / MAX_STEP

    obj_distance = float(np.linalg.norm(obj - zone))
    obj_inside = bool(np.all(np.abs(obj - zone) <= ZONE_HALF))
    if env.phase == "OPEN" and obj_inside and obj_distance <= release_margin:
        action[2] = +1.0
    else:
        action[2] = -1.0
    return action
