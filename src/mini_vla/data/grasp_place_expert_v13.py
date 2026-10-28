"""V1.3 三色场景采样与专家策略。"""

from itertools import product

import numpy as np

from mini_vla.data.grasp_place import (
    MIN_ROBOT_CLEAR,
    MIN_TARGET_SEP,
    ROBOT_MARGIN,
    TARGET_MARGIN,
    ZONE_CLEAR,
    ZONE_HALF,
)
from mini_vla.data.grasp_place_expert import expert_action_for
from mini_vla.data.grasp_place_v13 import (
    COLORS,
    GraspPlaceEnvV13,
    SceneV13,
)

ZONE_MARGIN = 0.72
MIN_ZONE_SEP = 2.6 * ZONE_HALF
ROBOT_ZONE_CLEAR = 0.32
V13_FINE_TOL = 0.08
COMBOS = tuple(product(COLORS, repeat=2))

COLOR_CN = {"red": "红", "blue": "蓝", "yellow": "黄"}
INSTRUCTION_TEMPLATES = (
    "把{block}色方块放到{zone}色放置区",
    "把{block}色目标放进{zone}色区域",
    "抓取{block}色方块并放置到{zone}色位置",
)


def valid_combo_mask_v13(holding, exec_block, exec_zone):
    """保留第一次 V1.3 的监督规则：每个观测都监督九种指令组合。"""
    if exec_block not in COLORS or exec_zone not in COLORS:
        raise ValueError("不支持的实际执行颜色组合")
    return np.ones(len(COMBOS), dtype=np.bool_)


def instruction_v13(block_color, zone_color, template_index=0):
    if block_color not in COLORS or zone_color not in COLORS:
        raise ValueError("不支持的方块或放置区颜色")
    if not 0 <= template_index < len(INSTRUCTION_TEMPLATES):
        raise ValueError("template_index 超出范围")
    return INSTRUCTION_TEMPLATES[template_index].format(
        block=COLOR_CN[block_color],
        zone=COLOR_CN[zone_color],
    )


def sample_scene_v13(rng, max_tries=200_000):
    """采样合法场景：三种方块、三种颜色放置区、一个机器人起点。"""
    for _ in range(max_tries):
        objects = {
            color: rng.uniform(-TARGET_MARGIN, TARGET_MARGIN, 2)
            for color in COLORS
        }
        zones = {
            color: rng.uniform(-ZONE_MARGIN, ZONE_MARGIN, 2)
            for color in COLORS
        }
        robot = rng.uniform(-ROBOT_MARGIN, ROBOT_MARGIN, 2)

        object_positions = list(objects.values())
        zone_positions = list(zones.values())

        if any(
            np.linalg.norm(object_positions[i] - object_positions[j])
            < MIN_TARGET_SEP
            for i in range(len(object_positions))
            for j in range(i + 1, len(object_positions))
        ):
            continue

        if any(
            np.linalg.norm(zone_positions[i] - zone_positions[j])
            < MIN_ZONE_SEP
            for i in range(len(zone_positions))
            for j in range(i + 1, len(zone_positions))
        ):
            continue

        # 初始方块不能压住任何放置区。
        if any(
            np.linalg.norm(obj - zone) < ZONE_CLEAR
            for obj in object_positions
            for zone in zone_positions
        ):
            continue

        if any(
            np.linalg.norm(robot - obj) < MIN_ROBOT_CLEAR
            for obj in object_positions
        ):
            continue

        if any(
            np.linalg.norm(robot - zone) < ROBOT_ZONE_CLEAR
            for zone in zone_positions
        ):
            continue

        return SceneV13(
            objects={k: v.astype(np.float64) for k, v in objects.items()},
            robot_start=robot.astype(np.float64),
            zones={k: v.astype(np.float64) for k, v in zones.items()},
        )

    raise RuntimeError(f"采样 {max_tries} 次仍未找到合法场景")


def expert_action_v13(env, block_color):
    """V1.2 的专家动作规则，应用于三色环境。"""
    return expert_action_for(
        env,
        block_color,
        fine_tol=V13_FINE_TOL,
        require_zone=True,
    )


def run_expert_episode(scene, block_color, zone_color, max_ticks=1500):
    env = GraspPlaceEnvV13(scene, block_color, zone_color)

    while not env.done and env.ticks < max_ticks:
        action = expert_action_v13(env, block_color)
        env.step(action)

    if not env.done:
        env._fail("MAX_TICKS")

    return env.info()
