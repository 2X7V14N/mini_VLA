"""V1.2 场景采样：两个方块 + 两个彩色放置区。

    from mini_vla.data.grasp_place_v12 import sample_scene_v12
    scene = sample_scene_v12(rng)
    env = GraspPlaceEnv(scene, "red", zone_color="blue",
                        close_enter_tol=CLOSE_ENTER_TOL)

和 V1.1 的区别只有 Scene：多了 zones={"red": xy, "blue": xy}。
环境、状态机、动作语义、失败码全部复用，不另起一套。

采样约束 = V1.1 的约束 + 四条放置区约束：
    ① 两个放置区互不重叠
    ② 放置区不压住方块
    ③ 放置区完整落在画布内（靠 ZONE_MARGIN 保证）
    ④ 放置区不压住机器人起点
"""

import numpy as np

from mini_vla.data.grasp_place import (
    MIN_ROBOT_CLEAR, MIN_TARGET_SEP, ROBOT_MARGIN, TARGET_MARGIN,
    ZONE_CLEAR, ZONE_HALF, Scene,
)

ZONE_MARGIN = 0.72               # 放置区中心的采样范围 -> 区域边缘最远 0.83
MIN_ZONE_SEP = 2.6 * ZONE_HALF   # 两区中心最小距离（0.286 > 0.22，保证不叠）
ROBOT_ZONE_CLEAR = 0.32          # 机器人起点离任一放置区中心的最小距离

# 64x64 图 1 像素约 0.03125 个世界单位；0.06 约 1.9 像素。
# 容差仍小于放置区半宽 0.11，避免要求低于图像可分辨精度的松手标签。
V12_FINE_TOL = 0.06


def sample_scene_v12(rng, max_tries=1000):
    """采一个 V1.2 场景：两个方块 + 红/蓝两个随机放置区。"""
    for _ in range(max_tries):
        red = rng.uniform(-TARGET_MARGIN, TARGET_MARGIN, 2)
        blue = rng.uniform(-TARGET_MARGIN, TARGET_MARGIN, 2)
        robot = rng.uniform(-ROBOT_MARGIN, ROBOT_MARGIN, 2)
        zr = rng.uniform(-ZONE_MARGIN, ZONE_MARGIN, 2)
        zb = rng.uniform(-ZONE_MARGIN, ZONE_MARGIN, 2)

        # V1.1 的三条
        if np.linalg.norm(red - blue) < MIN_TARGET_SEP:
            continue
        if np.linalg.norm(robot - red) < MIN_ROBOT_CLEAR:
            continue
        if np.linalg.norm(robot - blue) < MIN_ROBOT_CLEAR:
            continue

        # V1.2 的四条
        if np.linalg.norm(zr - zb) < MIN_ZONE_SEP:
            continue
        if (np.linalg.norm(red - zr) < ZONE_CLEAR
                or np.linalg.norm(blue - zr) < ZONE_CLEAR):
            continue
        if (np.linalg.norm(red - zb) < ZONE_CLEAR
                or np.linalg.norm(blue - zb) < ZONE_CLEAR):
            continue
        if (np.linalg.norm(robot - zr) < ROBOT_ZONE_CLEAR
                or np.linalg.norm(robot - zb) < ROBOT_ZONE_CLEAR):
            continue

        return Scene(red=red.astype(np.float64),
                     blue=blue.astype(np.float64),
                     robot_start=robot.astype(np.float64),
                     zones={"red": zr.astype(np.float64),
                            "blue": zb.astype(np.float64)})

    raise RuntimeError(f"采样失败：{max_tries} 次都没有找到合法场景")

# ===========================================================================
# 指令模板与配对样本
# ===========================================================================
import copy

from mini_vla.data.grasp_place import CLOSE_ENTER_TOL, GraspPlaceEnv
from mini_vla.data.grasp_place_expert import expert_action_for
from mini_vla.data.grasp_place_render import render

CN = {"red": "红", "blue": "蓝"}

# 同一个 index 的句子只换颜色，保证红蓝成对
PATTERNS = (
    "把{block}色方块放到{zone}色放置区",
    "把{block}色目标放进{zone}色区域",
    "抓取{block}色方块并放置到{zone}色位置",
)


def instruction_v12(block_color, zone_color, k=0):
    return PATTERNS[k].format(block=CN[block_color], zone=CN[zone_color])


# ★ 每 tick 出 4 条，顺序【固定】：
#     区域在外层、方块在内层 -> 相邻两条成为一对（配对方块颜色）
COMBOS = (("red", "red"), ("blue", "red"),
          ("red", "blue"), ("blue", "blue"))


def expert_action_v12(env, color, fine_tol=V12_FINE_TOL):
    """V1.2 专用专家：放宽开合容差，并要求机器人已在放置区内。

    V1.1 的 expert_action_for 默认行为不变。
    """
    return expert_action_for(env, color, fine_tol=fine_tol,
                             require_zone=True)


def plan_chunk_v12(env, block_color, zone_color, chunk_size=4,
                   fine_tol=V12_FINE_TOL):
    """从当前状态出发、指令为 (block, zone) 时，接下来 chunk_size 步专家会怎么做。

    ★ 用环境副本真实推演：
      空手时换 color（决定去哪个方块），任何时候换 target_zone（决定送到哪）。
      夹住时 color 不能换 —— 手上那个东西的颜色是既成事实。
    """
    sim = copy.deepcopy(env)
    if not env.holding:
        sim.color = block_color
    sim.zone_color = zone_color
    sim.target_zone = sim.zones[zone_color]

    out = []
    for _ in range(chunk_size):
        if sim.done:
            # episode 结束：剩下补零。不能继续"规划下一步"——
            # 那会变成去抓另一个方块，红蓝两条就分叉了。
            out.append(np.zeros(3, dtype=np.float32))
            continue
        a = expert_action_v12(sim, block_color, fine_tol=fine_tol)
        out.append(a.copy())
        sim.step(a)
    return np.asarray(out, dtype=np.float32)


def record_rollout_v12(scene, exec_block, exec_zone, template_index=0,
                       chunk_size=4):
    """跑一条物理 rollout，每 tick 出 4 条配对样本。"""
    env = GraspPlaceEnv(scene, exec_block, zone_color=exec_zone,
                        close_enter_tol=CLOSE_ENTER_TOL)
    samples = []
    t = 0
    while not env.done and t < 1500:
        img = render(env)
        st = env.observation()["robot_state"]

        for block_color, zone_color in COMBOS:      # ★ 顺序即配对契约
            samples.append({
                "image": img.copy(),
                "instruction": instruction_v12(block_color, zone_color,
                                               template_index),
                "robot_state": st.copy(),
                "action_chunk": plan_chunk_v12(env, block_color, zone_color,
                                               chunk_size),
                "phase": env.phase,
                "target_color": block_color,
                "zone_color": zone_color,
                "object_position": env.objects[block_color].copy(),
                "zone_position": np.asarray(env.zones[zone_color]).copy(),
                "pair_id": t,
                "timestep": t,
                "template_index": template_index,
            })

        env.step(expert_action_v12(env, exec_block))
        t += 1

    return samples, env.info()

