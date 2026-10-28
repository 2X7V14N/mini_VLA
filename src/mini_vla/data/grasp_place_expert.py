"""V1.1 专家策略：给定环境状态和【指令颜色】，输出 [dx, dy, dgrip]。

★ 两个入口：
    expert_action(env)            —— 按 env.color（这条 rollout 实际执行的指令）
    expert_action_for(env, color) —— 按指定颜色

  为什么需要第二个：配对数据要在【同一个状态】下算出"如果指令是另一个颜色，
  专家会怎么做"。两个样本共享完全相同的观测，所以两者是同一个输入的两种反应。

★ 它只读观测里能看到的东西：holding、自己的位置、两个物体的位置、区域中心。
  **不读 env.phase** —— 闭环时模型看不到 phase。

★ V1.2 通过 fine_tol=0.04 + require_zone=True 放宽开合条件：
  64x64 图像 1 像素约等于 0.031 个世界单位，默认的 0.005 精度无法从图像中学到。
  默认参数保持 V1.1 行为完全不变。
"""

import numpy as np

from mini_vla.data.grasp_place import (
    CLOSE_ENTER_TOL, MAX_STEP, REACH_TOL, ZONE_CENTER, ZONE_TOL,
)

FINE_TOL = 0.005      # "目标已对准区中心"的细判据


def expert_action_for(env, color, fine_tol=FINE_TOL, require_zone=False):
    """从当前物理状态出发，如果指令是 color，专家该做什么。

    color 是【指令】指定的颜色，不一定等于 env.color。
    fine_tol / require_zone 默认保持 V1.1 行为；V1.2 传
    fine_tol=0.04、require_zone=True。
    """
    a = np.zeros(3, dtype=np.float32)

    if not env.holding:
        # ---- 空手：朝指令指定的目标走，到了就闭合 ----
        obj = env.objects[color]
        d = obj - env.robot
        n = float(np.linalg.norm(d))
        # V1.2 由调用方传入 close_enter_tol；V1.1 保持 REACH_TOL。
        close_tol = getattr(env, "close_enter_tol", REACH_TOL)
        if n <= close_tol:
            a[2] = -1.0
            return a
        a[:2] = d / n * min(1.0, n / MAX_STEP)
        a[2] = +1.0
        return a

    # ---- 夹住东西：把【目标】送进放置区，不是把【自己】送到区心 ----
    # 目标相对夹爪的偏移在夹住后就固定了，所以反解出夹爪该去哪。
    offset = env.target_object - env.robot

    # ★ 环境判"到达放置区"看的是【夹爪】离区心的距离（ZONE_TOL=0.10），
    #   而这里保证的是【目标】落在区心。抓取偏移最大能到 CAPTURE_RADIUS=0.10，
    #   两者相等时，偏移接近上限的场景里夹爪会永远差一点点进不了 OPEN 阶段，
    #   专家于是提前松手 -> OBJECT_SLIPPED（实测 V1.2 800 条里正好 1 条）。
    #   按比例缩短偏移：夹爪一定停在 ZONE_TOL 以内，代价是目标离区心差一点点
    #   （仍在区内，ZONE_HALF=0.11）。
    reach = float(np.linalg.norm(offset))
    if reach > 1e-9:
        # require_zone 时给开合精度留出余量，保证松手时机器人已经在
        # 环境的 OPEN 条件内（离区域中心 <= ZONE_TOL），不会途中误松手。
        limit = (max(0.0, ZONE_TOL - fine_tol - 0.01) if require_zone
                 else ZONE_TOL - 0.01)
        offset = offset * min(1.0, limit / reach)

    goal = env.target_zone - offset      # V1.1 时就是 ZONE_CENTER
    d = goal - env.robot
    n = float(np.linalg.norm(d))
    at_zone = float(np.linalg.norm(env.robot - env.target_zone)) <= ZONE_TOL
    # V1.2 的放置区有限宽度：机器人已靠近补偿后的目标点，并不保证
    # 夹持中的方块也已进入区域（夹爪到方块的偏移可能接近 CAPTURE_RADIUS）。
    # 只有方块实际落点已在指令指定区域内，才允许开始松手；否则继续微调。
    object_in_target_zone = (
        not require_zone or env.in_target_zone(env.target_object)
    )
    if n <= fine_tol and (at_zone or not require_zone) and object_in_target_zone:
        a[2] = +1.0
        return a
    if n > 1e-9:
        a[:2] = d / n * min(1.0, n / MAX_STEP)
    a[2] = -1.0
    return a


def expert_action(env):
    """这条 rollout 实际执行的那条指令对应的动作。"""
    return expert_action_for(env, env.color)
