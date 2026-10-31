"""V1.1 第 1 步验收：用【写死的动作序列】把四个阶段跑通。

    python scripts/v1_1/test_grasp_place_env.py
    
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mini_vla.data.grasp_place import (
    GraspPlaceEnv, Scene, ZONE_CENTER, ZONE_HALF, MAX_STEP, REACH_TOL,
)

PASS, FAIL = "[OK]  ", "[FAIL]"
ncases = 0
nfail = 0


def check(name, cond, detail=""):
    global ncases, nfail
    ncases += 1
    if not cond:
        nfail += 1
    print(f"  {PASS if cond else FAIL} {name:<46}{detail}")


# ---------------------------------------------------------------------------
# 用写死的动作驱动环境
# ---------------------------------------------------------------------------
def goto(env, target, stop_phase, dgrip, max_ticks=500):
    """朝 target 走，直到阶段离开 stop_phase。返回用了多少 tick。"""
    for t in range(max_ticks):
        if env.phase != stop_phase:
            return t
        d = np.asarray(target, dtype=np.float64) - env.robot
        n = float(np.linalg.norm(d))
        a = np.zeros(3, dtype=np.float32)
        if n > 1e-9:
            a[:2] = d / n * min(1.0, n / MAX_STEP)
        a[2] = dgrip
        env.step(a)
    return max_ticks


def stand(env, dgrip, stop_phase, max_ticks=100):
    """原地不动，只发夹爪命令，直到阶段离开 stop_phase。"""
    for t in range(max_ticks):
        if env.phase != stop_phase:
            return t
        env.step(np.array([0.0, 0.0, dgrip], dtype=np.float32))
    return max_ticks


# ---------------------------------------------------------------------------
# 固定场景：坐标写死，方便手算和复现
# ---------------------------------------------------------------------------
def fixed_scene():
    return Scene(red=np.array([0.40, 0.20]),
                 blue=np.array([-0.45, 0.30]),
                 robot_start=np.array([0.00, 0.00]))


# ===========================================================================
# ① 完整成功流程 + 搬运连续性
# ===========================================================================
print("\n① 完整流程：走到红方块 -> 闭合 -> 搬到放置区 -> 释放")

env = GraspPlaceEnv(fixed_scene(), "red")
check("初始阶段是 MOVE_TO_OBJECT", env.phase == "MOVE_TO_OBJECT", env.phase)

goto(env, env.target_object, "MOVE_TO_OBJECT", dgrip=+1.0)
check("到达目标后进入 CLOSE", env.phase == "CLOSE", env.phase)
check("到达时还没夹住", env.holding is False)

stand(env, -1.0, "CLOSE")
check("夹爪闭到底后夹住目标", env.holding is True)
check("进入 MOVE_TO_ZONE", env.phase == "MOVE_TO_ZONE", env.phase)

# ---- 搬运：逐 tick 记录，检查目标是否只跟随夹爪的实际位移 ----
roll = [env.info().copy()]
goto(env, ZONE_CENTER, "MOVE_TO_ZONE", dgrip=-1.0)
roll.append(env.info().copy())
check("到达放置区后进入 OPEN", env.phase == "OPEN", env.phase)

stand(env, +1.0, "OPEN")
check("张开后进入 DONE", env.phase == "DONE", env.phase)
check("最终成功", env.info()["success"] is True)
check("目标最终在放置区内", GraspPlaceEnv.in_zone(env.info()["object"]),
      np.round(env.info()["object"], 3).tolist())

# 搬运连续性的硬检查：重新跑一遍，逐 tick 比对
env2 = GraspPlaceEnv(fixed_scene(), "red")
goto(env2, env2.target_object, "MOVE_TO_OBJECT", dgrip=+1.0)
stand(env2, -1.0, "CLOSE")
prev_robot, prev_obj = env2.robot.copy(), env2.target_object.copy()
max_jump, worst = 0.0, 0
while env2.phase == "MOVE_TO_ZONE":
    env2.step(np.array([1.0, 1.0, -1.0], dtype=np.float32))
    drobot = env2.robot - prev_robot
    dobj = env2.target_object - prev_obj
    jump = float(np.linalg.norm(dobj))
    if jump > max_jump:
        max_jump, worst = jump, env2.ticks
    if np.linalg.norm(drobot) > 1e-12:
        delta_err = float(np.linalg.norm(dobj - drobot))
        if delta_err > 1e-9:
            check("目标位移 == 夹爪位移", False, f"误差 {delta_err:.2e}")
            break
    prev_robot, prev_obj = env2.robot.copy(), env2.target_object.copy()

check("目标单步位移不超过动作上限", max_jump <= MAX_STEP + 1e-9,
      f"最大 {max_jump:.4f} @tick {worst}")

# ===========================================================================
# ② CLOSE_MISS：在远离目标的地方闭合
# ===========================================================================
print("\n② 失败路径：在远处闭合 -> CLOSE_MISS")

env = GraspPlaceEnv(fixed_scene(), "red")
goto(env, env.target_object, "MOVE_TO_OBJECT", dgrip=+1.0)
env.robot = env.robot + np.array([0.5, 0.0])      # 强行走开（模拟没对准）
stand(env, -1.0, "CLOSE")
check("失败码是 CLOSE_MISS", env.failure == "CLOSE_MISS", str(env.failure))
check("失败时 holding 仍为 False", env.holding is False)

# ===========================================================================
# ③ OBJECT_SLIPPED：搬运途中松手
# ===========================================================================
print("\n③ 失败路径：搬运途中松手 -> OBJECT_SLIPPED")

env = GraspPlaceEnv(fixed_scene(), "red")
goto(env, env.target_object, "MOVE_TO_OBJECT", dgrip=+1.0)
stand(env, -1.0, "CLOSE")
stand(env, +1.0, "MOVE_TO_ZONE", max_ticks=20)   # 持续张开，直到夹爪真的开到底
check("松手后失败码是 OBJECT_SLIPPED",
      env.failure == "OBJECT_SLIPPED", str(env.failure))
check("松手后 holding 变成 False", env.holding is False)

# ===========================================================================
# ④ DROP_OUTSIDE：在放置区外释放
# ===========================================================================
print("\n④ 失败路径：在放置区外释放 -> DROP_OUTSIDE")
# ★ 这个失败码在当前设计下【很罕见】：目标严格跟随夹爪，
#   夹爪站在区心时目标必然也在区内。要触发它必须同时满足：
#     ① 抓取时目标就偏离夹爪接近捕获半径（"擦边夹住"）
#     ② 松手时夹爪已进入区域容差，但目标的偏移方向朝外
#   所以这里手工构造这个极端场景，而不是靠随机采样碰运气。
u = np.array([1.0, 1.0]) / np.sqrt(2.0)      # 朝区域外的单位向量
obj0 = ZONE_CENTER + 0.18 * u                # 目标在区外（区半宽 0.11）
robot0 = obj0 - 0.10 * u                     # 机器人离目标刚好 0.10（捕获半径上限）
scene4 = Scene(red=obj0, blue=np.array([-0.50, -0.50]), robot_start=robot0)
env = GraspPlaceEnv(scene4, "red")

env.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))   # 原地闭合，触发进入 CLOSE
check("进入 CLOSE 阶段", env.phase == "CLOSE", env.phase)
stand(env, -1.0, "CLOSE", max_ticks=40)
check("擦边抓取（距离 0.10）仍然夹住", env.holding is True, env.phase)

# 机器人此刻在区心外 0.08（已进入 ZONE_TOL），目标在外 0.18
stand(env, -1.0, "MOVE_TO_ZONE", max_ticks=5)
check("夹爪进入 OPEN 阶段", env.phase == "OPEN", env.phase)

obj_at_open = env.target_object.copy()
check("松手前目标已经在区外", not GraspPlaceEnv.in_zone(obj_at_open),
      np.round(obj_at_open, 3).tolist())

stand(env, +1.0, "OPEN", max_ticks=20)
check("目标在区外时才报 DROP_OUTSIDE",
      env.failure == "DROP_OUTSIDE", str(env.failure))

# ===========================================================================
# ⑤ TARGET_NOT_REACHED：一步不动
# ===========================================================================
print("\n⑤ 失败路径：一直不动 -> TARGET_NOT_REACHED")

env = GraspPlaceEnv(fixed_scene(), "red")
for _ in range(210):
    if env.done:
        break
    env.step(np.array([0.0, 0.0, +1.0], dtype=np.float32))
check("超时后失败码是 TARGET_NOT_REACHED",
      env.failure == "TARGET_NOT_REACHED", str(env.failure))

# ===========================================================================
# ⑥ 观测契约
# ===========================================================================
print("\n⑥ 观测契约")

env = GraspPlaceEnv(fixed_scene(), "blue")
obs = env.observation()
rs = obs["robot_state"]
check("robot_state 形状 (3,)", rs.shape == (3,), str(rs.shape))
check("robot_state dtype float32", rs.dtype == np.float32, str(rs.dtype))
check("grip 在接任务时为 0", rs[2] == 0.0)
check("decode_grip 三段正确",
      (GraspPlaceEnv.decode_grip(-1.0) == -1
       and GraspPlaceEnv.decode_grip(0.0) == 0
       and GraspPlaceEnv.decode_grip(+1.0) == +1))

# ===========================================================================
print("\n" + "=" * 64)
print(f"{ncases - nfail}/{ncases} 通过" if nfail == 0 else f"有 {nfail} 项失败")
sys.exit(0 if nfail == 0 else 1)
