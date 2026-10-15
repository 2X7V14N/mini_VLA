import numpy as np
import mujoco

from mini_vla.simulation.reach_env import (
    sample_positions, make_model, hover_target, make_camera, render, expert_action,
)
from mini_vla.simulation.ik import ik

STEP_SIZE = 0.05
MAX_STEPS = 30
SUCCESS_RADIUS = 0.03
HOVER = 0.10


def main():
    rng = np.random.default_rng(0)
    pts = sample_positions(rng)
    model = make_model(pts[0], pts[1])
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # 红方块上方
    target = hover_target(pts[0])
    print(f"红方块 {pts[0].round(3).tolist()}   目标点 {target.round(3).tolist()}")

    # 从竖直姿态出发
    data.qpos[:3] = [0.0, 0.5, 1.0]
    mujoco.mj_forward(model, data)

    for step in range(MAX_STEPS):
        ee = np.asarray(data.site(sid).xpos)
        dist = np.linalg.norm(ee - target)
        print(f"  第 {step:2d} 步  末端 {ee.round(4).tolist()}  距目标 {dist:.4f}")

        if dist < SUCCESS_RADIUS:
            print(f"\n✅ 第 {step} 步到达")
            return

        a = expert_action(data, target, STEP_SIZE)
        new_target = ee + STEP_SIZE * a                     # 新的末端目标位置
        ik(model, data, new_target, sid)                    # IK 解算

    print(f"\n❌ {MAX_STEPS} 步没到达")


if __name__ == "__main__":
    main()
