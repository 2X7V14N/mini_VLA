import math
import time

import numpy as np
import mujoco
import mujoco.viewer

from mini_vla.simulation.ik import ik      



def main():
    model = mujoco.MjModel.from_xml_path("src/mini_vla/simulation/scene.xml")
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # 两个目标点（方块上方 0.10 米）
    TARGETS = {
        "红块上方": np.array([0.20,  0.07, 0.175]),
        "蓝块上方": np.array([0.15, -0.17, 0.175]),
    }

    # 检查两个目标都能到达（先离线验证一遍）
    print("检查可达性:")
    for name, t in TARGETS.items():
        data.qpos[:3] = [0, 0.5, 1.0]
        q, err = ik(model, data, t, site_id)
        print(f"  {name}: 误差 {err:.6f}  {'✅' if err < 1e-4 else '❌'}")

    names = list(TARGETS)
    print("\n窗口里末端会在两点之间来回移动。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t_frac, idx = 0.0, 0

        while viewer.is_running():
            a = TARGETS[names[idx % 2]]
            b = TARGETS[names[(idx + 1) % 2]]

            # 平滑插值（缓入缓出）
            s = 0.5 - 0.5 * math.cos(math.pi * t_frac)
            target = a + (b - a) * s

            # 用 IK 求关节角，直接设定（V2.0 是"几何移动"，不加动力学）
            ik(model, data, target, site_id, n_iter=60)

            viewer.sync()

            t_frac += 1 / 180.0
            if t_frac >= 1.0:
                t_frac, idx = 0.0, idx + 1

            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
