import math
import time

import mujoco
import mujoco.viewer


def main():
    model = mujoco.MjModel.from_xml_path("src/mini_vla/simulation/scene.xml")
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 用【仿真时钟】驱动摆动
            data.ctrl[1] = 0.8 * math.sin(data.time)

            # 每帧走 8 步物理 = 8 × 0.002 = 0.016 秒 ≈ 1/60 秒
            for _ in range(8):
                mujoco.mj_step(model, data)

            viewer.sync()

            # 睡到下一帧（让仿真速度和真实时间一致）
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
