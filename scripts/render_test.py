import os
import sys

import numpy as np
import mujoco
from PIL import Image

from ik_test import ik

# ===========================================================================
# 相机参数 —— 改这里，重跑，看图
# ===========================================================================
LOOKAT    = [0.10, -0.04, 0.14]   # 画面中心对准哪个点
DISTANCE  = 0.62                  # 远近（越小物体越大）
AZIMUTH   = 135                   # 水平角（度）
ELEVATION = -30                   # 俯仰角（度，负数=俯视）

# 三个要看的姿态
POSES = [
    ("home",       None),
    ("above_red",  [0.20,  0.07, 0.175]),
    ("above_blue", [0.15, -0.17, 0.175]),
]
# ===========================================================================


def main():
    model = mujoco.MjModel.from_xml_path("src/mini_vla/simulation/scene.xml")
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = LOOKAT
    cam.distance = DISTANCE
    cam.azimuth = AZIMUTH
    cam.elevation = ELEVATION

    os.makedirs("logs", exist_ok=True)
    big = mujoco.Renderer(model, 256, 256)
    small = mujoco.Renderer(model, 64, 64)

    tiles_big, tiles_small = [], []
    for name, target in POSES:
        if target is None:
            data.qpos[:3] = [0.0, 0.5, 1.0]        # 一个竖直姿态
            mujoco.mj_forward(model, data)
        else:
            ik(model, data, np.array(target), site_id)

        big.update_scene(data, cam)
        small.update_scene(data, cam)
        img_big, img_small = big.render(), small.render()

        Image.fromarray(img_big).save(f"logs/render_{name}_256.png")
        Image.fromarray(img_small).save(f"logs/render_{name}_64.png")
        tiles_big.append(img_big)
        tiles_small.append(np.array(Image.fromarray(img_small).resize((256, 256), Image.NEAREST)))

        # 统计像素数 —— 判断"模型能不能看清"
        a = img_small
        r, g, b = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)

        n_red    = int(((r > g + 30) & (r > b + 30) & (g < 90)).sum())   # 红：R 主导，且 G 很低
        n_blue   = int(((b > r + 30) & (b > g + 20)).sum())               # 蓝：B 主导
        n_green  = int(((g > r + 40) & (g > b + 40)).sum())               # 绿：G 主导
        n_orange = int(((r > g + 30) & (g > b + 30) & (g > 60)).sum())    # 橙：R>G>B，但 G 不低
        print(f"  {name:11s} 64x64: 红 {n_red:4d}px   蓝 {n_blue:4d}px")
        print(f"  {name:11s} 64x64: 红 {n_red:4d}px  蓝 {n_blue:4d}px  "
              f"绿球 {n_green:4d}px  橙臂 {n_orange:4d}px")
    gap = np.full((8, 256 * 3, 3), 255, np.uint8)
    out = np.concatenate([np.concatenate(tiles_big, 1), gap,
                        np.concatenate(tiles_small, 1)], 0)
    Image.fromarray(out).save("logs/render_test.png")
    print("\n已保存 logs/render_test.png")
    print("  上排 = 256x256（人看）   下排 = 64x64 放大 4 倍（模型看）")


if __name__ == "__main__":
    main()