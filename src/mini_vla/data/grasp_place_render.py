"""V1.1 64×64 renderer and V1.2 resolution-aware renderer.

★ 复用 V1 的坐标映射和画圆函数（synthetic.py 里的 _world_to_pixel / _draw_disk），
  这样 V1 和 V1.1 的图像约定完全一致：同一个世界坐标 -> 同一个像素。
  **只 import，不修改 V1 的文件。**

★ 图里画四层，从下到上：
     黑色背景
     黄色方块   放置区
     红点/蓝点  两个目标（被搬走的那个会跟着动 -> 这是"夹住了"的视觉线索）
     绿色小圆   机器人

★ 夹爪开合【不画】。它由观测里的 robot_state[2] = grip 提供，
  不需要让模型从像素里猜。少一个变量，图更干净。
"""

import numpy as np

from mini_vla.data.grasp_place import ZONE_CENTER, ZONE_HALF
from mini_vla.data.synthetic import _draw_disk, _world_to_pixel

IMAGE_SIZE = 64
V12_IMAGE_SIZE = 128
TARGET_PX = 3        # 目标半径（像素），沿用 V1
ROBOT_PX = 2         # 机器人半径（像素），沿用 V1

BG   = (0, 0, 0)
RED  = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
ZONE = (255, 220, 0)
ZONE_COLORS = {"red": (120, 30, 30), "blue": (30, 40, 120),
               "target": (255, 220, 0)}


def render(env, image_size=IMAGE_SIZE):
    """把当前环境画成 (image_size, image_size, 3) uint8。"""
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # ① 放置区（最底层）
    if getattr(env, "zone_color", None) is None:
        # V1.1：一个固定区、实心黄色 —— 保持原样，不改动 V1.1 的输入分布
        r0, c0 = _world_to_pixel(ZONE_CENTER[0] - ZONE_HALF,
                                 ZONE_CENTER[1] + ZONE_HALF, image_size)
        r1, c1 = _world_to_pixel(ZONE_CENTER[0] + ZONE_HALF,
                                 ZONE_CENTER[1] - ZONE_HALF, image_size)
        rs, re = sorted((r0, r1))
        cs, ce = sorted((c0, c1))
        img[rs:re + 1, cs:ce + 1] = ZONE
    else:
        # V1.2：多个彩色区。暗填充 + 亮边框，和【实心圆】的方块在形状上分开，
        #   避免"红方块"和"红区域"只靠颜色区分而混淆。
        for key, center in env.zones.items():
            col = ZONE_COLORS.get(key, ZONE)
            r0, c0 = _world_to_pixel(center[0] - ZONE_HALF,
                                     center[1] + ZONE_HALF, image_size)
            r1, c1 = _world_to_pixel(center[0] + ZONE_HALF,
                                     center[1] - ZONE_HALF, image_size)
            rs, re = sorted((r0, r1))
            cs, ce = sorted((c0, c1))
            img[rs:re + 1, cs:ce + 1] = col
            bright = tuple(min(255, v + 90) for v in col)
            img[rs:rs + 1, cs:ce + 1] = bright
            img[re:re + 1, cs:ce + 1] = bright
            img[rs:re + 1, cs:cs + 1] = bright
            img[rs:re + 1, ce:ce + 1] = bright

    # Scale rendered objects with resolution so they retain the same world-space
    # size (and become sharper) instead of appearing half-size at 128px.
    pixel_scale = image_size / IMAGE_SIZE
    target_px = max(1, int(round(TARGET_PX * pixel_scale)))
    robot_px = max(1, int(round(ROBOT_PX * pixel_scale)))

    # ② 两个目标
    for name, color in (("red", RED), ("blue", BLUE)):
        p = env.objects[name]
        r, c = _world_to_pixel(p[0], p[1], image_size)
        _draw_disk(img, r, c, target_px, color)

    # ③ 机器人（画在最上面，挡住身下的东西 —— 和 V1 一致）
    r, c = _world_to_pixel(env.robot[0], env.robot[1], image_size)
    _draw_disk(img, r, c, robot_px, GREEN)

    return img


def render_v12(env, image_size=V12_IMAGE_SIZE):
    """Render V1.2 at a selectable resolution (64 or 128 pixels)."""
    if image_size not in (64, 128):
        raise ValueError("V1.2 image_size 必须是 64 或 128")
    return render(env, image_size=image_size)
