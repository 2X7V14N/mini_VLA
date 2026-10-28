"""V1.3 三色环境的 64×64 图像渲染。"""

import numpy as np

from mini_vla.data.grasp_place import ZONE_HALF
from mini_vla.data.synthetic import _draw_disk, _world_to_pixel

IMAGE_SIZE = 64
TARGET_PX = 3
ROBOT_PX = 2

BLOCK_COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}

# 深色填充，外加亮色边框。即使黄色放置区对应黄色方块，
# 方框和圆点的形状也能帮助视觉区分两者。
ZONE_COLORS = {
    "red": ((120, 30, 30), (210, 100, 100)),
    "blue": ((30, 40, 120), (100, 120, 210)),
    "yellow": ((130, 100, 20), (230, 190, 70)),
}

ROBOT_COLOR = (0, 255, 0)


def render_v13(env, image_size=IMAGE_SIZE):
    """将三色环境渲染为 (H, W, 3) 的 uint8 RGB 图像。"""
    if image_size != 64:
        raise ValueError("V1.3 当前固定使用 64×64 分辨率")

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # 先画放置区：深色方框加亮边框。
    for color, center in env.zones.items():
        fill, border = ZONE_COLORS[color]
        x, y = map(float, center)

        r0, c0 = _world_to_pixel(
            x - ZONE_HALF, y + ZONE_HALF, image_size
        )
        r1, c1 = _world_to_pixel(
            x + ZONE_HALF, y - ZONE_HALF, image_size
        )
        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))

        image[r0:r1 + 1, c0:c1 + 1] = fill
        image[r0, c0:c1 + 1] = border
        image[r1, c0:c1 + 1] = border
        image[r0:r1 + 1, c0] = border
        image[r0:r1 + 1, c1] = border

    # 再画三种方块。
    for color, position in env.objects.items():
        row, col = _world_to_pixel(
            float(position[0]), float(position[1]), image_size
        )
        _draw_disk(
            image, row, col, TARGET_PX, BLOCK_COLORS[color]
        )

    # 机器人最后绘制，保持 V1.2 的图层顺序。
    row, col = _world_to_pixel(
        float(env.robot[0]), float(env.robot[1]), image_size
    )
    _draw_disk(image, row, col, ROBOT_PX, ROBOT_COLOR)

    return image
