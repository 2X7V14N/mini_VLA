import numpy as np

from mini_vla.data.grasp_place_v13 import GraspPlaceEnvV13, SceneV13
from mini_vla.data.grasp_place_render_v13 import render_v13


def test_v13_renderer_returns_64_pixel_rgb_image_with_all_block_colors():
    scene = SceneV13(
        objects={
            "red": np.array([-0.45, -0.40]),
            "blue": np.array([0.00, 0.45]),
            "yellow": np.array([0.45, -0.35]),
        },
        robot_start=np.array([0.00, 0.00]),
        zones={
            "red": np.array([-0.65, 0.65]),
            "blue": np.array([0.65, 0.65]),
            "yellow": np.array([0.00, -0.70]),
        },
    )
    env = GraspPlaceEnvV13(scene, "yellow", "blue")

    image = render_v13(env)

    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert np.any(np.all(image == (255, 0, 0), axis=-1))
    assert np.any(np.all(image == (0, 0, 255), axis=-1))
    assert np.any(np.all(image == (255, 255, 0), axis=-1))
    assert np.any(np.all(image == (0, 255, 0), axis=-1))