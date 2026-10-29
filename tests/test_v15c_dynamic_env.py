import numpy as np

from mini_vla.data.grasp_place_dynamic_v15c import DynamicBothEnvV15
from mini_vla.data.grasp_place_v13 import SceneV13


def test_v15c_block_moves_before_capture_and_zone_moves():
    scene = SceneV13(
        objects={"red": np.array([-0.45, -0.40]), "blue": np.array([0.0, 0.45]), "yellow": np.array([0.45, -0.35])},
        robot_start=np.array([0.0, 0.0]),
        zones={"red": np.array([-0.65, 0.65]), "blue": np.array([0.0, 0.0]), "yellow": np.array([0.65, -0.65])},
    )
    env = DynamicBothEnvV15(scene, "red", "blue", zone_velocity=(0.01, 0.0), block_velocity=(0.0, 0.01))
    before_block, before_zone = env.target_object.copy(), env.target_zone.copy()
    env.step(np.zeros(3, dtype=np.float32))
    assert np.allclose(env.target_object - before_block, [0.0, 0.01])
    assert np.allclose(env.target_zone - before_zone, [0.01, 0.0])


def test_v15c_block_stops_free_motion_after_capture():
    scene = SceneV13(
        objects={"red": np.array([-0.45, -0.40]), "blue": np.array([0.0, 0.45]), "yellow": np.array([0.45, -0.35])},
        robot_start=np.array([-0.45, -0.40]),
        zones={"red": np.array([-0.65, 0.65]), "blue": np.array([0.0, 0.0]), "yellow": np.array([0.65, -0.65])},
    )
    env = DynamicBothEnvV15(scene, "red", "blue", zone_velocity=(0.01, 0.0), block_velocity=(0.0, 0.01))
    env.holding = True
    env.phase = "MOVE_TO_ZONE"
    env.gripper_open = 0.0
    before = env.target_object.copy()
    env.step(np.zeros(3, dtype=np.float32))
    assert np.allclose(env.target_object, before)


def test_v15c_reset_restores_target_block_velocity():
    scene = SceneV13(
        objects={"red": np.array([-0.45, -0.40]), "blue": np.array([0.0, 0.45]), "yellow": np.array([0.45, -0.35])},
        robot_start=np.array([0.0, 0.0]),
        zones={"red": np.array([-0.65, 0.65]), "blue": np.array([0.0, 0.0]), "yellow": np.array([0.65, -0.65])},
    )
    env = DynamicBothEnvV15(scene, "red", "blue", zone_velocity=(0.01, 0.0), block_velocity=(0.0, 0.01))
    env.step(np.zeros(3, dtype=np.float32))
    env.reset()
    assert np.allclose(env.target_object, scene.objects["red"])
    assert np.allclose(env.block_velocity["red"], [0.0, 0.01])
