import numpy as np
import pytest

from mini_vla.data.grasp_place_v13 import (
    COLORS,
    GraspPlaceEnvV13,
    SceneV13,
)


@pytest.fixture
def scene():
    return SceneV13(
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


def test_v13_environment_accepts_all_nine_instruction_pairs(scene):
    for block in COLORS:
        for zone in COLORS:
            env = GraspPlaceEnvV13(scene, block, zone)
            assert env.color == block
            assert env.zone_color == zone
            assert env.phase == "MOVE_TO_OBJECT"
            assert env.target_zone is env.zones[zone]


def test_v13_environment_moves_held_object_with_robot(scene):
    env = GraspPlaceEnvV13(scene, "yellow", "blue")
    env.holding = True
    before_robot = env.robot.copy()
    before_object = env.target_object.copy()

    env.step(np.array([0.5, 0.0, 0.0], dtype=np.float32))

    assert np.allclose(env.target_object - before_object, env.robot - before_robot)


def test_v13_environment_rejects_unknown_color(scene):
    with pytest.raises(ValueError):
        GraspPlaceEnvV13(scene, "green", "red")