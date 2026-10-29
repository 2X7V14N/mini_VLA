import numpy as np
import pytest

from mini_vla.data.grasp_place_dynamic_v15 import (
    MAX_ZONE_SPEED,
    ZONE_MOTION_BOUND,
    DynamicZoneEnvV15,
)
from mini_vla.data.grasp_place_v13 import SceneV13


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
            "blue": np.array([0.00, 0.00]),
            "yellow": np.array([0.65, -0.65]),
        },
    )


def make_env(scene, moving_zone="blue", velocity=(0.01, 0.0), **kwargs):
    return DynamicZoneEnvV15(
        scene, "red", "blue",
        moving_zone=moving_zone,
        zone_velocity=velocity,
        **kwargs,
    )


def test_moving_zone_advances_by_velocity(scene):
    env = make_env(scene, velocity=(0.01, 0.0))
    before = {key: value.copy() for key, value in env.zones.items()}

    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert np.allclose(env.zones["blue"] - before["blue"], [0.01, 0.0])
    assert np.allclose(env.zones["red"], before["red"])
    assert np.allclose(env.zones["yellow"], before["yellow"])


def test_target_zone_tracks_moving_zone(scene):
    env = make_env(scene, moving_zone="blue", velocity=(0.012, 0.0))
    assert env.target_zone is env.zones["blue"]

    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.isclose(env.target_zone[0], 0.012)


def test_zone_bounces_at_bound(scene):
    env = make_env(scene, velocity=(MAX_ZONE_SPEED, 0.0))
    env.zones["blue"][:] = [ZONE_MOTION_BOUND - MAX_ZONE_SPEED, 0.0]

    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert np.isclose(env.zones["blue"][0], ZONE_MOTION_BOUND)
    assert np.isclose(env.zone_velocities["blue"][0], -MAX_ZONE_SPEED)

    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert np.isclose(env.zones["blue"][0],
                      ZONE_MOTION_BOUND - MAX_ZONE_SPEED)
    assert np.isclose(env.zone_velocities["blue"][0], -MAX_ZONE_SPEED)


def test_held_object_follows_robot_while_zone_moves(scene):
    env = make_env(scene, velocity=(0.01, 0.0))
    env.zones["blue"][:] = [0.5, 0.5]
    env.phase = "MOVE_TO_ZONE"
    env.holding = True
    env.gripper_open = 0.0
    env.cmd = -1
    robot_before = env.robot.copy()
    object_before = env.target_object.copy()
    zone_before = env.target_zone.copy()

    env.step(np.array([0.5, 0.0, -1.0], dtype=np.float32))

    assert np.allclose(env.target_object - object_before,
                       env.robot - robot_before)
    assert np.allclose(env.target_zone - zone_before, [0.01, 0.0])


def test_observation_has_four_dimensions(scene):
    env = make_env(scene)
    state = env.observation()["robot_state"]
    assert state.shape == (4,)
    assert state[2] == 0.0
    assert state[3] == 1.0


def test_reset_restores_zone_positions_and_velocity(scene):
    env = make_env(scene, velocity=(0.01, 0.01))
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    env.reset()

    assert np.allclose(env.zones["blue"], scene.zones["blue"])
    assert np.allclose(env.zone_velocities["blue"], [0.01, 0.01])
    assert np.allclose(env.zone_velocities["red"], [0.0, 0.0])
    assert np.allclose(env.robot, scene.robot_start)


def test_static_zones_never_move(scene):
    env = make_env(scene, moving_zone="blue", velocity=(0.01, 0.0))
    before = {key: value.copy() for key, value in env.zones.items()}

    for _ in range(20):
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))

    assert np.allclose(env.zones["red"], before["red"])
    assert np.allclose(env.zones["yellow"], before["yellow"])


def test_zone_stays_in_bounds_over_many_steps(scene):
    env = make_env(
        scene,
        velocity=(MAX_ZONE_SPEED, MAX_ZONE_SPEED),
        phase_max_ticks={
            "MOVE_TO_OBJECT": 10000,
            "CLOSE": 10000,
            "MOVE_TO_ZONE": 10000,
            "OPEN": 10000,
        },
    )

    for _ in range(500):
        env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        assert np.all(np.isfinite(env.zones["blue"]))
        assert np.all(np.abs(env.zones["blue"])
                      <= ZONE_MOTION_BOUND + 1e-12)


def test_info_reports_zone_motion(scene):
    env = make_env(scene, velocity=(0.01, 0.0))
    env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    info = env.info()
    assert info["moving_zone"] == "blue"
    assert np.allclose(info["zone_velocities"]["blue"], [0.01, 0.0])
    assert "zone_positions" in info


def test_rejects_unknown_moving_zone(scene):
    with pytest.raises(ValueError):
        make_env(scene, moving_zone="green")


def test_rejects_bad_velocity(scene):
    with pytest.raises(ValueError):
        make_env(scene, velocity=(0.01,))
    with pytest.raises(ValueError):
        make_env(scene, velocity=(np.nan, 0.0))
