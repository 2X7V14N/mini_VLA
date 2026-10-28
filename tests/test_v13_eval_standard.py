import numpy as np

from mini_vla.data.grasp_place import ZONE_HALF
from mini_vla.data.grasp_place_v13 import GraspPlaceEnvV13, SceneV13
from mini_vla.data.v13_eval_standards import OFFICIAL_V13, V13_EVAL_STANDARDS


def test_official_is_the_only_v13_evaluation_standard():
    assert tuple(V13_EVAL_STANDARDS) == ("official",)
    assert OFFICIAL_V13.close_enter_tol == 0.085
    assert OFFICIAL_V13.zone_tol == 0.106
    assert OFFICIAL_V13.total_max_ticks == 1950
    assert OFFICIAL_V13.zone_half * 2 == 0.30


def test_v13_environment_accepts_official_standard():
    scene = SceneV13(
        objects={
            "red": np.array([-.4, -.4]),
            "blue": np.array([0., .4]),
            "yellow": np.array([.4, -.4]),
        },
        robot_start=np.array([0., 0.]),
        zones={
            "red": np.array([-.7, .7]),
            "blue": np.array([.7, .7]),
            "yellow": np.array([0., -.7]),
        },
    )
    env = GraspPlaceEnvV13(
        scene, "yellow", "yellow",
        close_enter_tol=OFFICIAL_V13.close_enter_tol,
        zone_tol=OFFICIAL_V13.zone_tol,
        phase_max_ticks=OFFICIAL_V13.phase_max_ticks,
    )
    assert env.zone_tol == OFFICIAL_V13.zone_tol
    assert env.phase_max_ticks == OFFICIAL_V13.phase_max_ticks
    assert ZONE_HALF == 0.11


