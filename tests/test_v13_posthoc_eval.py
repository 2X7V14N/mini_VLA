import numpy as np

from scripts.v1_3.eval_v13_closed_loop import placement_success
from mini_vla.data.grasp_place import ZONE_HALF
from mini_vla.data.v13_eval_standards import OFFICIAL_V13


def test_official_standard_uses_30cm_scoring_square_without_changing_environment():
    assert OFFICIAL_V13.zone_half == 0.15
    assert OFFICIAL_V13.zone_half > ZONE_HALF

    target = np.array([0.0, 0.0])

    assert placement_success([0.14, 0.14], target, True, OFFICIAL_V13.zone_half)
    assert not placement_success([0.151, 0.0], target, True, OFFICIAL_V13.zone_half)
    assert not placement_success([0.0, 0.0], target, False, OFFICIAL_V13.zone_half)
