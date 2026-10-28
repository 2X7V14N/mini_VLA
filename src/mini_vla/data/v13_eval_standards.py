"""V1.3 闭环评测标准。"""

from dataclasses import dataclass

from mini_vla.data.grasp_place import (
    CAPTURE_RADIUS,
    ZONE_HALF,
)


@dataclass(frozen=True)
class V13EvalStandard:
    name: str
    close_enter_tol: float
    zone_tol: float
    phase_max_ticks: dict[str, int]
    total_max_ticks: int
    # Final placement scoring half-width; it does not change the rollout environment.
    zone_half: float = ZONE_HALF

    def as_dict(self):
        return {
            "name": self.name,
            "close_enter_tol": self.close_enter_tol,
            "zone_tol": self.zone_tol,
            "phase_max_ticks": dict(self.phase_max_ticks),
            "total_max_ticks": self.total_max_ticks,
            "capture_radius": CAPTURE_RADIUS,
            "zone_half": self.zone_half,
            "placement_zone_side_length": 2 * self.zone_half,
        }


OFFICIAL_V13 = V13EvalStandard(
    name="official",
    close_enter_tol=0.085,
    zone_tol=0.106,
    phase_max_ticks={
        "MOVE_TO_OBJECT": 260,
        "CLOSE": 28,
        "MOVE_TO_ZONE": 400,
        "OPEN": 28,
    },
    total_max_ticks=1950,
    zone_half=0.15,
)


V13_EVAL_STANDARDS = {OFFICIAL_V13.name: OFFICIAL_V13}

