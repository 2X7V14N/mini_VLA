"""Small deterministic smoke check for the V1.5a dynamic environment."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_dynamic_v15 import DynamicZoneEnvV15  # noqa: E402
from mini_vla.data.grasp_place_expert_v13 import sample_scene_v13  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(20261027)
    checks = 0
    for _ in range(args.scenes):
        scene = sample_scene_v13(rng)
        env = DynamicZoneEnvV15(scene, "red", "blue", zone_velocity=(0.01, 0.0))
        before = env.zones["blue"].copy()
        env.step(np.zeros(3, dtype=np.float32))
        assert np.allclose(env.zones["blue"] - before, [0.01, 0.0])
        assert np.allclose(env.zones["red"], scene.zones["red"])
        assert np.allclose(env.zones["yellow"], scene.zones["yellow"])
        checks += 3
    print(f"[OK] V1.5a environment checks={checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
