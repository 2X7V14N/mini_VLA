import numpy as np

from mini_vla.data.grasp_place_dynamic_expert_v15 import expert_action_v15
from mini_vla.data.grasp_place_dynamic_v15 import DynamicZoneEnvV15
from mini_vla.data.grasp_place_expert_v13 import COMBOS, sample_scene_v13


def test_dynamic_expert_completes_all_color_pairs():
    rng = np.random.default_rng(1234)
    scene = sample_scene_v13(rng)
    for block, zone in COMBOS:
        velocity = np.array([0.012, -0.006], dtype=np.float64)
        env = DynamicZoneEnvV15(scene, block, zone, zone_velocity=velocity)
        for _ in range(500):
            if env.done:
                break
            env.step(expert_action_v15(env, block))
        assert env.done, (block, zone, env.info())
        assert env.info()["success"], (block, zone, env.info())
