import numpy as np
import torch

from mini_vla.data.dynamic_dataset_v15a import DynamicZoneDatasetV15a
from mini_vla.models.text import SimpleCharTokenizer


def test_dynamic_dataset_builds_causal_history_and_dynamic_targets(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    n = 3
    prompts = np.asarray(["把红色方块放到蓝色放置区"] * n)
    np.savez_compressed(
        split / "ep_0000_00.npz",
        image=np.zeros((n, 64, 64, 3), dtype=np.uint8),
        instruction=prompts,
        robot_state=np.zeros((n, 4), dtype=np.float32),
        action_chunk=np.zeros((n, 4, 3), dtype=np.float32),
        phase=np.asarray(["MOVE_TO_OBJECT"] * n),
        zone_positions=np.zeros((n, 3, 2), dtype=np.float32),
        zone_velocities=np.ones((n, 3, 2), dtype=np.float32),
        exec_zone=np.asarray("blue"), scene_id=np.asarray("train_0000"),
        exec_block=np.asarray("red"),
    )
    tok = SimpleCharTokenizer(sorted(set(prompts[0])), max_length=32)
    ds = DynamicZoneDatasetV15a(tmp_path, "train", tok, history=4)
    item = ds[0]
    assert item["image_history"].shape == (4, 3, 64, 64)
    assert item["action_chunk"].shape == (4, 3)
    assert item["zone_position"].shape == (2,)
    assert torch.allclose(item["zone_velocity"], torch.ones(2))
