import numpy as np
import torch

from mini_vla.data.dynamic_dataset_v15c import DynamicBothDatasetV15c
from mini_vla.models.text import SimpleCharTokenizer


def test_v15c_dataset_builds_history_and_future_zone_target(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    prompt, count = "把红色方块放到蓝色放置区", 5
    zones = np.zeros((count, 3, 2), dtype=np.float32)
    zones[:, 1, 0] = np.arange(count, dtype=np.float32) * 0.01
    np.savez_compressed(
        split / "ep_0000_00.npz",
        image=np.zeros((count, 64, 64, 3), dtype=np.uint8),
        instruction=np.asarray([prompt] * count),
        robot_state=np.zeros((count, 4), dtype=np.float32),
        action_chunk=np.zeros((count, 4, 3), dtype=np.float32),
        object_positions=np.zeros((count, 3, 2), dtype=np.float32),
        object_velocities=np.zeros((count, 3, 2), dtype=np.float32),
        zone_positions=zones, zone_velocities=np.zeros((count, 3, 2), dtype=np.float32),
        exec_block=np.asarray("red"), exec_zone=np.asarray("blue"), scene_id=np.asarray("train_0000"),
    )
    tokenizer = SimpleCharTokenizer(sorted(set(prompt)), max_length=32)
    dataset = DynamicBothDatasetV15c(tmp_path, "train", tokenizer, history=4, future_ticks=3)
    item = dataset[1]
    assert item["image_history"].shape == (4, 3, 64, 64)
    assert item["action_chunk"].shape == (4, 3)
    assert torch.allclose(item["future_zone_position"], torch.tensor([0.04, 0.0]))
