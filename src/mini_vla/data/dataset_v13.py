"""V1.3 数据读取：每个样本是一组同观测下的 9 种指令标签。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


COLORS = ("red", "blue", "yellow")
EXPECTED_COMBOS = [
    (block, zone)
    for block in COLORS
    for zone in COLORS
]
PHASE_IDS = {name: index for index, name in enumerate(
    ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")
)}


class V13GroupedDataset(Dataset):
    """每个索引对应一个 tick，而不是单条指令样本。

    返回：
      image:       (3, 64, 64)
      token_ids:   (9, max_length)
      robot_state: (4,) = x, y, holding, gripper_open
      action_chunk:(9, 4, 3)
      valid_mask:  (9,)；语义和物理状态一致的标签为 True
    """

    def __init__(self, data_root, split, tokenizer):
        self.data_root = Path(data_root)
        self.split = split
        self.tokenizer = tokenizer
        split_dir = self.data_root / split
        paths = sorted(split_dir.glob("ep_*.npz"))
        if not paths:
            raise FileNotFoundError(f"没有找到 episode 文件：{split_dir}")

        self.episodes = []
        self.cumulative_ticks = [0]
        self.balance_buckets = []

        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if "valid_combo_mask" not in data:
                    raise ValueError(
                        f"{path.name}: 缺少 valid_combo_mask；"
                        "这是旧版 V1.3 数据，不能用于修复后的训练。"
                    )
                n = len(data["instruction"])
                if n == 0 or n % 9 != 0:
                    raise ValueError(f"{path.name}: 样本数不是 9 的倍数")
                ticks = n // 9

                combo_ids = data["combo_index"].reshape(ticks, 9)
                expected_ids = np.tile(np.arange(9), (ticks, 1))
                if not np.array_equal(combo_ids, expected_ids):
                    raise ValueError(f"{path.name}: 9 种组合顺序不正确")

                images = data["image"].reshape(ticks, 9, 64, 64, 3)
                states = data["robot_state"].reshape(ticks, 9, 4)
                instructions = data["instruction"].reshape(ticks, 9)
                actions = data["action_chunk"].reshape(ticks, 9, 4, 3)
                valid_masks = data["valid_combo_mask"].reshape(ticks, 9)
                valid_masks = valid_masks.astype(bool, copy=False)
                if not np.all(valid_masks.any(axis=1)):
                    raise ValueError(f"{path.name}: 存在没有有效标签的 tick")

                # 同一 tick 的 9 条样本必须共享观测和机器人状态。
                if not np.all(images == images[:, :1]):
                    raise ValueError(f"{path.name}: 同组图像不一致")
                if not np.allclose(states, states[:, :1]):
                    raise ValueError(f"{path.name}: 同组机器人状态不一致")

                episode = {
                    # 只保留每组第一张图，避免在内存里存 9 份重复图像。
                    "image": images[:, 0].copy(),
                    "state": states[:, 0].copy(),
                    "instruction": instructions.astype(str).copy(),
                    "action": actions.copy(),
                    "valid_mask": valid_masks.copy(),
                }
                if "phase" in data and "balance_source" in data:
                    phases = data["phase"].reshape(ticks, 9)[:, 0].astype(str)
                    sources = data["balance_source"].reshape(ticks, 9)[:, 0].astype(int)
                    active_combos = valid_masks.argmax(axis=1)
                    try:
                        phase_ids = np.asarray([PHASE_IDS[p] for p in phases])
                    except KeyError as exc:
                        raise ValueError(f"{path.name}: 未知阶段 {exc.args[0]}") from exc
                    if np.any(valid_masks.sum(axis=1) != 1):
                        raise ValueError(
                            f"{path.name}: balanced mixture requires exactly one valid instruction per tick"
                        )
                    if np.any((sources < 0) | (sources > 1)):
                        raise ValueError(f"{path.name}: balance_source must be 0 or 1")
                    self.balance_buckets.extend(
                        (sources * 36 + phase_ids * 9 + active_combos).tolist()
                    )
                else:
                    self.balance_buckets.extend([-1] * ticks)

            self.episodes.append(episode)
            self.cumulative_ticks.append(
                self.cumulative_ticks[-1] + ticks
            )

        self.n_ticks = self.cumulative_ticks[-1]

    def __len__(self):
        return self.n_ticks

    def __getitem__(self, index):
        if index < 0 or index >= self.n_ticks:
            raise IndexError(index)

        # 找到全局 tick 属于哪个 episode。
        episode_index = 0
        while self.cumulative_ticks[episode_index + 1] <= index:
            episode_index += 1

        local_tick = index - self.cumulative_ticks[episode_index]
        episode = self.episodes[episode_index]

        image = torch.from_numpy(
            episode["image"][local_tick]
        ).permute(2, 0, 1).float() / 255.0

        instructions = episode["instruction"][local_tick].tolist()
        token_ids = self.tokenizer.batch_encode(instructions)

        state = torch.from_numpy(episode["state"][local_tick]).float()
        actions = torch.from_numpy(
            episode["action"][local_tick]
        ).float()
        valid_mask = torch.from_numpy(
            episode["valid_mask"][local_tick]
        ).bool()

        return {
            "image": image,
            "token_ids": token_ids,
            "robot_state": state,
            "action_chunk": actions,
            "valid_mask": valid_mask,
        }


class V13TickBatchSampler(Sampler):
    """打乱 tick 顺序，但每个 tick 的九种组合始终留在同一批次。"""

    def __init__(self, n_ticks, ticks_per_batch, seed=0):
        if n_ticks <= 0:
            raise ValueError("n_ticks 必须大于 0")
        if ticks_per_batch <= 0:
            raise ValueError("ticks_per_batch 必须大于 0")

        self.n_ticks = n_ticks
        self.ticks_per_batch = ticks_per_batch
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(self.n_ticks)

        for start in range(0, self.n_ticks, self.ticks_per_batch):
            yield order[start:start + self.ticks_per_batch].tolist()

    def __len__(self):
        return (
            self.n_ticks + self.ticks_per_batch - 1
        ) // self.ticks_per_batch


class V13BalancedMixtureBatchSampler(Sampler):
    """Sample across source, phase, and active combo with optional emphasis."""

    def __init__(
        self, dataset, ticks_per_batch, samples_per_bucket=32, seed=0,
        open_correction_multiplier=1, yellow_combo_multiplier=1,
        old_replay_multiplier=1, correction_multiplier=1,
        close_correction_multiplier=1, move_to_zone_correction_multiplier=1,
    ):
        if ticks_per_batch <= 0 or samples_per_bucket <= 0:
            raise ValueError("batch 和每个 bucket 的 tick 数必须大于 0")
        if open_correction_multiplier < 1:
            raise ValueError("OPEN correction multiplier 必须至少为 1")
        if yellow_combo_multiplier < 1:
            raise ValueError("yellow combo multiplier 必须至少为 1")
        if old_replay_multiplier < 1 or correction_multiplier < 1:
            raise ValueError("source multiplier 必须至少为 1")
        if close_correction_multiplier < 1 or move_to_zone_correction_multiplier < 1:
            raise ValueError("阶段 correction multiplier 必须至少为 1")
        if len(dataset.balance_buckets) != len(dataset):
            raise ValueError("dataset 缺少 balanced mixture metadata")
        self.ticks_per_batch = ticks_per_batch
        self.samples_per_bucket = samples_per_bucket
        self.open_correction_multiplier = open_correction_multiplier
        self.yellow_combo_multiplier = yellow_combo_multiplier
        self.old_replay_multiplier = old_replay_multiplier
        self.correction_multiplier = correction_multiplier
        self.close_correction_multiplier = close_correction_multiplier
        self.move_to_zone_correction_multiplier = move_to_zone_correction_multiplier
        self.seed = seed
        self.epoch = 0
        self.groups = [[] for _ in range(72)]
        for index, bucket in enumerate(dataset.balance_buckets):
            if bucket >= 0:
                self.groups[bucket].append(index)
        missing_old = [index for index in range(36) if not self.groups[index]]
        if missing_old:
            raise ValueError(
                f"balanced mixture 缺少旧示范分层 bucket：{missing_old}"
            )
        self.active_buckets = [
            index for index, group in enumerate(self.groups) if group
        ]
        if not any(index >= 36 for index in self.active_buckets):
            raise ValueError("balanced mixture 至少需要一个纠错分层 bucket")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = []
        for bucket in self.active_buckets:
            group = self.groups[bucket]
            draws = self.samples_per_bucket
            source_phase_combo = bucket % 36
            phase_id, combo_id = divmod(source_phase_combo, 9)
            source_id = bucket // 36
            if source_id == 0:
                draws *= self.old_replay_multiplier
            else:
                draws *= self.correction_multiplier
            if source_id == 1 and phase_id == 3:
                draws *= self.open_correction_multiplier
            if source_id == 1 and phase_id == 1:
                draws *= self.close_correction_multiplier
            if source_id == 1 and phase_id == 2:
                draws *= self.move_to_zone_correction_multiplier
            if combo_id in (2, 5, 6, 7, 8):
                draws *= self.yellow_combo_multiplier
            order.extend(
                rng.choice(group, size=draws, replace=True).tolist()
            )
        rng.shuffle(order)
        for start in range(0, len(order), self.ticks_per_batch):
            yield order[start:start + self.ticks_per_batch]

    def __len__(self):
        n = 0
        for bucket in self.active_buckets:
            source_phase_combo = bucket % 36
            phase_id, combo_id = divmod(source_phase_combo, 9)
            source_id = bucket // 36
            draws = self.samples_per_bucket
            draws *= (
                self.old_replay_multiplier
                if source_id == 0
                else self.correction_multiplier
            )
            if source_id == 1 and phase_id == 3:
                draws *= self.open_correction_multiplier
            if source_id == 1 and phase_id == 1:
                draws *= self.close_correction_multiplier
            if source_id == 1 and phase_id == 2:
                draws *= self.move_to_zone_correction_multiplier
            if combo_id in (2, 5, 6, 7, 8):
                draws *= self.yellow_combo_multiplier
            n += draws
        return (n + self.ticks_per_batch - 1) // self.ticks_per_batch
