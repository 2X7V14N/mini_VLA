"""Frame dataset for V1.5c episodes with causal image history."""

from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


COLORS = ("red", "blue", "yellow")


class DynamicBothDatasetV15c(Dataset):
    def __init__(self, data_root, split, tokenizer, history=4, future_ticks=4,
                 cache_episodes=8, phase_metadata=False,
                 defer_image_validation=False):
        if history < 1 or future_ticks < 1:
            raise ValueError("history 和 future_ticks 必须为正整数")
        self.root = Path(data_root) / split
        self.tokenizer = tokenizer
        self.history = int(history)
        self.future_ticks = int(future_ticks)
        self.cache_episodes = max(1, int(cache_episodes))
        self.phase_metadata = bool(phase_metadata)
        self.defer_image_validation = bool(defer_image_validation)
        paths = sorted(self.root.glob("ep_*.npz"))
        if not paths:
            raise FileNotFoundError(f"没有找到 episode 文件：{self.root}")
        self.episodes, self.index, self._cache = [], [], OrderedDict()
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                required = {
                    "image", "instruction", "robot_state", "action_chunk",
                    "object_positions", "object_velocities", "zone_positions",
                    "zone_velocities", "exec_block", "exec_zone", "scene_id",
                }
                missing = required.difference(data.files)
                if missing:
                    raise ValueError(f"{path.name}: 缺少字段 {sorted(missing)}")
                # V1.5d has thousands of compressed episodes. Reading every
                # image solely to inspect its shape dominates startup time;
                # defer that check until the episode is actually decoded by a
                # worker. The default keeps V1.5c's eager validation intact.
                n = len(data["instruction"] if self.defer_image_validation else data["image"])
                if not self.defer_image_validation:
                    if data["image"].shape != (n, 64, 64, 3) or data["image"].dtype != np.uint8:
                        raise ValueError(f"{path.name}: image 必须为 (T,64,64,3) uint8")
                if data["robot_state"].shape != (n, 4) or data["action_chunk"].shape != (n, 4, 3):
                    raise ValueError(f"{path.name}: state/action 形状错误")
                if any(data[key].shape != (n, 3, 2) for key in (
                    "object_positions", "object_velocities", "zone_positions", "zone_velocities",
                )):
                    raise ValueError(f"{path.name}: 动态标签形状错误")
                block, zone = str(data["exec_block"]), str(data["exec_zone"])
                if block not in COLORS or zone not in COLORS:
                    raise ValueError(f"{path.name}: 未知颜色组合")
                if len(data["instruction"]) != n:
                    raise ValueError(f"{path.name}: instruction 长度错误")
                phase = None
                if self.phase_metadata:
                    if "phase" not in data.files:
                        raise ValueError(f"{path.name}: 缺少 V1.5d 所需 phase 字段")
                    phase = data["phase"].astype(str)
                    if len(phase) != n:
                        raise ValueError(f"{path.name}: phase 长度错误")
                meta = {
                    "path": path, "n": n, "block": block, "zone": zone,
                    "scene_id": str(data["scene_id"]),
                }
                if phase is not None:
                    meta["phase"] = phase
            episode_id = len(self.episodes)
            self.episodes.append(meta)
            self.index.extend((episode_id, tick) for tick in range(n))
        self.episode_ranges = []
        offset = 0
        for episode in self.episodes:
            self.episode_ranges.append((offset, offset + episode["n"]))
            offset += episode["n"]

    def _load(self, episode_id):
        if episode_id in self._cache:
            result = self._cache.pop(episode_id)
            self._cache[episode_id] = result
            return result
        meta = self.episodes[episode_id]
        with np.load(meta["path"], allow_pickle=False) as data:
            block_i, zone_i = COLORS.index(meta["block"]), COLORS.index(meta["zone"])
            result = {
                "image": data["image"].copy(),
                "instruction": data["instruction"].astype(str),
                "state": data["robot_state"].astype(np.float32),
                "action": data["action_chunk"].astype(np.float32),
                "block_position": data["object_positions"][:, block_i].astype(np.float32),
                "block_velocity": data["object_velocities"][:, block_i].astype(np.float32),
                "zone_position": data["zone_positions"][:, zone_i].astype(np.float32),
                "zone_velocity": data["zone_velocities"][:, zone_i].astype(np.float32),
                "scene_id": meta["scene_id"],
            }
        if result["image"].shape != (meta["n"], 64, 64, 3) or result["image"].dtype != np.uint8:
            raise ValueError(f"{meta['path'].name}: image 必须为 (T,64,64,3) uint8")
        self._cache[episode_id] = result
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return result

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode_id, tick = self.index[index]
        episode = self._load(episode_id)
        start = max(0, tick - self.history + 1)
        history = episode["image"][start:tick + 1]
        if len(history) < self.history:
            history = np.concatenate([
                np.repeat(history[:1], self.history - len(history), axis=0), history
            ], axis=0)
        future_tick = min(tick + self.future_ticks, len(episode["image"]) - 1)
        return {
            "image_history": torch.from_numpy(history.copy()).permute(0, 3, 1, 2).float() / 255.0,
            "token_ids": self.tokenizer.batch_encode([episode["instruction"][tick]])[0],
            "robot_state": torch.from_numpy(episode["state"][tick].copy()),
            "action_chunk": torch.from_numpy(episode["action"][tick].copy()),
            "block_position": torch.from_numpy(episode["block_position"][tick].copy()),
            "block_velocity": torch.from_numpy(episode["block_velocity"][tick].copy()),
            "zone_position": torch.from_numpy(episode["zone_position"][tick].copy()),
            "zone_velocity": torch.from_numpy(episode["zone_velocity"][tick].copy()),
            "future_zone_position": torch.from_numpy(episode["zone_position"][future_tick].copy()),
            "scene_id": episode["scene_id"],
        }


class V15cShuffleBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, seed=0):
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, int(batch_size), int(seed), 0
        if self.batch_size < 1:
            raise ValueError("batch_size 必须大于 0")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = []
        for episode_id in rng.permutation(len(self.dataset.episodes)):
            start, end = self.dataset.episode_ranges[episode_id]
            local = np.arange(start, end)
            rng.shuffle(local)
            order.extend(local.tolist())
        for start in range(0, len(order), self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
