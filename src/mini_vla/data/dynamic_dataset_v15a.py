"""Dataset for V1.5a moving-zone episodes.

Each item is one frame from one episode.  A causal four-frame image window is
returned; frames before the beginning of an episode are replicated, so scene
boundaries never leak into one another.
"""

from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class DynamicZoneDatasetV15a(Dataset):
    def __init__(self, data_root, split, tokenizer, history=4, cache_episodes=8):
        if history < 1:
            raise ValueError("history 必须为正整数")
        self.root = Path(data_root) / split
        self.tokenizer = tokenizer
        self.history = int(history)
        self.cache_episodes = max(1, int(cache_episodes))
        paths = sorted(self.root.glob("ep_*.npz"))
        if not paths:
            raise FileNotFoundError(f"没有找到 episode 文件：{self.root}")

        self.episodes = []
        self._cache = OrderedDict()
        self.index = []
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                required = {
                    "image", "instruction", "robot_state", "action_chunk",
                    "phase", "zone_positions", "zone_velocities",
                    "exec_zone", "scene_id",
                }
                missing = required.difference(z.files)
                if missing:
                    raise ValueError(f"{path.name}: 缺少字段 {sorted(missing)}")
                image = z["image"]
                n = len(image)
                if image.shape != (n, 64, 64, 3) or image.dtype != np.uint8:
                    raise ValueError(f"{path.name}: image 必须为 (T,64,64,3) uint8")
                action = z["action_chunk"].astype(np.float32, copy=True)
                if action.shape != (n, 4, 3):
                    raise ValueError(f"{path.name}: action_chunk 必须为 (T,4,3)")
                state = z["robot_state"].astype(np.float32, copy=True)
                if state.shape != (n, 4):
                    raise ValueError(f"{path.name}: robot_state 必须为 (T,4)")
                zone_pos = z["zone_positions"].astype(np.float32, copy=True)
                zone_vel = z["zone_velocities"].astype(np.float32, copy=True)
                if zone_pos.shape != (n, 3, 2) or zone_vel.shape != (n, 3, 2):
                    raise ValueError(f"{path.name}: zone 标签形状错误")
                zone = str(z["exec_zone"])
                colors = ["red", "blue", "yellow"]
                if zone not in colors:
                    raise ValueError(f"{path.name}: 未知 exec_zone={zone}")
                episode = {
                    "path": path,
                    "n": n,
                    "scene_id": str(z["scene_id"]),
                    "exec_block": str(z["exec_block"]),
                    "exec_zone": zone,
                }
                if not (len(z["instruction"]) == n
                        and len(z["phase"]) == n):
                    raise ValueError(f"{path.name}: 字段长度不一致")
            episode_id = len(self.episodes)
            self.episodes.append(episode)
            self.index.extend((episode_id, t) for t in range(n))
        self.episode_ranges = []
        cursor = 0
        for episode in self.episodes:
            self.episode_ranges.append((cursor, cursor + episode["n"]))
            cursor += episode["n"]

    def _load_episode(self, episode_id):
        if episode_id in self._cache:
            episode = self._cache.pop(episode_id)
            self._cache[episode_id] = episode
            return episode
        meta = self.episodes[episode_id]
        with np.load(meta["path"], allow_pickle=False) as z:
            colors = ["red", "blue", "yellow"]
            zone_index = colors.index(meta["exec_zone"])
            episode = {
                "image": z["image"].copy(),
                "instruction": z["instruction"].astype(str),
                "state": z["robot_state"].astype(np.float32),
                "action": z["action_chunk"].astype(np.float32),
                "phase": z["phase"].astype(str),
                "zone_position": z["zone_positions"][:, zone_index].astype(np.float32),
                "zone_velocity": z["zone_velocities"][:, zone_index].astype(np.float32),
                "scene_id": meta["scene_id"],
            }
        self._cache[episode_id] = episode
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)
        return episode

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.index):
            raise IndexError(index)
        episode_id, t = self.index[index]
        ep = self._load_episode(episode_id)
        start = max(0, t - self.history + 1)
        history = ep["image"][start:t + 1]
        if len(history) < self.history:
            history = np.concatenate(
                [np.repeat(history[:1], self.history - len(history), axis=0),
                 history], axis=0)
        images = torch.from_numpy(history.copy()).permute(0, 3, 1, 2).float() / 255.0
        token_ids = self.tokenizer.batch_encode([ep["instruction"][t]])[0]
        return {
            "image_history": images,
            "token_ids": token_ids,
            "robot_state": torch.from_numpy(ep["state"][t].copy()),
            "action_chunk": torch.from_numpy(ep["action"][t].copy()),
            "zone_position": torch.from_numpy(ep["zone_position"][t].copy()),
            "zone_velocity": torch.from_numpy(ep["zone_velocity"][t].copy()),
            "phase": ep["phase"][t],
            "scene_id": ep["scene_id"],
        }


class V15aShuffleBatchSampler(Sampler):
    """Scene-safe sampler; items are frames, but episode windows stay local."""
    def __init__(self, dataset, batch_size, seed=0):
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        episode_order = rng.permutation(len(self.dataset.episodes)).tolist()
        order = []
        for episode_id in episode_order:
            start, end = self.dataset.episode_ranges[episode_id]
            local = np.arange(start, end)
            rng.shuffle(local)
            order.extend(local.tolist())
        for start in range(0, len(order), self.batch_size):
            yield order[start:start + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
