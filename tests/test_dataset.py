import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mini_vla.data.dataset import RobotDataset
from mini_vla.data.synthetic import generate_episode
from mini_vla.models.text import SimpleCharTokenizer


def make_dataset(n_episodes=3):
    """生成几条 episode，拼成 Dataset。"""
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(n_episodes):
        samples.extend(generate_episode(rng))

    # 词表从数据里【自动收集】，而不是手写
    vocab = sorted(set("".join(s["instruction"] for s in samples)))
    tokenizer = SimpleCharTokenizer(vocab, max_length=16)
    return RobotDataset(samples, tokenizer), tokenizer

def test_single_sample_format():
    ds, tokenizer = make_dataset()
    s = ds[0]

    assert set(s.keys()) == {"image", "token_ids", "robot_state", "action_chunk"}

    # 图像
    assert s["image"].shape == (3, 64, 64)
    assert s["image"].dtype == torch.float32
    assert float(s["image"].min()) >= 0.0
    assert float(s["image"].max()) <= 1.0

    # token
    assert s["token_ids"].shape == (16,)
    assert s["token_ids"].dtype == torch.long

    # 状态
    assert s["robot_state"].shape == (2,)
    assert s["robot_state"].dtype == torch.float32

    # 动作
    assert s["action_chunk"].shape == (4, 2)
    assert s["action_chunk"].dtype == torch.float32
    assert float(s["action_chunk"].min()) >= -1.0
    assert float(s["action_chunk"].max()) <= 1.0

def test_dataloader_batch_shape():
    ds, _ = make_dataset()
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    assert batch["image"].shape == (8, 3, 64, 64)
    assert batch["token_ids"].shape == (8, 16)
    assert batch["robot_state"].shape == (8, 2)
    assert batch["action_chunk"].shape == (8, 4, 2)

def test_no_unexpected_unk():
    """指令里的字符必须全在词表里，不能出现 UNK。

    抓的是"词表没覆盖全部字符"——那样指令会变成一串 <unk>，
    模型看不到任何语言信息，而且不报错。
    """
    ds, tokenizer = make_dataset()

    for i in range(len(ds)):
        ids = ds[i]["token_ids"]
        # 真实的字符位置（前 7 个）不应该出现 unk_id
        real = ids[:7]
        assert tokenizer.unk_id not in real.tolist(), \
            f"第 {i} 条样本出现了 UNK: {real.tolist()}"
