import numpy as np
import pytest

from mini_vla.data.split import split_episodes, flatten


def make_episodes(n_episodes, frames_per_episode=3):
    """造假数据。第 i 个 episode 的每个样本都带 i 这个标记。

    为什么要打标记：这样能精确追踪"哪个 episode 去了哪里"，
    比用 id() 或内存地址可靠得多。
    """
    return [
        [{"episode_id": i, "frame": j} for j in range(frames_per_episode)]
        for i in range(n_episodes)
    ]


def episode_ids(episodes):
    """取出一组 episode 的编号集合。"""
    return {ep[0]["episode_id"] for ep in episodes}

def test_split_counts():
    eps = make_episodes(10)
    train, val, test = split_episodes(eps, train_ratio=0.8, val_ratio=0.1)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1

def test_no_overlap_and_no_loss():
    """每个 episode 必须恰好出现在一个集合里。"""
    eps = make_episodes(10)
    train, val, test = split_episodes(eps)

    tr, va, te = episode_ids(train), episode_ids(val), episode_ids(test)

    # 两两不相交
    assert tr & va == set(), "训练集和验证集有重叠"
    assert tr & te == set(), "训练集和测试集有重叠"
    assert va & te == set(), "验证集和测试集有重叠"

    # 合起来正好是全部
    assert tr | va | te == episode_ids(eps), "有 episode 丢失了"

def test_split_is_reproducible():
    eps = make_episodes(20)
    a = split_episodes(eps, seed=7)
    b = split_episodes(eps, seed=7)
    for x, y in zip(a, b):
        assert episode_ids(x) == episode_ids(y)

    # 换个 seed 应该不同
    c = split_episodes(eps, seed=8)
    assert episode_ids(a[0]) != episode_ids(c[0])

def test_input_not_modified():
    eps = make_episodes(10)
    before = [ep[0]["episode_id"] for ep in eps]      # 记录原始顺序

    split_episodes(eps)

    after = [ep[0]["episode_id"] for ep in eps]
    assert before == after, "输入的 episodes 被修改了"

def test_too_few_episodes_raises():
    eps = make_episodes(3)          # 3 个按 80/10/10 -> train=2, val=0, test=1
    with pytest.raises(ValueError):
        split_episodes(eps)

def test_flatten():
    eps = make_episodes(3, frames_per_episode=4)
    flat = flatten(eps)
    assert len(flat) == 12
    assert isinstance(flat[0], dict)

