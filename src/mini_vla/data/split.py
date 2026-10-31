import numpy as np


def split_episodes(episodes, train_ratio=0.8, val_ratio=0.1, seed=42):
    """把 episodes 划分为训练/验证/测试三组。

    参数:
        episodes: list[list[dict]]，外层是 episode，内层是该 episode 的样本
        train_ratio, val_ratio: 前两者的比例，测试集拿剩下的
        seed: 随机种子，保证可复现

    返回:
        (train_episodes, val_episodes, test_episodes)

    注意:
        划分的单位是【episode】，不是单帧样本。
        同一个 episode 的所有帧必须待在同一个集合里。
    """
    n = len(episodes)

    # ① 算出每组多少个 episode
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    # ② 边界检查
    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(
            f"{n} 个 episode 无法按 {train_ratio}/{val_ratio} 划分："
            f"train={n_train}, val={n_val}, test={n_test}。"
            f"存在空集合，请增加 episode 数量。"
        )


    # ③ 打乱 episode 的索引
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)


    # ④ 按顺序切片
    train = [episodes[i] for i in order[:n_train]]
    val   = [episodes[i] for i in order[n_train:n_train + n_val]]
    test  = [episodes[i] for i in order[n_train + n_val:]]

    return train, val, test


def flatten(episodes):
    """把 list[list[sample]] 摊平成 list[sample]。"""
    return [s for ep in episodes for s in ep]
