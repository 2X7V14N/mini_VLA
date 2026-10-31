"""把录制下来的 .npz episode 读成训练样本。

★ 输出的结构【故意和 V2.0 的 reach_env.generate_episode 完全一致】：
      list[episode]，每个 episode 是 list[dict]，
      dict 的 key = image / instruction / robot_state / action_chunk
  这样 RobotDataset、split_episodes、flatten 一行都不用改。

★ 唯一的转换：录下来的是【每 tick 一个 5 维动作】，训练要的是【动作块】。
      action_chunk[t] = action[t], action[t+1], ..., action[t+K-1]
  所以一条 T tick 的 episode 只能产出 T-K+1 条样本 —— 末尾 K-1 帧没有完整的
  未来动作，直接丢掉，不要补零。补零等于教模型"以后什么都不用做"。
"""

from pathlib import Path

import numpy as np


def load_episodes(npz_dir, chunk_size: int = 4, verbose: bool = False):
    """读取目录下所有 ep_*.npz。

    参数:
        npz_dir     含 ep_XXXX.npz 的目录（不要传 rejected_ 目录）
        chunk_size  动作块长度
    返回:
        list[list[dict]]，格式同 V2.0
    """
    files = sorted(Path(npz_dir).glob("ep_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"{npz_dir} 里没有 ep_*.npz。"
            f"（注意别指向 rejected_* 目录，那里只有诊断轨迹、没有图像）")

    episodes = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        image = z["image"]              # (T, 64, 64, 3) uint8
        action = z["action"]            # (T, 5)        float32
        state = z["robot_state"]        # (T, 3)        float32
        instruction = str(z["instruction"])

        T = len(action)
        if T < chunk_size:
            raise ValueError(f"{f.name} 只有 {T} tick，凑不出 {chunk_size} 的动作块")

        ep = []
        for t in range(T - chunk_size + 1):
            ep.append({
                "image": image[t],
                "instruction": instruction,
                "robot_state": state[t].astype(np.float32),
                "action_chunk": action[t:t + chunk_size].astype(np.float32),
            })
        episodes.append(ep)

        if verbose:
            print(f"  {f.name}  {T:>4} tick -> {len(ep):>4} 条样本")

    return episodes
