"""验证 expert_chunk。"""

import numpy as np

from mini_vla.simulation.reach_env import expert_chunk


def main():
    current = np.array([0.0, 0.0, 0.15])      # 末端当前位置
    target  = np.array([0.1, 0.1, 0.15])      # 目标

    chunk = expert_chunk(current, target)
    print(f"形状: {chunk.shape}   dtype: {chunk.dtype}")
    print(f"值域: [{chunk.min():.3f}, {chunk.max():.3f}]")
    print("\n4 步动作:")
    for i, a in enumerate(chunk):
        print(f"  第 {i} 步: {a.round(4).tolist()}")

    # 反复应用第一步，能到达吗
    pos, n = current.copy(), 0
    for n in range(20):
        c = expert_chunk(pos, target)
        pos = pos + 0.05 * c[0]
        if np.linalg.norm(pos - target) < 1e-6:
            break
    print(f"\n{n+1} 步到达   最终位置 {pos.round(6).tolist()}")


if __name__ == "__main__":
    main()
