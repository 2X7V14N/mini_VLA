"""验证 generate_episode。"""

import numpy as np

from mini_vla.simulation.reach_env import generate_episode


def main():
    rng = np.random.default_rng(0)

    ep = generate_episode(rng, paired=True)
    print(f"帧数: {len(ep)}  (配对后，每帧 2 条)")

    s = ep[0]
    print(f"\n第一条样本:")
    print(f"  image        {s['image'].shape} {s['image'].dtype}")
    print(f"  instruction  {s['instruction']!r}")
    print(f"  robot_state  {s['robot_state'].shape} {s['robot_state'].round(4).tolist()}")
    print(f"  action_chunk {s['action_chunk'].shape} {s['action_chunk'].dtype}")

    print(f"\n配对检查:")
    for i in range(0, min(6, len(ep)), 2):
        a, b = ep[i], ep[i + 1]
        same_img = np.array_equal(a["image"], b["image"])
        same_st = np.array_equal(a["robot_state"], b["robot_state"])
        diff_instr = a["instruction"] != b["instruction"]
        diff_act = not np.allclose(a["action_chunk"], b["action_chunk"])
        print(f"  第{i//2}对: 同图{same_img}  同状态{same_st}  "
              f"不同指令{diff_instr}  不同动作{diff_act}")

    print(f"\n统计 20 个 episode:")
    rng = np.random.default_rng(1)
    lens = [len(generate_episode(rng)) for _ in range(20)]
    print(f"  每个 episode 的样本数: {lens}")
    print(f"  平均 {np.mean(lens):.1f} 条")


if __name__ == "__main__":
    main()
