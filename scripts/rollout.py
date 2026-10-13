"""闭环评估：模型 vs 专家 vs 随机。"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))   # ← 见下面的说明

from mini_vla.data.synthetic import _sample_two_goals, _sample_robot
from mini_vla.evaluation.rollout import (
    run_episode, run_expert_episode, run_random_episode,
)
from mini_vla.models.text import SimpleCharTokenizer
from mini_vla.models.mini_vla import MiniVLA


N_SCENES = 200
INSTRUCTIONS = ["移动到红色目标", "移动到蓝色目标"]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ① 加载模型和词表
    ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
    tokenizer = SimpleCharTokenizer(ckpt["vocab"], max_length=ckpt["max_length"])
    print(f"模型的最佳验证 loss: {ckpt['best_val']:.5f}")
    print(f"设备: {device}")

    model = MiniVLA(vocab_size=tokenizer.vocab_size, max_length=ckpt["max_length"])
    model.to(device)                                    # ← 关键：模型必须和输入在同一个设备上
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ② 生成测试场景（用固定的种子）
    rng = np.random.default_rng(12345)
    scenes = []
    for _ in range(N_SCENES):
        red, blue = _sample_two_goals(rng)
        robot = _sample_robot(rng, [red, blue])
        scenes.append((robot, red, blue))

    # ③ 三种策略各跑一遍
    results = {"模型": [], "专家": [], "随机": []}
    for robot, red, blue in scenes:
        for instr in INSTRUCTIONS:
            results["模型"].append(run_episode(model, tokenizer, robot, red, blue, instr))
            results["专家"].append(run_expert_episode(robot, red, blue, instr))
            results["随机"].append(run_random_episode(robot, red, blue, instr, rng))

    # ④ 打印对比表
    print(f"\n{'策略':<8} {'成功率':>8} {'平均最终距离':>14} {'平均步数':>10}")
    print("-" * 46)
    for name, rs in results.items():
        rate = np.mean([r["success"] for r in rs]) * 100
        dist = np.mean([r["final_dist"] for r in rs])
        steps = np.mean([r["steps"] for r in rs])
        print(f"{name:<8} {rate:>7.1f}% {dist:>14.4f} {steps:>10.1f}")


if __name__ == "__main__":
    main()
