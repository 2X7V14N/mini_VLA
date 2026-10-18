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

# 两个用途不同的随机种子。分开是为了让"测试场景"不受随机基线的影响。
SCENE_SEED = 12345      # 生成测试场景
RANDOM_SEED = 999       # 随机基线的动作噪声


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ① 加载模型和词表
    ckpt = torch.load(
        "checkpoints/v1_natural_language.pt",
        map_location=device,
        weights_only=False,
    )
    tokenizer = SimpleCharTokenizer(ckpt["vocab"], max_length=ckpt["max_length"])
    print(f"模型的最佳验证 loss: {ckpt['best_val']:.5f}")
    print(f"设备: {device}")

    model = MiniVLA(vocab_size=tokenizer.vocab_size, max_length=ckpt["max_length"])
    model.to(device)                                    # ← 关键：模型必须和输入在同一个设备上
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ② 生成测试场景
    # ⚠️ 场景生成用一个【独立的】rng，随机基线用另一个。
    #    如果两者共用一个 rng，任何一方的调用次数变化都会改变测试场景，
    #    导致同一份模型在不同次运行中拿到不同的分数 —— 无法对比。
    scene_rng = np.random.default_rng(SCENE_SEED)
    scenes = []
    for _ in range(N_SCENES):
        red, blue = _sample_two_goals(scene_rng)
        robot = _sample_robot(scene_rng, [red, blue])
        scenes.append((robot, red, blue))

    # ③ 三种策略各跑一遍
    # 按指令分组存放：合并成一个总数会掩盖"红蓝表现不对称"这种重要信息
    random_rng = np.random.default_rng(RANDOM_SEED)
    STRATEGIES = ["模型", "专家", "随机"]
    results = {s: {"红": [], "蓝": []} for s in STRATEGIES}

    for robot, red, blue in scenes:
        for instr, key in zip(INSTRUCTIONS, ["红", "蓝"]):
            results["模型"][key].append(
                run_episode(model, tokenizer, robot, red, blue, instr, device=device))
            results["专家"][key].append(
                run_expert_episode(robot, red, blue, instr))
            results["随机"][key].append(
                run_random_episode(robot, red, blue, instr, random_rng))

    # ④ 打印对比表
    print(f"\n{'策略':<6} {'红色':>8} {'蓝色':>8} {'总体':>8} "
          f"{'平均最终距离':>13} {'平均步数':>9}")
    print("-" * 60)
    for name in STRATEGIES:
        d = results[name]
        both = d["红"] + d["蓝"]
        red_rate = np.mean([r["success"] for r in d["红"]]) * 100
        blue_rate = np.mean([r["success"] for r in d["蓝"]]) * 100
        all_rate = np.mean([r["success"] for r in both]) * 100
        dist = np.mean([r["final_dist"] for r in both])
        steps = np.mean([r["steps"] for r in both])
        print(f"{name:<6} {red_rate:>7.1f}% {blue_rate:>7.1f}% {all_rate:>7.1f}% "
              f"{dist:>13.4f} {steps:>9.1f}")

    print(f"\n(每个格子 N={N_SCENES}；模型/专家/随机跑的是同一批场景)")


if __name__ == "__main__":
    main()
