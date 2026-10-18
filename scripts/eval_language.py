"""V1：语言泛化评估。

同一批场景，三种说法，对比闭环成功率。
差异只能归因于"说法"，没有其他混淆变量。
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_vla.data.instructions import (
    SEEN_TEMPLATES, UNSEEN_KNOWN_CHARS, UNSEEN_NOVEL_CHARS,
)
from mini_vla.data.synthetic import _sample_two_goals, _sample_robot
from mini_vla.evaluation.rollout import run_episode, run_expert_episode
from mini_vla.models.text import SimpleCharTokenizer
from mini_vla.models.mini_vla import MiniVLA


N_SCENES = 100
SCENE_SEED = 12345

TEMPLATE_SETS = {
    "SEEN（见过）":      SEEN_TEMPLATES,
    "UNSEEN_KNOWN（新说法/旧字）": UNSEEN_KNOWN_CHARS,
    "UNSEEN_NOVEL（新说法/新字）": UNSEEN_NOVEL_CHARS,
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ① 加载模型（用 V1 的 checkpoint）
    ckpt = torch.load(
        "checkpoints/v1_natural_language.pt",
        map_location=device,
        weights_only=False,
    )
    tokenizer = SimpleCharTokenizer(ckpt["vocab"], max_length=ckpt["max_length"])
    model = MiniVLA(vocab_size=tokenizer.vocab_size, max_length=ckpt["max_length"])
    model.to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"模型最佳验证 loss: {ckpt['best_val']:.5f}   词表 {tokenizer.vocab_size}")

    # ② 固定测试场景 —— 【所有模板组共用这一批】
    scene_rng = np.random.default_rng(SCENE_SEED)
    scenes = []
    for _ in range(N_SCENES):
        red, blue = _sample_two_goals(scene_rng)
        robot = _sample_robot(scene_rng, [red, blue])
        scenes.append((robot, red, blue))

    # ③ 逐组评估
    print(f"\n{'说法组':<28} {'红色':>8} {'蓝色':>8} {'总体':>8} {'评估次数':>8}")
    print("-" * 66)
    summary = {}
    for name, pool in TEMPLATE_SETS.items():
        res = {"红": [], "蓝": []}
        for robot, red, blue in scenes:
            for key in ("红", "蓝"):
                for tmpl in pool[key]:                        # 该颜色的【每一条】说法都跑
                    res[key].append(
                        run_episode(model, tokenizer, robot, red, blue, tmpl, device=device))
        r = np.mean([x["success"] for x in res["红"]]) * 100
        b = np.mean([x["success"] for x in res["蓝"]]) * 100
        t = np.mean([x["success"] for x in res["红"] + res["蓝"]]) * 100
        n = len(res["红"]) + len(res["蓝"])
        summary[name] = t
        print(f"{name:<28} {r:>7.1f}% {b:>7.1f}% {t:>7.1f}% {n:>8}")

    # ④ 专家基线（不依赖说法，只需跑一次作参照）
    expert = []
    for robot, red, blue in scenes:
        for key in ("红", "蓝"):
            tmpl = SEEN_TEMPLATES[key][0]
            expert.append(run_expert_episode(robot, red, blue, tmpl))
    print(f"{'专家（参照）':<28} {'':>8} {'':>8} "
          f"{np.mean([x['success'] for x in expert])*100:>7.1f}% {len(expert):>8}")


if __name__ == "__main__":
    main()
