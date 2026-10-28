import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from mini_vla.data.grasp_place_expert_v13 import sample_scene_v13
from mini_vla.data.grasp_place_render_v13 import render_v13
from mini_vla.data.grasp_place_v13 import COLORS, GraspPlaceEnvV13


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--out", default="reports/v13_scene_preview.png"
    )
    args = parser.parse_args()

    if args.scenes <= 0:
        parser.error("--scenes 必须大于 0")

    rng = np.random.default_rng(args.seed)
    cols = 4
    rows = (args.scenes + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 3.4 * rows),
        squeeze=False,
    )

    for index in range(rows * cols):
        ax = axes[index // cols][index % cols]
        ax.axis("off")

        if index >= args.scenes:
            continue

        scene = sample_scene_v13(rng)
        # 图像里显示完整场景；选红->红只是为了初始化环境对象，
        # 不会改变画面中的方块、区域和机器人位置。
        env = GraspPlaceEnvV13(scene, "red", "red")
        image = render_v13(env)

        ax.imshow(image, interpolation="nearest")
        ax.set_title(f"Scene {index + 1}", fontsize=10)

    fig.suptitle(
        f"V1.3 three-color scenes ({args.scenes}, seed={args.seed})",
        fontsize=14,
    )
    fig.tight_layout()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"预览图已保存：{out}")
    print(f"场景数：{args.scenes}，随机种子：{args.seed}")


if __name__ == "__main__":
    main()
