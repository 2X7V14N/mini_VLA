"""Create the reproducible, untrained V1.5a baseline from V1.3 final.

This is a migration artifact, not a training run: the temporal residual gate
is zero, so the action path is exactly the V1.3 policy on the latest frame.
It is retained for before/after dynamic closed-loop comparisons.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.models.mini_vla_v15a_dynamic import (  # noqa: E402
    LanguageConditionedMiniVLA3ColorDynamic,
)
from mini_vla.models.text import SimpleCharTokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="checkpoints/v1.3_final.pt")
    parser.add_argument("--out", default="checkpoints/v1.5_init.pt")
    parser.add_argument("--history", type=int, default=4)
    args = parser.parse_args()

    source = ROOT / args.source
    output = ROOT / args.out
    if not source.is_file():
        raise FileNotFoundError(f"找不到 V1.3 初始化 checkpoint：{source}")
    if output.exists():
        raise FileExistsError(f"为避免覆盖而停止：{output}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "LanguageConditionedMiniVLA3Color":
        raise ValueError("source 必须是 V1.3 三色 checkpoint")
    tokenizer = SimpleCharTokenizer(
        checkpoint["vocab"], int(checkpoint["max_length"])
    )
    model = LanguageConditionedMiniVLA3ColorDynamic.from_v13_checkpoint(
        checkpoint, tokenizer, history=args.history
    )
    if model.temporal_gate.detach().item() != 0.0:
        raise RuntimeError("V1.5a temporal gate 必须以 0 初始化")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_type": "LanguageConditionedMiniVLA3ColorDynamic",
        "model_state": model.state_dict(),
        "vocab": checkpoint["vocab"],
        "max_length": checkpoint["max_length"],
        "state_dim": checkpoint["state_dim"],
        "action_dim": checkpoint["action_dim"],
        "chunk_size": checkpoint["chunk_size"],
        "history": int(args.history),
        "source_checkpoint": str(source),
        "initialization": "v1.3_final + zero-gated temporal residual",
    }, output)
    print(f"saved untrained V1.5a baseline: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
