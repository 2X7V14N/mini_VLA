"""准备 V1.2 的纠偏微调数据。

默认策略：训练集保留全部闭环纠偏 episode，再从原始专家训练集按比例
抽取旧数据回放；验证集保留原始验证集和全部纠偏验证集，测试集只复制
原始测试集。脚本只复制文件，不改写样本内容，便于追溯和重复实验。

示例：
    python scripts/v1_2/prepare_v12_finetune_data.py \
        --source data/v1.2_correction_64_close08_policy_v2 \
        --out data/v1.2_correction_64_close08_policy_ft70
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def is_corrected(path: Path) -> bool:
    with np.load(path, allow_pickle=True) as z:
        # 原始专家 episode 也可能含有 corrected=True 的困难状态标签；
        # episode 来源由标量 correction_stage 判断，policy_closed_loop
        # 才是本次新增的模型闭环纠偏 rollout。
        return str(np.asarray(z["correction_stage"]).item()) == "policy_closed_loop"


def split_sources(root: Path, split: str):
    paths = sorted((root / split).glob("ep_*.npz"))
    corrected = [p for p in paths if is_corrected(p)]
    base = [p for p in paths if not is_corrected(p)]
    return base, corrected


def copy_as_sequential(paths: list[Path], out_dir: Path, start: int = 0) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(paths, start=start):
        shutil.copy2(src, out_dir / f"ep_{i:04d}.npz")
    return start + len(paths)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="policy_v2 合并数据目录")
    ap.add_argument("--out", required=True, help="新的微调数据目录；必须不存在")
    ap.add_argument("--replay-ratio", type=float, default=0.30,
                    help="保留的原始训练 episode 比例，默认 0.30")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if not source.is_dir():
        raise FileNotFoundError(f"源数据目录不存在：{source}")
    if out.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖请换 --out：{out}")
    if not 0.0 <= args.replay_ratio <= 1.0:
        raise ValueError("--replay-ratio 必须在 [0, 1] 范围内")

    rng = np.random.default_rng(args.seed)
    summary = {
        "source": str(source),
        "out": str(out),
        "replay_ratio": args.replay_ratio,
        "seed": args.seed,
        "policy": "all corrected train episodes + sampled base replay; full mixed val; base test",
        "splits": {},
    }

    for split in ("train", "val", "test"):
        base, corrected = split_sources(source, split)
        if split == "train":
            keep_n = int(round(len(base) * args.replay_ratio))
            keep_n = min(len(base), max(0, keep_n))
            chosen_base = sorted(rng.choice(base, size=keep_n, replace=False).tolist(),
                                 key=lambda p: p.name)
            selected = chosen_base + corrected
        elif split == "val":
            # 验证同时覆盖旧分布和纠偏分布，避免只优化困难状态。
            chosen_base = base
            selected = base + corrected
        else:
            # 测试集保持源目录的原始干净测试集，不混入策略 rollout。
            chosen_base = base
            selected = base

        copy_as_sequential(selected, out / split)
        summary["splits"][split] = {
            "base_available": len(base),
            "corrected_available": len(corrected),
            "base_selected": len(chosen_base),
            "corrected_selected": len(corrected) if split != "test" else 0,
            "episodes_selected": len(selected),
        }

    out.mkdir(parents=True, exist_ok=True)
    (out / "finetune_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n微调数据已准备：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
