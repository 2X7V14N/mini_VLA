"""V1.1 诊断：小数据过拟合探针（专测语言通路）。

    python scripts/v1_1/overfit_v11.py --pairs 64 --epochs 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from torch.utils.data import DataLoader                    # noqa: E402
from mini_vla.data.dataset import RobotDataset             # noqa: E402
from mini_vla.data.grasp_place_data import load_split      # noqa: E402
from mini_vla.models.mini_vla import MiniVLA              # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer       # noqa: E402
from train import PairedBatchSampler, compute_loss         # noqa: E402

DIMS = ["dx", "dy", "dgrip"]


def plumbing_checks(batch):
    """把'配对监督到底有没有可能工作'的六项前提逐条验一遍。"""
    tok_ids = batch["token_ids"]
    act = batch["action_chunk"]
    st = batch["robot_state"]
    img = batch["image"]

    print("\n=== 管线检查 ===")
    n_pairs = len(tok_ids) // 2

    same_tok = (tok_ids[0::2] == tok_ids[1::2]).all(dim=1)
    print(f"  ① 相邻 token_ids 不同          {int((~same_tok).sum())}/{n_pairs}"
          f"  {'[OK]' if not same_tok.any() else '[FAIL]'}")

    hold = st[0::2, 2] > 0.5
    d_act = (act[0::2] - act[1::2]).abs().amax(dim=(1, 2))
    free = (~hold)
    print(f"  ② 空手时相邻动作不同          "
          f"{int((d_act[free] > 1e-6).sum())}/{int(free.sum())}"
          f"  {'[OK]' if (d_act[free] > 1e-6).all() else '[FAIL]'}")
    print(f"     夹住时相邻动作相同          "
          f"{int((d_act[hold] <= 1e-6).sum())}/{int(hold.sum())}"
          f"  {'[OK]' if (d_act[hold] <= 1e-6).all() else '[FAIL]'}")

    same_img = (img[0::2] == img[1::2]).all(dim=(1, 2, 3))
    same_st = torch.allclose(st[0::2], st[1::2])
    print(f"  ③ 相邻图像/状态完全相同        "
          f"{'[OK]' if same_img.all() and same_st else '[FAIL]'}")

    # ④ compute_loss 比的是不是相邻两条
    zero = torch.zeros_like(act)
    _, _, pair_ref = compute_loss(zero, act, torch.nn.MSELoss())
    manual = ((act[0::2] - act[1::2]) ** 2).mean()
    print(f"  ④ compute_loss 的 pair 项 == E[|Δ动作|^2]？  "
          f"{pair_ref.item():.6f} vs {manual.item():.6f}"
          f"  {'[OK]' if abs(pair_ref.item() - manual.item()) < 1e-6 else '[FAIL]'}")

    print(f"  ⑤ 文本编码器收到不同 token     "
          f"{'[OK]' if not same_tok.any() else '[FAIL]'}（同 ①）")
    return manual.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_tr = load_split(ROOT / "data" / "v1.1" / "train")
    samples = all_tr[:args.pairs * 2]
    vocab = sorted(set("".join(s["instruction"] for s in all_tr)))
    tok = SimpleCharTokenizer(vocab, max_length=16)
    ds = RobotDataset(samples, tok)
    loader = DataLoader(ds, batch_sampler=PairedBatchSampler(
        len(ds), pairs_per_batch=args.pairs, seed=0))
    b = next(iter(loader))

    img = b["image"].to(dev); tok_ids = b["token_ids"].to(dev)
    st = b["robot_state"].to(dev); tgt = b["action_chunk"].to(dev)

    # compute_loss 读的是 train.py 的模块常量，这里直接改它
    import train as train_mod
    train_mod.PAIR_LOSS_WEIGHT = args.pair_weight
    print(f"{args.pairs} 对样本（{len(ds)} 条）　设备 {dev}　lr {args.lr}"
          f"　clip {args.clip}　pair_weight {args.pair_weight}")
    baseline = plumbing_checks(b)
    print(f"\n  ★ 不看指令时的 pair loss 下限 = {baseline:.6f}")
    print(f"    pair loss 降不到这个数以下，就说明模型开始读指令了。")

    model = MiniVLA(vocab_size=tok.vocab_size, max_length=16,
                    state_dim=3, action_dim=3, chunk_size=4).to(dev)
    print(f"  参数量 {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = torch.nn.MSELoss()

    print(f"\n{'epoch':>6}{'base':>10}{'pair':>10}{'total':>10}"
          f"{'红蓝差异':>10}{'grad':>8}   " + "  ".join(f"{d:>8}" for d in DIMS))
    best_pair = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(img, tok_ids, st)
        total, base, pair = compute_loss(pred, tgt, crit)
        total.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if ep % max(1, args.epochs // 25) == 0 or ep == 1:
            with torch.no_grad():
                model.eval()
                p = model(img, tok_ids, st)
                _, bb, pp = compute_loss(p, tgt, crit)
                n = len(p)
                r = tok.encode("抓住红色方块并放到目标区域").unsqueeze(0).expand(n, -1).to(dev)
                bl = tok.encode("抓住蓝色方块并放到目标区域").unsqueeze(0).expand(n, -1).to(dev)
                lang = float((model(img, r, st) - model(img, bl, st)).abs().mean())
                pd = ((p - tgt) ** 2).mean(dim=(0, 1)).cpu().numpy()
            best_pair = min(best_pair, pp.item())
            print(f"{ep:>6}{bb.item():>10.4f}{pp.item():>10.4f}{total.item():>10.4f}"
                  f"{lang:>10.4f}{float(g):>8.2f}   "
                  + "  ".join(f"{v:>8.4f}" for v in pd), flush=True)

    print(f"\n最佳 pair loss {best_pair:.6f}（不看指令的下限 {baseline:.6f}）")
    if best_pair < baseline * 0.9:
        print("[通过] 语言通路能学 —— 小数据上 pair loss 明显低于下限。")
        print("       说明架构没问题，正式训练是 epoch 不够或学习率/批次的问题。")
    else:
        print("[不通过] 小数据也学不动语言 —— 问题在管线或架构，不是 epoch 数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
