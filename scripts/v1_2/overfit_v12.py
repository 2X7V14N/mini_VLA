"""V1.2 诊断：小数据过拟合探针（专测方块颜色 + 区域颜色两路语言）。

    python scripts/v1_2/overfit_v12.py --ticks 64 --epochs 500

契约：
    只取 train 的前 N 个 tick（每个 tick 4 条样本），整个探针集作为一个 batch
    反复训练。目的不是泛化，而是验证：
        ① 标签和配对结构允许模型拟合；
        ② 方块颜色和区域颜色都能改变输出；
        ③ 图像 / 指令 / 状态三路输入都能影响输出。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from torch.utils.data import DataLoader                          # noqa: E402
from mini_vla.data.dataset import RobotDataset                   # noqa: E402
from mini_vla.data.grasp_place_data import load_split            # noqa: E402
from mini_vla.data.grasp_place_v12 import instruction_v12        # noqa: E402
from mini_vla.models.mini_vla import LanguageConditionedMiniVLA   # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer             # noqa: E402
from train_v12 import V12BatchSampler, compute_loss_v12          # noqa: E402

DIMS = ["dx", "dy", "dgrip"]


def plumbing_checks(batch):
    """配对监督能否工作的前提，逐条验一遍。"""
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

    zero = torch.zeros_like(act)
    _, _, block_ref, zone_ref = compute_loss_v12(
        zero, act, torch.nn.MSELoss(), pair_weight=1.0)
    pair_ref = 0.5 * (block_ref + zone_ref)
    manual = 0.5 * (
        ((act[0::4] - act[1::4]) ** 2).mean()
        + 0.5 * (
            ((act[0::4] - act[2::4]) ** 2).mean()
            + ((act[1::4] - act[3::4]) ** 2).mean()
        )
    )
    print(f"  ④ compute_loss 的 pair 项 == E[|Δ动作|^2]？  "
          f"{pair_ref.item():.6f} vs {manual.item():.6f}"
          f"  {'[OK]' if abs(pair_ref.item() - manual.item()) < 1e-6 else '[FAIL]'}")
    return manual.item()


@torch.no_grad()
def language_deltas(model, batch, tok, dev):
    """同一张图 + 同一个状态，分别只换方块颜色 / 只换区域颜色。"""
    model.eval()
    img = batch["image"].to(dev)
    st = batch["robot_state"].to(dev)
    n = img.size(0)

    def one(b1, z1, b2, z2):
        t1 = tok.encode(instruction_v12(b1, z1, 0))
        t2 = tok.encode(instruction_v12(b2, z2, 0))
        t1 = t1.unsqueeze(0).expand(n, -1).to(dev)
        t2 = t2.unsqueeze(0).expand(n, -1).to(dev)
        return float((model(img, t1, st) - model(img, t2, st)).abs().mean())

    block = 0.5 * (one("red", "red", "blue", "red")
                   + one("red", "blue", "blue", "blue"))
    zone = 0.5 * (one("red", "red", "red", "blue")
                  + one("blue", "red", "blue", "blue"))
    return block, zone


@torch.no_grad()
def batch_ablation(model, batch, dev):
    """同一批数据上破坏一路输入，看 loss 变化。"""
    model.eval()
    img = batch["image"].to(dev)
    tok_ids = batch["token_ids"].to(dev)
    st = batch["robot_state"].to(dev)
    tgt = batch["action_chunk"].to(dev)

    def loss_of(i, t, s):
        return torch.nn.functional.mse_loss(model(i, t, s), tgt).item()

    perm = torch.randperm(tok_ids.size(0))
    return {
        "完整输入": loss_of(img, tok_ids, st),
        "图像置零": loss_of(torch.zeros_like(img), tok_ids, st),
        "指令打乱": loss_of(img, tok_ids[perm], st),
        "状态置零": loss_of(img, tok_ids, torch.zeros_like(st)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v1.2"))
    ap.add_argument("--ticks", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--pair-weight", type=float, default=1.0)
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_tr = load_split(Path(args.data) / "train")
    samples = all_tr[:args.ticks * 4]
    if len(samples) % 4:
        raise ValueError(f"取到的样本数 {len(samples)} 不是 4 的倍数")

    # 标签本身的输出差上限：模型再怎么拟合，也不会超过这个量级
    A = np.stack([s["action_chunk"] for s in samples]).reshape(-1, 4, 4, 3)
    ceil_block = 0.5 * (np.abs(A[:, 0] - A[:, 1]).mean()
                        + np.abs(A[:, 2] - A[:, 3]).mean())
    ceil_zone = 0.5 * (np.abs(A[:, 0] - A[:, 2]).mean()
                       + np.abs(A[:, 1] - A[:, 3]).mean())

    vocab = sorted(set("".join(s["instruction"] for s in all_tr)))
    tok = SimpleCharTokenizer(vocab, max_length=16)
    ds = RobotDataset(samples, tok)
    loader = DataLoader(ds, batch_sampler=V12BatchSampler(
        len(ds), ticks_per_batch=len(ds) // 4, seed=0))
    b = next(iter(loader))

    img = b["image"].to(dev); tok_ids = b["token_ids"].to(dev)
    st = b["robot_state"].to(dev); tgt = b["action_chunk"].to(dev)

    print(f"{args.ticks} 个 tick（{len(ds)} 条样本）　设备 {dev}"
          f"　lr {args.lr}　clip {args.clip}　pair_weight {args.pair_weight}")
    baseline = plumbing_checks(b)
    print(f"\n  ★ 不看指令时的 pair loss 下限 = {baseline:.6f}")
    print(f"  ★ 标签本身：方块D = {ceil_block:.4f}　区域D = {ceil_zone:.4f}")

    model = LanguageConditionedMiniVLA(vocab_size=tok.vocab_size,
                                      max_length=16, state_dim=3,
                                      action_dim=3, chunk_size=4,
                                      red_token_id=tok.stoi["红"],
                                      blue_token_id=tok.stoi["蓝"]).to(dev)
    print(f"  参数量 {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = torch.nn.MSELoss()

    print(f"\n{'epoch':>6}{'base':>10}{'pair':>10}{'total':>10}"
          f"{'方块D':>9}{'区域D':>9}{'grad':>8}   "
          + "  ".join(f"{d:>8}" for d in DIMS))

    last = {}
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(img, tok_ids, st)
        total, base, block_pair, zone_pair = compute_loss_v12(
            pred, tgt, crit, pair_weight=args.pair_weight)
        pair = 0.5 * (block_pair + zone_pair)
        if not torch.isfinite(total):
            print(f"\n[中止] 第 {ep} 轮 loss 不是有限值，保留最后一次有效权重。")
            break
        total.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if torch.isnan(g):
            print(f"\n[中止] 第 {ep} 轮梯度出现 NaN，保留最后一次有效权重。")
            opt.zero_grad()
            break
        opt.step()

        if ep % max(1, args.epochs // 25) == 0 or ep == 1 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                p = model(img, tok_ids, st)
                _, bb, bp, zp = compute_loss_v12(
                    p, tgt, crit, pair_weight=args.pair_weight)
                pp = 0.5 * (bp + zp)
                pd = ((p - tgt) ** 2).mean(dim=(0, 1)).cpu().numpy()
            block, zone = language_deltas(model, b, tok, dev)
            last = {"base": bb.item(), "pair": pp.item(),
                    "block": block, "zone": zone}
            print(f"{ep:>6}{bb.item():>10.4f}{pp.item():>10.4f}"
                  f"{total.item():>10.4f}{block:>9.4f}{zone:>9.4f}"
                  f"{float(g):>8.2f}   "
                  + "  ".join(f"{v:>8.4f}" for v in pd), flush=True)

    # 不管有没有提前中止，都用当前权重重新算一次最终指标
    model.eval()
    with torch.no_grad():
        p = model(img, tok_ids, st)
        _, bb, bp, zp = compute_loss_v12(
            p, tgt, crit, pair_weight=args.pair_weight)
        pp = 0.5 * (bp + zp)
    block, zone = language_deltas(model, b, tok, dev)
    last = {"base": bb.item(), "pair": pp.item(),
            "block": block, "zone": zone}
    print(f"\n最终：base {bb.item():.4f}  pair {pp.item():.4f}  "
          f"方块D {block:.4f}  区域D {zone:.4f}")

    abl = batch_ablation(model, b, dev)
    base_loss = abl["完整输入"]
    print("\n=== 消融（同一批过拟合数据）===")
    for k, v in abl.items():
        change = (v - base_loss) / max(base_loss, 1e-12) * 100
        print(f"  {k:<8}{v:.6f}   变化 {change:+.1f}%")

    ok_loss = last.get("base", 9) < 0.02 and last.get("pair", 9) < 0.02
    ok_lang = (last.get("block", 0) >= max(0.15, 0.6 * ceil_block)
               and last.get("zone", 0) >= max(0.15, 0.6 * ceil_zone))
    ok_abl = all(abl[k] > 1.5 * base_loss
                 for k in ("图像置零", "指令打乱", "状态置零"))

    print("\n" + "=" * 60)
    print(f"  base/pair < 0.02                  {'[OK]' if ok_loss else '[FAIL]'}"
          f"  base={last.get('base', float('nan')):.4f}"
          f" pair={last.get('pair', float('nan')):.4f}")
    print(f"  方块D / 区域D >= 0.6*标签上限      "
          f"{'[OK]' if ok_lang else '[FAIL]'}"
          f"  {last.get('block', float('nan')):.4f}"
          f" / {last.get('zone', float('nan')):.4f}"
          f"  (上限 {ceil_block:.3f} / {ceil_zone:.3f})")
    print(f"  三路消融都 >= 完整输入的 1.5 倍     "
          f"{'[OK]' if ok_abl else '[FAIL]'}")
    passed = ok_loss and ok_lang and ok_abl
    print(f"\n{'[通过]' if passed else '[不通过]'} 过拟合探针")
    if not passed:
        print("  定位：base 不降 -> 模型/优化；pair 不降 -> 配对；"
              "区域D 不起来 -> 区域语言；图像消融不变 -> 视觉通路。")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
