"""V1.1 第 5 步：训练最小改动模型。

    python scripts/v1_1/train_v11.py --limit 2000 --epochs 3      # 冒烟
    python scripts/v1_1/train_v11.py                              # 正式
    python scripts/v1_1/train_v11.py --epochs 40 --no-plot

和 V1 相比只改两处：
    状态编码 2 -> 3 维
    动作头 (4,2) -> (4,3)
视觉 / 文本 / 融合 / 配对采样 / 损失 / 优化器 全部保持不变。

★ 不覆盖 checkpoints/v1_natural_language.pt。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ★ 行缓冲：否则 print 进了缓冲区，控制台看起来像"没输出"
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from torch.utils.data import DataLoader                      # noqa: E402
from mini_vla.data.dataset import RobotDataset               # noqa: E402
from mini_vla.data.grasp_place_data import (                 # noqa: E402
    TEMPLATES, load_split,
)
from mini_vla.models.mini_vla import MiniVLA                 # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer         # noqa: E402
from train import PairedBatchSampler, compute_loss           # noqa: E402

DIMS = ["dx", "dy", "dgrip"]
STATE_DIM = 3
ACTION_DIM = 3
CHUNK = 4
MAX_LENGTH = 16
GRAD_CLIP = 1.0


# ===========================================================================
# 实时监视
# ===========================================================================
class Live:
    """四格：
        ① loss 分解（base / pair / total，对数轴）—— 配对损失有没有在工作
        ② 三个动作维各自的 val MSE —— 哪一维学不动
        ③ 梯度范数（裁剪前）
        ④ 【红蓝输出差异】—— 同一张图、同一个状态，只换指令，输出变了多少
           这是"语言有没有被用上"最直接的指标；V1 就是靠它发现语言塌缩的
    """

    def __init__(self, epochs):
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        self.plt = plt
        plt.ion()
        self.fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.6))
        self.fig.suptitle("V1.1 训练 —— 实时")
        self.ax_loss, self.ax_dim, self.ax_gn, self.ax_lang = ax.ravel()
        self.n = epochs

        self.ax_loss.set_xlabel("epoch"); self.ax_loss.set_ylabel("MSE (log)")
        self.ax_loss.set_yscale("log"); self.ax_loss.grid(alpha=0.3, which="both")
        self.l_tr, = self.ax_loss.plot([], [], "-", lw=2, color="#2b7bba", label="train total")
        self.l_va, = self.ax_loss.plot([], [], "-", lw=2, color="#d62728", label="val total")
        self.l_pair, = self.ax_loss.plot([], [], "--", lw=1.2, color="#7c3aed", label="train pair")
        self.ax_loss.legend(fontsize=8)

        self.ax_dim.set_xlabel("epoch"); self.ax_dim.set_ylabel("val MSE / 维度 (log)")
        self.ax_dim.set_yscale("log"); self.ax_dim.grid(alpha=0.3, which="both")
        self.dim_lines = [self.ax_dim.plot([], [], lw=1.6, label=d)[0] for d in DIMS]
        self.ax_dim.legend(fontsize=8)

        self.ax_gn.set_xlabel("epoch"); self.ax_gn.set_ylabel("梯度范数（裁剪前）")
        self.ax_gn.grid(alpha=0.3)
        self.gn_line, = self.ax_gn.plot([], [], lw=1.2, color="#2ca02c")
        self.ax_gn.axhline(GRAD_CLIP, color="0.6", ls="--", lw=1, label=f"clip={GRAD_CLIP}")
        self.ax_gn.legend(fontsize=8)

        self.ax_lang.set_xlabel("epoch"); self.ax_lang.set_ylabel("红蓝输出差异")
        self.ax_lang.grid(alpha=0.3)
        self.lang_line, = self.ax_lang.plot([], [], lw=1.8, color="#ea580c")
        self.ax_lang.axhline(0.05, color="0.6", ls="--", lw=1, label="0.05：约等于忽略语言")
        self.ax_lang.legend(fontsize=8)

        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, h, lang):
        e = np.arange(1, len(h["tr"]) + 1)
        self.l_tr.set_data(e, h["tr"]); self.l_va.set_data(e, h["va"])
        self.l_pair.set_data(e, h["pair"])
        for ax in (self.ax_loss, self.ax_dim, self.ax_gn, self.ax_lang):
            ax.set_xlim(0, max(10, e.max()))
        for ax in (self.ax_loss, self.ax_dim):
            ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
        for i, ln in enumerate(self.dim_lines):
            ln.set_data(e, np.array(h["dim"])[:, i])
        self.ax_dim.relim(); self.ax_dim.autoscale_view(scalex=False, scaley=True)
        self.gn_line.set_data(e, h["gn"]); self.ax_gn.relim()
        self.ax_gn.autoscale_view(scalex=False, scaley=True)
        self.lang_line.set_data(e, h["lang"]); self.ax_lang.relim()
        self.ax_lang.autoscale_view(scalex=False, scaley=True)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def close(self):
        self.plt.ioff(); self.plt.show(block=True)


# ===========================================================================
# 训练 / 评估
# ===========================================================================
def train_epoch(model, loader, opt, device):
    model.train()
    sums = {"total": 0.0, "base": 0.0, "pair": 0.0}
    per_dim = np.zeros(ACTION_DIM); gnorm = 0.0; n = 0
    for b in loader:
        img = b["image"].to(device); tok = b["token_ids"].to(device)
        st = b["robot_state"].to(device); tgt = b["action_chunk"].to(device)

        opt.zero_grad()
        pred = model(img, tok, st)
        total, base, pair = compute_loss(pred, tgt, torch.nn.MSELoss())
        total.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        nb = img.size(0)
        for k in sums:
            sums[k] += {"total": total, "base": base, "pair": pair}[k].item() * nb
        per_dim += ((pred.detach() - tgt) ** 2).mean(dim=(0, 1)).cpu().numpy() * nb
        gnorm += float(g) * nb
        n += nb
    return {k: v / n for k, v in sums.items()}, per_dim / n, gnorm / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {"total": 0.0, "base": 0.0, "pair": 0.0}
    per_dim = np.zeros(ACTION_DIM); n = 0
    for b in loader:
        img = b["image"].to(device); tok = b["token_ids"].to(device)
        st = b["robot_state"].to(device); tgt = b["action_chunk"].to(device)
        pred = model(img, tok, st)
        total, base, pair = compute_loss(pred, tgt, torch.nn.MSELoss())
        nb = img.size(0)
        for k in sums:
            sums[k] += {"total": total, "base": base, "pair": pair}[k].item() * nb
        per_dim += ((pred - tgt) ** 2).mean(dim=(0, 1)).cpu().numpy() * nb
        n += nb
    return {k: v / n for k, v in sums.items()}, per_dim / n


@torch.no_grad()
def language_delta(model, batch, tokenizer, device):
    """同一张图 + 同一个状态，只换指令，输出差多少。"""
    model.eval()
    img = batch["image"].to(device); st = batch["robot_state"].to(device)
    n = img.size(0)
    r = tokenizer.encode(TEMPLATES["red"][0]).unsqueeze(0).expand(n, -1).to(device)
    b = tokenizer.encode(TEMPLATES["blue"][0]).unsqueeze(0).expand(n, -1).to(device)
    return float((model(img, r, st) - model(img, b, st)).abs().mean())


@torch.no_grad()
def ablation(model, loader, tokenizer, device):
    """整集消融：图像置零 / 指令打乱 / 状态置零。"""
    model.eval()
    res = {}
    for b in loader:
        img = b["image"].to(device); tok = b["token_ids"].to(device)
        st = b["robot_state"].to(device); tgt = b["action_chunk"].to(device)

        def loss_of(i, t, s):
            return F.mse_loss(model(i, t, s), tgt).item()

        perm = torch.randperm(tok.size(0))
        for name, v in (("完整输入", loss_of(img, tok, st)),
                        ("图像置零", loss_of(torch.zeros_like(img), tok, st)),
                        ("指令打乱", loss_of(img, tok[perm], st)),
                        ("状态置零", loss_of(img, tok, torch.zeros_like(st)))):
            res[name] = res.get(name, 0.0) + v * img.size(0)
        res["_n"] = res.get("_n", 0) + img.size(0)

    n = res.pop("_n")
    res = {k: v / n for k, v in res.items()}
    base = res["完整输入"]
    print("\n=== 消融（整个验证集）===")
    print(f"  完整输入   {base:.5f}")
    for k in ("图像置零", "指令打乱", "状态置零"):
        print(f"  {k}   {res[k]:.5f}   变化 {(res[k] - base) / base * 100:+.1f}%")
    print("  变化大 = 这一路真的在用；指令不变 = 语言通路塌缩")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v1.1"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本（冒烟）")
    ap.add_argument("--pair-weight", type=float, default=1.0)
    ap.add_argument("--out", default="v1.1_grasp_place")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    import train as train_mod
    train_mod.PAIR_LOSS_WEIGHT = args.pair_weight

    np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print(f"\n读取 {args.data} ...")
    tr = load_split(Path(args.data) / "train")
    va = load_split(Path(args.data) / "val")
    if args.limit:
        tr = tr[:args.limit // 2 * 2]          # ★ 必须是偶数，否则配对被截断
    print(f"样本: train={len(tr)}  val={len(va)}（都是偶数）")

    vocab = sorted(set("".join(s["instruction"] for s in tr)))
    tok = SimpleCharTokenizer(vocab, max_length=MAX_LENGTH)
    print(f"词表 {tok.vocab_size}　字符 {''.join(vocab)}")

    train_ds = RobotDataset(tr, tok)
    val_ds = RobotDataset(va, tok)

    train_loader = DataLoader(
        train_ds, batch_sampler=PairedBatchSampler(
            len(train_ds), pairs_per_batch=args.batch // 2, seed=0))
    val_loader = DataLoader(
        val_ds, batch_sampler=PairedBatchSampler(
            len(val_ds), pairs_per_batch=args.batch // 2, seed=1))

    model = MiniVLA(vocab_size=tok.vocab_size, max_length=MAX_LENGTH,
                    state_dim=STATE_DIM, action_dim=ACTION_DIM,
                    chunk_size=CHUNK).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    h = {"tr": [], "va": [], "pair": [], "dim": [], "gn": [], "lang": []}
    best_val, best_state, bad = float("inf"), None, 0
    patience = 15

    live = None
    if not args.no_plot:
        try:
            live = Live(args.epochs)
        except Exception as e:
            print(f"[提示] 实时图开不起来（{e}）")

    val_batch = next(iter(val_loader))
    print(f"\n开始训练 {args.epochs} 轮（batch {args.batch}，"
          f"pair_weight {args.pair_weight}）")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        trn, _, gn = train_epoch(model, train_loader, opt, device)
        val, vdim = evaluate(model, val_loader, device)
        lang = language_delta(model, val_batch, tok, device)

        if not np.isfinite(trn["total"]) or not np.isfinite(val["total"]):
            print(f"\n[中止] 第 {ep} 轮出现 NaN/Inf，训练已发散。"
                  f"丢掉的权重不会保存。")
            break

        h["tr"].append(trn["total"]); h["va"].append(val["total"])
        h["pair"].append(trn["pair"]); h["dim"].append(vdim)
        h["gn"].append(gn); h["lang"].append(lang)

        if val["total"] < best_val:
            best_val, bad = val["total"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep % max(1, args.epochs // 40) == 0 or ep == 1:
            print(f"epoch {ep:>3}/{args.epochs}  "
                  f"train tot {trn['total']:.5f} base {trn['base']:.5f} "
                  f"pair {trn['pair']:.5f}  val {val['total']:.5f}  "
                  f"红蓝差异 {lang:.4f}  |  "
                  + "  ".join(f"{d} {v:.4f}" for d, v in zip(DIMS, vdim))
                  + f"  {time.time() - t0:.0f}s", flush=True)
        if live:
            live.update(h, lang)
        if bad >= patience:
            print(f"\n验证 loss 连续 {patience} 轮没改善，第 {ep} 轮提前停止")
            break

    model.load_state_dict(best_state)
    print(f"\n最佳验证 loss: {best_val:.5f}")

    ck = ROOT / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state, "vocab": vocab,
        "max_length": MAX_LENGTH, "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM, "chunk_size": CHUNK,
        "best_val": best_val, "data": str(args.data),
    }, ck / f"{args.out}.pt")
    print(f"已保存 checkpoints/{args.out}.pt")

    ablation(model, val_loader, tok, device)
    if live:
        live.close()


if __name__ == "__main__":
    main()
