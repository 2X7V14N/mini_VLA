"""V1.2 ：训练（多放置区 + 放置区随机化）。

    python scripts/v1_2/train_v12.py --limit 4000 --epochs 3 --out v1.2_smoke   # 冒烟
    python scripts/v1_2/train_v12.py --batch 64 --epochs 150                    # 正式
    python scripts/v1_2/train_v12.py --batch 64 --epochs 150 --no-plot

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

from torch.utils.data import DataLoader, Sampler             # noqa: E402
from mini_vla.data.dataset import RobotDataset               # noqa: E402
from mini_vla.data.grasp_place_data import load_split
from mini_vla.data.grasp_place_v12 import instruction_v12    # noqa: E402

from mini_vla.models.mini_vla import LanguageConditionedMiniVLA  # noqa: E402
from mini_vla.models.text import SimpleCharTokenizer         # noqa: E402

DIMS = ["dx", "dy", "dgrip"]
STATE_DIM = 3
ACTION_DIM = 3
CHUNK = 4
MAX_LENGTH = 16
GRAD_CLIP = 1.0
PAIR_WEIGHT = 1.0


class V12BatchSampler(Sampler[list[int]]):
    """按完整 tick 组采样，保持四条反事实样本在同一 batch 中。

    V1.2 每个 tick 的顺序是：
        red->red, blue->red, red->blue, blue->blue
    前两条监督方块颜色变化，后两条监督放置区颜色变化。
    只保持两条相邻样本会让 zone pair 可能不在同一 batch，无法计算
    ``red->red`` 与 ``red->blue`` 的同观测差分。
    """

    def __init__(self, n_samples, ticks_per_batch, seed=0):
        if n_samples % 4:
            raise ValueError("V1.2 样本数必须是 4 的倍数")
        if ticks_per_batch < 1:
            raise ValueError("ticks_per_batch 必须为正数")
        self.n_ticks = n_samples // 4
        self.ticks_per_batch = ticks_per_batch
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(self.n_ticks)
        for start in range(0, self.n_ticks, self.ticks_per_batch):
            ticks = order[start:start + self.ticks_per_batch]
            yield [4 * int(t) + j for t in ticks for j in range(4)]

    def __len__(self):
        return (self.n_ticks + self.ticks_per_batch - 1) // self.ticks_per_batch


def compute_loss_v12(prediction, target, criterion, pair_weight=1.0):
    """V1.2 的动作损失 + 两路同观测语言差分损失。

    batch 必须由 ``V12BatchSampler`` 产生，因此每 4 条样本属于同一 tick。
    ``block_pair`` 比较 (red,red) 与 (blue,red)，
    ``zone_pair`` 比较两个相同方块、不同区域的动作。
    """
    if prediction.size(0) % 4:
        raise ValueError("V1.2 loss 的 batch 大小必须是 4 的倍数")
    base_loss = criterion(prediction, target)

    block_pred = prediction[0::4] - prediction[1::4]
    block_tgt = target[0::4] - target[1::4]
    block_pair = criterion(block_pred, block_tgt)

    zone_pred_red = prediction[0::4] - prediction[2::4]
    zone_tgt_red = target[0::4] - target[2::4]
    zone_pred_blue = prediction[1::4] - prediction[3::4]
    zone_tgt_blue = target[1::4] - target[3::4]
    zone_pair = 0.5 * (
        criterion(zone_pred_red, zone_tgt_red)
        + criterion(zone_pred_blue, zone_tgt_blue)
    )

    # 两路语言差分取平均，保持其量级与 V1.1 的单路 pair loss 相近。
    total = base_loss + pair_weight * 0.5 * (block_pair + zone_pair)
    return total, base_loss, block_pair, zone_pair


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
        self.fig.suptitle("V1.2 训练 —— 实时")
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

        self.ax_lang.set_xlabel("epoch"); self.ax_lang.set_ylabel("方块 / 区域差异")
        self.ax_lang.grid(alpha=0.3)
        self.block_line, = self.ax_lang.plot(
            [], [], lw=1.8, color="#ea580c", label="方块差异")
        self.zone_line, = self.ax_lang.plot(
            [], [], lw=1.8, color="#2563eb", label="区域差异")
        self.ax_lang.axhline(0.05, color="0.6", ls="--", lw=1,
                             label="0.05：约等于忽略这一维")
        self.ax_lang.legend(fontsize=8)

        self.fig.tight_layout()
        plt.show(block=False)

    def update(self, h):
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
        self.block_line.set_data(e, h["lang"]); self.ax_lang.relim()
        self.zone_line.set_data(e, h["lang_zone"]); self.ax_lang.relim()
        self.ax_lang.autoscale_view(scalex=False, scaley=True)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()

    def close(self):
        self.plt.ioff(); self.plt.show(block=True)


# ===========================================================================
# 训练 / 评估
# ===========================================================================
def train_epoch(model, loader, opt, device):
    model.train()
    sums = {"total": 0.0, "base": 0.0, "pair": 0.0,
            "block_pair": 0.0, "zone_pair": 0.0}
    per_dim = np.zeros(ACTION_DIM); gnorm = 0.0; n = 0
    for b in loader:
        img = b["image"].to(device); tok = b["token_ids"].to(device)
        st = b["robot_state"].to(device); tgt = b["action_chunk"].to(device)

        opt.zero_grad()
        pred = model(img, tok, st)
        total, base, block_pair, zone_pair = compute_loss_v12(
            pred, tgt, torch.nn.MSELoss(), PAIR_WEIGHT)
        total.backward()
        g = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        nb = img.size(0)
        pair = 0.5 * (block_pair + zone_pair)
        values = {"total": total, "base": base, "pair": pair,
                  "block_pair": block_pair, "zone_pair": zone_pair}
        for k in sums:
            sums[k] += values[k].item() * nb
        per_dim += ((pred.detach() - tgt) ** 2).mean(dim=(0, 1)).cpu().numpy() * nb
        gnorm += float(g) * nb
        n += nb
    return {k: v / n for k, v in sums.items()}, per_dim / n, gnorm / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    sums = {"total": 0.0, "base": 0.0, "pair": 0.0,
            "block_pair": 0.0, "zone_pair": 0.0}
    per_dim = np.zeros(ACTION_DIM); n = 0
    for b in loader:
        img = b["image"].to(device); tok = b["token_ids"].to(device)
        st = b["robot_state"].to(device); tgt = b["action_chunk"].to(device)
        pred = model(img, tok, st)
        total, base, block_pair, zone_pair = compute_loss_v12(
            pred, tgt, torch.nn.MSELoss(), PAIR_WEIGHT)
        pair = 0.5 * (block_pair + zone_pair)
        nb = img.size(0)
        values = {"total": total, "base": base, "pair": pair,
                  "block_pair": block_pair, "zone_pair": zone_pair}
        for k in sums:
            sums[k] += values[k].item() * nb
        per_dim += ((pred - tgt) ** 2).mean(dim=(0, 1)).cpu().numpy() * nb
        n += nb
    return {k: v / n for k, v in sums.items()}, per_dim / n


@torch.no_grad()
def pair_loss_floors(loader, device):
    """Dataset-wide V1.2 pair losses for a predictor that ignores language."""
    block_sum = zone_sum = n = 0
    criterion = torch.nn.MSELoss()
    for batch in loader:
        target = batch["action_chunk"].to(device)
        zero = torch.zeros_like(target)
        _, _, block, zone = compute_loss_v12(
            zero, target, criterion, pair_weight=PAIR_WEIGHT)
        count = target.size(0)
        block_sum += block.item() * count
        zone_sum += zone.item() * count
        n += count
    return block_sum / n, zone_sum / n


@torch.no_grad()
def language_delta(model, batch, tokenizer, device):
    """同一张图 + 同一个状态，只换【方块颜色】，输出差多少。

    ★ 用 batch 里的【第 0 条和第 1 条】：
      V12BatchSampler 保证它们是同一个观测的一对
      （V1.2 是红/蓝【方块】指令，区域颜色相同）。
      所以这个诊断不依赖任何具体模板，不会因为版本换了模板就失效。
    """
    model.eval()
    img = batch["image"].to(device)
    st = batch["robot_state"].to(device)
    tok = batch["token_ids"].to(device)
    n = img.size(0)
    a = tok[0:1].expand(n, -1)
    b = tok[1:2].expand(n, -1)
    return float((model(img, a, st) - model(img, b, st)).abs().mean())


@torch.no_grad()
def zone_language_delta(model, batch, tokenizer, device):
    """同一张图 + 同一个状态，只换【区域颜色】，输出差多少。

    ★ 必须显式构造指令，不能借 batch 里的第 2 条：
      V12BatchSampler 还保证 (0,2) / (1,3) 是同一个 tick 的区域对，
      第 2 条可能来自另一个 tick。
    """
    model.eval()
    img = batch["image"].to(device)
    st = batch["robot_state"].to(device)
    n = img.size(0)

    pairs = ((("red", "red"), ("red", "blue")),
             (("blue", "red"), ("blue", "blue")))
    vals = []
    for (b1, z1), (b2, z2) in pairs:
        t1 = tokenizer.encode(instruction_v12(b1, z1, 0))
        t2 = tokenizer.encode(instruction_v12(b2, z2, 0))
        t1 = t1.unsqueeze(0).expand(n, -1).to(device)
        t2 = t2.unsqueeze(0).expand(n, -1).to(device)
        vals.append(float(
            (model(img, t1, st) - model(img, t2, st)).abs().mean()))
    return sum(vals) / len(vals)


@torch.no_grad()
def ablation(model, loader, tokenizer, device, tag=""):
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
    print(f"\n=== 消融（整个验证集）{tag} ===")
    print(f"  完整输入   {base:.5f}")
    for k in ("图像置零", "指令打乱", "状态置零"):
        print(f"  {k}   {res[k]:.5f}   变化 {(res[k] - base) / base * 100:+.1f}%")
    print("  变化大 = 这一路真的在用；指令不变 = 语言通路塌缩")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "v1.2"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本（冒烟）")
    ap.add_argument("--val-limit", type=int, default=0,
                    help="只用前 N 条验证样本（快速对照；必须是 4 的倍数）")
    ap.add_argument("--pair-weight", type=float, default=1.0,
                    help="方块差分和区域差分损失的共同权重")
    ap.add_argument("--min-language-delta", type=float, default=0.05,
                    help="保存 checkpoint 前，方块/区域输出差异都必须达到此值")
    ap.add_argument("--min-epochs", type=int, default=30,
                    help="最早允许早停的轮数；语言门槛未通过时始终训练到最后")
    ap.add_argument("--init-ckpt", default="",
                    help="可选：从已有 V1.2 checkpoint 加载模型权重进行微调")
    ap.add_argument("--out", default="v1.2_zone")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    global PAIR_WEIGHT
    PAIR_WEIGHT = args.pair_weight

    np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    print(f"\n读取 {args.data} ...")
    tr = load_split(Path(args.data) / "train")
    va = load_split(Path(args.data) / "val")
    train_image_size = tuple(tr[0]["image"].shape[:2])
    val_image_size = tuple(va[0]["image"].shape[:2])
    if (train_image_size != val_image_size
            or train_image_size[0] != train_image_size[1]
            or train_image_size[0] not in (64, 128)):
        raise ValueError(
            "V1.2 train/val 图像分辨率必须一致，且为 64x64 或 128x128；"
            f"train={train_image_size}, val={val_image_size}")
    image_size = train_image_size[0]
    if args.limit:
        tr = tr[:args.limit // 4 * 4]          # ★ 必须是 4 的倍数，否则 tick 组被截断
    if args.val_limit:
        va = va[:args.val_limit // 4 * 4]
    if len(tr) % 4 or len(va) % 4 or args.batch % 4:
        raise ValueError(
            f"V1.2 每 tick 4 条，样本数和 batch 都必须是 4 的倍数："
            f"train={len(tr)} val={len(va)} batch={args.batch}")
    print(f"样本: train={len(tr)}  val={len(va)}（都是 4 的倍数）")
    print(f"图像分辨率: {image_size}x{image_size}")
    if image_size == 128 and args.batch > 32:
        print("[提示] 128x128 会显著增加显存占用；4GB 级 GPU 建议先用 --batch 16 或 32。")

    vocab = sorted(set("".join(s["instruction"] for s in tr)))
    tok = SimpleCharTokenizer(vocab, max_length=MAX_LENGTH)
    print(f"词表 {tok.vocab_size}　字符 {''.join(vocab)}")

    train_ds = RobotDataset(tr, tok)
    val_ds = RobotDataset(va, tok)

    train_loader = DataLoader(
        train_ds, batch_sampler=V12BatchSampler(
            len(train_ds), ticks_per_batch=args.batch // 4, seed=0))
    val_loader = DataLoader(
        val_ds, batch_sampler=V12BatchSampler(
            len(val_ds), ticks_per_batch=args.batch // 4, seed=1))

    model = LanguageConditionedMiniVLA(
        vocab_size=tok.vocab_size, max_length=MAX_LENGTH,
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        chunk_size=CHUNK,
        red_token_id=tok.stoi["红"], blue_token_id=tok.stoi["蓝"]).to(device)
    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if not init_path.is_file():
            raise FileNotFoundError(f"微调初始 checkpoint 不存在：{init_path}")
        init_ck = torch.load(init_path, map_location=device, weights_only=False)
        if init_ck.get("model_type") != "LanguageConditionedMiniVLA":
            raise ValueError(
                "--init-ckpt 必须是 LanguageConditionedMiniVLA checkpoint")
        if init_ck.get("vocab") != vocab:
            raise ValueError(
                "初始 checkpoint 与当前训练集词表不一致；拒绝按错误 token 顺序加载")
        expected = {
            "image_size": image_size,
            "max_length": MAX_LENGTH,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "chunk_size": CHUNK,
            "red_token_id": tok.stoi["红"],
            "blue_token_id": tok.stoi["蓝"],
        }
        mismatched = {
            key: (init_ck.get(key), value)
            for key, value in expected.items()
            if init_ck.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"初始 checkpoint 与当前数据/模型配置不兼容：{mismatched}")
        model.load_state_dict(init_ck["model_state"], strict=True)
        print(f"已从初始 checkpoint 加载权重：{init_path}")
        print("注意：优化器会重新初始化；这是权重微调，不是断点续训。")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    h = {"tr": [], "va": [], "pair": [], "dim": [], "gn": [],
         "lang": [], "lang_zone": []}
    best_val, best_state, bad = float("inf"), None, 0
    patience = 15

    live = None
    if not args.no_plot:
        try:
            live = Live(args.epochs)
        except Exception as e:
            print(f"[提示] 实时图开不起来（{e}）")

    val_batch = next(iter(val_loader))
    # A checkpoint that only fits image/state while ignoring instructions is
    # not useful for V1.2. Require both counterfactual paths to beat their
    # instruction-agnostic zero-output baselines before selecting by val loss.
    val_block_floor, val_zone_floor = pair_loss_floors(val_loader, device)
    print(f"\n开始训练 {args.epochs} 轮（batch {args.batch}，"
          f"pair_weight {args.pair_weight}）")
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        trn, _, gn = train_epoch(model, train_loader, opt, device)
        val, vdim = evaluate(model, val_loader, device)
        lang_block = language_delta(model, val_batch, tok, device)
        lang_zone = zone_language_delta(model, val_batch, tok, device)

        if not np.isfinite(trn["total"]) or not np.isfinite(val["total"]):
            print(f"\n[中止] 第 {ep} 轮出现 NaN/Inf，训练已发散。"
                  f"丢掉的权重不会保存。")
            break

        h["tr"].append(trn["total"]); h["va"].append(val["total"])
        h["pair"].append(trn["pair"]); h["dim"].append(vdim)
        h["gn"].append(gn)
        h["lang"].append(lang_block); h["lang_zone"].append(lang_zone)

        language_engaged = (
            lang_block >= args.min_language_delta
            and lang_zone >= args.min_language_delta
            and val["block_pair"] < 0.95 * val_block_floor
            and val["zone_pair"] < 0.95 * val_zone_floor
        )
        if language_engaged:
            if val["total"] < best_val:
                best_val, bad = val["total"], 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1

        if ep % max(1, args.epochs // 40) == 0 or ep == 1:
            print(f"epoch {ep:>3}/{args.epochs}  "
                  f"train tot {trn['total']:.5f} base {trn['base']:.5f} "
                  f"pair {trn['pair']:.5f} (block {trn['block_pair']:.5f}, "
                  f"zone {trn['zone_pair']:.5f})  "
                  f"val {val['total']:.5f} base {val['base']:.5f} "
                  f"pair {val['pair']:.5f} (block {val['block_pair']:.5f}, "
                  f"zone {val['zone_pair']:.5f})  "
                  f"方块差异 {lang_block:.4f} 区域差异 {lang_zone:.4f} "
                  f"语言门槛 {'OK' if language_engaged else '未达'}  |  "
                  f"grad {gn:.2f}  "
                  + "  ".join(f"{d} {v:.4f}" for d, v in zip(DIMS, vdim))
                  + f"  {time.time() - t0:.0f}s", flush=True)
        if live:
            live.update(h)
        # Before the first language-aware checkpoint exists, do not use the
        # patience counter: otherwise the model could stop before learning to
        # condition on either instruction role.
        if (ep >= args.min_epochs and best_state is not None
                and bad >= patience):
            print(f"\n语言条件 checkpoint 的验证 loss 连续 {patience} 轮没改善，"
                  f"第 {ep} 轮提前停止")
            break

    if best_state is None:
        print("\n[中止] 训练期间没有 checkpoint 同时满足方块/区域语言差异门槛，"
              "不保存会忽略指令的权重。")
        if live:
            live.close()
        return

    print("\n########## 消融：最后一轮 ##########")
    ablation(model, val_loader, tok, device, tag="最后一轮")

    model.load_state_dict(best_state)
    print(f"\n最佳验证 loss: {best_val:.5f}")

    ck = ROOT / "checkpoints"
    ck.mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state, "vocab": vocab,
        "model_type": "LanguageConditionedMiniVLA",
        "red_token_id": tok.stoi["红"], "blue_token_id": tok.stoi["蓝"],
        "max_length": MAX_LENGTH, "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM, "chunk_size": CHUNK,
        "image_size": image_size,
        "best_val": best_val, "data": str(args.data),
        "args": vars(args), "seed": 0,
    }, ck / f"{args.out}.pt")
    print(f"已保存 checkpoints/{args.out}.pt")

    print("\n########## 消融：best ##########")
    ablation(model, val_loader, tok, device, tag="best")
    if live:
        live.close()


if __name__ == "__main__":
    main()
