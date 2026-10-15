from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler


from mini_vla.data.dataset import RobotDataset
from mini_vla.data.split import split_episodes, flatten
from mini_vla.simulation.reach_env import (
    generate_episode, RED_INSTRUCTION, BLUE_INSTRUCTION,
)
from mini_vla.models.mini_vla import MiniVLA
from mini_vla.models.text import SimpleCharTokenizer


SEED = 0
MAX_LENGTH = 16
N_EPISODES = 3000
BATCH_SIZE = 16
LR = 1e-3
N_EPOCHS = 120
PAIR_LOSS_WEIGHT = 1.0
GRAD_CLIP = 1.0          # 梯度范数上限，压住 pair_loss 造成的尖峰
DIAG_EVERY = 1          # 每多少轮打印一次"红蓝输出差异"


class PairedBatchSampler(Sampler[list[int]]):
    """Shuffle counterfactual red/blue samples as intact pairs.

    ``generate_episode(..., paired=True)`` emits samples in the order
    ``red_0, blue_0, red_1, blue_1, ...``.  Keeping each pair together lets
    the training loss supervise the instruction-dependent action difference.
    """

    def __init__(self, n_samples: int, pairs_per_batch: int, seed: int = SEED):
        if n_samples % 2 != 0:
            raise ValueError("paired training data must contain an even number of samples")
        if pairs_per_batch < 1:
            raise ValueError("pairs_per_batch must be positive")
        self.n_pairs = n_samples // 2
        self.pairs_per_batch = pairs_per_batch
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        pair_order = rng.permutation(self.n_pairs)
        for start in range(0, self.n_pairs, self.pairs_per_batch):
            pair_ids = pair_order[start:start + self.pairs_per_batch]
            yield [
                index
                for pair_id in pair_ids
                for index in (2 * int(pair_id), 2 * int(pair_id) + 1)
            ]

    def __len__(self):
        return (self.n_pairs + self.pairs_per_batch - 1) // self.pairs_per_batch


def main():
    # ① 固定随机种子
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ② 选设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"生成 {N_EPISODES} 个 episode...")
    rng = np.random.default_rng(SEED)
    episodes = []
    for i in range(N_EPISODES):
        episodes.append(generate_episode(rng, paired=True))
        if (i + 1) % 500 == 0:
            print(f"  已生成 {i + 1}/{N_EPISODES} 个 episode ...", flush=True)





    train_eps, val_eps, test_eps = split_episodes(episodes, seed=SEED)
    print(f"划分: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)} 个 episode")

    # ① 从【全部】数据建词表
    all_instructions = [s["instruction"] for ep in episodes for s in ep]
    vocab = sorted(set("".join(all_instructions)))
    tokenizer = SimpleCharTokenizer(vocab, max_length=MAX_LENGTH)
    print(f"词表大小: {tokenizer.vocab_size}")

    # ② 先把 episode 展开成样本，再建 Dataset
    train_ds = RobotDataset(flatten(train_eps), tokenizer)
    val_ds   = RobotDataset(flatten(val_eps),   tokenizer)
    test_ds  = RobotDataset(flatten(test_eps),  tokenizer)
    print(f"样本数: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PairedBatchSampler(
            len(train_ds), pairs_per_batch=BATCH_SIZE // 2, seed=SEED
        ),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=PairedBatchSampler(
            len(val_ds), pairs_per_batch=BATCH_SIZE // 2, seed=SEED + 1
        ),
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=PairedBatchSampler(
            len(test_ds), pairs_per_batch=BATCH_SIZE // 2, seed=SEED + 2
        ),
    )

    model = MiniVLA(
    vocab_size=tokenizer.vocab_size,
    max_length=MAX_LENGTH,
    state_dim=3,        
    action_dim=3,       
    chunk_size=4,       
    )
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    best_val = float("inf")
    best_state = None
    patience = 20
    bad_epochs = 0

    for epoch in range(1, N_EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)

        if epoch % DIAG_EVERY == 0 or epoch == 1:
            delta = language_diagnostic(model, train_loader, tokenizer, device)
            print(f"epoch {epoch:3d}  "
                  f"train base={tr['base']:.4f} pair={tr['pair']:.4f} tot={tr['total']:.4f}  "
                  f"val base={va['base']:.4f} pair={va['pair']:.4f} tot={va['total']:.4f}  "
                  f"红蓝差异={delta:.4f}", flush=True)

        # 早停逻辑（比的是 total —— 训练和验证用的是同一个目标函数）
        if va["total"] < best_val:
            best_val = va["total"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"\n验证 loss 连续 {patience} 轮没改善，第 {epoch} 轮提前停止")
                break

    model.load_state_dict(best_state)          # ← 恢复最佳权重
    print(f"最佳验证 loss: {best_val:.5f}")

    # 保存最佳模型。词表必须一起存 —— 推理时要用同一个 tokenizer，
    # 否则同一个字的 token id 会对不上。
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save({
        "model_state": best_state,
        "vocab": vocab,
        "max_length": MAX_LENGTH,
        "best_val": best_val,
    }, "checkpoints/best.pt")
    print("已保存 checkpoints/best.pt")

    # 消融诊断：训练集上看"有没有走记忆路径"，验证集上看"会不会用语言"
    print("\n########## 消融：训练集 ##########")
    ablation_check(model, train_loader, device, tokenizer)
    print("\n########## 消融：验证集 ##########")
    ablation_check(model, val_loader, device, tokenizer)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()                    # 切到训练模式
    sums = {"total": 0.0, "base": 0.0, "pair": 0.0}
    total_samples = 0

    for batch in loader:
        image       = batch["image"].to(device)
        token_ids   = batch["token_ids"].to(device)
        robot_state = batch["robot_state"].to(device)
        target      = batch["action_chunk"].to(device)

        optimizer.zero_grad()                                   # ①
        prediction = model(image, token_ids, robot_state)        # ②
        total, base, pair = compute_loss(prediction, target, criterion)
        total.backward()                                         # ④

        # 梯度裁剪：pair_loss 的梯度较大，偶尔会把参数推得过远
        # （V0 训练时出现过 loss 突然飙回 1.5 的震荡）。
        # 裁剪把梯度范数限制在 max_norm 以内，压住这种尖峰。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

        optimizer.step()                                         # ⑤

        n = image.size(0)
        sums["total"] += total.item() * n
        sums["base"]  += base.item()  * n
        sums["pair"]  += pair.item()  * n
        total_samples += n

    return {k: v / total_samples for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()                     # 关掉 dropout
    sums = {"total": 0.0, "base": 0.0, "pair": 0.0}
    total_samples = 0

    for batch in loader:
        image       = batch["image"].to(device)
        token_ids   = batch["token_ids"].to(device)
        robot_state = batch["robot_state"].to(device)
        target      = batch["action_chunk"].to(device)

        prediction = model(image, token_ids, robot_state)
        total, base, pair = compute_loss(prediction, target, criterion)

        n = image.size(0)
        sums["total"] += total.item() * n
        sums["base"]  += base.item()  * n
        sums["pair"]  += pair.item()  * n
        total_samples += n

    return {k: v / total_samples for k, v in sums.items()}


def compute_loss(prediction, target, criterion):
    """训练和验证用同一个目标函数。

    返回 (total, base, pair) 三个标量。
    【必须拆开返回】—— 只看 total 无法判断卡住的是动作误差还是语言差分误差，
    这两种情况要查的方向完全相反。
    """
    base_loss = criterion(prediction, target)

    # 偶数位和奇数位是配对样本：同图、同状态、【同句式】、不同颜色
    prediction_delta = prediction[0::2] - prediction[1::2]
    target_delta = target[0::2] - target[1::2]
    pair_loss = criterion(prediction_delta, target_delta)

    total = base_loss + PAIR_LOSS_WEIGHT * pair_loss
    return total, base_loss, pair_loss


@torch.no_grad()
def language_diagnostic(model, loader, tokenizer, device):
    """测"同图同状态、换指令，输出变多少"。

    这是最直接的"语言有没有在被使用"的指标：
        ≈ 0.00   模型完全忽略指令
        > 0.30   模型在用指令

    【在训练中途测】—— 不用等到最后才知道语言通路是在发育还是已经塌缩。
    V0 的教训：塌缩发生后梯度为零，永远出不来。越早看到越好。
    """
    was_training = model.training
    model.eval()
    batch = next(iter(loader))
    image = batch["image"].to(device)
    robot_state = batch["robot_state"].to(device)
    n = image.size(0)

    enc_r = tokenizer.encode(RED_INSTRUCTION).unsqueeze(0).expand(n, -1).to(device)
    enc_b = tokenizer.encode(BLUE_INSTRUCTION).unsqueeze(0).expand(n, -1).to(device)


    a_r = model(image, enc_r, robot_state)
    a_b = model(image, enc_b, robot_state)

    if was_training:
        model.train()
    return (a_r - a_b).abs().mean().item()


def ablation_check(model, loader, device, tokenizer):
    """检查模型是否真的用到了三路输入。

    做法：每次把一路输入破坏掉，看 loss 变不变。
    如果破坏某一路后 loss 几乎不变，说明模型根本没用它。
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        image       = batch["image"].to(device)
        token_ids   = batch["token_ids"].to(device)
        robot_state = batch["robot_state"].to(device)
        target      = batch["action_chunk"].to(device)

        base = nn.functional.mse_loss(model(image, token_ids, robot_state), target).item()

        # ① 图像置零
        zero_img = nn.functional.mse_loss(
            model(torch.zeros_like(image), token_ids, robot_state), target).item()

        # ② 指令打乱（把每条样本的指令换成同批次里另一条的）
        perm = torch.randperm(token_ids.size(0))
        shuf_tok = nn.functional.mse_loss(
            model(image, token_ids[perm], robot_state), target).item()

        # ③ 状态置零
        zero_state = nn.functional.mse_loss(
            model(image, token_ids, torch.zeros_like(robot_state)), target).item()

        # ④ Directly measure whether the instruction changes the action.
        n = token_ids.size(0)
        red_tokens  = tokenizer.encode(RED_INSTRUCTION).unsqueeze(0).expand(n, -1).to(device)
        blue_tokens = tokenizer.encode(BLUE_INSTRUCTION).unsqueeze(0).expand(n, -1).to(device)

        red_action = model(image, red_tokens, robot_state)
        blue_action = model(image, blue_tokens, robot_state)
        instruction_delta = (red_action - blue_action).abs().mean().item()

    print("\n=== 消融诊断（破坏一路输入，看 loss 变化）===")
    print(f"  完整输入      {base:.4f}")
    print(f"  图像置零      {zero_img:.4f}   变化 {(zero_img-base)/base*100:+.0f}%")
    print(f"  指令打乱      {shuf_tok:.4f}   变化 {(shuf_tok-base)/base*100:+.0f}%")
    print(f"  状态置零      {zero_state:.4f}   变化 {(zero_state-base)/base*100:+.0f}%")
    print(f"  红蓝输出差异  {instruction_delta:.5f}      (≈0 表示忽略语言，>0.1 表示在用)")
    print("\n  判断：变化很大 = 这一路真的在用；几乎不变 = 这一路被忽略了")

if __name__ == "__main__":
    main()
