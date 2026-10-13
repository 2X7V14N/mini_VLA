# Mini-VLA

从零搭建、训练、评估一个小型视觉-语言-动作（VLA）模型。

**目标不是训练通用机器人模型，而是亲手走通 VLA 的每一环**：数据契约 → 编码器 → 多模态融合 → 动作预测 → 训练 → 闭环评估。

---

## 当前状态

**v1（2026-10-13）：合成 2D 到达任务，闭环成功率 99.8%，与专家（100%）几乎持平。**

| 指标 | 值 |
|---|---|
| 参数量 | 990,632 |
| 验证 loss | 0.015538 |
| 语言无关下限 | 0.3044（本模型比它低 20 倍） |
| 闭环成功率（模型 / 专家 / 随机） | **99.8% / 100% / 7.5%** |
| 消融：指令打乱 | loss **+8672%** |

**完整实验记录**：[docs/experiments/exp001_synthetic_reach.md](docs/experiments/exp001_synthetic_reach.md)
**权重**：`checkpoints/v0_synthetic_reach.pt`（元数据见同名 `.json`）

> **"验证 loss 低于语言无关下限 20 倍"是本项目最关键的结论。**
> 任何完全不看指令的模型，loss 都不可能低于 0.3044。低于它，只可能是真的在用语言。

---

## 任务

```
输入:  图像 (64,64,3)  指令「移动到红色/蓝色目标」  机器人状态 [x, y]
输出:  未来 4 步动作块 (4, 2)，每步 [dx, dy]，归一化到 [-1, 1]
成功:  最终位置距指令指定的目标 < 0.1
```

**两个刻意的设计，保证三条输入支路都不可省略**：

- 目标位置**不在状态里** → 必须从图像看
- 场景里**有两个不同颜色的目标** → 指令必须决定选哪个

---

## 快速开始

```bash
conda activate mini-vla

# 1. 环境自检
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. 跑测试（应该有 60+ 个通过）
pytest -q

# 3. 训练（约 2-4 小时，早停可提前结束）
python -u scripts/train.py

# 4. 闭环评估
python -u scripts/rollout.py
```

---

## 项目结构

```
mini-VLA/
├── src/mini_vla/
│   ├── contracts.py              Observation / Action 数据契约
│   ├── data/
│   │   ├── synthetic.py          合成数据生成器（渲染 + 专家动作 + episode）
│   │   ├── dataset.py            样本 → 张量；PairedBatchSampler（保持配对）
│   │   └── split.py              按 episode 划分
│   ├── models/
│   │   ├── vision.py             CNN 视觉编码器    (B,3,64,64) → (B,16,128)
│   │   ├── text.py               字符级 tokenizer + 文本编码器
│   │   ├── state.py              状态编码器        (B,2) → (B,1,128)
│   │   ├── fusion.py             多模态融合（Transformer）
│   │   ├── action.py             动作头            (B,128) → (B,4,2)
│   │   └── mini_vla.py           组装全部模块
│   └── evaluation/
│       └── rollout.py            闭环 rollout + 专家/随机基线
├── scripts/
│   ├── train.py                  训练入口
│   └── rollout.py                闭环评估入口
├── tests/                        60+ 个测试
├── docs/experiments/             实验记录
└── checkpoints/                  权重（不进版本库）
```

---

## 核心设计决策

**为什么动作头输出 `(B, 4, 2)` 而不是 `(B, 8)`**
保留时间步维度，让"取第一步"（`chunk[:, 0]`）和"算相邻步差"（`chunk[:, 1:] - chunk[:, :-1]`）不用手写下标。**改 `chunk_size` 时上层代码不用动。**

**为什么用 action chunk + receding horizon**
单步预测每步都要重推理，相邻两次预测互不耦合 → 动作抖动。chunk 让 H 步来自同一次前向，天然自洽。执行时只走第一步再重新观察，训练和推理保持一致。

**为什么融合模块入口必须加 `LayerNorm`**
视觉 token 的标准差是 0.0078，文本是 0.87 —— 差 110 倍。不归一化的话，注意力会被文本完全主导。

**为什么需要 `pair_loss`（本项目最重要的发现）**
只用 `MSE(pred, target)` 时，模型会收敛到"输出红蓝目标的平均方向"——因为**逐样本的 loss 里没有交叉项**，平均策略对每条样本都不算太糟。模型完全忽略指令，而且 loss 正常下降。

`pair_loss = MSE(pred_红 - pred_蓝, target_红 - target_蓝)` 引入交叉项后，平均策略的 loss 从 0.31 涨到 2.0，**梯度被迫去读指令**。

详细分析见 [exp001](docs/experiments/exp001_synthetic_reach.md) 第四、五节。

---

## 已知限制

- 只适用于 64×64 合成 2D 场景，动作 2 维
- 只有两条固定指令，语言分支实质是**二分类**，不是真正的语言理解
- 任务动力学平凡（无接触、无摩擦），是模仿学习的"最好情况"
- 闭环成功率对测试场景的随机数流敏感（见 exp001 第九节）

**下一步**：接入真实数据（LeRobot / LIBERO），验证管线能否迁移到 256×256 双相机、7 自由度动作。

---

## 参考

- [ACT: Action Chunking with Transformers](https://arxiv.org/abs/2304.13705) — action chunk 与 receding horizon
- [Octo](https://arxiv.org/abs/2405.12213)、[SmolVLA](https://arxiv.org/abs/2506.01844) — 小型 VLA 的架构参考
