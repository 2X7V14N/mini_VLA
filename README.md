# Mini-VLA

从零搭建、训练、评估一个小型视觉-语言-动作（VLA）模型。

**目标不是训练通用机器人模型，而是亲手走通 VLA 的每一环**：数据契约 → 编码器 → 多模态融合 → 动作预测 → 训练 → 闭环评估。

---

## 当前状态

### V2.0：三维末端到达（2026-10-15）

**环境从自写的 numpy 2D 场景升级为 MuJoCo 物理仿真，机器人从一个点变成 3 自由度机械臂。语言刻意退回固定 2 条 —— 只增加"三维"这一个主要变化。**

| 策略 | 红色 | 蓝色 | 总体 | **无效动作率** | 步数 | 最终距离 |
|---|---|---|---|---|---|---|
| **模型** | 100.0% | 99.0% | **99.5%** | **0.3%** | 6.0 | 0.0130 |
| 专家 | 100.0% | 100.0% | 100.0% | 0.3% | 5.9 | 0.0091 |
| 随机 | 1.0% | 2.0% | 1.5% | **21.6%** | 19.8 | 0.2848 |

| 指标 | 值 |
|---|---|
| 参数量 | 991,340 |
| 验证 loss | 0.004516 |
| 语言无关下限 | 0.1528（本模型比它低 34 倍） |
| 消融：指令打乱 | loss **+1032%** |
| 训练平台期 | **80 轮**（V1 是 37 轮 —— 三维确实更难） |

**⭐ 无效动作率是本轮最有价值的指标**：模型 0.3% 和专家相同，随机策略 21.6%。**这说明模型学会了"把末端保持在够程内"** —— 只看成功率看不出这个差别。

- 完整记录：[exp003_3d_reach.md](docs/experiments/exp003_3d_reach.md)
- 权重：`checkpoints/v2_3d_reach.pt`

> **结论的边界**：3 自由度简化臂，只做"到达方块上方"，**没有夹爪、接触、抓取**。语言本轮被刻意简化。

### V1：自然语言泛化（2026-10-14）

**指令从 2 条固定字符串扩展为多说法自然语言。验证模型理解的是【含义】而非记住固定文本。**

| 说法组 | 闭环成功率 | 评估次数 |
|---|---|---|
| SEEN（见过的说法） | **100.0%** | 1000 |
| UNSEEN_KNOWN（新说法 / 已知字） | **100.0%** | 600 |
| **UNSEEN_NOVEL（新说法 / 含生字）** | **100.0%** | 600 |
| 专家（参照） | 100.0% | 200 |

**最有说服力的是最后一行**：`「请找到红色方块并过去」` 经 tokenize 后有一半是 `<UNK>`（词表来自训练集），模型仍然认出"红"并走了过去。**如果是查表，这串字符它从没训练过。**

| 指标 | 值 |
|---|---|
| 参数量 | 991,784 |
| 验证 loss | 0.022528 |
| 消融：指令打乱 | loss **+9982%** |
| 红蓝输出差异 | 1.075 |

- 完整记录：[exp002_natural_language.md](docs/experiments/exp002_natural_language.md)
- 权重：`checkpoints/v1_natural_language.pt`

> **结论的边界**：模型学到的是"在句子里找红/蓝"—— 是关键词匹配层面的理解，**不是句法/语义理解**。词序、否定、空间关系、多步指令都未验证。

### V0：合成 2D 到达（2026-10-13）

固定指令，闭环成功率 99.8%，验证 loss 0.015538（比语言无关下限 0.3044 低 20 倍）。

- 完整记录：[exp001_synthetic_reach.md](docs/experiments/exp001_synthetic_reach.md)
- 权重：`checkpoints/v0_synthetic_reach.pt`

> **"验证 loss 低于语言无关下限 20 倍"是 V0 最关键的结论。**
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
│   ├── contracts.py              Observation / Action 数据契约（V0 遗留，未接入管道）
│   ├── data/
│   │   ├── synthetic.py          【V0/V1】合成 2D 数据生成器
│   │   ├── instructions.py       【V1】指令模板池（多说法）
│   │   ├── dataset.py            样本 → 张量；PairedBatchSampler（保持配对）
│   │   └── split.py              按 episode 划分
│   ├── models/                   ← V0/V1/V2 全部复用，未改代码
│   │   ├── vision.py             CNN 视觉编码器    (B,3,64,64) → (B,16,128)
│   │   ├── text.py               字符级 tokenizer + 文本编码器
│   │   ├── state.py              状态编码器        (B,D_state) → (B,1,128)
│   │   ├── fusion.py             多模态融合（Transformer）
│   │   ├── action.py             动作头            (B,128) → (B,H,D_action)
│   │   └── mini_vla.py           组装全部模块
│   ├── simulation/               ← 【V2.0 新增】MuJoCo 三维环境
│   │   ├── scene.xml             MJCF 场景（3 自由度臂 + 2 个方块）
│   │   ├── ik.py                 Jacobian IK 求解器
│   │   └── reach_env.py          采样 / 渲染 / 专家 / 数据生成
│   └── evaluation/
│       └── rollout.py            【V0/V1】2D 闭环 rollout
├── scripts/
│   ├── train.py                  训练入口（V2.0 的数据源 + state_dim=3）
│   ├── rollout.py                【V0/V1】2D 闭环评估
│   ├── eval_3d.py                【V2.0】3D 闭环评估
│   ├── view_scene.py             MuJoCo 交互窗口（调试用）
│   ├── render_test.py            渲染成图 + 像素统计（调相机用）
│   ├── ik_test.py                IK 单点测试
│   ├── reach_demo.py             末端在两点间移动（交互演示）
│   ├── chunk_test.py             expert_chunk 验证
│   └── episode_test.py           generate_episode 验证
├── tests/                        60+ 个测试
├── docs/
│   ├── ROADMAP.md                版本路线图与待办
│   └── experiments/              实验记录（exp001 / exp002 / exp003）
└── checkpoints/                  权重（.pt 不进版本库，.json 元数据进）
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
