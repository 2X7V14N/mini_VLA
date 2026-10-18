# Mini-VLA

从零搭建、训练和评估一个小型视觉-语言-动作（VLA）模型。项目按版本逐步增加难度，当前正式版本是 **V1.1**：在 V1 的二维红蓝目标选择基础上，完成抓取、搬运和放置。

## 版本状态

| 版本 | 任务 | 状态 |
|---|---|---|
| V0 | 二维固定指令到达 | 已完成 |
| V1 | 二维红蓝双目标 + 自然语言指令，到达指定目标 | 已完成，保留为回归基线 |
| **V1.1** | 二维红蓝双目标抓取并放入固定区域 | **当前正式版本** |

V1.1 仍然是二维任务，不引入三维高度、关节角、逆运动学或 MuJoCo。这样可以单独验证视觉、语言、状态和动作闭环，而不把问题混入三维控制误差。

## V1.1 任务与接口

每个场景包含一个红色目标、一个蓝色目标、随机二维机器人起点和一个固定放置区域。指令指定要抓取的颜色。模型完成：移动到目标 → 闭合夹爪 → 搬运到放置区 → 打开夹爪。

```text
输入：
  image          (64, 64, 3) uint8 RGB
  instruction    红色/蓝色目标的自然语言指令
  robot_state    [x, y, holding]

输出：
  action_chunk   (4, 3)，四步 [dx, dy, dgrip]

dgrip 解码：
  dgrip <= -0.5  闭合
  dgrip >= +0.5  张开
  其余            保持
```

`holding` 是实际夹持状态，不是夹爪命令。抓取后目标按夹爪每个 tick 的位移跟随移动，禁止瞬移到放置区。

## V1.1 验收结果

本次正式基线使用 1000 个训练 rollout、200 个验证 rollout 和 200 个测试 rollout。

数据检查：

```text
39/39 项通过
场景跨集合重叠：0
MOVE_TO_OBJECT 状态方向 R²：-0.180
```

状态泄露检查表明，仅凭机器人状态不能推断目标方向，图像是必要输入。

模型闭环评测使用 200 次测试 episode：

| 指标 | 结果 |
|---|---:|
| 完整成功率 | **96.0%** |
| 红色指令成功率 | **96.0%** |
| 蓝色指令成功率 | **96.0%** |
| 颜色选择正确率 | **100.0%** |
| 捕获目标 | **96.5%** |
| 已捕获后搬运成功 | **99.5%** |
| 随机策略完整成功率 | 0.0% |

回归测试：`36 passed`。

详细记录见 [V1.1 发布记录](docs/V1.1_RELEASE.md)。

## 安装

建议使用 Python 3.10 或更高版本，并创建独立环境：

```powershell
conda create -n mini-vla python=3.10 -y
conda activate mini-vla
python -m pip install -e ".[train,dev]"
```

CPU 运行可以只安装基础依赖和测试依赖；训练时需要安装与本机 CUDA 匹配的 PyTorch。

## 运行 V1

V1 的代码和评测仍然保留，用于回归比较：

```powershell
python -m pytest -q
python -u scripts/train.py
python -u scripts/rollout.py
python -u scripts/eval_language.py
```

V1 输入为 `image + instruction + [x, y]`，输出为四步二维移动动作。V1 的实验记录见 [exp002_natural_language.md](docs/experiments/exp002_natural_language.md)。

## 运行 V1.1

### 1. 生成数据

```powershell
python scripts/record_grasp_place.py
```

默认生成 `data/v1.1`，包含 train/val/test 三个划分。数据文件是实验产物，默认被 `.gitignore` 排除；需要时可用相同脚本和随机种子重建。

### 2. 检查数据契约

```powershell
python scripts/check_v11_dataset.py --data data/v1.1
```

### 3. 训练模型

```powershell
python scripts/train_v11.py --data data/v1.1
```

训练结果默认写入 `checkpoints/v1.1_grasp_place.pt`。权重文件默认不提交到 Git，避免仓库被二进制实验产物污染。

### 4. 闭环评测

```powershell
python scripts/eval_v11_closed_loop.py \
  --data data/v1.1 \
  --ckpt checkpoints/v1.1_grasp_place.pt \
  --scenes 100
```

### 5. 交互式可视化演示

```powershell
python scripts/demo_grasp_place.py \
  --checkpoint checkpoints/v1.1_grasp_place.pt \
  --interval 0.05
```

窗口会显示红蓝目标、机器人起点、固定放置区、夹爪轨迹、当前阶段和 `holding` 状态。终端输入 `red`/`blue` 或 `红色`/`蓝色` 选择任务。

## 项目结构

```text
src/mini_vla/
  data/
    synthetic.py             V1 二维场景与数据
    episodes.py              V1 episode 生成
    grasp_place.py           V1.1 抓取放置环境
    grasp_place_expert.py    V1.1 专家动作
    grasp_place_data.py      V1.1 配对数据与加载
    grasp_place_render.py    V1.1 64×64 渲染
  models/                    视觉、文本、状态、融合和动作头
  evaluation/                V1 评测模块
scripts/
  train.py                   V1 训练
  rollout.py                 V1 闭环评测
  train_v11.py               V1.1 训练
  eval_v11_closed_loop.py    V1.1 模型闭环评测
  eval_grasp_place_expert.py V1.1 专家验收
  check_v11_dataset.py       V1.1 数据契约和泄露检查
  demo_grasp_place.py        V1.1 交互式模型演示
tests/                       V1 回归测试
docs/                        计划、路线图和实验记录
```

## 可复现性约定

每次实验应记录代码 commit、随机种子、数据划分、checkpoint、最佳验证 loss、闭环成功率和失败构成。V1.1 的配对样本必须保持：

```text
red_t, blue_t, red_t+1, blue_t+1, ...
```

同一对样本必须共享图像和机器人状态，才能保证语言反事实损失的含义正确。

## 许可证与后续

当前项目用于学习和实验验证。V1.1 是稳定基线；后续 V1.2 应在保持 V1/V1.1 回归测试通过的前提下，一次只增加一个主要难度因素。
