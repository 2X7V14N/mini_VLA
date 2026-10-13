"""闭环 rollout：让模型真正控制机器人走一趟。"""

import numpy as np
import torch

from mini_vla.data.synthetic import render_scene, generate_expert_chunk


# ===========================================================================
# 辅助函数：把一帧观测转成模型输入
# ===========================================================================

def _to_model_input(image, robot, instruction, tokenizer, device):
    """把 (图像, 状态, 指令) 转成三个模型输入张量。

    ⚠️ 这里的处理和 RobotDataset.__getitem__ 必须【一字不差】。
       如果哪里不一样（比如忘了 permute、或者用整数除以 255），
       模型收到的是分布不同的输入 —— 不会报错，只会输出垃圾。
    """
    # 图像：(64,64,3) uint8  ->  (1,3,64,64) float32, [0,1]
    img = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
    img = img.unsqueeze(0).to(device)

    # 指令："移动到红色目标"  ->  (1, 16) long
    tok = tokenizer.encode(instruction).unsqueeze(0).to(device)

    # 状态：(2,) float32  ->  (1, 2) float32
    st = torch.from_numpy(np.asarray(robot, dtype=np.float32)).unsqueeze(0).to(device)

    return img, tok, st


# ===========================================================================
# 三种策略：模型 / 专家 / 随机
# ===========================================================================
# 三个函数的返回格式完全一样，方便后面并排比较：
#     {"success": bool, "steps": int, "final_dist": float, "trajectory": (T,2)}

def run_episode(model, tokenizer, robot_start, red, blue, instruction,
                step_size=0.1, max_steps=50, success_radius=0.1, device="cuda"):
    """让【模型】控制机器人走一趟。"""
    goal = red if "红" in instruction else blue         # 指令决定目标
    robot = np.array(robot_start, dtype=np.float32)

    trajectory = [robot.copy()]
    model.eval()                                        # 关掉 dropout

    with torch.no_grad():                               # 推理不需要梯度
        for _ in range(max_steps):
            # ① 渲染当前观测
            image = render_scene(robot, red, blue)

            # ② 转成模型输入
            img, tok, st = _to_model_input(image, robot, instruction, tokenizer, device)

            # ③ 推理，得到动作块 (1, 4, 2)
            chunk = model(img, tok, st)

            # ④ 只执行【第一步】—— 和生成训练数据时的 receding horizon 一致
            action = chunk[0, 0].cpu().numpy()          # (2,)
            robot = robot + step_size * action
            trajectory.append(robot.copy())

            # ⑤ 到达就成功
            if np.linalg.norm(robot - goal) < success_radius:
                return {"success": True, "steps": len(trajectory) - 1,
                        "final_dist": float(np.linalg.norm(robot - goal)),
                        "trajectory": np.stack(trajectory)}

    # 跑满 max_steps 还没到 -> 失败
    return {"success": False, "steps": max_steps,
            "final_dist": float(np.linalg.norm(robot - goal)),
            "trajectory": np.stack(trajectory)}


def run_expert_episode(robot_start, red, blue, instruction,
                       step_size=0.1, max_steps=50, success_radius=0.1):
    """专家策略基线 —— 应该 100% 成功。用来验证 rollout 代码本身没问题。"""
    goal = red if "红" in instruction else blue
    robot = np.array(robot_start, dtype=np.float32)

    trajectory = [robot.copy()]
    for _ in range(max_steps):
        chunk = generate_expert_chunk(robot, goal, chunk_size=4, step_size=step_size)
        robot = robot + step_size * chunk[0]
        trajectory.append(robot.copy())
        if np.linalg.norm(robot - goal) < success_radius:
            return {"success": True, "steps": len(trajectory) - 1,
                    "final_dist": float(np.linalg.norm(robot - goal)),
                    "trajectory": np.stack(trajectory)}

    return {"success": False, "steps": max_steps,
            "final_dist": float(np.linalg.norm(robot - goal)),
            "trajectory": np.stack(trajectory)}


def run_random_episode(robot_start, red, blue, instruction, rng,
                       step_size=0.1, max_steps=50, success_radius=0.1):
    """随机策略基线 —— 应该接近 0%。给出"什么都不学"的下限。"""
    goal = red if "红" in instruction else blue
    robot = np.array(robot_start, dtype=np.float32)

    trajectory = [robot.copy()]
    for _ in range(max_steps):
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        robot = robot + step_size * action
        trajectory.append(robot.copy())
        if np.linalg.norm(robot - goal) < success_radius:
            return {"success": True, "steps": len(trajectory) - 1,
                    "final_dist": float(np.linalg.norm(robot - goal)),
                    "trajectory": np.stack(trajectory)}

    return {"success": False, "steps": max_steps,
            "final_dist": float(np.linalg.norm(robot - goal)),
            "trajectory": np.stack(trajectory)}
