"""V2.0 闭环评估：模型 vs 专家 vs 随机。

指标:
    成功率          最终位置距目标 < SUCCESS_RADIUS
    无效动作比例    IK 残差超过阈值的步数占比
    平均步数
    平均最终距离
"""

import numpy as np
import mujoco
import torch

from mini_vla.simulation.reach_env import (
    sample_positions, make_model, hover_target, make_camera, render,
    expert_chunk, IMAGE_SIZE,
    STEP_SIZE, CHUNK_SIZE, SUCCESS_RADIUS, MAX_STEPS,
    RED_INSTRUCTION, BLUE_INSTRUCTION,
)
from mini_vla.simulation.ik import ik
from mini_vla.models.text import SimpleCharTokenizer
from mini_vla.models.mini_vla import MiniVLA


N_SCENES = 100
SCENE_SEED = 12345      # 场景生成专用
RANDOM_SEED = 999       # 随机基线专用（和场景分开，避免互相影响）
IK_TOL = 1e-3           # IK 残差超过它 = 无效动作


# ===========================================================================
# 工具
# ===========================================================================

def _to_model_input(image, ee, instruction, tokenizer, device):
    """一帧观测 -> 三个模型输入张量。必须和 RobotDataset 一字不差。"""
    img = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    tok = tokenizer.encode(instruction).unsqueeze(0).to(device)
    st = torch.from_numpy(np.asarray(ee, dtype=np.float32)).unsqueeze(0).to(device)
    return img, tok, st


def sample_scene(rng):
    """采样一个场景：方块位置 + 初始关节角。"""
    pts = sample_positions(rng)
    start_qpos = [rng.uniform(-3.0, 3.0), rng.uniform(0.0, 1.5), rng.uniform(0.0, 2.0)]
    return pts, np.asarray(start_qpos, dtype=np.float64)


def make_env(scene):
    """按场景建 MuJoCo 环境。"""
    pts, start_qpos = scene
    model = make_model(pts[0], pts[1])
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    data.qpos[:3] = start_qpos
    mujoco.mj_forward(model, data)
    return model, data, sid, pts


def target_for(pts, instruction):
    """指令 -> 目标点。"""
    return hover_target(pts[0] if instruction == RED_INSTRUCTION else pts[1])


def result_of(data, sid, target, steps, invalid):
    ee = np.asarray(data.site(sid).xpos)
    dist = float(np.linalg.norm(ee - target))
    return {
        "success": dist < SUCCESS_RADIUS,
        "steps": steps,
        "final_dist": dist,
        "invalid_ratio": invalid / max(1, steps),
    }


# ===========================================================================
# 三种策略
# ===========================================================================

def run_expert_episode(scene, instruction):
    model, data, sid, pts = make_env(scene)
    target = target_for(pts, instruction)

    invalid, steps = 0, 0
    ee = np.asarray(data.site(sid).xpos)
    while steps < MAX_STEPS and np.linalg.norm(ee - target) >= SUCCESS_RADIUS:
        chunk = expert_chunk(ee, target, CHUNK_SIZE, STEP_SIZE)
        _, err = ik(model, data, ee + STEP_SIZE * chunk[0], sid)
        if err > IK_TOL:
            invalid += 1
        steps += 1
        ee = np.asarray(data.site(sid).xpos)          # ★ 必须重新读
    return result_of(data, sid, target, steps, invalid)


def run_random_episode(scene, instruction, rng):
    model, data, sid, pts = make_env(scene)
    target = target_for(pts, instruction)

    invalid, steps = 0, 0
    ee = np.asarray(data.site(sid).xpos)
    while steps < MAX_STEPS and np.linalg.norm(ee - target) >= SUCCESS_RADIUS:
        action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        _, err = ik(model, data, ee + STEP_SIZE * action, sid)
        if err > IK_TOL:
            invalid += 1
        steps += 1
        ee = np.asarray(data.site(sid).xpos)
    return result_of(data, sid, target, steps, invalid)


def run_model_episode(net, tokenizer, scene, instruction, device):
    model, data, sid, pts = make_env(scene)
    target = target_for(pts, instruction)

    camera = make_camera()
    renderer = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)

    invalid, steps = 0, 0
    ee = np.asarray(data.site(sid).xpos)
    while steps < MAX_STEPS and np.linalg.norm(ee - target) >= SUCCESS_RADIUS:
        image = render(model, data, camera, renderer)
        img_t, tok_t, st_t = _to_model_input(image, ee, instruction, tokenizer, device)

        with torch.no_grad():
            out = net(img_t, tok_t, st_t)             # (1, 4, 3)
        action = out[0, 0].cpu().numpy()              # (3,) 只取第一步
        assert action.shape == (3,), f"动作形状应为 (3,)，实际 {action.shape}"

        _, err = ik(model, data, ee + STEP_SIZE * action, sid)
        if err > IK_TOL:
            invalid += 1
        steps += 1
        ee = np.asarray(data.site(sid).xpos)

    renderer.close()
    return result_of(data, sid, target, steps, invalid)


# ===========================================================================
# 主流程
# ===========================================================================

def summarize(results):
    n = len(results)
    return {
        "success": 100 * np.mean([r["success"] for r in results]),
        "invalid": 100 * np.mean([r["invalid_ratio"] for r in results]),
        "steps": np.mean([r["steps"] for r in results]),
        "dist": np.mean([r["final_dist"] for r in results]),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载模型 ----
    ckpt = torch.load("checkpoints/v2_3d_reach.pt", map_location=device, weights_only=False)
    tokenizer = SimpleCharTokenizer(ckpt["vocab"], max_length=ckpt["max_length"])
    net = MiniVLA(vocab_size=tokenizer.vocab_size, max_length=ckpt["max_length"],
                  state_dim=3, action_dim=3, chunk_size=4)
    net.to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    print(f"模型 best_val={ckpt['best_val']:.6f}   设备={device}")

    # ---- 固定测试场景（三种策略共用）----
    scene_rng = np.random.default_rng(SCENE_SEED)
    scenes = [sample_scene(scene_rng) for _ in range(N_SCENES)]
    print(f"测试场景: {N_SCENES} 个")

    # ---- 三种策略 x 两条指令 ----
    random_rng = np.random.default_rng(RANDOM_SEED)
    rows = {}
    for name in ("模型", "专家", "随机"):
        res = {"红": [], "蓝": []}
        for scene in scenes:
            for instr, key in ((RED_INSTRUCTION, "红"), (BLUE_INSTRUCTION, "蓝")):
                if name == "模型":
                    r = run_model_episode(net, tokenizer, scene, instr, device)
                elif name == "专家":
                    r = run_expert_episode(scene, instr)
                else:
                    r = run_random_episode(scene, instr, random_rng)
                res[key].append(r)
        rows[name] = res
        print(f"  {name} 完成")

    # ---- 打印 ----
    print(f"\n{'策略':<6} {'红色':>8} {'蓝色':>8} {'总体':>8} "
          f"{'无效动作':>9} {'步数':>7} {'最终距离':>10}")
    print("-" * 62)
    for name, res in rows.items():
        both = res["红"] + res["蓝"]
        s = summarize(both)
        r_ = summarize(res["红"])["success"]
        b_ = summarize(res["蓝"])["success"]
        print(f"{name:<6} {r_:>7.1f}% {b_:>7.1f}% {s['success']:>7.1f}% "
              f"{s['invalid']:>8.1f}% {s['steps']:>7.1f} {s['dist']:>10.4f}")

    print(f"\n(每个格子 N={N_SCENES}；三种策略跑同一批场景)")


if __name__ == "__main__":
    main()
