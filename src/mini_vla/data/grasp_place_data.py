"""V1.1 数据生成：一条物理 rollout -> 每 tick 两条配对样本。

配对规则（计划 §7 第 3 步）：
    每个场景只跑【一条】物理 rollout；
    每个 timestep 对【同一张图 + 同一个状态】生成红、蓝两条相邻样本；
    holding == 0：红样本指向红，蓝样本指向蓝；
    holding == 1：两条都指向放置区、都保持闭合（pair_loss 自然趋零）；
    释放后 episode 结束，不再伪造另一种颜色的后续轨迹。

存盘顺序必须是 red_0, blue_0, red_1, blue_1, …
—— V1 的 PairedBatchSampler 假设相邻两条是同一个输入。
"""

import copy

import numpy as np

from mini_vla.data.grasp_place import GraspPlaceEnv, ZONE_CENTER
from mini_vla.data.grasp_place_expert import expert_action_for
from mini_vla.data.grasp_place_render import render

COLORS = ("red", "blue")

# 红蓝模板：把"红"换成"蓝"后句式完全一致，同一个 index 构成一对
TEMPLATES = {
    "red": ["抓住红色方块并放到目标区域",
            "把红色目标放进区域",
            "抓取红色方块并放置到指定位置"],
    "blue": ["抓住蓝色方块并放到目标区域",
             "把蓝色目标放进区域",
             "抓取蓝色方块并放置到指定位置"],
}


def plan_chunk(env, color, chunk_size=4):
    """从当前状态出发、指令为 color 时，接下来 chunk_size 步专家会做什么。

    ★ 用【环境副本】真实推演，不是近似公式：
      复制一份 env（机器人、两个物体、holding 全带上），把它的 color 换成
      color（**只在空手时换** —— 夹住时手上那个东西的颜色是既成事实），
      然后连走 chunk_size 步，收集专家动作。

      这样 chunk 的第 1 步一定和"真的执行时会做的动作"完全一致，
      后面几步也走的是真实环境动力学。
    """
    sim = copy.deepcopy(env)
    if not env.holding:
        sim.color = color

    out = []
    for _ in range(chunk_size):
        if sim.done:
            # ★ episode 已经结束：剩下的步补零。
            #   不能继续让专家"规划下一步"—— 那会变成"去抓另一个颜色"，
            #   红蓝两条就会分叉，破坏"夹住时两条标签相同"的规则。
            out.append(np.zeros(3, dtype=np.float32))
            continue
        a = expert_action_for(sim, color)
        out.append(a.copy())
        sim.step(a)
    return np.asarray(out, dtype=np.float32)



def record_rollout(scene, exec_color, template_index=0, chunk_size=4):
    """跑一条物理 rollout，返回 (配对样本列表, 结果 info)。

    exec_color 是【实际执行】的那条指令的颜色；样本里红蓝两条都会有。
    """
    env = GraspPlaceEnv(scene, exec_color)
    samples = []
    t = 0
    while not env.done and t < 1500:
        img = render(env)
        st = env.observation()["robot_state"]

        for color in COLORS:                     # ★ 严格 red_t, blue_t 相邻
            samples.append({
                "image": img.copy(),
                "instruction": TEMPLATES[color][template_index],
                "robot_state": st.copy(),
                "action_chunk": plan_chunk(env, color, chunk_size),
                "phase": env.phase,
                "target_color": color,
                "object_position": env.objects[color].copy(),
                "zone_position": ZONE_CENTER.copy(),
                "pair_id": t,
                "timestep": t,
                "template_index": template_index,
            })

        env.step(expert_action_for(env, exec_color))
        t += 1

    return samples, env.info()

def load_rollout_samples(path):
    """读一条 rollout 的 npz，展开成样本列表（保持配对顺序）。

    ★ 每个数组只能取【一次】。np.load 是惰性的：z["image"] 每次访问都会
      重新解压整个数组；而 img[i] 是【视图】，会一直引用住那个完整数组。
      如果写成 z["image"][i]，每条 rollout 的每个样本都会钉住一份完整图像
      —— 内存放大 N 倍，1000 条 rollout 会需要几十 GB（实测直接 OOM）。
    """
    z = np.load(path, allow_pickle=False)
    img = z["image"]
    instr = z["instruction"]
    st = z["robot_state"]
    chunk = z["action_chunk"]
    n = len(img)
    if n % 2:
        raise ValueError(f"{path} 样本数是奇数 {n}，配对结构坏了")
    out = []
    for i in range(n):
        out.append({
            "image": img[i],
            "instruction": str(instr[i]),
            "robot_state": st[i],
            "action_chunk": chunk[i],
        })
    return out


def load_split(split_dir):
    """按文件名顺序拼接整个 split。

    ★ 顺序即契约：拼接后第 2k / 2k+1 条必须还是同一对。
      PairedBatchSampler 只认下标奇偶，不认 pair_id。
    """
    from pathlib import Path
    files = sorted(Path(split_dir).glob("ep_*.npz"))
    if not files:
        raise FileNotFoundError(f"{split_dir} 里没有 ep_*.npz")
    samples = []
    for f in files:
        samples.extend(load_rollout_samples(f))
    if len(samples) % 2:
        raise ValueError(f"{split_dir} 总样本数是奇数，配对结构坏了")
    return samples
