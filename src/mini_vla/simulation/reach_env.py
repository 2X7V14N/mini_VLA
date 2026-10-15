from pathlib import Path

import numpy as np
import mujoco

from mini_vla.simulation.ik import ik
# 场景 XML 模板（含 {red_pos} / {blue_pos} 占位符）
TEMPLATE_PATH = Path(__file__).parent / "scene.xml"

# ---- 可调的几何参数（都由实测确定，不要随便改）----
BLOCK_Z = 0.04           # 方块中心高度（躺着桌面上）
HOVER = 0.10             # 悬停高度：末端停在方块上方多远
R_MIN, R_MAX = 0.10, 0.26   # 方块到机械臂底座的水平距离范围（够程实测 0.28）
MIN_SEP = 0.12           # 两个方块的最小间距（保证图像上分得开）

# 相机参数 —— 用你调好的那组
CAM_LOOKAT    = [0.00, 0.00, 0.12]
CAM_DISTANCE  = 0.62
CAM_AZIMUTH   = 135
CAM_ELEVATION = -30
IMAGE_SIZE    = 64

def load_template() -> str:
    """读 XML 模板。"""
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def make_model(red_xy, blue_xy) -> mujoco.MjModel:
    """按给定的方块水平位置，生成一个模型。"""
    xml = load_template().format(
        red_pos=f"{red_xy[0]:.4f} {red_xy[1]:.4f} {BLOCK_Z}",
        blue_pos=f"{blue_xy[0]:.4f} {blue_xy[1]:.4f} {BLOCK_Z}",
    )
    return mujoco.MjModel.from_xml_string(xml)


def sample_positions(rng, n=2, max_tries=500):
    """在【够程内】采样 n 个方块的水平位置。

    约束:
        1. 半径在 [R_MIN, R_MAX] 内  -> 保证 IK 有解
        2. 两两距离 > MIN_SEP        -> 图像上分得开

    注意：这里【不检查 IK】，因为半径范围是实测出来的，
          留了 2cm 余量，正常情况下都能到达。
    """
    for _ in range(max_tries):
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(R_MIN, R_MAX, n)
        pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

        # 检查两两距离
        ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pts[i] - pts[j]) < MIN_SEP:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return pts.astype(np.float32)

    raise RuntimeError(f"采样失败：{max_tries} 次都没找到满足约束的 {n} 个位置")


def hover_target(block_xy) -> np.ndarray:
    """方块上方悬停点的世界坐标 —— 这就是专家要到达的目标。"""
    return np.array([block_xy[0], block_xy[1], BLOCK_Z + HOVER], dtype=np.float64)

def make_camera(template_model=None) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = CAM_LOOKAT
    cam.distance = CAM_DISTANCE
    cam.azimuth = CAM_AZIMUTH
    cam.elevation = CAM_ELEVATION
    return cam


def render(model, data, camera, renderer) -> np.ndarray:
    """渲染一帧 (64,64,3) uint8。

    渲染器要【复用】，不要每次新建 —— 创建 Renderer 有开销。
    """
    renderer.update_scene(data, camera)
    return renderer.render()

def expert_action(data, target, step_size):
    """专家策略：朝目标走一步。

    返回 (dx, dy, dz) ∈ [-1, 1]³，和模型输出同格式。
    """
    ee = np.asarray(data.site("ee_site").xpos)
    direction = np.asarray(target) - ee
    dist = np.linalg.norm(direction)
    if dist < 1e-8:
        return np.zeros(3, dtype=np.float32)

    # 比例控制：远了走满，近了收小（和 V1 的 generate_expert_chunk 同一个思路）
    step = min(1.0, dist / step_size)
    return (direction / dist * step).astype(np.float32)

def expert_chunk(current_ee, target, chunk_size=4, step_size=0.05):
    """未来 chunk_size 步的专家动作，形状 (chunk_size, 3)。
    返回: (chunk_size, 3) float32，每个分量 ∈ [-1, 1]
    """
    actions = []
    pos = np.asarray(current_ee, dtype=np.float64).copy()     # 副本，不动原数据

    for _ in range(chunk_size):
        direction = np.asarray(target, dtype=np.float64) - pos
        dist = np.linalg.norm(direction)

        if dist < 1e-8:                    # 已经到目标了
            actions.append(np.zeros(3))
            continue

        step = min(1.0, dist / step_size)  # 近了就收小步（避免冲过头）
        a = direction / dist * step
        actions.append(a)
        pos = pos + step_size * a          # 在【副本】上前进，不动真实状态

    return np.asarray(actions, dtype=np.float32)

# ---- episode 相关参数 ----
CHUNK_SIZE = 4
STEP_SIZE = 0.05
SUCCESS_RADIUS = 0.03
MAX_STEPS = 20

RED_INSTRUCTION = "移动到红色方块上方"
BLUE_INSTRUCTION = "移动到蓝色方块上方"


def generate_episode(rng, paired=True):
    """生成一个 episode 的所有样本。返回 list[dict]。

    paired=True 时，每一帧产出【两条】样本：
        同一张图 + 同一个末端状态 + 两条不同的指令 + 两个不同的动作标签
    这堵死了"靠记忆绕过语言"的路 —— 和 V1 用的是同一套方法。

    每个样本：
        image         (64, 64, 3) uint8
        instruction   str
        robot_state   (3,)  float32   末端世界坐标
        action_chunk  (4, 3) float32
    """
    # ① 采样场景
    pts = sample_positions(rng)
    red_xy, blue_xy = pts[0], pts[1]
    model = make_model(red_xy, blue_xy)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

    # ② 随机初始关节角（保证末端在可达空间内）
    data.qpos[:3] = [rng.uniform(-3.0, 3.0), rng.uniform(0.0, 1.5), rng.uniform(0.0, 2.0)]
    mujoco.mj_forward(model, data)

    # ③ 两个悬停目标
    target_red = hover_target(red_xy)
    target_blue = hover_target(blue_xy)

    # ④ 随机决定哪条指令驱动运动（避免数据偏向某一种颜色）
    if rng.random() < 0.5:
        main_instr, main_target = RED_INSTRUCTION, target_red
        other_instr, other_target = BLUE_INSTRUCTION, target_blue
    else:
        main_instr, main_target = BLUE_INSTRUCTION, target_blue
        other_instr, other_target = RED_INSTRUCTION, target_red

    camera = make_camera()
    renderer = mujoco.Renderer(model, IMAGE_SIZE, IMAGE_SIZE)

    samples = []
    for _ in range(MAX_STEPS):
        image = render(model, data, camera, renderer)
        ee = np.asarray(data.site(sid).xpos, dtype=np.float32)

        # 主样本
        samples.append({
            "image": image,
            "instruction": main_instr,
            "robot_state": ee,
            "action_chunk": expert_chunk(ee, main_target, CHUNK_SIZE, STEP_SIZE),
        })

        # 配对样本：同图、同状态、另一条指令、另一个目标
        if paired:
            samples.append({
                "image": image,
                "instruction": other_instr,
                "robot_state": ee,
                "action_chunk": expert_chunk(ee, other_target, CHUNK_SIZE, STEP_SIZE),
            })

        # ⑤ 到达就停
        if np.linalg.norm(ee - main_target) < SUCCESS_RADIUS:
            break

        # ⑥ 执行主指令的第一步（receding horizon —— 和 V1 一致）
        a0 = expert_chunk(ee, main_target, CHUNK_SIZE, STEP_SIZE)[0]
        ik(model, data, ee + STEP_SIZE * a0, sid)

    return samples
