import numpy as np
from mini_vla.data.instructions import SEEN_TEMPLATES

def _world_to_pixel(x: float, y: float, image_size: int = 64) -> tuple[int, int]:
    """世界坐标 (x, y) ∈ [-1,1]² -> 图像像素 (row, col)。

    ⚠️ 返回顺序是 (row, col)，和 numpy 的 image[row, col] 一致。
       注意 row 随 y 增大而【减小】，所以有 1-y 这个翻转。
    """
    col = int((x + 1) * 0.5 * (image_size - 1))
    row = int((1 - y) * 0.5 * (image_size - 1))
    return row, col

def _draw_disk(image, row: int, col: int, radius: int, color) -> None:
    """在 (row, col) 画一个实心圆。直接修改 image，不返回值。"""
    h, w = image.shape[:2]
    yy, xx = np.ogrid[:h, :w]        # yy 形状 (h,1)，xx 形状 (1,w)
    mask = (xx - col) ** 2 + (yy - row) ** 2 <= radius ** 2
    image[mask] = color

def render_scene(robot_position, red_position, blue_position,
                 image_size=64, target_radius=3, robot_radius=2):
    # ① 建画布
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    # ② 画红色目标
    row, col = _world_to_pixel(red_position[0], red_position[1], image_size)
    _draw_disk(image, row, col, target_radius, (255, 0, 0))      # 红色

    # ③ 画蓝色目标
    row, col = _world_to_pixel(blue_position[0], blue_position[1], image_size)
    _draw_disk(image, row, col, target_radius, (0, 0, 255))      # 蓝色

    # ④ 画机器人
    row, col = _world_to_pixel(robot_position[0], robot_position[1], image_size)
    _draw_disk(image, row, col, robot_radius, (0, 255, 0))       # 绿色

    return image

def generate_expert_chunk(robot_position, goal_position,
                          chunk_size=4, step_size=0.1):
    """生成专家动作块。返回 (chunk_size, 2) float32，值域 [-1, 1]。

    normalized_action = 真实位移 / step_size，是给模型学的标签。
    推理时：real_delta = predicted_action * step_size
    """
    # ① 复制一份模拟位置（不能改传进来的 robot_position）
    sim_position = np.array(robot_position, dtype=np.float32)

    actions = []
    for _ in range(chunk_size):
        direction = goal_position - sim_position

        delta = np.clip(direction, -step_size, step_size)

        normalized_action = delta / step_size
        actions.append(normalized_action)

        sim_position = sim_position + delta

    return np.asarray(actions, dtype=np.float32)

def _sample_two_goals(rng, min_dist=0.4, margin=0.8, max_tries=200):
    """采样红蓝目标位置，保证两者距离 > min_dist。"""
    for _ in range(max_tries):
        red  = rng.uniform(-margin, margin, size=2).astype(np.float32)
        blue = rng.uniform(-margin, margin, size=2).astype(np.float32)
        if np.linalg.norm(red - blue) > min_dist:
            return red, blue
    raise RuntimeError(f"采样失败：{max_tries} 次都没找到距离 > {min_dist} 的两点")

def _sample_robot(rng, goals, min_dist=0.2, margin=0.8, max_tries=200):
    """采样机器人位置，保证离所有目标都 > min_dist。"""
    for _ in range(max_tries):
        pos = rng.uniform(-margin, margin, size=2).astype(np.float32)
        if all(np.linalg.norm(pos - g) > min_dist for g in goals):
            return pos
    raise RuntimeError(...)

def generate_episode(rng, max_steps=20, step_size=0.1,
                     chunk_size=4, success_radius=0.1,
                     paired=False, templates=SEEN_TEMPLATES):                    # ← 新增参数
    """生成一个 episode 的所有训练样本。

    paired=True 时，每一帧会产出【两条】样本：
        同一张图 + 同一个状态 + 两条不同的指令 + 两个不同的动作标签
    这消除了"靠记忆绕过语言"的可能。
    """
    red, blue = _sample_two_goals(rng)
    robot = _sample_robot(rng, [red, blue])

    # ① 主指令（驱动机器人运动的那个）+ 另一条指令
    if rng.random() < 0.5:
        key, goal, other_key, other_goal = "红", red, "蓝", blue
    else:
        key, goal, other_key, other_goal = "蓝", blue, "红", red

    # 每条 episode 固定一种说法（和之前"固定一条指令"的道理一样）
    #
    # ⚠️ 配对样本用【同一个模板编号】：
    #     移动到红色目标  ↔  移动到蓝色目标
    #   而不是各自随机抽：
    #     移动到红色目标  ↔  靠近蓝色目标     ← 同时引入了句式差异
    #   这样 pair_loss 只需要关注【颜色差异】这一个变量，
    #   句式的不变性交给 base_loss（同一目标的所有说法，标签动作相同）。
    #   前提：两个颜色的模板池长度相同（tests 里有 test_templates_are_balanced 保证）。
    template_index = rng.integers(len(templates[key]))
    instruction = templates[key][template_index]
    other_instruction = templates[other_key][template_index]

    samples = []
    for _ in range(max_steps):
        image = render_scene(robot, red, blue)
        state = robot.astype(np.float32)

        # ② 主样本（动作先算出来存好，推进机器人时要用【同一个】）
        chunk_main = generate_expert_chunk(robot, goal, chunk_size, step_size)
        samples.append({
            "image": image,
            "instruction": instruction,
            "robot_state": state,
            "action_chunk": chunk_main,
        })

        # ③ 配对样本：同图、同状态、另一条指令、另一个目标
        if paired:
            samples.append({
                "image": image,                    # 同一张图（不用复制，只读）
                "instruction": other_instruction,
                "robot_state": state,              # 同一个状态
                "action_chunk": generate_expert_chunk(robot, other_goal, chunk_size, step_size),
            })

        robot = robot + step_size * chunk_main[0]
        if np.linalg.norm(robot - goal) < success_radius:
            break

    return samples
