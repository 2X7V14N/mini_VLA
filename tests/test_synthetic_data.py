"""合成数据的测试。"""
import numpy as np
import pytest
from mini_vla.data.instructions import (
    SEEN_TEMPLATES, UNSEEN_KNOWN_CHARS, UNSEEN_NOVEL_CHARS,
)

from mini_vla.data.synthetic import (
    _world_to_pixel,
    render_scene,
    generate_expert_chunk,
    generate_episode,
)

# 全文件共用的测试数据
ROBOT = np.array([-0.5, -0.5], dtype=np.float32)
RED   = np.array([ 0.6,  0.6], dtype=np.float32)
BLUE  = np.array([-0.6,  0.6], dtype=np.float32)

GREEN_COLOR = (0, 255, 0)
RED_COLOR   = (255, 0, 0)
BLUE_COLOR  = (0, 0, 255)

# 词表全集（SEEN + 两组 UNSEEN 的所有字符）
ALL_INSTRUCTIONS = (
    SEEN_TEMPLATES["红"] + SEEN_TEMPLATES["蓝"]
)


def _centroid_of(image, color):
    """找出某种颜色像素的质心 (row, col)。找不到返回 None。"""
    mask = np.all(image == np.array(color, dtype=np.uint8), axis=-1)
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    return rows.mean(), cols.mean()

def test_render_scene_shape_and_dtype():
    image = render_scene(ROBOT, RED, BLUE)
    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8

def test_world_to_pixel_corners():
    """四个角必须映射到图像的四个角。

    抓的是 y 轴翻转写错——那种错误所有形状断言都会通过。
    """
    assert _world_to_pixel(-1, -1) == (63, 0)     # 左下 -> row=63, col=0
    assert _world_to_pixel( 1,  1) == (0, 63)     # 右上 -> row=0,  col=63
    assert _world_to_pixel(-1,  1) == (0, 0)      # 左上
    assert _world_to_pixel( 1, -1) == (63, 63)    # 右下

def test_render_scene_colors_at_correct_positions():
    """每种颜色必须出现在对应的世界坐标处。"""
    image = render_scene(ROBOT, RED, BLUE)

    # 机器人（绿）应该在左下
    r, c = _centroid_of(image, (0, 255, 0))
    assert r is not None, "图像里找不到绿色机器人"
    exp_r, exp_c = _world_to_pixel(*ROBOT)
    assert abs(r - exp_r) < 1.5 and abs(c - exp_c) < 1.5    # 允许 1.5 像素误差

    # 红目标在右上
    r, c = _centroid_of(image, (255, 0, 0))
    exp_r, exp_c = _world_to_pixel(*RED)
    assert abs(r - exp_r) < 1.5 and abs(c - exp_c) < 1.5    

    # 蓝目标在左上
    r, c = _centroid_of(image, (0, 0, 255))
    exp_r, exp_c = _world_to_pixel(*BLUE)
    assert abs(r - exp_r) < 1.5 and abs(c - exp_c) < 1.5

def test_robot_drawn_on_top():
    """机器人和目标重合时，机器人必须可见。
    抓的是绘制顺序写错——那会让观测在任务成功的一刻突变。"""
    same = np.array([0.3, 0.3], dtype=np.float32)
    image = render_scene(same, same, BLUE)     # 机器人和红目标在同一位置
    assert _centroid_of(image, (0, 255, 0)) is not None, "机器人被目标盖住了"

def test_expert_chunk_shape_and_range():
    chunk = generate_expert_chunk(ROBOT, RED)
    assert chunk.shape == (4, 2)
    assert chunk.dtype == np.float32
    assert np.isfinite(chunk).all()
    assert (chunk >= -1).all()
    assert (chunk <= 1).all()

def test_expert_points_toward_goal():
    """第一步动作必须指向目标。

    抓的是坐标系搞反——那样动作会指向反方向，
    而形状和范围断言全都能通过。
    """
    chunk = generate_expert_chunk(ROBOT, RED)
    direction = RED - ROBOT
    # 点积 > 0 说明两个向量夹角小于 90 度，即方向大致一致
    assert np.dot(chunk[0], direction) > 0

def test_expert_does_not_modify_input():
    """函数不能改调用者传进来的数组。"""
    robot = ROBOT.copy()
    before = robot.copy()
    generate_expert_chunk(robot, RED)
    assert np.array_equal(robot, before), "robot_position 被函数修改了"

def test_expert_decelerates_near_goal():
    """离目标近时，动作幅度必须变小。

    抓的是 clip 逻辑写错——如果永远走满一步，
    机器人会在目标附近来回震荡。
    """
    far_pos  = np.array([-0.9, -0.9], dtype=np.float32)
    near_pos = np.array([ 0.55, 0.55], dtype=np.float32)

    far  = generate_expert_chunk(far_pos,  RED)
    near = generate_expert_chunk(near_pos, RED)

    assert np.linalg.norm(near[0]) < np.linalg.norm(far[0])

def test_expert_reaches_goal():
    """反复执行第一步动作，必须能在有限步内到达目标。

    这是数据质量的第一道防线：
    如果专家自己都走不到目标，生成的数据全是废的，
    模型学出来必然失败。
    """
    pos  = np.array([-0.9, -0.9], dtype=np.float32)
    goal = np.array([ 0.9,  0.9], dtype=np.float32)

    for _ in range(100):
        chunk = generate_expert_chunk(pos, goal)
        pos = pos + 0.1 * chunk[0]
        if np.linalg.norm(pos - goal) < 0.02:
            break

    assert np.linalg.norm(pos - goal) < 0.02, "专家策略走不到目标"

def _pixel_to_world(row, col, image_size=64):
    """像素 (row, col) -> 世界坐标 (x, y)。

    这是 _world_to_pixel 的【逆运算】：
        col = (x+1) * 0.5 * 63   =>   x = col / 63 * 2 - 1
        row = (1-y) * 0.5 * 63   =>   y = 1 - row / 63 * 2
    """
    x = col / (image_size - 1) * 2 - 1
    y = 1 - row / (image_size - 1) * 2
    return np.array([x, y], dtype=np.float32)


def _find_goal_world_position(image, color):
    """从图像里找出某种颜色的目标，返回它的世界坐标。"""
    centroid = _centroid_of(image, color)
    assert centroid is not None, f"图像里找不到颜色 {color}"
    row, col = centroid
    return _pixel_to_world(row, col)

def test_episode_is_reproducible():
    """相同种子的两个 rng 必须产生完全相同的 episode。

    抓的是"用了未播种的全局随机数"——
    那样每次跑结果都不同，实验无法复现。
    """
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)

    ep_a = generate_episode(rng_a)
    ep_b = generate_episode(rng_b)

    assert len(ep_a) == len(ep_b), "两次生成的帧数不同"

    for i, (sa, sb) in enumerate(zip(ep_a, ep_b)):
        assert sa["instruction"] == sb["instruction"], f"第 {i} 帧指令不同"
        assert np.array_equal(sa["image"], sb["image"]), f"第 {i} 帧图像不同"
        assert np.array_equal(sa["robot_state"], sb["robot_state"]), f"第 {i} 帧状态不同"
        assert np.array_equal(sa["action_chunk"], sb["action_chunk"]), f"第 {i} 帧动作不同"

def test_episode_respects_max_steps():
    """帧数不能超过 max_steps，也不能为空。"""
    ep = generate_episode(np.random.default_rng(0), max_steps=5)
    assert 0 < len(ep) <= 5


def test_episode_stops_when_reaching_goal():
    """到达目标后必须提前结束，不该跑满 max_steps。

    抓的是"忘了写 break"——那样机器人到达目标后会在原地
    打转，产生一堆近乎静止的帧，模型会偏向输出小动作。
    """
    ep = generate_episode(np.random.default_rng(0), max_steps=100)
    assert len(ep) < 100, "跑满了 100 步说明没有提前结束"

def test_episode_sample_schema():
    """episode 里每一帧都必须符合数据契约。"""
    episode = generate_episode(np.random.default_rng(0))
    assert len(episode) > 0, "生成了空 episode"

    for i, s in enumerate(episode):
        ctx = f"第 {i} 帧"

        # 图像
        assert s["image"].shape == (64, 64, 3), f"{ctx} image 形状错"
        assert s["image"].dtype == np.uint8, f"{ctx} image dtype 错"

        # 指令
        assert isinstance(s["instruction"], str), f"{ctx} instruction 不是字符串"
        pool = set(SEEN_TEMPLATES["红"] + SEEN_TEMPLATES["蓝"])
        assert s["instruction"] in pool, f"{ctx} instruction 内容非法"

        # 状态
        assert s["robot_state"].shape == (2,), f"{ctx} robot_state 形状错"
        assert s["robot_state"].dtype == np.float32, f"{ctx} robot_state dtype 错"
        assert np.isfinite(s["robot_state"]).all(), f"{ctx} robot_state 有 nan/inf"
        assert (-1 <= s["robot_state"]).all() and (s["robot_state"] <= 1).all(), \
            f"{ctx} robot_state 越界"

        # 动作
        assert s["action_chunk"].shape == (4, 2), f"{ctx} action_chunk 形状错"
        assert s["action_chunk"].dtype == np.float32, f"{ctx} action_chunk dtype 错"
        assert np.isfinite(s["action_chunk"]).all(), f"{ctx} action_chunk 有 nan/inf"
        assert (-1 <= s["action_chunk"]).all() and (s["action_chunk"] <= 1).all(), \
            f"{ctx} action_chunk 越界"

def test_instruction_matches_action_direction():
    """第一帧的动作必须指向【指令所说那个颜色】的目标。

    这个测试能抓到一种极隐蔽的错误：
    图像、形状、数值范围全都正确，但红蓝指令选反了。
    所有形状断言都测不出来。
    """
    rng = np.random.default_rng(7)
    checked = 0

    for _ in range(20):
        episode = generate_episode(rng)
        s = episode[0]                       # 只查第一帧：那时离目标最远，动作最大

        # ① 判断指令里有没有"红"字
        color = RED_COLOR if "红" in s["instruction"] else BLUE_COLOR

        # ② 从图像里找出那个目标的实际位置
        goal = _find_goal_world_position(s["image"], color)

        # ③ 动作方向应该和"目标 - 当前"一致
        direction = goal - s["robot_state"]
        dot = float(np.dot(s["action_chunk"][0], direction))

        assert dot > 0, (
            f"指令是 {s['instruction']}，但动作没指向对应目标\n"
            f"  robot_state = {s['robot_state']}\n"
            f"  推断的目标位置 = {goal}\n"
            f"  第一步动作 = {s['action_chunk'][0]}\n"
            f"  点积 = {dot}"
        )
        checked += 1

    assert checked == 20, "循环没跑满"

def test_paired_samples_share_inputs():
    """配对的两条样本，图像和状态完全相同，指令和动作必须不同。"""
    ep = generate_episode(np.random.default_rng(0), paired=True)
    assert len(ep) % 2 == 0

    for i in range(0, len(ep) - 1, 2):
        a, b = ep[i], ep[i + 1]
        assert np.array_equal(a["image"], b["image"])          # 同图
        assert np.array_equal(a["robot_state"], b["robot_state"])  # 同状态
        assert a["instruction"] != b["instruction"]            # 不同指令
        assert not np.allclose(a["action_chunk"], b["action_chunk"])  # 不同动作

def test_templates_are_balanced():
    """红蓝两组的模板必须一一对应，唯一的区别是颜色字。

    抓的是"某个颜色有更多变的说法"——那会污染
    "红蓝是否都能区分"的结论。
    """
    red, blue = SEEN_TEMPLATES["红"], SEEN_TEMPLATES["蓝"]
    assert len(red) == len(blue)
    for r, b in zip(red, blue):
        # 把颜色字替换成占位符后，两句必须完全相同
        assert r.replace("红", "X") == b.replace("蓝", "X"), f"{r} 和 {b} 不平行"


def test_instruction_comes_from_template_pool():
    """生成的指令必须来自模板池，不能是硬编码的字符串。"""
    ep = generate_episode(np.random.default_rng(0))
    pool = set(SEEN_TEMPLATES["红"] + SEEN_TEMPLATES["蓝"])
    for s in ep:
        assert s["instruction"] in pool


def test_paired_samples_use_different_colors():
    """配对样本必须指向【不同颜色】的目标。

    抓的是"other_instruction 也用了同一个颜色的模板"——
    那样 pair_loss 就学不到颜色区分了，而且不报错。
    """
    ep = generate_episode(np.random.default_rng(0), paired=True)
    for i in range(0, len(ep), 2):
        a, b = ep[i]["instruction"], ep[i + 1]["instruction"]
        assert ("红" in a) != ("红" in b), "配对样本用了同一种颜色"


def test_template_pool_has_no_single_color_words():
    """同一个词不能只出现在一个颜色的模板里。

    抓的是"泄漏"：如果「靠近」只出现在红色模板里，
    模型可以靠这个词作弊，而不是靠颜色。
    """
    red_words = set("".join(SEEN_TEMPLATES["红"]))
    blue_words = set("".join(SEEN_TEMPLATES["蓝"]))
    only_red = red_words - blue_words - {"红"}
    only_blue = blue_words - red_words - {"蓝"}
    assert not only_red, f"这些字只出现在红色模板里: {only_red}"
    assert not only_blue, f"这些字只出现在蓝色模板里: {only_blue}"
