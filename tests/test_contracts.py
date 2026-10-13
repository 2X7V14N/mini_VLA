"""数据契约的测试。

这些测试回答两个方向的同一个问题：
    合法输入 —— 必须被接受
    非法输入 —— 必须被拒绝

只测一个方向是不够的：
    只测合法 -> 漏掉"合法输入被误拒"（比如检查写反、elif 在正常路径上执行）
    只测非法 -> 漏掉"非法输入被放行"（比如 and 写成了 or）

运行：
    conda run -n mini-vla python -m pytest tests/ -v
"""

import numpy as np
import pytest

from mini_vla.contracts import Action, Observation


def make_valid_observation():
    """构造一个完全合法的 Observation，供各个测试复用。

    单独抽出来的好处：合法输入的定义只有一处。将来契约变了
    （比如图像尺寸从 64 改成 96），只需要改这里，不用改六个测试。
    """
    return Observation(
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        instruction="移动到红色方块",
        robot_state=np.zeros(2, dtype=np.float32),
    )


# ===========================================================================
# Observation
# ===========================================================================


def test_valid_observation():
    """合法输入必须构造成功，且字段值原样保留。

    这个测试专门抓"合法输入被误拒"这一类 bug。它看起来平淡，
    但正是它能发现"某个检查写反了"或"elif 在正常路径上被触发"——
    那两种 bug 会让整个 pipeline 一步都跑不动。
    """
    obs = make_valid_observation()

    assert obs.image.shape == (64, 64, 3)

    # 注意：numpy 数组不能用 == 断言。
    # obs.robot_state == np.zeros(2) 返回的是一个布尔"数组"，
    # assert 撞上多元素数组会抛 "truth value is ambiguous"。
    # 必须用 np.array_equal，它返回单个 bool。
    assert np.array_equal(obs.robot_state, np.zeros(2, dtype=np.float32))

    assert obs.instruction == "移动到红色方块"


def test_invalid_image_shape():
    """image 形状不是 (64, 64, 3) 时必须抛 ValueError。

    只把 image 一个字段改坏，其余保持合法。
    这样一旦测试通过，就能确定拦截它的确实是形状检查，
    而不是碰巧被别的检查拦下的。
    """
    with pytest.raises(ValueError):
        Observation(
            image=np.zeros((32, 32, 3), dtype=np.uint8),  # 形状错
            instruction="移动到红色方块",
            robot_state=np.zeros(2, dtype=np.float32),
        )


def test_invalid_instruction():
    """空字符串和全空白字符串都必须被拒绝。

    这两个 case 走的是不同的代码分支：
        ""      -> len(...) == 0 就能抓到
        "   "   -> 长度是 3，必须靠 strip() 才能抓到

    所以两个都要测。只测 "" 的话，strip() 没写也发现不了。
    """
    # case 1: 空字符串
    with pytest.raises(ValueError):
        Observation(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            instruction="",
            robot_state=np.zeros(2, dtype=np.float32),
        )

    # case 2: 只有空白字符，长度不为 0
    with pytest.raises(ValueError):
        Observation(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            instruction="   ",
            robot_state=np.zeros(2, dtype=np.float32),
        )


def test_invalid_robot_state_range():
    """越界必须被拒绝，而且要测两个方向。

    为什么两个方向都要测：
    范围检查最典型的写法错误是把 or 写成 and，
    而那种错误通常只在一个方向上暴露——
    比如 [5.0, 0.0] 能溜过去，[0.0, -5.0] 却被拦住。
    只测一边，这个 bug 会活下来。
    """
    # 越上界：某个值 > 1
    with pytest.raises(ValueError):
        Observation(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            instruction="移动到红色方块",
            robot_state=np.array([5.0, 0.0], dtype=np.float32),
        )

    # 越下界：某个值 < -1
    with pytest.raises(ValueError):
        Observation(
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            instruction="移动到红色方块",
            robot_state=np.array([0.0, -5.0], dtype=np.float32),
        )


# ===========================================================================
# Action
# ===========================================================================


def test_valid_action():
    """合法 Action 必须构造成功，且字段值原样保留。"""
    action = Action(delta_xy=np.array([0.1, 0.2], dtype=np.float32))

    assert action.delta_xy.shape == (2,)
    assert np.array_equal(
        action.delta_xy, np.array([0.1, 0.2], dtype=np.float32)
    )


def test_invalid_action_range():
    """delta_xy 越界时必须抛 ValueError。

    和 robot_state 同理，两个方向都值得测，这里各写一个断言块。
    """
    # 越上界
    with pytest.raises(ValueError):
        Action(delta_xy=np.array([5.0, 0.0], dtype=np.float32))

    # 越下界
    with pytest.raises(ValueError):
        Action(delta_xy=np.array([0.0, -5.0], dtype=np.float32))
