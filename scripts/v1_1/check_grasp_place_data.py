"""V1.1 第 3 步（上）验收：录 5 条 rollout，检查配对结构。

    python scripts/v1_1/check_grasp_place_data.py              # 5 条
    python scripts/v1_1/check_grasp_place_data.py --rollouts 20

只查结构，不存大文件：
    ① 样本顺序是 red_t, blue_t, red_{t+1}, blue_{t+1}, …
    ② 同一对内：图像逐元素相同、状态逐元素相同
    ③ 同一对内：指令只差颜色，模板编号相同
    ④ holding == 0 时两个动作指向不同目标；holding == 1 时两者相同
    ⑤ 形状 / dtype 符合契约
    ⑥ 用录下来的动作回放，结果和专家一致
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mini_vla.data.grasp_place import (      # noqa: E402
    GraspPlaceEnv, sample_scene, MAX_STEP, ZONE_CENTER,
)
from mini_vla.data.grasp_place_data import record_rollout, TEMPLATES   # noqa: E402
from mini_vla.data.grasp_place_expert import expert_action_for         # noqa: E402

PASS, FAIL = "[OK]  ", "[FAIL]"
nc = nf = 0


def check(name, cond, detail=""):
    global nc, nf
    nc += 1
    if not cond:
        nf += 1
    print(f"  {PASS if cond else FAIL} {name:<44}{detail}")


def angle_between(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(
        np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    total_pairs = 0

    for i in range(args.rollouts):
        scene = sample_scene(rng)
        exec_color = "red" if i % 2 == 0 else "blue"
        tidx = i % len(TEMPLATES["red"])
        samples, info = record_rollout(scene, exec_color, template_index=tidx)

        print(f"\n--- rollout {i}（实际执行 {exec_color}，模板 {tidx}，"
              f"{len(samples)} 条样本 / {len(samples) // 2} 对，"
              f"{info['phase']}）---")

        check("样本数是偶数", len(samples) % 2 == 0, str(len(samples)))

        n_bad_img = n_bad_state = n_bad_instr = n_bad_refl = 0
        for t in range(len(samples) // 2):
            a, b = samples[2 * t], samples[2 * t + 1]

            # ① 顺序：红在前、蓝在后
            if a["target_color"] != "red" or b["target_color"] != "blue":
                n_bad_refl += 1

            # ② 图像 / 状态逐元素相同
            if not np.array_equal(a["image"], b["image"]):
                n_bad_img += 1
            if not np.array_equal(a["robot_state"], b["robot_state"]):
                n_bad_state += 1

            # ③ 指令只差颜色、模板编号相同
            if (a["template_index"] != b["template_index"]
                    or a["instruction"] == b["instruction"]):
                n_bad_instr += 1

        check("配对顺序 red_t, blue_t 严格相邻", n_bad_refl == 0, f"错位 {n_bad_refl}")
        check("同一对图像逐元素相同", n_bad_img == 0, f"不等 {n_bad_img}")
        check("同一对状态逐元素相同", n_bad_state == 0, f"不等 {n_bad_state}")
        check("同一对指令只差颜色、模板同编号", n_bad_instr == 0, f"异常 {n_bad_instr}")

        # ④ holding==0 -> 两个动作指向不同目标
        bad_dir = bad_same = 0
        n_cmp = 0
        for t in range(len(samples) // 2):
            a, b = samples[2 * t], samples[2 * t + 1]
            holding = a["robot_state"][2] > 0.5
            if not holding:
                da = a["action_chunk"][0][:2]
                db = b["action_chunk"][0][:2]
                na, nb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
                # ★ 只有【两边都在动】时方向比较才有意义：
                #   到了目标的那几 tick，红样本动作是"原地闭合"(位移 0)，
                #   零向量的夹角无定义，原来会被误判成"雷同"。
                if na > 0.05 and nb > 0.05:
                    n_cmp += 1
                    if angle_between(da, db) < 5.0:
                        bad_same += 1
            else:
                if not np.allclose(a["action_chunk"], b["action_chunk"],
                                   atol=1e-6):
                    bad_dir += 1
        check("空手时红蓝动作方向不同", bad_same == 0,
              f"雷同 {bad_same}（可比 {n_cmp} 对）")
        check("夹住时红蓝动作完全相同", bad_dir == 0, f"不同 {bad_dir}")


        # ⑤ 契约
        s0 = samples[0]
        check("image 形状 (64,64,3) uint8",
              s0["image"].shape == (64, 64, 3) and s0["image"].dtype == np.uint8,
              str(s0["image"].shape))
        check("robot_state 形状 (3,) float32",
              s0["robot_state"].shape == (3,) and s0["robot_state"].dtype == np.float32)
        check("action_chunk 形状 (4,3) float32",
              s0["action_chunk"].shape == (4, 3)
              and s0["action_chunk"].dtype == np.float32)
        lo, hi = float(s0["action_chunk"].min()), float(s0["action_chunk"].max())
        check("动作值在 [-1,1]", lo >= -1.0 - 1e-6 and hi <= 1.0 + 1e-6,
              f"[{lo:+.3f}, {hi:+.3f}]")

        # ⑥ 真回放：把【录下来的】动作喂回一个全新的环境
        env = GraspPlaceEnv(scene, exec_color)
        mism, n_steps = 0, 0
        for t in range(len(samples) // 2):
            if env.done:
                break
            idx = 2 * t if exec_color == "red" else 2 * t + 1
            env.step(samples[idx]["action_chunk"][0])      # ★ 录下来的第一步
            n_steps += 1
            if 2 * (t + 1) < len(samples):
                exp = samples[2 * (t + 1)]["robot_state"]
                got = env.observation()["robot_state"]
                if not np.allclose(exp, got, atol=1e-6):
                    mism += 1
        check("回放逐 tick 状态一致", mism == 0, f"偏差 {mism}/{n_steps}")
        check("回放最终阶段一致", env.phase == info["phase"], env.phase)


    print("\n" + "=" * 60)
    print(f"共 {args.rollouts} 条 rollout，{total_pairs} 对样本")
    print(f"{nc - nf}/{nc} 通过" if nf == 0 else f"有 {nf} 项失败")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
