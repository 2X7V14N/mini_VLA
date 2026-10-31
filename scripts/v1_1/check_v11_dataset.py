"""V1.1 第 4 步：检查落盘的数据集 + 开训前的状态泄露检查。

    python scripts/v1_1/check_v11_dataset.py --data data/v1.1_smoke
    python scripts/v1_1/check_v11_dataset.py --data data/v1.1

分两段：
  A. 契约检查 —— 配对、指令、标签、grip、搬运、形状、划分
  B. 状态泄露检查 —— 只看 MOVE_TO_OBJECT + 大幅动作帧，
     用 train 训练 kNN、在 val 上测方向误差。
     如果状态能高精度预测方向，说明数据里图像可能是多余的。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mini_vla.data.grasp_place_data import TEMPLATES     # noqa: E402

PASS, FAIL = "[OK]  ", "[FAIL]"
nc = nf = 0


def check(name, cond, detail=""):
    global nc, nf
    nc += 1
    if not cond:
        nf += 1
    print(f"  {PASS if cond else FAIL} {name:<42}{detail}")


def load_split(d):
    return sorted(Path(d).glob("ep_*.npz"))


def angle_between(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(
        np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))))


# ===========================================================================
# A. 契约检查
# ===========================================================================
def check_split(name, files):
    print(f"\n[A] {name}：{len(files)} 条 rollout")

    e_even = e_order = e_img = e_state = e_pair = e_instr = 0
    e_grip = e_dgrip = e_carry = e_shape = 0
    e_hold_same = 0
    n_cmp = n_diff = 0
    n_pairs = 0
    phases = {}
    scenes = set()

    for f in files:
        z = np.load(f, allow_pickle=False)
        img, instr, st = z["image"], z["instruction"], z["robot_state"]
        chunk, phase, tgt = z["action_chunk"], z["phase"], z["target_color"]
        obj, pair, ts = z["object_position"], z["pair_id"], z["timestep"]
        tidx, exec_color = z["template_index"], str(z["exec_color"])
        N = len(img)

        if N % 2:
            e_even += 1
            continue

        # 形状 / dtype
        if (img.shape[1:] != (64, 64, 3) or img.dtype != np.uint8
                or st.shape[1:] != (3,) or st.dtype != np.float32
                or chunk.shape[1:] != (4, 3) or chunk.dtype != np.float32):
            e_shape += 1

        if not (np.all(tgt[0::2] == "red") and np.all(tgt[1::2] == "blue")):
            e_order += 1
        if not np.array_equal(img[0::2], img[1::2]):
            e_img += 1
        if not np.array_equal(st[0::2], st[1::2]):
            e_state += 1
        if not (np.array_equal(pair[0::2], pair[1::2])
                and np.array_equal(ts[0::2], ts[1::2])
                and np.array_equal(tidx, tidx[0:1].repeat(N))):
            e_pair += 1

        for i in range(N):
            if instr[i] != TEMPLATES[str(tgt[i])][int(tidx[i])]:
                e_instr += 1
                break

        # grip == holding；且 holding 与阶段一致
        grip = st[:, 2] > 0.5
        if not np.array_equal(grip[0::2], grip[1::2]):
            e_grip += 1
        should = np.isin(phase, ("MOVE_TO_ZONE", "OPEN"))
        if not np.array_equal(grip, should):
            e_grip += 1

        # 配对标签规则
        for t in range(N // 2):
            a, b = chunk[2 * t][0], chunk[2 * t + 1][0]
            if grip[2 * t]:
                if not np.allclose(chunk[2 * t], chunk[2 * t + 1], atol=1e-6):
                    e_hold_same += 1
            else:
                na, nb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
                if na > 0.3 and nb > 0.3:      # 归一化后 >0.3 才算真在动
                    n_cmp += 1
                    if angle_between(a[:2], b[:2]) > 5.0:
                        n_diff += 1
            n_pairs += 1

        # dgrip 解码与阶段一致
        # ★ 必须【逐样本】按它自己的指令颜色判断，不能拿执行 rollout 的阶段：
        #   配对里另一半是反事实样本，它在 CLOSE 阶段可能正"朝另一个颜色走"，
        #   dgrip 理应是 +1。
        dg = chunk[:, 0, 2]
        if not np.all(np.isin(dg, (-1.0, 0.0, 1.0))):
            e_dgrip += 1
        d_obj = np.linalg.norm(st[:, :2] - obj, axis=1)     # 到【本样本颜色】目标的距离
        d_zone = np.linalg.norm(obj - z["zone_position"], axis=1)
        if np.any(dg[(~grip) & (d_obj > 0.10)] < 0.5):       # 空手 + 还没到 -> 张开
            e_dgrip += 1
        # ★ 夹住时两条样本规划的是【同一个】被夹住的物体（执行颜色那个），
        #   所以"有没有对准区心"要看它，不能看样本自己的颜色物体
        #   —— 那个物体从头到尾没动过。
        _idx = 0 if exec_color == "red" else 1
        d_zone_held = np.repeat(
            np.linalg.norm(obj[_idx::2] - z["zone_position"][0], axis=1), 2)
        if np.any(dg[grip & (d_zone_held > 0.02)] > -0.5):
            e_dgrip += 1

        # 搬运：执行颜色那条，位移必须逐 tick 相等
        idx = 0 if exec_color == "red" else 1
        rob = st[idx::2, :2].astype(np.float64)
        ob = obj[idx::2, :].astype(np.float64)
        hold = grip[idx::2]
        drob, dob = np.diff(rob, axis=0), np.diff(ob, axis=0)
        want = np.where(hold[:-1, None], drob, 0.0)
        if not np.allclose(dob, want, atol=1e-6):   # 存盘是 float32
            e_carry += 1

        for p in phase:
            phases[str(p)] = phases.get(str(p), 0) + 1
        scenes.add((float(z["red"][0]), float(z["red"][1]),
                    float(z["blue"][0]), float(z["blue"][1]),
                    float(z["robot_start"][0]), float(z["robot_start"][1])))

    check("每条样本数是偶数", e_even == 0, f"奇数的 {e_even}")
    check("配对顺序 red_t, blue_t", e_order == 0, f"错位 {e_order}")
    check("同一对图像逐元素相同", e_img == 0, f"不等 {e_img}")
    check("同一对状态逐元素相同", e_state == 0, f"不等 {e_state}")
    check("pair_id / timestep / template_index 成对一致", e_pair == 0, f"异常 {e_pair}")
    check("指令 == 模板[颜色][编号]", e_instr == 0, f"不匹配 {e_instr}")
    check("grip == holding，且与阶段一致", e_grip == 0, f"异常 {e_grip}")
    check("夹住时红蓝动作完全相同", e_hold_same == 0, f"违背 {e_hold_same}")
    frac = n_diff / max(1, n_cmp)
    check("空手时红蓝方向确实不同（>=80%）", frac >= 0.80,
          f"{frac:.1%}（可比 {n_cmp} 对）")
    check("dgrip 解码与阶段一致", e_dgrip == 0, f"违背 {e_dgrip}")
    check("搬运位移 == 夹爪位移（逐 tick）", e_carry == 0, f"违背 {e_carry}")
    check("形状 / dtype 符合契约", e_shape == 0, f"异常 {e_shape}")

    print(f"       {n_pairs} 对样本　阶段分布 {phases}")
    return {"pairs": n_pairs, "phases": phases, "scenes": scenes}


# ===========================================================================
# B. 状态泄露检查
# ===========================================================================
def collect_frames(files):
    """只取 MOVE_TO_OBJECT 且 |a_xy| > 0.1 的帧。"""
    S, D, R = [], [], []
    for k, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        st, chunk, phase = z["robot_state"], z["action_chunk"], z["phase"]
        m = (phase == "MOVE_TO_OBJECT")
        a = chunk[:, 0, :2]
        mag = np.linalg.norm(a, axis=1)
        keep = m & (mag > 0.1)
        if not keep.any():
            continue
        S.append(st[keep])
        D.append(a[keep] / mag[keep, None])
        R.append(np.full(int(keep.sum()), k))
    if not S:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0)
    return np.concatenate(S), np.concatenate(D), np.concatenate(R)


def circ_mean(v):
    m = v.mean(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-12 else np.array([1.0, 0.0])


def leak_check(train_files, val_files, k=8):
    print("\n[B] 状态泄露检查（只用 MOVE_TO_OBJECT + |a_xy|>0.1 的帧）")
    Str, Dtr, _ = collect_frames(train_files)
    Sva, Dva, _ = collect_frames(val_files)
    print(f"  train 帧 {len(Str)}　val 帧 {len(Sva)}")
    if len(Str) < 20 or len(Sva) < 5:
        print("  帧太少，判不了（小数据集/小场景数时会这样）")
        return None

    base = circ_mean(Dtr)                    # 基线：一律猜 train 的平均方向
    errs, preds = [], []
    for i in range(len(Sva)):
        d = np.linalg.norm(Str - Sva[i], axis=1)
        nb = np.argsort(d)[:k]
        p = circ_mean(Dtr[nb])
        preds.append(p)
        errs.append(np.degrees(np.arccos(np.clip(float(p @ Dva[i]), -1, 1))))
    errs = np.array(errs)
    preds = np.array(preds)
    base_err = np.degrees(np.arccos(np.clip(Dtr @ base, -1, 1)))

    # R²：把方向当二维向量回归
    ss_res = float(((preds - Dva) ** 2).sum())
    ss_tot = float(((Dva - Dva.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    print(f"  kNN 角误差 中位 {np.median(errs):.1f}deg　90分位 "
          f"{np.percentile(errs, 90):.1f}deg")
    print(f"  基线角误差 中位 {np.median(base_err):.1f}deg")
    print(f"  方向 R2 {r2:+.3f}　（1=状态完全决定方向，0=毫无信息）")
    if r2 > 0.90:
        print("  ⚠ 状态能很高精度地预测方向 —— 数据里图像可能是多余的。")
        print("    按规定这【不是硬拦截】，但要同时看图像置零消融才算数。")
    else:
        print("  [OK] 状态不足以决定方向，图像是必要信息。")
    return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1.1")
    args = ap.parse_args()
    root = Path(args.data)
    if not root.exists():
        print(f"{root} 不存在")
        return 1

    files = {s: load_split(root / s) for s in ("train", "val", "test")}
    if not files["train"]:
        print(f"{root} 下没有 train/ep_*.npz")
        return 1

    res = {s: check_split(s, f) for s, f in files.items() if f}

    # 划分完整性：三个 split 的场景不能重叠
    all_sets = [set(v["scenes"]) for v in res.values()]
    overlap = 0
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            overlap += len(all_sets[i] & all_sets[j])
    print(f"\n[A] split 间场景重叠：{overlap}　（必须为 0）")
    check("train/val/test 按场景划分、互不重叠", overlap == 0, str(overlap))

    # 覆盖：train 和 val 都要有红蓝 + 各阶段
    for s in ("train", "val"):
        if s not in res:
            continue
        ph = res[s]["phases"]
        need = ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")
        check(f"{s} 覆盖四个阶段", all(ph.get(n, 0) > 0 for n in need),
              str({k: ph.get(k, 0) for k in need}))

    if "val" in files and files["val"]:
        leak_check(files["train"], files["val"])

    print("\n" + "=" * 62)
    print(f"{nc - nf}/{nc} 通过" if nf == 0 else f"有 {nf} 项失败")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
