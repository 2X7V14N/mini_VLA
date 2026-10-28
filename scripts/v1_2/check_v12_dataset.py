"""V1.2 第 2 步：数据检查（落盘后）。

    python scripts/v1_2/check_v12_dataset.py --data data/v1.2_smoke

在 V1.1 的全套检查上加 V1.2 专属的：
    ① 每 tick 4 条，方块色 [红,蓝,红,蓝]、区域色 [红,红,蓝,蓝]
    ② 同一 tick 的 4 张图 / 4 个状态逐元素相同
    ③ 配对 = (0,1) 和 (2,3)：同区域、异方块
    ④ 几何：区域不重叠、在画布内、不压方块、不压机器人起点
外加开训前的状态泄露检查。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mini_vla.data.grasp_place import (                      # noqa: E402
    MIN_ROBOT_CLEAR, ZONE_CLEAR, ZONE_HALF,
)
from mini_vla.data.grasp_place_v12 import (                  # noqa: E402
    MIN_ZONE_SEP, ROBOT_ZONE_CLEAR, instruction_v12,
)

PASS, FAIL = "[OK]  ", "[FAIL]"
nc = nf = 0
BLOCKS = ("red", "blue")


def check(name, cond, detail=""):
    global nc, nf
    nc += 1
    if not cond:
        nf += 1
    print(f"  {PASS if cond else FAIL} {name:<42}{detail}")


def angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(
        np.clip(float(np.dot(a, b)) / (na * nb), -1.0, 1.0))))


def load_split(d):
    return sorted(Path(d).glob("ep_*.npz"))


def check_split(name, files):
    print(f"\n[A] {name}：{len(files)} 条 rollout")
    e_n = e_order = e_img = e_st = e_instr = e_pair = 0
    e_grip = e_carry = e_shape = e_dgrip = 0
    e_zone_overlap = e_zone_bounds = e_block_zone = 0
    e_robot_zone = e_robot_block = 0
    n_pairs = n_cmp = n_diff = 0
    phases = {}
    scenes = set()
    resolutions = set()

    for f in files:
        z = np.load(f, allow_pickle=False)
        img, instr, st = z["image"], z["instruction"], z["robot_state"]
        chunk, phase = z["action_chunk"], z["phase"]
        tb, zc = z["target_color"], z["zone_color"]
        obj, zp, pair, ts = (z["object_position"], z["zone_position"],
                             z["pair_id"], z["timestep"])
        tidx, eb, ez = z["template_index"], str(z["exec_block"]), str(z["exec_zone"])
        red, blue = z["red"], z["blue"]
        zr, zb = z["zones_red"], z["zones_blue"]
        robot0 = z["robot_start"]
        N = len(img)
        if img.ndim == 4:
            resolutions.add(tuple(img.shape[1:3]))

        # ---- 几何契约（每个 rollout 一条场景）----
        if np.linalg.norm(zr - zb) < MIN_ZONE_SEP:
            e_zone_overlap += 1
        if (not np.all(np.abs(zr) + ZONE_HALF <= 1.0)
                or not np.all(np.abs(zb) + ZONE_HALF <= 1.0)):
            e_zone_bounds += 1
        zones = (zr, zb)
        if any(np.linalg.norm(obj0 - c) < ZONE_CLEAR
               for obj0 in (red, blue) for c in zones):
            e_block_zone += 1
        if (np.linalg.norm(robot0 - zr) < ROBOT_ZONE_CLEAR
                or np.linalg.norm(robot0 - zb) < ROBOT_ZONE_CLEAR):
            e_robot_zone += 1
        if (np.linalg.norm(robot0 - red) < MIN_ROBOT_CLEAR
                or np.linalg.norm(robot0 - blue) < MIN_ROBOT_CLEAR):
            e_robot_block += 1

        if N % 4:
            e_n += 1
            continue

        if (img.ndim != 4 or img.shape[1] != img.shape[2]
                or img.shape[1] not in (64, 128) or img.shape[-1] != 3
                or img.dtype != np.uint8
                or st.shape[1:] != (3,) or chunk.shape[1:] != (4, 3)):
            e_shape += 1

        # ① 每 4 条的固定顺序
        if not (np.all(tb[0::4] == "red") and np.all(tb[1::4] == "blue")
                and np.all(tb[2::4] == "red") and np.all(tb[3::4] == "blue")):
            e_order += 1
        if not (np.all(zc[0::4] == "red") and np.all(zc[1::4] == "red")
                and np.all(zc[2::4] == "blue") and np.all(zc[3::4] == "blue")):
            e_order += 1

        # ② 同一 tick 的 4 张图 / 4 个状态相同
        for k in range(1, 4):
            if not np.array_equal(img[k::4], img[0::4]):
                e_img += 1
                break
        for k in range(1, 4):
            if not np.array_equal(st[k::4], st[0::4]):
                e_st += 1
                break

        # 指令与模板一致
        for i in range(N):
            if instr[i] != instruction_v12(str(tb[i]), str(zc[i]), int(tidx[i])):
                e_instr += 1
                break

        # ③ 配对：同区域、异方块
        for t in range(N // 4):
            for a_i, b_i in ((4 * t, 4 * t + 1), (4 * t + 2, 4 * t + 3)):
                if (tb[a_i] == tb[b_i]) or (zc[a_i] != zc[b_i]):
                    e_pair += 1
                if pair[a_i] != pair[b_i] or ts[a_i] != ts[b_i]:
                    e_pair += 1

        grip = st[:, 2] > 0.5
        should = np.isin(phase, ("MOVE_TO_ZONE", "OPEN"))
        if not np.array_equal(grip, should):
            e_grip += 1
        for k in range(1, 4):
            if not np.array_equal(grip[k::4], grip[0::4]):
                e_grip += 1
                break

        # 配对标签：空手时同区域的两个方块方向不同；夹住时同一对完全相同
        for t in range(N // 4):
            a1, b1 = 4 * t, 4 * t + 1
            if grip[a1]:
                if not np.allclose(chunk[a1], chunk[b1], atol=1e-6):
                    e_pair += 1
            else:
                na, nb = np.linalg.norm(chunk[a1][0][:2]), np.linalg.norm(chunk[b1][0][:2])
                if na > 0.3 and nb > 0.3:
                    n_cmp += 1
                    if angle(chunk[a1][0][:2], chunk[b1][0][:2]) > 5.0:
                        n_diff += 1
            n_pairs += 1

        # dgrip 离散 + 搬运跟随
        dg = chunk[:, 0, 2]
        if not np.all(np.isin(dg, (-1.0, 0.0, 1.0))):
            e_dgrip += 1
        # 执行组合在 COMBOS 里的位置
        pos = {"red->red": 0, "blue->red": 1,
               "red->blue": 2, "blue->blue": 3}[f"{eb}->{ez}"]
        rob = st[pos::4, :2].astype(np.float64)
        ob = obj[pos::4].astype(np.float64)
        hold = grip[pos::4]
        drob, dob = np.diff(rob, axis=0), np.diff(ob, axis=0)
        if not np.allclose(dob, np.where(hold[:-1, None], drob, 0.0), atol=1e-6):
            e_carry += 1

        for p in phase:
            phases[str(p)] = phases.get(str(p), 0) + 1
        scenes.add(tuple(np.round(np.concatenate(
            [red, blue, robot0, zr, zb]), 6).tolist()))

    check("每 tick 4 条（N % 4 == 0）", e_n == 0, str(e_n))
    check("方块色 [红,蓝,红,蓝] / 区域色 [红,红,蓝,蓝]", e_order == 0, str(e_order))
    check("同一 tick 的 4 张图相同", e_img == 0, str(e_img))
    check("同一 tick 的 4 个状态相同", e_st == 0, str(e_st))
    check("指令 == 模板[方块][区域][编号]", e_instr == 0, str(e_instr))
    check("配对 = 同区域 + 异方块 + 同 timestep", e_pair == 0, str(e_pair))
    check("grip == holding，且与阶段一致", e_grip == 0, str(e_grip))
    check("dgrip 取值在 {-1,0,+1}", e_dgrip == 0, str(e_dgrip))
    check("搬运位移 == 夹爪位移（逐 tick）", e_carry == 0, str(e_carry))
    check("形状 / dtype 符合契约", e_shape == 0,
          f"{e_shape} 错误，分辨率 {sorted(resolutions)}")
    check("两个放置区不重叠", e_zone_overlap == 0, str(e_zone_overlap))
    check("放置区完整落在画布内", e_zone_bounds == 0, str(e_zone_bounds))
    check("方块不压住放置区", e_block_zone == 0, str(e_block_zone))
    check(f"机器人起点离放置区 >= {ROBOT_ZONE_CLEAR}", e_robot_zone == 0,
          str(e_robot_zone))
    check(f"机器人起点离方块 >= {MIN_ROBOT_CLEAR}", e_robot_block == 0,
          str(e_robot_block))
    frac = n_diff / max(1, n_cmp)
    check("空手时同区域异方块方向不同（>=80%）", frac >= 0.80,
          f"{frac:.1%}（可比 {n_cmp} 对）")

    print(f"       {n_pairs} 对样本　图像分辨率 {sorted(resolutions)}　阶段分布 {phases}")
    return {"pairs": n_pairs, "phases": phases, "scenes": scenes}


def collect_frames(files):
    S, D = [], []
    for f in files:
        z = np.load(f, allow_pickle=False)
        st, chunk, phase = z["robot_state"], z["action_chunk"], z["phase"]
        a = chunk[:, 0, :2]
        mag = np.linalg.norm(a, axis=1)
        keep = (phase == "MOVE_TO_OBJECT") & (mag > 0.1)
        if keep.any():
            S.append(st[keep])
            D.append(a[keep] / mag[keep, None])
    if not S:
        return np.zeros((0, 3)), np.zeros((0, 2))
    return np.concatenate(S), np.concatenate(D)


def circ_mean(v):
    m = v.mean(axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 1e-12 else np.array([1.0, 0.0])


def leak_check(train_files, val_files, k=8):
    print("\n[B] 状态泄露检查（只用 MOVE_TO_OBJECT + |a_xy|>0.1 的帧）")
    Str, Dtr = collect_frames(train_files)
    Sva, Dva = collect_frames(val_files)
    print(f"  train 帧 {len(Str)}　val 帧 {len(Sva)}")
    if len(Str) < 20 or len(Sva) < 5:
        print("  帧太少，判不了（小数据集会这样）")
        return
    base = circ_mean(Dtr)
    errs, preds = [], []
    for i in range(len(Sva)):
        d = np.linalg.norm(Str - Sva[i], axis=1)
        p = circ_mean(Dtr[np.argsort(d)[:k]])
        preds.append(p)
        errs.append(np.degrees(np.arccos(np.clip(float(p @ Dva[i]), -1, 1))))
    errs, preds = np.array(errs), np.array(preds)
    base_err = np.degrees(np.arccos(np.clip(Dtr @ base, -1, 1)))
    ss_res = float(((preds - Dva) ** 2).sum())
    ss_tot = float(((Dva - Dva.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"  kNN 角误差 中位 {np.median(errs):.1f}deg　基线 {np.median(base_err):.1f}deg")
    print(f"  方向 R2 {r2:+.3f}")
    print("  [OK] 状态不足以决定方向，图像是必要信息。" if r2 <= 0.90
          else "  ⚠ 状态能较高精度预测方向 —— 结合图像置零消融再判断")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/v1.2")
    args = ap.parse_args()
    root = Path(args.data)
    if not root.exists():
        print(f"{root} 不存在")
        return 1

    files = {s: load_split(root / s) for s in ("train", "val", "test")}
    if not files["train"]:
        print(f"{root}/train 下没有 ep_*.npz")
        return 1

    res = {s: check_split(s, f) for s, f in files.items() if f}

    sets = [v["scenes"] for v in res.values()]
    overlap = sum(len(sets[i] & sets[j])
                  for i in range(len(sets)) for j in range(i + 1, len(sets)))
    check("split 间场景不重叠", overlap == 0, str(overlap))

    for s in ("train", "val"):
        if s in res:
            ph = res[s]["phases"]
            need = ("MOVE_TO_OBJECT", "CLOSE", "MOVE_TO_ZONE", "OPEN")
            check(f"{s} 覆盖四个阶段", all(ph.get(n, 0) > 0 for n in need),
                  str({k: ph.get(k, 0) for k in need}))

    if files.get("val"):
        leak_check(files["train"], files["val"])

    print("\n" + "=" * 62)
    print(f"{nc - nf}/{nc} 通过" if nf == 0 else f"有 {nf} 项失败")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
