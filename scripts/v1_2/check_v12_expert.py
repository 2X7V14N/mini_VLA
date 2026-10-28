"""V1.2 第 1 步：专家验收（两个彩色放置区）。

    python scripts/v1_2/check_v12_expert.py --scenes 200

一个场景跑 4 种指令（方块色 × 区域色）= 800 条。
只验专家 + 环境，不接模型、不录数据。
使用 V12_FINE_TOL 的放宽开合容差，和录制数据保持同一份专家。

为什么这里 100% 是【预期】的：
    专家知道所有坐标，这个任务对它来说是确定的。
    这一步的价值是把仪表接上（WRONG_ZONE 能不能触发、四种组合各自通不通），
    真正难的是后面模型能不能从图里看出"哪个方块、哪个区域"。
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mini_vla.data.grasp_place import (                      # noqa: E402
    CLOSE_ENTER_TOL, GraspPlaceEnv,
)
from mini_vla.data.grasp_place_v12 import (                  # noqa: E402
    V12_FINE_TOL, expert_action_v12, sample_scene_v12,
)

FAILS = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
         "ZONE_NOT_REACHED", "DROP_OUTSIDE", "WRONG_ZONE", "MAX_TICKS")
COLORS = ("red", "blue")


def run(scene, obj_color, zone_color, fine_tol):
    env = GraspPlaceEnv(scene, obj_color, zone_color=zone_color,
                        close_enter_tol=CLOSE_ENTER_TOL)
    while not env.done and env.ticks < 1500:
        env.step(expert_action_v12(env, obj_color, fine_tol=fine_tol))
    info = env.info()
    obj = info["object"]
    return {
        "obj": obj_color, "zone": zone_color,
        "success": bool(info["success"]),
        "failure": info["failure"],
        "in_target": env.in_target_zone(obj),
        "in_any": env.in_any_zone(obj),
        "zone_distance": float(np.linalg.norm(
            env.objects[obj_color] - env.zones[zone_color])),
        "ticks": info["ticks"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--fine-tol", type=float, default=V12_FINE_TOL,
                    help="专家到达指定区域后允许松手的中心误差")
    args = ap.parse_args()
    if not (0 < args.fine_tol < 0.11):
        ap.error("--fine-tol 必须在 (0, 0.11) 内，小于放置区半宽")

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.scenes):
        sc = sample_scene_v12(rng)
        for oc in COLORS:
            for zc in COLORS:
                rows.append(run(sc, oc, zc, args.fine_tol))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.scenes} 个场景 ...", flush=True)

    n = len(rows)
    ok = sum(r["success"] for r in rows)
    print(f"\n{'=' * 64}")
    print(f"场景 {args.scenes} × 4 种指令 = {n} 条")
    print(f"松手容差 {args.fine_tol:.3f}；放置区半宽 0.110")
    print(f"专家成功率 {ok / n:.2%}")
    distances = np.asarray([r["zone_distance"] for r in rows], dtype=float)
    print("松手时物体到目标区域中心："
          f"max={distances.max():.4f}, p90={np.percentile(distances, 90):.4f}; "
          f"落在目标区 {sum(r['in_target'] for r in rows)}/{n}")

    print(f"\n{'方块→区域':<14}{'成功':>8}{'落在指令区域':>14}{'落在任一区域':>14}")
    for oc in COLORS:
        for zc in COLORS:
            sub = [r for r in rows if r["obj"] == oc and r["zone"] == zc]
            print(f"{oc + ' -> ' + zc:<14}"
                  f"{sum(r['success'] for r in sub) / len(sub):>7.1%}"
                  f"{sum(r['in_target'] for r in sub) / len(sub):>13.1%}"
                  f"{sum(r['in_any'] for r in sub) / len(sub):>13.1%}")

    fails = {}
    for r in rows:
        if not r["success"]:
            k = r["failure"] or "UNKNOWN"
            fails[k] = fails.get(k, 0) + 1
    print("\n失败构成")
    if not fails:
        print("  没有失败")
    else:
        for k in FAILS:
            if fails.get(k):
                print(f"  {k:<18}{fails[k]:>4}")

    # 专家必须每一条都对：成功率 100%，且不允许 WRONG_ZONE
    good = (ok == n) and not fails.get("WRONG_ZONE")
    print("\n" + "=" * 64)
    print(f"{'[通过]' if good else '[不通过]'} 专家必须 100%，且不应出现 WRONG_ZONE")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
