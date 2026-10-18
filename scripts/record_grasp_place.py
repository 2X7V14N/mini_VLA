"""V1.1 第 3 步：批量录制数据集。

    python scripts/record_grasp_place.py                       # 1000/200/200 场景
    python scripts/record_grasp_place.py --train 50 --val 10 --test 10   # 先试小的

每个场景跑【一条】物理 rollout，每 tick 生成红蓝两条配对样本。
产物：data/v1.1/{train,val,test}/ep_XXXX.npz + summary.json

★ 划分单位是【场景 / rollout】，不是帧。同一 rollout 的所有帧、同一对样本，
  必须在同一个集合里。这是 V1 就定死的规矩。

★ 执行颜色按 rollout 序号交替（偶红奇蓝），模板按序号轮换，
  保证红蓝、三套模板在各 split 里都均衡。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mini_vla.data.grasp_place import sample_scene          # noqa: E402
from mini_vla.data.grasp_place_data import (                # noqa: E402
    record_rollout, TEMPLATES, COLORS,
)

ROOT = Path(__file__).resolve().parent.parent
FAIL_CODES = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
              "ZONE_NOT_REACHED", "DROP_OUTSIDE", "MAX_TICKS")


def save_rollout(path, samples, exec_color, scene, info):
    """一条 rollout 存成一个 npz。字符串数组存指令和阶段（诊断用）。"""
    n = len(samples)
    np.savez_compressed(
        path,
        image=np.stack([s["image"] for s in samples]),
        instruction=np.array([s["instruction"] for s in samples]),
        robot_state=np.stack([s["robot_state"] for s in samples]),
        action_chunk=np.stack([s["action_chunk"] for s in samples]),
        phase=np.array([s["phase"] for s in samples]),
        target_color=np.array([s["target_color"] for s in samples]),
        object_position=np.stack([s["object_position"] for s in samples]),
        zone_position=np.stack([s["zone_position"] for s in samples]),
        pair_id=np.array([s["pair_id"] for s in samples], dtype=np.int32),
        timestep=np.array([s["timestep"] for s in samples], dtype=np.int32),
        template_index=np.array([s["template_index"] for s in samples],
                                dtype=np.int32),
        exec_color=np.array(exec_color),
        # 场景本身也存下来：复现、闭环评测都要用
        red=np.asarray(scene.red, dtype=np.float64),
        blue=np.asarray(scene.blue, dtype=np.float64),
        robot_start=np.asarray(scene.robot_start, dtype=np.float64),
        success=np.bool_(info["success"]),
        failure=np.array(info["failure"] or ""),
        ticks=np.int32(info["ticks"]),
    )


def record_split(name, n_scenes, seed, out_root, t_start):
    rng = np.random.default_rng(seed)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    st = {"scenes": 0, "rollouts": 0, "samples": 0, "ok": 0,
          "fails": {}, "lengths": [], "colors": {"red": 0, "blue": 0},
          "templates": {}}

    print(f"\n[{name}] {n_scenes} 个场景 -> {out_dir}", flush=True)
    for i in range(n_scenes):
        scene = sample_scene(rng)
        exec_color = COLORS[i % 2]                 # 交替，保证红蓝均衡
        tidx = i % len(TEMPLATES["red"])

        samples, info = record_rollout(scene, exec_color, template_index=tidx)
        save_rollout(out_dir / f"ep_{i:04d}.npz", samples, exec_color, scene, info)

        st["scenes"] += 1
        st["rollouts"] += 1
        st["samples"] += len(samples)
        st["lengths"].append(len(samples) // 2)
        st["colors"][exec_color] += 1
        st["templates"][str(tidx)] = st["templates"].get(str(tidx), 0) + 1
        if info["success"]:
            st["ok"] += 1
        else:
            k = info["failure"] or "UNKNOWN"
            st["fails"][k] = st["fails"].get(k, 0) + 1

        if (i + 1) % 50 == 0 or i + 1 == n_scenes:
            rate = st["ok"] / st["rollouts"]
            print(f"  {i + 1:>4}/{n_scenes}　成功 {st['ok']:>4}　"
                  f"成功率 {rate:.1%}　样本 {st['samples']:>6}　"
                  f"{time.time() - t_start:.0f}s", flush=True)

    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1.1")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out_root = ROOT / args.out
    if out_root.exists() and any(out_root.rglob("*.npz")):
        print(f"[中止] {out_root} 里已经有 .npz 了。删掉或换 --out。")
        return 1

    t0 = time.time()
    summary = {"seed": args.seed, "splits": {}}
    for name, n in (("train", args.train), ("val", args.val),
                    ("test", args.test)):
        # 三个 split 用不同种子，场景互不重叠
        summary["splits"][name] = record_split(
            name, n, args.seed + {"train": 0, "val": 10000, "test": 20000}[name],
            out_root, t0)

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print(f"{'split':<8}{'场景':>6}{'样本':>9}{'成功率':>9}{'红/蓝':>10}{'长度中位':>10}")
    for name, st in summary["splits"].items():
        rate = st["ok"] / max(1, st["rollouts"])
        lm = int(np.median(st["lengths"])) if st["lengths"] else 0
        print(f"{name:<8}{st['scenes']:>6}{st['samples']:>9}{rate:>8.1%}"
              f"{st['colors']['red']:>5}/{st['colors']['blue']:<4}{lm:>10}")

    fails = {}
    for st in summary["splits"].values():
        for k, v in st["fails"].items():
            fails[k] = fails.get(k, 0) + v
    print("\n失败构成（全部 split）")
    if not fails:
        print("  没有失败")
    else:
        for k in FAIL_CODES:
            if fails.get(k):
                print(f"  {k:<18}{fails[k]:>4}")

    total = sum(st["samples"] for st in summary["splits"].values())
    size = sum(f.stat().st_size for f in out_root.rglob("*.npz"))
    print(f"\n总样本 {total}　总大小 {size / 1e6:.1f} MB　"
          f"用时 {time.time() - t0:.0f}s")
    print(f"产物：{out_root.relative_to(ROOT)}")

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary.json 已写入")

    ok = all(st["ok"] == st["rollouts"] for st in summary["splits"].values())
    print(f"\n{'[通过]' if ok else '[不通过]'} 专家成功率必须 100%（确定性环境）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
