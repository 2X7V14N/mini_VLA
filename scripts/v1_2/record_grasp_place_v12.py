"""V1.2 第 2 步：批量录制数据集。

    python scripts/v1_2/record_grasp_place_v12.py --train 30 --val 10 --test 10 --out data/v1.2_smoke
    python scripts/v1_2/record_grasp_place_v12.py                      # 1000/200/200

每个场景跑【一条】物理 rollout（实际执行的组合按序号轮换），
每 tick 出 4 条配对样本（红蓝方块 × 红蓝区域）。

产物：data/v1.2/{train,val,test}/ep_XXXX.npz + summary.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mini_vla.data.grasp_place_v12 import (      # noqa: E402
    COMBOS, sample_scene_v12, record_rollout_v12,
)

FAIL_CODES = ("TARGET_NOT_REACHED", "CLOSE_MISS", "OBJECT_SLIPPED",
              "ZONE_NOT_REACHED", "DROP_OUTSIDE", "WRONG_ZONE", "MAX_TICKS")


def save_rollout(path, samples, exec_block, exec_zone, scene, info):
    np.savez_compressed(
        path,
        image=np.stack([s["image"] for s in samples]),
        instruction=np.array([s["instruction"] for s in samples]),
        robot_state=np.stack([s["robot_state"] for s in samples]),
        action_chunk=np.stack([s["action_chunk"] for s in samples]),
        phase=np.array([s["phase"] for s in samples]),
        target_color=np.array([s["target_color"] for s in samples]),
        zone_color=np.array([s["zone_color"] for s in samples]),
        object_position=np.stack([s["object_position"] for s in samples]),
        zone_position=np.stack([s["zone_position"] for s in samples]),
        pair_id=np.array([s["pair_id"] for s in samples], dtype=np.int32),
        timestep=np.array([s["timestep"] for s in samples], dtype=np.int32),
        template_index=np.array([s["template_index"] for s in samples],
                                dtype=np.int32),
        exec_block=np.array(exec_block),
        exec_zone=np.array(exec_zone),
        red=np.asarray(scene.red, dtype=np.float64),
        blue=np.asarray(scene.blue, dtype=np.float64),
        robot_start=np.asarray(scene.robot_start, dtype=np.float64),
        zones_red=np.asarray(scene.zones["red"], dtype=np.float64),
        zones_blue=np.asarray(scene.zones["blue"], dtype=np.float64),
        success=np.bool_(info["success"]),
        failure=np.array(info["failure"] or ""),
        ticks=np.int32(info["ticks"]),
    )


def record_split(name, n_scenes, seed, out_root, t_start):
    rng = np.random.default_rng(seed)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    st = {"scenes": 0, "samples": 0, "ok": 0, "fails": {},
          "lengths": [], "combos": {}}

    print(f"\n[{name}] {n_scenes} 个场景 -> {out_dir}", flush=True)
    for i in range(n_scenes):
        scene = sample_scene_v12(rng)
        exec_block, exec_zone = COMBOS[i % len(COMBOS)]   # 四种组合轮换
        tidx = i % 3

        samples, info = record_rollout_v12(scene, exec_block, exec_zone,
                                           template_index=tidx)
        save_rollout(out_dir / f"ep_{i:04d}.npz", samples,
                     exec_block, exec_zone, scene, info)

        st["scenes"] += 1
        st["samples"] += len(samples)
        st["lengths"].append(len(samples) // 4)
        key = f"{exec_block}->{exec_zone}"
        st["combos"][key] = st["combos"].get(key, 0) + 1
        if info["success"]:
            st["ok"] += 1
        else:
            k = info["failure"] or "UNKNOWN"
            st["fails"][k] = st["fails"].get(k, 0) + 1

        if (i + 1) % 50 == 0 or i + 1 == n_scenes:
            print(f"  {i + 1:>4}/{n_scenes}　成功 {st['ok']:>4}　"
                  f"成功率 {st['ok'] / st['scenes']:.1%}　"
                  f"样本 {st['samples']:>6}　{time.time() - t_start:.0f}s",
                  flush=True)
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v1.2")
    ap.add_argument("--train", type=int, default=1000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=5150)
    args = ap.parse_args()

    out_root = ROOT / args.out
    if out_root.exists() and any(out_root.rglob("*.npz")):
        print(f"[中止] {out_root} 里已经有 .npz 了。删掉或换 --out。")
        return 1

    t0 = time.time()
    summary = {"seed": args.seed, "splits": {}}
    for name, n, off in (("train", args.train, 0), ("val", args.val, 10000),
                         ("test", args.test, 20000)):
        summary["splits"][name] = record_split(name, n, args.seed + off,
                                               out_root, t0)

    print("\n" + "=" * 70)
    print(f"{'split':<8}{'场景':>6}{'样本':>9}{'成功率':>9}"
          f"{'长度中位':>10}　组合分布")
    for name, s in summary["splits"].items():
        lm = int(np.median(s["lengths"])) if s["lengths"] else 0
        print(f"{name:<8}{s['scenes']:>6}{s['samples']:>9}"
              f"{s['ok'] / max(1, s['scenes']):>8.1%}{lm:>10}　{s['combos']}")

    fails = {}
    for s in summary["splits"].values():
        for k, v in s["fails"].items():
            fails[k] = fails.get(k, 0) + v
    print("\n失败构成（全部 split）")
    if not fails:
        print("  没有失败")
    else:
        for k in FAIL_CODES:
            if fails.get(k):
                print(f"  {k:<18}{fails[k]:>4}")

    total = sum(s["samples"] for s in summary["splits"].values())
    size = sum(f.stat().st_size for f in out_root.rglob("*.npz"))
    print(f"\n总样本 {total}　总大小 {size / 1e6:.1f} MB　"
          f"用时 {time.time() - t0:.0f}s")
    print(f"产物：{out_root.relative_to(ROOT)}")

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary.json 已写入")

    ok = all(s["ok"] == s["scenes"] for s in summary["splits"].values())
    print(f"\n{'[通过]' if ok else '[不通过]'} 专家成功率必须 100%")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
