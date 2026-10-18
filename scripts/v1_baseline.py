"""V1 回归基线：跑一遍 V1 的测试和语言评测，把结果存档。

    python scripts/v1_baseline.py            # 建立 / 刷新基线
    python scripts/v1_baseline.py --check    # 和已存基线比对，有差异就退出码 1

★ 为什么必须先做这一步
  V1.1 会新增模块。如果哪天 V1 的指标变了，我们得能立刻回答
  "是我改坏的，还是它本来就这样"。没有基线，这个判断做不了。

★ 它只【读】V1：不改任何文件、不覆盖任何 checkpoint。
  全部通过子进程调用，所以 V1 的代码一行都不用动。
"""

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT_DIR = ROOT / "docs" / "experiments"
OUT_JSON = OUT_DIR / "v1_baseline.json"
OUT_RAW = OUT_DIR / "v1_baseline_raw.txt"

GROUPS = ("SEEN", "UNSEEN_KNOWN", "UNSEEN_NOVEL")


def run(cmd):
    """跑一条命令，把 stdout+stderr 一起抓回来（解码失败也不要崩）。"""
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True)
    text = (p.stdout.decode(errors="replace")
            + p.stderr.decode(errors="replace"))
    return p.returncode, text


def parse_pytest(text):
    m = re.search(r"(\d+) passed", text)
    return int(m.group(1)) if m else -1


def parse_eval(text):
    """从 eval_language.py 的输出里抽出三组说法 + 专家。

    只匹配【ASCII 前缀】(SEEN / UNSEEN_KNOWN / UNSEEN_NOVEL) 和数字，
    不依赖中文——这样控制台编码怎么变都不影响解析。
    """
    rates, expert = {}, None
    for line in text.splitlines():
        m = re.match(r"(SEEN|UNSEEN_KNOWN|UNSEEN_NOVEL)\S*\s+"
                     r"([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)", line)
        if m:
            rates[m.group(1)] = {
                "red": float(m.group(2)), "blue": float(m.group(3)),
                "overall": float(m.group(4)), "n": int(m.group(5)),
            }
            continue
        # 专家那一行只有【一个】百分比，后面跟评估次数
        if line.count("%") == 1:
            m = re.search(r"([\d.]+)%\s+(\d+)\s*$", line)
            if m:
                expert = {"overall": float(m.group(1)), "n": int(m.group(2))}
    return rates, expert


def collect():
    print("① 单元测试 ...", flush=True)
    rc_t, out_t = run([PY, "-m", "pytest", "-q"])
    n_passed = parse_pytest(out_t)

    print("② V1 语言评测（100 场景 × 3 组说法，约 20~60 秒）...", flush=True)
    rc_e, out_e = run([PY, "scripts/eval_language.py"])
    rates, expert = parse_eval(out_e)

    return {
        "date": datetime.date.today().isoformat(),
        "python": PY,
        "pytest_returncode": rc_t,
        "pytest_passed": n_passed,
        "eval_returncode": rc_e,
        "language_generalization": rates,
        "expert": expert,
    }, out_t + "\n" + "=" * 70 + "\n" + out_e


def show(b):
    print(f"\n{'项目':<24}{'值'}")
    print("-" * 48)
    print(f"{'pytest 通过数':<24}{b['pytest_passed']}")
    for g in GROUPS:
        r = b["language_generalization"].get(g)
        print(f"{g:<24}{r['overall']:.1f}%  (n={r['n']})" if r else f"{g:<24}(缺失)")
    e = b.get("expert")
    print(f"{'专家':<24}{e['overall']:.1f}%  (n={e['n']})" if e else f"{'专家':<24}(缺失)")


def main():
    check = "--check" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if check and not OUT_JSON.exists():
        print(f"[中止] 还没有基线文件 {OUT_JSON}，先不带 --check 跑一次。")
        return 1

    cur, raw = collect()
    show(cur)

    if not check:
        OUT_JSON.write_text(json.dumps(cur, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        OUT_RAW.write_text(raw, encoding="utf-8")
        print(f"\n基线已存: {OUT_JSON.relative_to(ROOT)}")
        print(f"原始输出: {OUT_RAW.relative_to(ROOT)}")
        return 0

    # ---- 比对模式 ----
    base = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    bad = []
    if cur["pytest_returncode"] != 0 or cur["pytest_passed"] != base["pytest_passed"]:
        bad.append(f"pytest: {base['pytest_passed']} -> {cur['pytest_passed']}")
    for g in GROUPS:
        b0 = base["language_generalization"].get(g)
        c0 = cur["language_generalization"].get(g)
        if not b0 or not c0:
            bad.append(f"{g}: 缺失")
        elif abs(b0["overall"] - c0["overall"]) > 1e-6:
            bad.append(f"{g}: {b0['overall']}% -> {c0['overall']}%")
    b0, c0 = base.get("expert"), cur.get("expert")
    if b0 and c0 and abs(b0["overall"] - c0["overall"]) > 1e-6:
        bad.append(f"专家: {b0['overall']}% -> {c0['overall']}%")

    if bad:
        print("\n[回归] V1 的指标变了：")
        for line in bad:
            print("  -", line)
        return 1
    print("\n[通过] V1 指标与基线完全一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
