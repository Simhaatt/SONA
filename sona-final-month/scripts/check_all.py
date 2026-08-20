#!/usr/bin/env python3
"""Single green-check command for the whole sona-final-month package.

Chains every verifiable stage — results consolidation, realtime unit tests,
realtime selftest, and the scene-classifier harness selftest — and prints one
pass/fail summary. Exits non-zero if any stage fails, so it can be used as a
pre-flight check before a demo or a thesis draft push.

    python scripts/check_all.py
    python scripts/check_all.py --results-dir "../results"
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(label: str, cmd: list, cwd: Path | None = None) -> tuple[bool, float, str]:
    print(f"\n---- {label} ----")
    print(f"$ {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable] + [str(c) for c in cmd], cwd=cwd,
                       capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = (r.stdout or "") + (r.stderr or "")
    print(out.strip()[-2000:])  # keep the log readable
    ok = r.returncode == 0
    print(f"[{'OK' if ok else 'FAIL'}] {label} ({dt:.2f}s, exit {r.returncode})")
    return ok, dt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT.parent / "results")
    ap.add_argument("--skip-significance", action="store_true",
                    help="skip the (slower) per_sample.csv significance stage")
    args = ap.parse_args()
    rd = args.results_dir

    stages = []

    ok, dt, _ = run("1/6 build_final_tables.py",
                    [HERE / "build_final_tables.py", "--results-dir", rd,
                     "--out-dir", ROOT / "FINAL_TABLES"])
    stages.append(("build_final_tables", ok, dt))

    if not args.skip_significance:
        ok, dt, _ = run("2/6 significance.py --preset all",
                        [HERE / "significance.py", "--preset", "all",
                         "--results-dir", rd, "--out-dir", ROOT / "SIGNIFICANCE"])
        # significance.py exits 0 even when every preset comparison is skipped
        # (per_sample.csv not present on this machine) — that's expected, not a
        # failure of the harness itself.
        stages.append(("significance", ok, dt))
    else:
        stages.append(("significance", None, 0.0))

    ok, dt, _ = run("3/6 make_figures.py",
                    [HERE / "make_figures.py", "--tables-dir", ROOT / "FINAL_TABLES",
                     "--out-dir", ROOT / "figures"])
    stages.append(("make_figures", ok, dt))

    ok, dt, _ = run("4/6 realtime unit tests (pytest)",
                    ["-m", "pytest", "-q", "test_stream_processor.py"],
                    cwd=ROOT / "realtime")
    stages.append(("realtime_tests", ok, dt))

    ok, dt, _ = run("5/6 realtime_demo.py --mode selftest",
                    [ROOT / "realtime" / "realtime_demo.py", "--mode", "selftest"])
    stages.append(("realtime_selftest", ok, dt))

    ok, dt, _ = run("6/6 eval_scene_classifier.py --self-test",
                    [HERE / "eval_scene_classifier.py", "--self-test",
                     "--out-dir", ROOT / "SCENE_EVAL_SELFTEST"])
    stages.append(("scene_classifier_selftest", ok, dt))

    print("\n================ SUMMARY ================")
    hard_fail = False
    for name, ok, dt in stages:
        if ok is None:
            status = "SKIPPED"
        elif ok:
            status = "PASS"
        else:
            status = "FAIL"
            hard_fail = True
        print(f"  {status:8s} {name:28s} {dt:6.2f}s")
    print("==========================================")

    if hard_fail:
        print("\ncheck_all: FAILED — see stage log above.")
        sys.exit(1)
    print("\ncheck_all: all stages green.")
    sys.exit(0)


if __name__ == "__main__":
    main()
