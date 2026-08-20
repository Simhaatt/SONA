#!/usr/bin/env python3
"""Week-1 pipeline: build tables -> significance tests -> figures.

    python scripts/run_all.py --results-dir "../results"

Each stage is also runnable on its own (see the individual scripts). Stages
degrade gracefully: significance needs per_sample.csv files, figures need the
tables produced by stage 1.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if r.returncode != 0:
        print(f"  stage exited with code {r.returncode} (continuing)")
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=ROOT.parent / "results")
    ap.add_argument("--skip-significance", action="store_true")
    args = ap.parse_args()
    rd = args.results_dir

    run([HERE / "build_final_tables.py", "--results-dir", rd,
         "--out-dir", ROOT / "FINAL_TABLES"])
    if not args.skip_significance:
        run([HERE / "significance.py", "--preset", "all", "--results-dir", rd,
             "--out-dir", ROOT / "SIGNIFICANCE"])
    run([HERE / "make_figures.py", "--tables-dir", ROOT / "FINAL_TABLES",
         "--out-dir", ROOT / "figures"])
    print("\nrun_all complete.")


if __name__ == "__main__":
    main()
