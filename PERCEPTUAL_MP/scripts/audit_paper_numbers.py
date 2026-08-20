"""Check every number asserted in PAPER_DRAFT.md against the artefacts it came from.

Mirrors AudioVQ/scripts/audit_paper_numbers.py. Run before circulating the draft:

    python PERCEPTUAL_MP/scripts/audit_paper_numbers.py

Exits non-zero if any claim has drifted from its source, so a stale number cannot reach a
reviewer. Claims are checked against results/phase4/phase4_summary.json (frozen out-of-sample
evaluation) and results/onnx/edge_bench_idle_2026-08-20/summary.csv (idle-machine timings).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MP = HERE.parent
REPO = MP.parent

FAILURES: list[str] = []
CHECKS = 0


def check(section: str, label: str, claimed: float, source: float, tol: float) -> None:
    global CHECKS
    CHECKS += 1
    ok = abs(claimed - source) <= tol
    status = "ok      " if ok else "MISMATCH"
    print(f"  {status} {label:<52s} claimed {claimed:<12.6g} source {source:<12.6g}")
    if not ok:
        FAILURES.append(f"[{section}] {label}: claimed {claimed}, source {source}")


def main() -> int:
    global CHECKS
    summary = json.loads((MP / "results" / "phase4" / "phase4_summary.json").read_text())

    print("[§5.2] model sizes (KB)")
    sizes = summary["table1_sizes_kb"]
    check("5.2", "FP32 size", 2842.7, sizes["fp32"], 0.05)
    check("5.2", "mixed (stem FP32) size", 916.9, sizes["mixed_stemfp32"], 0.05)
    check("5.2", "uniform INT8 size", 950.7, sizes["uniform_int8"], 0.05)

    # The size inversion is the claim, so assert the direction, not just the values.
    CHECKS_OK = sizes["mixed_stemfp32"] < sizes["uniform_int8"]
    print(f"  {'ok      ' if CHECKS_OK else 'MISMATCH'} "
          f"{'mixed is SMALLER than uniform INT8':<52s} "
          f"{sizes['mixed_stemfp32']:.1f} < {sizes['uniform_int8']:.1f}")
    if not CHECKS_OK:
        FAILURES.append("[5.2] size inversion claim is false")

    print("\n[§5.2] out-of-sample significance (VoiceBank 824)")
    sig = {s["comparison"]: s for s in summary["significance_out_of_sample"]}

    a = sig["mixed_stemfp32 - uniform_int8"]
    check("5.2", "stem exemption vs uniform, mean", 0.108, a["mean"], 0.001)
    check("5.2", "  CI low", 0.091, a["ci_lo"], 0.001)
    check("5.2", "  CI high", 0.126, a["ci_hi"], 0.001)
    check("5.2", "  n", 824, a["n"], 0)

    b = sig["fp32 - mixed_stemfp32"]
    check("5.2", "mixed vs FP32, mean (want ~0)", 0.0014, b["mean"], 0.0001)
    check("5.2", "mixed vs FP32, p (want n.s.)", 0.25, b["p"], 0.005)
    if b["p"] <= 0.05:
        FAILURES.append("[5.2] mixed vs FP32 is significant - the headline claim is false")

    c = sig["fp32 - uniform_int8"]
    check("5.2", "total quantization cost", 0.109, c["mean"], 0.001)

    # "the stem carries ~97% of the degradation": total cost minus what the mixed model still loses
    stem_share = (c["mean"] - b["mean"]) / c["mean"] * 100.0
    print(f"  derived  {'stem share of total INT8 degradation':<52s} {stem_share:.1f}%")
    if stem_share < 95.0:
        FAILURES.append(f"[5.2] stem share fell to {stem_share:.1f}% - '~97%' no longer holds")

    # stem = 3840 bytes of the 722209-parameter FP32 model
    pct = 3840 / (722209 * 4) * 100
    check("5.2", "stem as % of model weights", 0.133, pct, 0.001)

    print("\n[§5.3] condition breakdown (all 9 must be significant)")
    for cond, d in summary["condition_breakdown"].items():
        CHECKS += 1
        ok = d["p"] < 0.05
        print(f"  {'ok      ' if ok else 'MISMATCH'} {cond:<52s} "
              f"delta {d['mean']:+.3f}  p {d['p']:.2g}  n {d['n']}")
        if not ok:
            FAILURES.append(f"[5.3] condition {cond} is not significant (p={d['p']})")

    print("\n[§5.6] idle-machine timings")
    bench = REPO / "results" / "onnx" / "edge_bench_idle_2026-08-20" / "summary.csv"
    if not bench.exists():
        FAILURES.append(f"[5.6] missing benchmark artefact: {bench}")
    else:
        rows = [r for r in csv.DictReader(bench.open(newline="", encoding="utf-8"))]
        by = {(r["backend"], r["threads"]): float(r["mean_rtf"]) for r in rows if r.get("backend")}
        for threads, claimed_ratio in (("1", 1.39), ("2", 1.43), ("4", 1.46)):
            fp32, int8 = by.get(("fp32", threads)), by.get(("int8", threads))
            if fp32 is None or int8 is None:
                FAILURES.append(f"[5.6] no fp32/int8 row at {threads} thread(s)")
                continue
            check("5.6", f"int8/fp32 slowdown at {threads} thread(s)", claimed_ratio,
                  int8 / fp32, 0.01)
        check("5.6", "fp32 RTF, 1 thread", 0.084, by[("fp32", "1")], 0.001)
        check("5.6", "int8 RTF, 1 thread", 0.117, by[("int8", "1")], 0.001)

    print("\n" + "=" * 84)
    if FAILURES:
        print(f"AUDIT FAILED: {len(FAILURES)} of {CHECKS} claims do not match their source")
        for f in FAILURES:
            print("  -", f)
        return 1
    print(f"AUDIT PASSED: all {CHECKS} checked claims match their source artefacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
