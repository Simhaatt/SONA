"""Analysis of the retrieval benchmark: matched-budget comparison and the bit-rate trend.

Separated from the benchmark run so the (slow) OPQ training is not repeated when only the
analysis changes. Budgets are matched with a tolerance because the data-oblivious method carries
an extra 32-bit norm per vector, so its budgets never land exactly on PQ's multiples of 8m --
comparing at "nearest budget" keeps the comparison honest rather than declaring a mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--tol", type=float, default=0.10, help="relative budget tolerance")
    args = ap.parse_args()

    res_dir = ROOT / "results" / "retrieval"
    rows = list(csv.DictReader(open(res_dir / f"benchmark_{args.model}.csv")))
    for r in rows:
        for k in ("bits_per_vector", "recall@1", "recall@10", "recall@100", "build_search_s",
                  "compression_vs_fp32"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])

    def get(prefix, bits_target):
        cands = [r for r in rows if r["method"].startswith(prefix)
                 and abs(r["bits_per_vector"] - bits_target) / bits_target <= args.tol]
        return max(cands, key=lambda r: r["recall@10"]) if cands else None

    obl = sorted([r for r in rows if r["method"].startswith("scalar")
                  and "no-rot" not in r["method"]], key=lambda r: r["bits_per_vector"])

    print("=" * 104)
    print("RQ1 -- MATCHED-BUDGET: data-oblivious (rotation+scalar) vs data-dependent (PQ / OPQ)")
    print("=" * 104)
    print(f"{'budget (bits/vec)':>18s} {'compress':>9s} {'oblivious':>10s} {'PQ':>10s} "
          f"{'OPQ':>10s} {'bestDD-obl':>12s} {'winner':>15s}")
    trend = []
    for o in obl:
        b = o["bits_per_vector"]
        pq, opq = get("PQ-", b), get("OPQ-", b)
        dd = [x for x in (pq, opq) if x]
        if not dd:
            print(f"{b:18.0f} {o['compression_vs_fp32']:8.1f}x {o['recall@10']:10.4f} "
                  f"{'--':>10s} {'--':>10s}   (no baseline at this budget)")
            continue
        best = max(dd, key=lambda r: r["recall@10"])
        gap = best["recall@10"] - o["recall@10"]
        winner = "data-dependent" if gap > 0 else "data-oblivious"
        trend.append((b, gap))
        pq_s = f"{pq['recall@10']:10.4f}" if pq else f"{'--':>10s}"
        opq_s = f"{opq['recall@10']:10.4f}" if opq else f"{'--':>10s}"
        print(f"{b:18.0f} {o['compression_vs_fp32']:8.1f}x {o['recall@10']:10.4f} "
              f"{pq_s} {opq_s} {gap:+12.4f} {winner:>15s}")

    if len(trend) >= 2:
        print("\n  Gap (best data-dependent minus data-oblivious) vs budget:")
        for b, g in trend:
            print(f"    {b:6.0f} bits/vec -> {g:+.4f}")
        # Comparing only the endpoints would call a non-monotone curve "confirmed"; test the
        # whole ordering instead, and report the rank correlation alongside it.
        gaps = [g for _, g in trend]
        budgets = [b for b, _ in trend]
        monotone = all(gaps[i] > gaps[i + 1] for i in range(len(gaps) - 1))
        endpoints_fall = gaps[0] > gaps[-1]
        try:
            from scipy.stats import spearmanr

            rho, _ = spearmanr(budgets, gaps)
        except Exception:
            rho = float("nan")
        print(f"\n  strictly monotone decreasing: {monotone}   Spearman(budget, gap) = {rho:+.2f}")
        if monotone:
            verdict = "CONFIRMED (gap shrinks at every step)"
        elif endpoints_fall:
            verdict = ("PARTIALLY SUPPORTED -- endpoints fall but the curve is not monotone; "
                       "do not report as a clean trend")
        else:
            verdict = "NOT CONFIRMED"
        print(f"  PREDICTION FROM THE ANISOTROPY ANALYSIS: {verdict}")

    # ------------------------------------------------------------------ RQ2 rotation ablation
    print("\n" + "=" * 104)
    print("RQ2 -- contribution of the random rotation (identical quantizer and codebook)")
    print("=" * 104)
    print(f"{'bits/coord':>10s} {'with rot':>10s} {'without':>10s} {'delta R@10':>11s} {'rel gain':>10s}")
    for bits in (1, 2, 3, 4):
        a = next((r for r in rows if r["method"] == f"scalar-{bits}b-rot-gaussian"), None)
        b = next((r for r in rows if r["method"] == f"scalar-{bits}b-no-rot"), None)
        if a and b:
            rel = (a["recall@10"] - b["recall@10"]) / max(b["recall@10"], 1e-9) * 100
            print(f"{bits:10d} {a['recall@10']:10.4f} {b['recall@10']:10.4f} "
                  f"{a['recall@10']-b['recall@10']:+11.4f} {rel:9.1f}%")

    # ------------------------------------------------------------------ build cost
    print("\n" + "=" * 104)
    print("BUILD COST -- the practical advantage of being data-oblivious (no training pass)")
    print("=" * 104)
    print(f"{'method':26s} {'bits/vec':>9s} {'R@10':>8s} {'build+search s':>15s}")
    for r in sorted(rows, key=lambda r: (r["bits_per_vector"], r["method"])):
        if r["method"] == "float32-exact":
            continue
        print(f"{r['method']:26s} {r['bits_per_vector']:9.0f} {r['recall@10']:8.4f} "
              f"{r['build_search_s']:15.1f}")

    obl_hi = max(obl, key=lambda r: r["bits_per_vector"])
    opq_hi = get("OPQ-", obl_hi["bits_per_vector"])
    if opq_hi:
        speed = opq_hi["build_search_s"] / max(obl_hi["build_search_s"], 1e-9)
        print(f"\n  At ~{obl_hi['bits_per_vector']:.0f} bits/vec: OPQ gains "
              f"{opq_hi['recall@10']-obl_hi['recall@10']:+.4f} R@10 but costs {speed:.0f}x more "
              f"to build ({opq_hi['build_search_s']:.0f}s vs {obl_hi['build_search_s']:.1f}s).")

    out = res_dir / f"analysis_{args.model}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"trend_gap_vs_budget": trend}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
