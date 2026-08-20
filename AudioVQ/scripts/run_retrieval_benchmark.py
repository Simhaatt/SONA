"""RQ1 + RQ2: retrieval quality vs bit budget, data-oblivious vs data-dependent.

Tests the prediction made by the anisotropy analysis: because audio embeddings occupy only ~3% of
their ambient dimensions, data-DEPENDENT quantizers (PQ/OPQ) should beat the data-OBLIVIOUS
rotation+scalar scheme, with the gap widening as the bit budget shrinks.

RQ2 is answered by the `no-rot` rows: the identical scalar quantizer with the rotation disabled,
which isolates the rotation's contribution from the codebook's.

All methods are compared at matched bits/vector. Ground truth is exhaustive float32 search over
the same database, so every number is directly comparable.

Usage:
    python scripts/run_retrieval_benchmark.py --model wavlm_base_plus
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.embeddings import load_bank  # noqa: E402
from src.quantizers import RotatedScalarQuantizer, UniformScalarQuantizer  # noqa: E402
from src.retrieval import (  # noqa: E402
    FaissPQIndex,
    Float32Index,
    ScalarVQIndex,
    distortion_stats,
    normalize,
    sorted_topk,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "results" / "retrieval"))
    args = ap.parse_args()

    emb_dir = ROOT / "embeddings" / args.model
    DB, _ = load_bank(emb_dir, "database")
    Q, _ = load_bank(emb_dir, "queries")
    d = DB.shape[1]
    print(f"database={DB.shape} queries={Q.shape} d={d}")

    # Work with unit-normalised vectors so inner product == cosine, the standard retrieval
    # setting for embeddings, and so the quantizer's norm handling is not what is being measured.
    DBn, Qn = normalize(DB), normalize(Q)

    print("computing exhaustive float32 ground truth ...")
    t0 = time.perf_counter()
    GT = sorted_topk(Qn, DBn, args.k, metric="ip")
    print(f"  done ({time.perf_counter()-t0:.1f}s)")

    # ---------------------------------------------------------------- methods at matched budgets
    methods = [Float32Index(d)]

    # Data-oblivious scalar: bits/coord in {1,2,3,4} -> 768/1536/2304/3072 bits (+32 norm)
    for bits in (1, 2, 3, 4):
        methods.append(ScalarVQIndex(
            RotatedScalarQuantizer(d=d, bits=bits, rotate=True, rotation="gaussian", seed=0)))
        # RQ2 ablation: same quantizer, rotation off
        methods.append(ScalarVQIndex(
            RotatedScalarQuantizer(d=d, bits=bits, rotate=False, seed=0)))
        methods.append(ScalarVQIndex(UniformScalarQuantizer(d=d, bits=bits)))

    # Data-dependent PQ/OPQ. m sub-quantizers x 8 bits = 8m bits/vector.
    # m in {96,192,288,384} matches 1,2,3,4 bits/coord for d=768.
    for m in (96, 192, 288, 384):
        methods.append(FaissPQIndex(d=d, m=m, nbits=8, opq=False))
        methods.append(FaissPQIndex(d=d, m=m, nbits=8, opq=True))

    rows = []
    for meth in methods:
        t0 = time.perf_counter()
        try:
            meth.build(DBn)
            ret = meth.search(Qn, args.k, metric="ip")
            row = {
                "method": meth.name,
                "bits_per_vector": meth.bits_per_vector,
                "bytes_per_vector": meth.bits_per_vector / 8.0,
                "compression_vs_fp32": (d * 32) / meth.bits_per_vector,
                "recall@1": _r(ret, GT, 1),
                "recall@10": _r(ret, GT, 10),
                "recall@100": _r(ret, GT, 100),
                "build_search_s": time.perf_counter() - t0,
            }
            if isinstance(meth, ScalarVQIndex):
                row.update({f"dist_{k}": v for k, v in
                            distortion_stats(DBn, meth._recon).items()})
            rows.append(row)
            print(f"  {meth.name:26s} {meth.bits_per_vector:7.0f} b/vec  "
                  f"R@1={row['recall@1']:.4f} R@10={row['recall@10']:.4f} "
                  f"R@100={row['recall@100']:.4f}  ({row['build_search_s']:.0f}s)", flush=True)
        except Exception as e:  # a failing baseline must not kill the sweep
            print(f"  {meth.name:26s} FAILED: {str(e)[:120]}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"benchmark_{args.model}.csv", "w", newline="", encoding="utf-8") as fh:
        cols = sorted({k for r in rows for k in r})
        cols = [c for c in ["method", "bits_per_vector", "bytes_per_vector",
                            "compression_vs_fp32", "recall@1", "recall@10", "recall@100"]
                if c in cols] + [c for c in cols if c not in (
                    "method", "bits_per_vector", "bytes_per_vector", "compression_vs_fp32",
                    "recall@1", "recall@10", "recall@100")]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- head-to-head at matched bits
    print("\n" + "=" * 96)
    print("MATCHED-BUDGET COMPARISON (RQ1: data-oblivious vs data-dependent)")
    print("=" * 96)
    print(f"{'bits/vec':>9s}  {'data-oblivious (rot+scalar)':>30s}  {'PQ':>10s}  {'OPQ':>10s}  {'winner':>14s}")
    by_bits = {}
    for r in rows:
        by_bits.setdefault(int(round(r["bits_per_vector"] / 8) * 8), []).append(r)
    verdicts = []
    for bits in sorted(by_bits):
        grp = by_bits[bits]
        obl = next((r for r in grp if r["method"].startswith("scalar") and "no-rot" not in r["method"]), None)
        pq = next((r for r in grp if r["method"].startswith("PQ")), None)
        opq = next((r for r in grp if r["method"].startswith("OPQ")), None)
        if not (obl and (pq or opq)):
            continue
        best_dd = max([x for x in (pq, opq) if x], key=lambda r: r["recall@10"])
        winner = "data-oblivious" if obl["recall@10"] > best_dd["recall@10"] else "data-dependent"
        verdicts.append(winner)
        print(f"{bits:9d}  {obl['recall@10']:30.4f}  "
              f"{(pq['recall@10'] if pq else float('nan')):10.4f}  "
              f"{(opq['recall@10'] if opq else float('nan')):10.4f}  {winner:>14s}")

    if verdicts:
        dd = verdicts.count("data-dependent")
        print(f"\n  data-dependent wins {dd}/{len(verdicts)} budgets"
              f" -- {'PREDICTION CONFIRMED' if dd > len(verdicts)/2 else 'PREDICTION NOT SUPPORTED'}")

    # ---------------------------------------------------------------- RQ2: rotation ablation
    print("\n" + "=" * 96)
    print("RQ2: does the random rotation help? (identical quantizer, rotation on vs off)")
    print("=" * 96)
    print(f"{'bits/coord':>10s} {'with rotation':>15s} {'without':>10s} {'delta':>9s}")
    for bits in (1, 2, 3, 4):
        a = next((r for r in rows if r["method"] == f"scalar-{bits}b-rot-gaussian"), None)
        b = next((r for r in rows if r["method"] == f"scalar-{bits}b-no-rot"), None)
        if a and b:
            print(f"{bits:10d} {a['recall@10']:15.4f} {b['recall@10']:10.4f} "
                  f"{a['recall@10']-b['recall@10']:+9.4f}")

    with open(out / f"benchmark_{args.model}.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out / f'benchmark_{args.model}.csv'}")


def _r(ret, gt, k):
    from src.retrieval import recall_at_k
    return recall_at_k(ret, gt, k)


if __name__ == "__main__":
    main()
