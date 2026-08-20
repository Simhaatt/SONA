"""Confidence intervals and paired tests for the retrieval recall numbers.

The benchmark reports mean Recall@k as a point estimate. Recall is computed per query, so the
per-query values support bootstrap CIs and paired tests between methods on identical queries --
which is what turns "OPQ leads by +0.24" into a defensible claim rather than an observation.

Recomputes per-query recall for the key methods at each budget, then reports:
  * mean Recall@10 with a 95% bootstrap CI
  * paired Wilcoxon of each data-dependent baseline against the data-oblivious method
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import wilcoxon

from src.embeddings import load_bank  # noqa: E402
from src.quantizers import RotatedScalarQuantizer  # noqa: E402
from src.retrieval import FaissPQIndex, normalize, sorted_topk  # noqa: E402


def per_query_recall(retrieved: np.ndarray, gt: np.ndarray, k: int) -> np.ndarray:
    r = retrieved[:, :k]
    g = gt[:, :k]
    return np.array([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(r, g)])


def boot_ci(v, n=5000, seed=0):
    v = np.asarray(v, dtype=np.float64)
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, size=(n, v.size))].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb = ROOT / "embeddings" / args.model
    DB, _ = load_bank(emb, "database")
    Q, _ = load_bank(emb, "queries")
    d = DB.shape[1]
    DBn, Qn = normalize(DB), normalize(Q)
    GT = sorted_topk(Qn, DBn, args.k, metric="ip")
    print(f"{args.model}: database={DB.shape} queries={Q.shape}\n")

    # budgets: bits/coord for the oblivious method, matched m for PQ/OPQ
    grid = [(1, 96), (2, 192), (3, 256), (4, 384)]
    rows, per_q = [], {}

    for bits, m in grid:
        q = RotatedScalarQuantizer(d=d, bits=bits, rotate=True, rotation="gaussian", seed=0)
        recon = q.decode(*q.encode(DBn))
        ret = sorted_topk(Qn, recon, args.k, metric="ip")
        per_q[("oblivious", bits)] = per_query_recall(ret, GT, args.k)

        for opq in (False, True):
            idx = FaissPQIndex(d=d, m=m, nbits=8, opq=opq).build(DBn)
            r = idx.search(Qn, args.k, metric="ip")
            per_q[("OPQ" if opq else "PQ", bits)] = per_query_recall(r, GT, args.k)
        print(f"  computed budget {bits} bit/coord (m={m})", flush=True)

    print("\n" + "=" * 96)
    print(f"Recall@{args.k} with 95% bootstrap CI, and paired Wilcoxon vs the data-oblivious method")
    print("=" * 96)
    print(f"{'bits':>5s} {'method':>11s} {'mean':>8s} {'95% CI':>20s} {'vs oblivious':>14s} {'p':>11s}")
    for bits, m in grid:
        base = per_q[("oblivious", bits)]
        mu, lo, hi = boot_ci(base)
        print(f"{bits:5d} {'oblivious':>11s} {mu:8.4f} [{lo:.4f}, {hi:.4f}]{'':>4s} "
              f"{'--':>14s} {'--':>11s}")
        rows.append({"bits": bits, "method": "oblivious", "mean": mu, "ci_lo": lo, "ci_hi": hi})
        for name in ("PQ", "OPQ"):
            v = per_q[(name, bits)]
            mu2, lo2, hi2 = boot_ci(v)
            delta = float(np.mean(v - base))
            try:
                _, p = wilcoxon(v, base)
            except Exception:
                p = float("nan")
            sig = "*" if p < 0.05 else " "
            print(f"{bits:5d} {name:>11s} {mu2:8.4f} [{lo2:.4f}, {hi2:.4f}]{'':>4s} "
                  f"{delta:+14.4f} {p:11.2e}{sig}")
            rows.append({"bits": bits, "method": name, "mean": mu2, "ci_lo": lo2, "ci_hi": hi2,
                          "delta_vs_oblivious": delta, "wilcoxon_p": float(p)})

    out = ROOT / "results" / "retrieval"
    with open(out / f"recall_significance_{args.model}.csv", "w", newline="",
              encoding="utf-8") as fh:
        cols = ["bits", "method", "mean", "ci_lo", "ci_hi", "delta_vs_oblivious", "wilcoxon_p"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out / f'recall_significance_{args.model}.csv'}")


if __name__ == "__main__":
    main()
