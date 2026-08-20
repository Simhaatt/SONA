"""E1 -- inverse approximation ratio (1/Ratio@k) alongside Recall@k.

Why. Dimitropoulos & Mamoulis (arXiv:2606.04522) show that Recall@k overstates the cost of
approximation in ANN *search*, and propose 1/Ratio@k -- which scores how much *worse in distance*
the retrieved neighbours are, rather than whether they are the same items -- as a replacement that
tracks downstream utility. Our setting is harder: they hold vectors exact and vary search effort,
so a missed neighbour is merely a slightly more distant true vector, whereas quantization damages
the vectors themselves.

The open question their paper leaves for us: does their proposed fix work here? Two outcomes, both
publishable.

  * 1/Ratio@k stays high while Recall@k collapses  ->  their metric corroborates our semantic
    result from a purely geometric direction, without needing task labels. Strong support.
  * 1/Ratio@k collapses too, while semantic retention stays ~96 %  ->  their fix is insufficient
    under quantization, and *no* purely geometric metric predicts deployment behaviour here. That
    is a stronger and more surprising claim than the one we currently make.

Definition. Vectors are unit-normalised, so Euclidean distance is a monotone function of the inner
product the index actually ranks by; we report the ratio in distance space, which is the form the
source paper uses.

    Ratio@k(q)   = mean_true_distance(retrieved top-k) / mean_true_distance(true top-k)   >= 1
    1/Ratio@k(q) = its reciprocal, in (0, 1], higher is better -- directly comparable to recall.

"True distance" always means distance in the ORIGINAL float32 space, for both sets. Only the
*selection* of neighbours is made in the quantized space; the scoring is exact. That isolates the
quality of the selection, which is what the metric is for.

Usage:
    python scripts/ratio_at_k.py --model wavlm_base_plus
    python scripts/ratio_at_k.py --model all
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

import numpy as np  # noqa: E402

from src.embeddings import load_bank  # noqa: E402
from src.quantizers import RotatedScalarQuantizer, UniformScalarQuantizer  # noqa: E402
from src.retrieval import (  # noqa: E402
    FaissPQIndex,
    Float32Index,
    ScalarVQIndex,
    normalize,
    recall_at_k,
    sorted_topk,
)

MODELS = ("wavlm_base_plus", "wav2vec2_base", "hubert_base")


def true_distances(queries: np.ndarray, database: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from each query to the items it selected. (nq, k)"""
    # ||q - d||^2 = ||q||^2 + ||d||^2 - 2 q.d ; vectors are unit-norm so this is 2 - 2*ip
    picked = database[idx]                                   # (nq, k, d)
    ip = np.einsum("qd,qkd->qk", queries, picked)
    return np.sqrt(np.maximum(2.0 - 2.0 * ip, 0.0))


def inverse_ratio_at_k(queries: np.ndarray, database: np.ndarray,
                       retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> dict:
    """1/Ratio@k, averaged over queries. Both sets scored in the original space."""
    d_ret = true_distances(queries, database, retrieved[:, :k]).mean(axis=1)
    d_gt = true_distances(queries, database, ground_truth[:, :k]).mean(axis=1)
    # A query whose true neighbours sit at distance 0 (duplicate vectors) would divide by zero.
    ok = d_gt > 1e-9
    ratio = np.ones_like(d_gt)
    ratio[ok] = d_ret[ok] / d_gt[ok]
    inv = 1.0 / np.maximum(ratio, 1e-12)
    return {
        "inv_ratio_mean": float(np.mean(inv)),
        "inv_ratio_p05": float(np.percentile(inv, 5)),
        "mean_dist_retrieved": float(np.mean(d_ret)),
        "mean_dist_true": float(np.mean(d_gt)),
        "n_degenerate_queries": int((~ok).sum()),
    }


def run_model(model: str, k: int, out_dir: Path) -> list[dict]:
    emb_dir = ROOT / "embeddings" / model
    DB, _ = load_bank(emb_dir, "database")
    Q, _ = load_bank(emb_dir, "queries")
    d = DB.shape[1]
    DBn, Qn = normalize(DB), normalize(Q)
    print(f"\n=== {model}: database={DBn.shape} queries={Qn.shape} ===", flush=True)

    GT = sorted_topk(Qn, DBn, k, metric="ip")

    methods = [Float32Index(d)]
    for bits in (1, 2, 3, 4):
        methods.append(ScalarVQIndex(
            RotatedScalarQuantizer(d=d, bits=bits, rotate=True, rotation="gaussian", seed=0)))
        methods.append(ScalarVQIndex(
            RotatedScalarQuantizer(d=d, bits=bits, rotate=False, seed=0)))
        methods.append(ScalarVQIndex(UniformScalarQuantizer(d=d, bits=bits)))
    for m in (96, 192, 288, 384):
        methods.append(FaissPQIndex(d=d, m=m, nbits=8, opq=False))
        methods.append(FaissPQIndex(d=d, m=m, nbits=8, opq=True))

    rows = []
    for meth in methods:
        t0 = time.perf_counter()
        try:
            meth.build(DBn)
            ret = meth.search(Qn, k, metric="ip")
            row = {"model": model, "method": meth.name,
                   "bits_per_vector": float(meth.bits_per_vector)}
            for kk in (1, 10, 100):
                if kk > k:
                    continue
                row[f"recall@{kk}"] = recall_at_k(ret, GT, kk)
                row[f"inv_ratio@{kk}"] = inverse_ratio_at_k(Qn, DBn, ret, GT, kk)["inv_ratio_mean"]
            row.update({f"{kk}_detail": v for kk, v in
                        [("k10", inverse_ratio_at_k(Qn, DBn, ret, GT, min(10, k)))]})
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
            print(f"  {meth.name:28s} {row['bits_per_vector']:7.0f} b/vec  "
                  f"R@10={row.get('recall@10', float('nan')):.4f}  "
                  f"1/Ratio@10={row.get('inv_ratio@10', float('nan')):.4f}", flush=True)
        except Exception as e:
            print(f"  {meth.name:28s} FAILED: {str(e)[:110]}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "results" / "ratio_at_k"))
    args = ap.parse_args()

    models = MODELS if args.model == "all" else (args.model,)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for m in models:
        all_rows += run_model(m, args.k, out)

    flat = [{k: v for k, v in r.items() if not k.endswith("_detail")} for r in all_rows]
    cols = [c for c in ["model", "method", "bits_per_vector",
                        "recall@1", "inv_ratio@1", "recall@10", "inv_ratio@10",
                        "recall@100", "inv_ratio@100", "seconds"]
            if any(c in r for r in flat)]
    with (out / "ratio_at_k.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat)
    (out / "ratio_at_k.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")

    # ------------------------------------------------------ the comparison the paper needs
    lines = ["# 1/Ratio@k vs Recall@k", "",
             "Both metrics are purely geometric and need no labels. Recall@k asks *are these the "
             "same items*; 1/Ratio@k asks *how much worse in distance*. Dimitropoulos & Mamoulis "
             "(arXiv:2606.04522) propose the latter as a better proxy for downstream utility "
             "under search approximation; this table tests whether it also holds under "
             "quantization, which damages the vectors themselves.", "",
             "| model | method | bits/vec | Recall@10 | 1/Ratio@10 | gap |",
             "|---|---|---|---|---|---|"]
    for r in flat:
        if "recall@10" not in r:
            continue
        gap = r["inv_ratio@10"] - r["recall@10"]
        lines.append(f"| {r['model']} | {r['method']} | {r['bits_per_vector']:.0f} | "
                     f"{r['recall@10']:.4f} | {r['inv_ratio@10']:.4f} | {gap:+.4f} |")
    (out / "ratio_at_k.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {out}")
    obl = [r for r in flat if "rot" in r["method"] and "recall@10" in r]
    if obl:
        print("\ndata-oblivious rows -- recall vs inverse ratio at 10:")
        for r in obl:
            print(f"  {r['model']:16s} {r['method']:26s} "
                  f"R@10={r['recall@10']:.4f}  1/Ratio@10={r['inv_ratio@10']:.4f}")


if __name__ == "__main__":
    main()
