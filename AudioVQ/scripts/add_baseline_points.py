"""Fill in PQ/OPQ baseline points that the main sweep missed.

faiss requires the number of sub-quantizers m to divide the dimension d, so m=288 (attempted for
the ~2300-bit budget) failed for d=768. Valid divisors near that budget are m=256 (2048 bits).
This adds the missing points and appends them to the existing benchmark CSV rather than re-running
the whole sweep, since OPQ training is the expensive part.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import load_bank  # noqa: E402
from src.retrieval import FaissPQIndex, normalize, recall_at_k, sorted_topk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--m", type=int, nargs="+", default=[256])
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    emb = ROOT / "embeddings" / args.model
    DB, _ = load_bank(emb, "database")
    Q, _ = load_bank(emb, "queries")
    d = DB.shape[1]
    DBn, Qn = normalize(DB), normalize(Q)
    GT = sorted_topk(Qn, DBn, args.k, metric="ip")
    print(f"d={d}; adding m={args.m}")

    csv_path = ROOT / "results" / "retrieval" / f"benchmark_{args.model}.csv"
    existing = list(csv.DictReader(open(csv_path)))
    cols = list(existing[0].keys())
    have = {r["method"] for r in existing}

    new_rows = []
    for m in args.m:
        if d % m:
            print(f"  skip m={m}: does not divide d={d}")
            continue
        for opq in (False, True):
            idx = FaissPQIndex(d=d, m=m, nbits=8, opq=opq)
            if idx.name in have:
                print(f"  {idx.name} already present, skipping")
                continue
            t0 = time.perf_counter()
            idx.build(DBn)
            ret = idx.search(Qn, args.k, metric="ip")
            row = {c: "" for c in cols}
            row.update({
                "method": idx.name,
                "bits_per_vector": idx.bits_per_vector,
                "bytes_per_vector": idx.bits_per_vector / 8.0,
                "compression_vs_fp32": (d * 32) / idx.bits_per_vector,
                "recall@1": recall_at_k(ret, GT, 1),
                "recall@10": recall_at_k(ret, GT, 10),
                "recall@100": recall_at_k(ret, GT, 100),
                "build_search_s": time.perf_counter() - t0,
            })
            new_rows.append(row)
            print(f"  {idx.name:22s} {idx.bits_per_vector:6.0f} b/vec "
                  f"R@10={row['recall@10']:.4f} ({row['build_search_s']:.0f}s)", flush=True)

    if new_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(existing + new_rows)
        print(f"appended {len(new_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
