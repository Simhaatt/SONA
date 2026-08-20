"""Index-scale study: do the findings hold as the database grows?

The main experiments use 10 000-vector indices. Recall generally *falls* as a database grows
(more candidates compete to displace the true neighbours), so a result measured at 10k may not
survive at deployment scale. This sweeps the database size and reports how the two headline
quantities move:

  * absolute Recall@10 for the data-oblivious method
  * the gap to OPQ at a matched budget

The largest available bank is 23 075 MS-SNSD clean_train utterances; this cannot reach 10^6, so
the result is a *trend* over the reachable range rather than proof at deployment scale, and is
reported as such.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.embeddings import load_bank  # noqa: E402
from src.quantizers import RotatedScalarQuantizer  # noqa: E402
from src.retrieval import FaissPQIndex, normalize, recall_at_k, sorted_topk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500, 5000, 10000])
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--m", type=int, default=192, help="PQ sub-quantizers at matched budget")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb = ROOT / "embeddings" / args.model
    DB_all, _ = load_bank(emb, "database")
    Q, _ = load_bank(emb, "queries")
    d = DB_all.shape[1]
    Qn = normalize(Q)
    print(f"{args.model}: full bank {DB_all.shape}, queries {Q.shape}")
    print(f"budget: {args.bits} bit/coord (oblivious) vs PQ/OPQ m={args.m}\n")

    rows = []
    print(f"{'N':>7s} {'oblivious':>10s} {'PQ':>9s} {'OPQ':>9s} {'OPQ-obl':>9s}")
    for n in args.sizes:
        if n > len(DB_all):
            print(f"{n:7d}  (skipped: bank has only {len(DB_all)})")
            continue
        DBn = normalize(DB_all[:n])
        GT = sorted_topk(Qn, DBn, args.k, metric="ip")

        q = RotatedScalarQuantizer(d=d, bits=args.bits, rotate=True, rotation="gaussian", seed=0)
        r_obl = recall_at_k(
            sorted_topk(Qn, q.decode(*q.encode(DBn)), args.k, metric="ip"), GT, args.k)

        r_pq = recall_at_k(
            FaissPQIndex(d=d, m=args.m, nbits=8, opq=False).build(DBn)
            .search(Qn, args.k, metric="ip"), GT, args.k)
        r_opq = recall_at_k(
            FaissPQIndex(d=d, m=args.m, nbits=8, opq=True).build(DBn)
            .search(Qn, args.k, metric="ip"), GT, args.k)

        gap = r_opq - r_obl
        rows.append({"n": n, "oblivious": r_obl, "pq": r_pq, "opq": r_opq, "gap": gap})
        print(f"{n:7d} {r_obl:10.4f} {r_pq:9.4f} {r_opq:9.4f} {gap:+9.4f}", flush=True)

    if len(rows) >= 2:
        print("\nTrend over the reachable range:")
        first, last = rows[0], rows[-1]
        print(f"  oblivious Recall@10: {first['oblivious']:.4f} at N={first['n']} -> "
              f"{last['oblivious']:.4f} at N={last['n']} "
              f"({last['oblivious']-first['oblivious']:+.4f})")
        print(f"  gap to OPQ:          {first['gap']:+.4f} -> {last['gap']:+.4f} "
              f"({last['gap']-first['gap']:+.4f})")
        gaps = [r["gap"] for r in rows]
        widening = gaps[-1] > gaps[0]
        print(f"\n  => As the index grows, the data-dependent advantage "
              f"{'WIDENS' if widening else 'NARROWS'}.")
        print("     Caveat: this covers 1k-10k vectors only; behaviour at 10^6 remains "
              "extrapolation and is stated as such in the paper.")

    out = ROOT / "results" / "scale"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"scale_{args.model}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "oblivious", "pq", "opq", "gap"])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out / f'scale_{args.model}.csv'}")


if __name__ == "__main__":
    main()
