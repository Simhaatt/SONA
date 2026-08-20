"""Downstream semantic retrieval: does quantization preserve MEANING, not just geometry?

Recall@k measures whether the true nearest neighbours survive compression. It does not tell us
whether the *useful* structure survives -- a quantizer could scramble the ranking among near-ties
while keeping semantically identical items retrievable, or vice versa. This script measures the
task a practitioner actually cares about, using labels already present in the corpora:

  speaker  VoiceBank-DEMAND test clean (824 clips, speaker IDs p232/p257)
           query = each clip, correct = another clip from the same speaker
  noise    MS-SNSD noisy mixtures (1100 clips, noise_type labels)
           query = each clip, correct = another clip with the same noise type

Metric is precision@k over leave-one-out queries against the same bank (the query itself is
excluded), which is the standard protocol for label-based retrieval. A random-guess baseline is
reported so the numbers are interpretable rather than free-floating.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.embeddings import load_bank  # noqa: E402
from src.quantizers import RotatedScalarQuantizer, UniformScalarQuantizer  # noqa: E402
from src.retrieval import FaissPQIndex, normalize  # noqa: E402


def precision_at_k(X: np.ndarray, labels: list[str], k: int = 10) -> float:
    """Leave-one-out precision@k: fraction of the k nearest neighbours sharing the query label."""
    Xn = normalize(X)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)  # exclude self
    topk = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    lab = np.asarray(labels)
    hits = (lab[topk] == lab[:, None]).sum(axis=1)
    return float(np.mean(hits / k))


def per_query_precision(X: np.ndarray, labels: list[str], k: int = 10) -> np.ndarray:
    """Per-query precision@k, so methods can be compared with a PAIRED test over queries."""
    Xn = normalize(X)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)
    topk = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    lab = np.asarray(labels)
    return (lab[topk] == lab[:, None]).sum(axis=1) / k


def random_baseline(labels: list[str], k: int = 10) -> float:
    """Expected precision@k if neighbours were drawn at random (leave-one-out)."""
    n = len(labels)
    cnt = Counter(labels)
    # P(random other item shares my label) averaged over queries
    return float(np.mean([(cnt[l] - 1) / (n - 1) for l in labels]))


def reconstruct(X: np.ndarray, method: str, d: int, bits: int,
                train: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct X after quantization.

    `train` supplies the codebook-fitting data for the data-DEPENDENT methods. It must be a
    separate, larger bank: fitting 256 k-means centroids on the 824/1100-vector evaluation set
    would both under-train the codebooks (faiss wants >=9984 points) and let them overfit the very
    vectors being scored. The data-oblivious methods ignore `train` by construction -- which is
    precisely the property under study.
    """
    Xn = normalize(X)
    if method == "rot":
        q = RotatedScalarQuantizer(d=d, bits=bits, rotate=True, rotation="gaussian", seed=0)
        return q.decode(*q.encode(Xn))
    if method == "norot":
        q = RotatedScalarQuantizer(d=d, bits=bits, rotate=False, seed=0)
        return q.decode(*q.encode(Xn))
    if method == "uniform":
        q = UniformScalarQuantizer(d=d, bits=bits).fit(normalize(train) if train is not None else Xn)
        return q.decode(*q.encode(Xn))
    if method.startswith("pq") or method.startswith("opq"):
        import faiss

        m = {1: 96, 2: 192, 3: 256, 4: 384}[bits]
        opq = method.startswith("opq")
        idx = FaissPQIndex(d=d, m=m, nbits=8, opq=opq)
        fit_on = normalize(train) if train is not None else Xn
        # train the codebooks on the held-out bank, then encode ONLY the evaluation vectors
        import numpy as _np

        X_fit = _np.ascontiguousarray(fit_on, dtype=_np.float32)
        X_eval = _np.ascontiguousarray(Xn, dtype=_np.float32)
        if opq:
            opq_mat = faiss.OPQMatrix(d, m)
            pq = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
            index = faiss.IndexPreTransform(opq_mat, pq)
        else:
            index = faiss.IndexPQ(d, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(X_fit)
        index.add(X_eval)
        recon = index.reconstruct_n(0, len(X_eval))
        return _np.asarray(recon, dtype=_np.float32)
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    emb = ROOT / "embeddings" / args.model
    # Held-out bank for fitting the data-dependent codebooks: 10k vectors from a DIFFERENT corpus
    # split than either semantic bank, so PQ/OPQ are neither under-trained nor fitted to the
    # vectors they are scored on.
    TRAIN, _ = load_bank(emb, "database")
    print(f"codebook-fitting bank (held out): {TRAIN.shape}\n")

    tasks = []
    Xs, ms = load_bank(emb, "speakers")
    tasks.append(("speaker", Xs, [m["speaker"] for m in ms]))
    Xn_, mn = load_bank(emb, "noisy")
    tasks.append(("noise_type", Xn_, [m["noise_type"] for m in mn]))

    out_rows = []
    per_query = {}
    for task, X, labels in tasks:
        d = X.shape[1]
        n_classes = len(set(labels))
        base = random_baseline(labels, args.k)
        fp32 = precision_at_k(X, labels, args.k)
        print("=" * 92)
        print(f"TASK: {task}   n={len(X)}  classes={n_classes}  "
              f"random baseline P@{args.k}={base:.4f}")
        print("=" * 92)
        print(f"  {'float32 (uncompressed)':32s} P@{args.k}={fp32:.4f}")
        out_rows.append({"task": task, "method": "float32", "bits": 32 * d,
                          "precision_at_k": fp32, "random_baseline": base})

        for bits in (1, 2, 3, 4):
            for meth in ("rot", "norot", "pq", "opq"):
                try:
                    Xq = reconstruct(X, meth, d, bits, train=TRAIN)
                    p = precision_at_k(Xq, labels, args.k)
                    per_query[(task, meth, bits)] = per_query_precision(Xq, labels, args.k)
                    retained = p / fp32 if fp32 > 0 else float("nan")
                    name = {"rot": f"scalar-{bits}b+rotation", "norot": f"scalar-{bits}b no-rot",
                            "pq": f"PQ({bits}b-equiv)", "opq": f"OPQ({bits}b-equiv)"}[meth]
                    print(f"  {name:32s} P@{args.k}={p:.4f}  ({retained*100:5.1f}% of float32)")
                    out_rows.append({"task": task, "method": name, "bits": bits * d,
                                      "precision_at_k": p, "retained_frac": retained,
                                      "random_baseline": base})
                except Exception as e:
                    print(f"  {meth}-{bits}b FAILED: {str(e)[:90]}")
        print()

    # ------------------------------------------------- significance of the rotation ablation
    from scipy.stats import wilcoxon

    print("=" * 92)
    print(f"ROTATION ABLATION significance (paired over queries, Wilcoxon; +ve favours rotation)")
    print("=" * 92)
    print(f"{'task':12s} {'bits':>5s} {'with rot':>9s} {'no rot':>9s} {'delta':>9s} {'p':>10s} {'verdict':>14s}")
    for task, _, _ in tasks:
        for bits in (1, 2, 3, 4):
            a = per_query.get((task, "rot", bits))
            b = per_query.get((task, "norot", bits))
            if a is None or b is None:
                continue
            delta = float(np.mean(a) - np.mean(b))
            try:
                _, p = wilcoxon(a, b)
            except Exception:
                p = float("nan")
            verdict = ("rotation helps" if delta > 0 and p < 0.05 else
                       "rotation HURTS" if delta < 0 and p < 0.05 else "no sig. difference")
            print(f"{task:12s} {bits:5d} {np.mean(a):9.4f} {np.mean(b):9.4f} "
                  f"{delta:+9.4f} {p:10.2e} {verdict:>14s}")

    # ------------------------------------------------------------------ summary
    print("=" * 92)
    print("SEMANTIC RETENTION at the most aggressive budget (1 bit/coord)")
    print("=" * 92)
    for task, _, _ in tasks:
        rows = [r for r in out_rows if r["task"] == task]
        f32 = next(r for r in rows if r["method"] == "float32")["precision_at_k"]
        rot = next((r for r in rows if r["method"] == "scalar-1b+rotation"), None)
        opq = next((r for r in rows if r["method"] == "OPQ(1b-equiv)"), None)
        if rot:
            print(f"  {task:12s} float32={f32:.4f}  rot-scalar={rot['precision_at_k']:.4f} "
                  f"({rot['retained_frac']*100:.1f}%)"
                  + (f"  OPQ={opq['precision_at_k']:.4f} ({opq['retained_frac']*100:.1f}%)"
                     if opq else ""))

    out = ROOT / "results" / "semantic"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"semantic_{args.model}.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["task", "method", "bits", "precision_at_k", "retained_frac", "random_baseline"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(out_rows)
    print(f"\nwrote {out / f'semantic_{args.model}.csv'}")


if __name__ == "__main__":
    main()
