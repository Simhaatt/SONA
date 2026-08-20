"""Does the choice of transformer layer change the quantization picture?

Every experiment so far uses the FINAL layer of the encoder. In speech SSL, different layers carry
different information (lower layers more acoustic/speaker, upper layers more phonetic), and layer
choice is a live decision for anyone building a retrieval index. If the structural properties that
drive our findings — within-vector crest factor, anisotropy — vary across layers, then layer
choice matters for quantization too, and reporting only the last layer would be misleading.

`extract_features` returns every layer in one forward pass, so the whole sweep costs roughly one
extraction. Uses a smaller bank than the main experiments (speed), which is adequate for measuring
structure and *relative* rotation benefit but not for headline recall numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.embeddings import _load_wav, load_model  # noqa: E402
from src.quantizers import RotatedScalarQuantizer  # noqa: E402
from src.retrieval import normalize, recall_at_k, sorted_topk  # noqa: E402

from src.config import MS_SNSD  # noqa: E402


@torch.no_grad()
def embed_all_layers(paths, model, device, sr, max_seconds=10.0):
    """Return (n_layers, n_files, d) mean-pooled embeddings from a single forward pass each."""
    out = None
    max_len = int(max_seconds * sr)
    for i, p in enumerate(paths):
        wav = _load_wav(p, sr)
        if wav.size > max_len:
            wav = wav[:max_len]
        if wav.size < sr // 10:
            wav = np.pad(wav, (0, sr // 10 - wav.size))
        x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
        feats, _ = model.extract_features(x)          # list of (1, T, d) per layer
        pooled = [f.mean(dim=1).squeeze(0).cpu().numpy() for f in feats]
        if out is None:
            out = [[] for _ in pooled]
        for li, v in enumerate(pooled):
            out[li].append(v)
        if (i + 1) % 250 == 0:
            print(f"    {i+1}/{len(paths)}", flush=True)
    return np.asarray([np.asarray(l, dtype=np.float32) for l in out])


def crest_and_kurtosis(X: np.ndarray) -> tuple[float, float]:
    Xn = normalize(X).astype(np.float64)
    rms = np.sqrt((Xn ** 2).mean(axis=1)) + 1e-12
    crest = float((np.abs(Xn).max(axis=1) / rms).mean())
    z = Xn / rms[:, None]
    kurt = float(((z ** 4).mean(axis=1) - 3.0).mean())
    return crest, kurt


def mean_cosine(X: np.ndarray, n: int = 800, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(len(X), n), replace=False)
    U = normalize(X[idx]).astype(np.float64)
    G = U @ U.T
    iu = np.triu_indices(len(idx), k=1)
    return float(G[iu].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--n-database", type=int, default=3000)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--bits", type=int, default=2)
    args = ap.parse_args()

    model, device, sr = load_model(args.model)
    db_files = sorted(MS_SNSD.glob("clean_train/*.wav"))[: args.n_database]
    q_files = sorted(MS_SNSD.glob("clean_test/*.wav"))[: args.n_queries]
    print(f"{args.model}: {len(db_files)} database + {len(q_files)} query clips, all layers\n")

    print("  extracting database ...")
    DB = embed_all_layers(db_files, model, device, sr)
    print("  extracting queries ...")
    QY = embed_all_layers(q_files, model, device, sr)
    n_layers, _, d = DB.shape
    print(f"\n  got {n_layers} layers, d={d}\n")

    rows = []
    print(f"{'layer':>6s} {'crest':>7s} {'kurtosis':>9s} {'cos':>7s} "
          f"{'R@10 rot':>9s} {'R@10 norot':>11s} {'rot gain':>9s}")
    for li in range(n_layers):
        DBn, Qn = normalize(DB[li]), normalize(QY[li])
        GT = sorted_topk(Qn, DBn, 10, metric="ip")
        crest, kurt = crest_and_kurtosis(DB[li])
        cos = mean_cosine(DB[li])

        qa = RotatedScalarQuantizer(d=d, bits=args.bits, rotate=True, rotation="gaussian", seed=0)
        qn = RotatedScalarQuantizer(d=d, bits=args.bits, rotate=False, seed=0)
        ra = recall_at_k(sorted_topk(Qn, qa.decode(*qa.encode(DBn)), 10, metric="ip"), GT, 10)
        rn = recall_at_k(sorted_topk(Qn, qn.decode(*qn.encode(DBn)), 10, metric="ip"), GT, 10)
        gain = (ra - rn) / max(rn, 1e-9) * 100

        rows.append({"layer": li, "crest": crest, "kurtosis": kurt, "mean_cosine": cos,
                      "recall_rot": ra, "recall_norot": rn, "rot_gain_pct": gain})
        print(f"{li:6d} {crest:7.2f} {kurt:9.2f} {cos:+7.3f} {ra:9.4f} {rn:11.4f} {gain:8.1f}%")

    out = ROOT / "results" / "layers"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"layers_{args.model}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    crests = [r["crest"] for r in rows]
    gains = [r["rot_gain_pct"] for r in rows]
    recs = [r["recall_rot"] for r in rows]
    best_r = int(np.argmax(recs))
    print(f"\n  crest factor across layers: {min(crests):.2f} -> {max(crests):.2f} "
          f"({max(crests)/max(min(crests),1e-9):.1f}x spread)")
    print(f"  rotation gain across layers: {min(gains):+.1f}% -> {max(gains):+.1f}%")
    print(f"  best layer for retrieval: layer {best_r} (R@10 {recs[best_r]:.4f}) vs "
          f"final layer {n_layers-1} (R@10 {recs[-1]:.4f})")
    if len(crests) > 2:
        rho = float(np.corrcoef(crests, gains)[0, 1])
        print(f"  Pearson r(crest, rotation gain) across {len(crests)} layers = {rho:+.3f}")
    print(f"\nwrote {out / f'layers_{args.model}.csv'}")


if __name__ == "__main__":
    main()
