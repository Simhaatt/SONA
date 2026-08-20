"""Robustness check: do the headline findings survive on a GOOD layer?

F14 showed the final encoder layer — used for every headline result — is the worst of twelve for
retrieval. That raises a direct threat to validity: if the geometry-vs-meaning dissociation (F7)
or the data-dependent advantage (F4) only appear at a poor operating point, they are artifacts of
layer choice rather than properties of the quantizer.

This re-runs the two central measurements on the best layer and compares:
  * Recall@10 retained under 1 bit/coord quantization
  * speaker and noise-type precision@10 retained under the same quantization

If F7 holds on both layers, the finding is about quantization. If it collapses on the good layer,
the paper's central claim needs rewriting.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.embeddings import _load_wav, load_model  # noqa: E402
from src.quantizers import RotatedScalarQuantizer  # noqa: E402
from src.retrieval import normalize, recall_at_k, sorted_topk  # noqa: E402

from src.config import MS_SNSD, NOISY_MANIFEST  # noqa: E402
from src.config import VBD_TEST_CLEAN as VBD_CLEAN  # noqa: E402


@torch.no_grad()
def embed_layer(paths, model, device, sr, layer: int, max_seconds=10.0):
    out, max_len = [], int(max_seconds * sr)
    for i, p in enumerate(paths):
        wav = _load_wav(p, sr)
        if wav.size > max_len:
            wav = wav[:max_len]
        if wav.size < sr // 10:
            wav = np.pad(wav, (0, sr // 10 - wav.size))
        x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
        feats, _ = model.extract_features(x)
        out.append(feats[layer].mean(dim=1).squeeze(0).cpu().numpy())
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(paths)}", flush=True)
    return np.asarray(out, dtype=np.float32)


def precision_at_k(X, labels, k=10):
    Xn = normalize(X)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, -np.inf)
    topk = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    lab = np.asarray(labels)
    return float(np.mean((lab[topk] == lab[:, None]).sum(axis=1) / k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 11],
                    help="best layer first, then the final layer used in the paper")
    ap.add_argument("--n-database", type=int, default=3000)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--bits", type=int, default=1)
    args = ap.parse_args()

    model, device, sr = load_model(args.model)
    db_files = sorted(MS_SNSD.glob("clean_train/*.wav"))[: args.n_database]
    q_files = sorted(MS_SNSD.glob("clean_test/*.wav"))[: args.n_queries]
    spk_files = sorted(VBD_CLEAN.glob("*.wav"))
    import re
    spk_labels = [re.match(r"(p\d+)", f.stem).group(1) for f in spk_files]

    with open(NOISY_MANIFEST, newline="", encoding="utf-8") as fh:
        nrows = [r for r in csv.DictReader(fh) if Path(r["noisy"]).exists()]
    noisy_files = [Path(r["noisy"]) for r in nrows]
    noisy_labels = [r.get("noise_type", "unknown") for r in nrows]

    print(f"{args.model}: comparing layers {args.layers} at {args.bits} bit/coord")
    print(f"  {len(db_files)} db, {len(q_files)} queries, {len(spk_files)} speaker clips, "
          f"{len(noisy_files)} noisy clips\n")

    results = {}
    for layer in args.layers:
        print(f"--- layer {layer} ---")
        DB = embed_layer(db_files, model, device, sr, layer)
        QY = embed_layer(q_files, model, device, sr, layer)
        SPK = embed_layer(spk_files, model, device, sr, layer)
        NOI = embed_layer(noisy_files, model, device, sr, layer)
        d = DB.shape[1]

        DBn, Qn = normalize(DB), normalize(QY)
        GT = sorted_topk(Qn, DBn, 10, metric="ip")
        q = RotatedScalarQuantizer(d=d, bits=args.bits, rotate=True, rotation="gaussian", seed=0)
        rec = recall_at_k(sorted_topk(Qn, q.decode(*q.encode(DBn)), 10, metric="ip"), GT, 10)

        row = {"layer": layer, "recall_kept": rec}
        for name, X, lab in [("speaker", SPK, spk_labels), ("noise_type", NOI, noisy_labels)]:
            f32 = precision_at_k(X, lab, 10)
            qq = RotatedScalarQuantizer(d=d, bits=args.bits, rotate=True,
                                        rotation="gaussian", seed=0)
            Xq = qq.decode(*qq.encode(normalize(X)))
            pq = precision_at_k(Xq, lab, 10)
            row[f"{name}_f32"] = f32
            row[f"{name}_quant"] = pq
            row[f"{name}_kept"] = pq / f32 if f32 else float("nan")
        results[layer] = row
        print(f"  Recall@10 kept        : {row['recall_kept']:.1%}")
        print(f"  speaker P@10 kept     : {row['speaker_kept']:.1%} "
              f"({row['speaker_f32']:.4f} -> {row['speaker_quant']:.4f})")
        print(f"  noise-type P@10 kept  : {row['noise_type_kept']:.1%} "
              f"({row['noise_type_f32']:.4f} -> {row['noise_type_quant']:.4f})\n")

    print("=" * 78)
    print("DOES THE GEOMETRY-vs-MEANING DISSOCIATION (F7) SURVIVE ON A GOOD LAYER?")
    print("=" * 78)
    print(f"{'layer':>6s} {'Recall@10 kept':>15s} {'speaker kept':>14s} {'noise kept':>12s}")
    for layer in args.layers:
        r = results[layer]
        print(f"{layer:6d} {r['recall_kept']:14.1%} {r['speaker_kept']:13.1%} "
              f"{r['noise_type_kept']:11.1%}")

    ok = all(results[l]["speaker_kept"] > 0.85 and results[l]["noise_type_kept"] > 0.85
             and results[l]["recall_kept"] < results[l]["speaker_kept"]
             for l in args.layers)
    print(f"\n  => dissociation (task retention >> recall retention) holds on ALL tested layers: "
          f"{'YES' if ok else 'NO'}")

    out = ROOT / "results" / "layers"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"best_layer_validation_{args.model}.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(results.values())).keys()))
        w.writeheader(); w.writerows(results.values())
    print(f"\nwrote {out / f'best_layer_validation_{args.model}.csv'}")


if __name__ == "__main__":
    main()
