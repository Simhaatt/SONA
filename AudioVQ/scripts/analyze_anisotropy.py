"""RQ3 premise check: how anisotropic is the audio embedding space, and does the random rotation
actually fix it?

TurboQuant's analysis assumes that after a random rotation each coordinate is Beta-distributed
and coordinates are near-independent, which is what licenses per-coordinate scalar quantization.
That assumption is about the *rotated* vector, so it should hold regardless of input structure --
but only if the rotation genuinely isotropises the data at the dimensionality in question. This
script measures, before and after rotation:

  * eigenvalue spectrum of the covariance, and the share of variance in the top few directions
  * participation ratio (effective number of dimensions actually used)
  * mean pairwise cosine similarity -- the standard "anisotropy" measure for embedding spaces;
    near 0 is isotropic, large positive means all vectors point the same way
  * per-coordinate kurtosis and the max absolute correlation between distinct coordinates,
    which are exactly the quantities the independence assumption needs to be small

A comparison against an isotropic Gaussian reference of the same shape gives a calibrated sense of
how far audio embeddings actually are from the ideal case.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.embeddings import load_bank  # noqa: E402
from src.quantizers import RotatedScalarQuantizer  # noqa: E402


def stats(X: np.ndarray, label: str) -> dict:
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)

    # covariance spectrum
    cov = (Xc.T @ Xc) / max(n - 1, 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.maximum(ev, 0)
    tot = ev.sum() + 1e-12
    top1 = ev[0] / tot
    top10 = ev[:10].sum() / tot
    top50 = ev[:50].sum() / tot
    # participation ratio: (sum l)^2 / sum l^2 -- effective dimensionality
    pr = (ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12)

    # mean pairwise cosine on a subsample (the standard embedding-anisotropy metric)
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 1500), replace=False)
    U = X[idx] / (np.linalg.norm(X[idx], axis=1, keepdims=True) + 1e-12)
    G = U @ U.T
    iu = np.triu_indices(len(idx), k=1)
    mean_cos = float(G[iu].mean())

    # coordinate-level assumptions behind per-coordinate scalar quantization
    Un = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Uc = Un - Un.mean(axis=0, keepdims=True)
    sd = Uc.std(axis=0) + 1e-12
    kurt = float(np.mean(((Uc / sd) ** 4).mean(axis=0) - 3.0))  # excess kurtosis, 0 for Gaussian
    C = np.corrcoef(Uc, rowvar=False)
    np.fill_diagonal(C, 0.0)
    max_abs_corr = float(np.nanmax(np.abs(C)))
    mean_abs_corr = float(np.nanmean(np.abs(C)))

    out = {
        "label": label, "n": int(n), "d": int(d),
        "var_top1": float(top1), "var_top10": float(top10), "var_top50": float(top50),
        "participation_ratio": float(pr), "effective_dim_frac": float(pr / d),
        "mean_pairwise_cosine": mean_cos,
        "mean_excess_kurtosis": kurt,
        "max_abs_coord_corr": max_abs_corr, "mean_abs_coord_corr": mean_abs_corr,
    }
    return out


def show(s: dict) -> None:
    print(f"  {s['label']:28s} "
          f"top1={s['var_top1']*100:5.1f}% top10={s['var_top10']*100:5.1f}% "
          f"PR={s['participation_ratio']:7.1f} ({s['effective_dim_frac']*100:4.1f}% of d) "
          f"cos={s['mean_pairwise_cosine']:+.3f} kurt={s['mean_excess_kurtosis']:+.2f} "
          f"maxcorr={s['max_abs_coord_corr']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--bank", default="database")
    ap.add_argument("--out", default=str(ROOT / "results" / "anisotropy"))
    args = ap.parse_args()

    emb_dir = ROOT / "embeddings" / args.model
    X, _ = load_bank(emb_dir, args.bank)
    n, d = X.shape
    print(f"bank={args.bank} model={args.model} shape={X.shape}\n")

    results = {}
    print("BEFORE / AFTER random rotation, vs an isotropic Gaussian reference:")

    s_raw = stats(X, "audio embeddings (raw)")
    show(s_raw); results["raw"] = s_raw

    # apply the same rotation the quantizer uses
    for kind in ("hadamard", "gaussian"):
        q = RotatedScalarQuantizer(d=d, bits=8, rotate=True, rotation=kind, seed=0)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Xr = q._forward(Xn.astype(np.float64))
        s = stats(Xr, f"after {kind} rotation")
        show(s); results[f"rot_{kind}"] = s

    rng = np.random.default_rng(0)
    s_gauss = stats(rng.standard_normal((n, d)), "isotropic Gaussian ref")
    show(s_gauss); results["gaussian_reference"] = s_gauss

    # -------------------------------------------------- within-vector coordinate distribution
    # THIS is the assumption the quantizer actually relies on. TurboQuant's claim is about the
    # coordinates of a SINGLE rotated vector (they become Beta-distributed by symmetry of the
    # random rotation), not about the dataset's covariance. A shared per-coordinate codebook is
    # appropriate only if, within one vector, the coordinates are spread evenly rather than a few
    # spikes dominating. Crest factor (max|c| / rms(c)) and within-vector excess kurtosis measure
    # exactly that.
    def within_vector(Y: np.ndarray) -> dict:
        Y = np.asarray(Y, dtype=np.float64)
        rms = np.sqrt((Y ** 2).mean(axis=1)) + 1e-12
        crest = np.abs(Y).max(axis=1) / rms
        z = Y / rms[:, None]
        kurt = (z ** 4).mean(axis=1) - 3.0
        return {"crest_mean": float(crest.mean()), "crest_p95": float(np.percentile(crest, 95)),
                "within_kurtosis_mean": float(kurt.mean())}

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    wv_raw = within_vector(Xn)
    q_g = RotatedScalarQuantizer(d=d, bits=8, rotate=True, rotation="gaussian", seed=0)
    wv_rot = within_vector(q_g._forward(Xn.astype(np.float64)))
    rng2 = np.random.default_rng(1)
    G = rng2.standard_normal((min(n, 5000), d))
    wv_iso = within_vector(G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12))
    results["within_vector"] = {"raw": wv_raw, "rotated": wv_rot, "isotropic_ref": wv_iso}

    print("\nWITHIN-VECTOR coordinate spread (what per-coordinate scalar quantization assumes):")
    print(f"  {'raw embeddings':22s} crest={wv_raw['crest_mean']:6.2f} "
          f"(p95 {wv_raw['crest_p95']:6.2f})  excess kurtosis={wv_raw['within_kurtosis_mean']:+8.2f}")
    print(f"  {'after rotation':22s} crest={wv_rot['crest_mean']:6.2f} "
          f"(p95 {wv_rot['crest_p95']:6.2f})  excess kurtosis={wv_rot['within_kurtosis_mean']:+8.2f}")
    print(f"  {'isotropic reference':22s} crest={wv_iso['crest_mean']:6.2f} "
          f"(p95 {wv_iso['crest_p95']:6.2f})  excess kurtosis={wv_iso['within_kurtosis_mean']:+8.2f}")

    # ------------------------------------------------------------------ interpretation
    print("\nINTERPRETATION")
    aniso_raw = s_raw["mean_pairwise_cosine"]
    pr_raw, pr_g = s_raw["effective_dim_frac"], s_gauss["effective_dim_frac"]

    print("  (1) Rotation-invariance sanity check: mean pairwise cosine and participation ratio "
          "are\n      mathematically unchanged by an orthogonal transform, and measured "
          f"{aniso_raw:+.3f} -> {results['rot_gaussian']['mean_pairwise_cosine']:+.3f}. "
          "Rotation is exactly orthogonal, as required.")

    print(f"\n  (2) DATASET anisotropy: mean pairwise cosine {aniso_raw:+.3f} vs "
          f"{s_gauss['mean_pairwise_cosine']:+.3f} for isotropic;\n      effective dimensionality "
          f"{s_raw['participation_ratio']:.0f}/{d} ({pr_raw*100:.1f}%) vs {pr_g*100:.1f}%.")
    if aniso_raw > 0.3 and pr_raw < 0.25 * pr_g:
        print("      => The embedding space is FAR from the worst case TurboQuant is optimal for.\n"
              "         Data-DEPENDENT methods (PQ/OPQ) can exploit this low-rank structure;\n"
              "         a data-oblivious quantizer spends bits uniformly over directions the data\n"
              "         never uses. PREDICTION: PQ/OPQ should beat the data-oblivious scheme here,\n"
              "         and the gap should widen as the bit budget shrinks. Tested in RQ1.")

    print(f"\n  (3) WITHIN-VECTOR spread: crest factor {wv_raw['crest_mean']:.2f} -> "
          f"{wv_rot['crest_mean']:.2f} after rotation (isotropic {wv_iso['crest_mean']:.2f}).")
    if wv_rot["crest_mean"] < wv_raw["crest_mean"] * 0.95:
        print("      => Rotation DOES flatten the within-vector coordinate profile, which is the\n"
              "         assumption a single shared per-coordinate codebook actually relies on.\n"
              "         So the rotation is doing its intended job even though dataset-level\n"
              "         anisotropy is untouched (it cannot be, by orthogonality).")
    else:
        print("      => Rotation does NOT flatten the within-vector profile much; the shared\n"
              "         codebook assumption is weaker here than the theory presumes.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"anisotropy_{args.model}_{args.bank}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out / f'anisotropy_{args.model}_{args.bank}.json'}")


if __name__ == "__main__":
    main()
