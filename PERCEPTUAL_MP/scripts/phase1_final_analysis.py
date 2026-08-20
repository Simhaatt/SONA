"""Phase 1 final analysis: does perceptual (PESQ) sensitivity ranking diverge from
fidelity/MSE-family (SI-SDR, mask-MSE) sensitivity ranking across the 6 quantizable layers?

Combines:
  - PERCEPTUAL_MP/pesq_wavs/pesq_per_clip.csv   (PESQ/STOI/ESTOI, WSL-scored, 200 dev clips)
  - PERCEPTUAL_MP/results/phase0/0a_dev_deltas.csv (SI-SDR/STOI/ESTOI/LSD/mask-MSE, same 200 clips)

Same paired-per-clip bootstrap CI methodology as Phase 0/1a for consistency.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PESQ_CSV = HERE.parent / "pesq_wavs" / "pesq_per_clip.csv"
PHASE0_CSV = HERE.parent / "results" / "phase0" / "0a_dev_deltas.csv"
OUT_DIR = HERE.parent / "results" / "phase1"

LAYERS = ["only_stem", "only_block0_d1", "only_block1_d2", "only_block2_d4", "only_block3_d8", "only_head"]


def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=1234):
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def main() -> None:
    # ---- PESQ per-clip deltas (variant - fp32), paired by uid ----
    pesq_by_uid_method: dict[str, dict[str, float]] = defaultdict(dict)
    with open(PESQ_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["pesq"] not in (None, ""):
                pesq_by_uid_method[row["uid"]][row["method"]] = float(row["pesq"])

    pesq_deltas: dict[str, list[float]] = defaultdict(list)
    for uid, by_method in pesq_by_uid_method.items():
        if "fp32" not in by_method:
            continue
        base = by_method["fp32"]
        for layer in LAYERS:
            if layer in by_method:
                pesq_deltas[layer].append(by_method[layer] - base)

    print("=" * 78)
    print("PHASE 1 FINAL -- PESQ sensitivity (paired per-clip delta vs FP32, 95% bootstrap CI)")
    print("=" * 78)
    pesq_summary = {}
    for layer in LAYERS:
        mean, lo, hi = bootstrap_ci(pesq_deltas[layer])
        sig = bool(lo is not None and (lo > 0 or hi < 0))
        pesq_summary[layer] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "significant": sig,
                                "mean_abs": float(np.mean(np.abs(pesq_deltas[layer])))}
        flag = "*" if sig else " "
        print(f"  {layer:16s} dPESQ = {mean:+.5f} [{lo:+.5f}, {hi:+.5f}]{flag}  "
              f"mean|delta|={pesq_summary[layer]['mean_abs']:.5f}")
    print("  (* = 95% CI excludes zero)")

    # ---- Phase 0 deltas (SI-SDR, STOI, mask_mse), same 200 clips, already computed ----
    other_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(PHASE0_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = row["variant"]
            if v not in LAYERS:
                continue
            for key in ("d_si_sdr", "d_stoi", "mask_mse"):
                val = row.get(key)
                if val not in (None, ""):
                    other_deltas[v][key].append(float(val))

    # ---- Rankings: mean |per-clip delta| as the sensitivity score, consistent across metrics ----
    sensitivity = {}
    for layer in LAYERS:
        sensitivity[layer] = {
            "pesq": pesq_summary[layer]["mean_abs"],
            "si_sdr": float(np.mean(np.abs(other_deltas[layer]["d_si_sdr"]))),
            "stoi": float(np.mean(np.abs(other_deltas[layer]["d_stoi"]))),
            "mask_mse": float(np.mean(other_deltas[layer]["mask_mse"])),  # already >=0
        }

    metrics = ["pesq", "si_sdr", "stoi", "mask_mse"]
    ranks = {}
    for m in metrics:
        vals = sorted(((layer, sensitivity[layer][m]) for layer in LAYERS), key=lambda x: -x[1])
        ranks[m] = {layer: i + 1 for i, (layer, _) in enumerate(vals)}

    print("\nSensitivity ranking (1 = most sensitive to quantizing that layer):")
    header = f"{'layer':16s}" + "".join(f"{m:>10s}" for m in metrics)
    print(header)
    for layer in LAYERS:
        row = f"{layer:16s}"
        for m in metrics:
            row += f"{ranks[m][layer]:10d}"
        print(row)

    print("\nSpearman rank correlation (the decisive test):")
    corr = {}
    pairs = [("pesq", "si_sdr"), ("pesq", "mask_mse"), ("pesq", "stoi"),
             ("si_sdr", "mask_mse"), ("si_sdr", "stoi"), ("stoi", "mask_mse")]
    for a, b in pairs:
        r1 = [ranks[a][layer] for layer in LAYERS]
        r2 = [ranks[b][layer] for layer in LAYERS]
        rho, p = spearmanr(r1, r2)
        corr[f"{a}_vs_{b}"] = {"rho": float(rho), "p": float(p)}
        verdict = "DIVERGE" if abs(rho) < 0.6 else "AGREE"
        print(f"  {a:10s} vs {b:10s}  rho={rho:+.3f}  p={p:.3f}  -> {verdict}")

    # ---- The headline finding: PESQ vs SI-SDR ranking, with the specific divergent layer named ----
    pesq_rank = ranks["pesq"]
    sisdr_rank = ranks["si_sdr"]
    biggest_flip, biggest_flip_delta = None, 0
    for layer in LAYERS:
        d = abs(pesq_rank[layer] - sisdr_rank[layer])
        if d > biggest_flip_delta:
            biggest_flip_delta, biggest_flip = d, layer

    print("\n" + "=" * 78)
    print("HEADLINE")
    print("=" * 78)
    rho_main = corr["pesq_vs_si_sdr"]["rho"]
    print(f"PESQ vs SI-SDR sensitivity ranking: rho={rho_main:+.3f} (n=6 layers, indicative only)")
    print(f"Largest single-layer rank disagreement: '{biggest_flip}' "
          f"(PESQ rank {pesq_rank[biggest_flip]}, SI-SDR rank {sisdr_rank[biggest_flip]})")
    print(f"  {biggest_flip}: dPESQ mean|delta|={sensitivity[biggest_flip]['pesq']:.5f} "
          f"(significant={pesq_summary[biggest_flip]['significant']}), "
          f"dSI-SDR mean|delta|={sensitivity[biggest_flip]['si_sdr']:.5f}")
    if abs(rho_main) < 0.6:
        print("\n=> The perceptual and fidelity rankings DIVERGE at the aggregate level.")
        print("   This supports the core hypothesis: MSE-guided allocation would misprioritize")
        print(f"   layers relative to what actually protects PESQ (notably '{biggest_flip}').")
    else:
        print("\n=> The perceptual and fidelity rankings largely AGREE at n=6 layers.")
        print("   This is the null result -- report it honestly, it does not support allocating")
        print("   bits differently from an MSE-guided baseline at this granularity.")

    out = {
        "pesq_per_layer": pesq_summary,
        "sensitivity": sensitivity,
        "rankings": ranks,
        "spearman": corr,
        "biggest_rank_disagreement": {"layer": biggest_flip, "rank_delta": biggest_flip_delta},
        "note": "n=6 layers -> Spearman here is indicative, not a well-powered hypothesis test on "
                "its own. The per-layer PESQ bootstrap CIs (from 200 paired dev clips) are the "
                "primary quantitative evidence for whether an individual layer's perceptual effect "
                "is distinguishable from zero.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "phase1_final_analysis.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT_DIR / 'phase1_final_analysis.json'}")


if __name__ == "__main__":
    main()
