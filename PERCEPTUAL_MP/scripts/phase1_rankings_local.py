"""Phase 1, part A: layer sensitivity rankings from data already collected in Phase 0.

No new inference. Reuses PERCEPTUAL_MP/results/phase0/0a_dev_deltas.csv (200 dev clips x 6
quantizable single-layer variants) to rank layers by |delta| under each metric, then compares
rankings with Spearman correlation. This is the cheap half of the hypothesis test; the PESQ half
(phase1_pesq_wsl.py) requires WSL and real audio.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DELTAS_CSV = HERE.parent / "results" / "phase0" / "0a_dev_deltas.csv"
OUT_DIR = HERE.parent / "results" / "phase1"

LAYERS = ["only_stem", "only_block0_d1", "only_block1_d2", "only_block2_d4", "only_block3_d8", "only_head"]
METRIC_ABS_SENSITIVITY = ["d_si_sdr", "d_stoi", "d_estoi", "d_lsd", "mask_mse"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_layer_metric: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with open(DELTAS_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            v = row["variant"]
            if v not in LAYERS:
                continue
            for m in METRIC_ABS_SENSITIVITY:
                val = row.get(m)
                if val not in (None, ""):
                    by_layer_metric[v][m].append(float(val))

    # Sensitivity score per layer per metric = mean absolute per-clip delta (magnitude of
    # disruption, not direction -- a layer that swings clips both ways is still "sensitive").
    sensitivity = {}
    for layer in LAYERS:
        sensitivity[layer] = {
            m: float(np.mean(np.abs(by_layer_metric[layer][m])))
            for m in METRIC_ABS_SENSITIVITY
            if by_layer_metric[layer][m]
        }

    print("=" * 78)
    print("PHASE 1a -- layer sensitivity from Phase 0 data (mean |per-clip delta|)")
    print("=" * 78)
    header = f"{'layer':16s}" + "".join(f"{m:>14s}" for m in METRIC_ABS_SENSITIVITY)
    print(header)
    for layer in LAYERS:
        row = f"{layer:16s}"
        for m in METRIC_ABS_SENSITIVITY:
            row += f"{sensitivity[layer].get(m, float('nan')):14.6f}"
        print(row)

    # Rank layers (1 = most sensitive) under each metric.
    ranks = {}
    for m in METRIC_ABS_SENSITIVITY:
        vals = [(layer, sensitivity[layer].get(m, 0.0)) for layer in LAYERS]
        vals.sort(key=lambda x: -x[1])
        ranks[m] = {layer: i + 1 for i, (layer, _) in enumerate(vals)}

    print("\nRanking (1 = most sensitive to quantizing that layer):")
    header = f"{'layer':16s}" + "".join(f"{m:>14s}" for m in METRIC_ABS_SENSITIVITY)
    print(header)
    for layer in LAYERS:
        row = f"{layer:16s}"
        for m in METRIC_ABS_SENSITIVITY:
            row += f"{ranks[m][layer]:14d}"
        print(row)

    # Pairwise Spearman correlation between metric rankings -- the core test.
    print("\nSpearman rank correlation between metrics' sensitivity orderings:")
    corr_results = {}
    for i, m1 in enumerate(METRIC_ABS_SENSITIVITY):
        for m2 in METRIC_ABS_SENSITIVITY[i + 1 :]:
            r1 = [ranks[m1][layer] for layer in LAYERS]
            r2 = [ranks[m2][layer] for layer in LAYERS]
            rho, p = spearmanr(r1, r2)
            corr_results[f"{m1}_vs_{m2}"] = {"rho": float(rho), "p": float(p)}
            print(f"  {m1:10s} vs {m2:10s}  rho={rho:+.3f}  p={p:.3f}")

    # The single most interesting number for the hypothesis: does the perceptual-ish metric
    # (STOI/ESTOI) rank layers differently from the fidelity metric (SI-SDR) or from raw
    # weight/mask distortion (mask_mse, the MSE-family proxy)?
    key_pairs = [("d_stoi", "d_si_sdr"), ("d_estoi", "d_si_sdr"), ("d_stoi", "mask_mse"),
                 ("d_estoi", "mask_mse")]
    print("\nKey comparisons for the hypothesis (perceptual-ish vs MSE-family):")
    for a, b in key_pairs:
        key = f"{a}_vs_{b}" if f"{a}_vs_{b}" in corr_results else f"{b}_vs_{a}"
        r = corr_results.get(key)
        if r:
            verdict = "DIVERGE" if abs(r["rho"]) < 0.6 else "AGREE"
            print(f"  {a:10s} vs {b:10s}  rho={r['rho']:+.3f}  -> {verdict}")

    out = {
        "sensitivity_mean_abs_delta": sensitivity,
        "rankings": ranks,
        "spearman": corr_results,
        "note": "n=6 layers -> Spearman/p-values here are indicative, not well-powered on their "
                "own; the per-clip bootstrap CIs from Phase 0 are the primary evidence for which "
                "individual layers have a real (non-zero) effect. STOI showed weaker sensitivity "
                "than SI-SDR at INT8 in Phase 0 -- PESQ (phase1_pesq_wsl.py) is the more decisive "
                "perceptual metric and should be weighted more heavily than this STOI-only pass.",
    }
    with open(OUT_DIR / "phase1a_local_rankings.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {OUT_DIR / 'phase1a_local_rankings.json'}")


if __name__ == "__main__":
    main()
