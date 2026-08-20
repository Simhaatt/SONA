"""E1b -- does a geometric metric predict downstream task retention?

The paper's central claim is that Recall@k understates deployment quality under quantization. That
raises the obvious practitioner question: if not recall, then what -- short of running the
downstream task, which needs labels a deployment may not have?

This script tests two candidate *label-free* geometric metrics against measured semantic retention:

    Recall@10       the field default
    1/Ratio@10      the inverse approximation ratio proposed by Dimitropoulos & Mamoulis
                    (arXiv:2606.04522) for ANN search, computed here under quantization

Semantic retention is the measured ground truth: downstream speaker- and noise-type-retrieval
precision after quantization, divided by its float32 value.

A metric is useful here if it is *both*

    calibrated -- close in value to the retention it is predicting, and
    discriminative -- low when a quantizer is genuinely broken.

The second condition matters. A metric pinned near 1.0 everywhere would look calibrated on good
quantizers and be worthless in practice, so uniform 1-bit (which destroys retrieval) is the case
that decides it.

Usage:
    python scripts/metric_predicts_task.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("wavlm_base_plus", "wav2vec2_base", "hubert_base")

# semantic CSV method names -> ratio_at_k method names
ALIAS = {
    "scalar-1b+rotation": "scalar-1b-rot-gaussian",
    "scalar-2b+rotation": "scalar-2b-rot-gaussian",
    "scalar-3b+rotation": "scalar-3b-rot-gaussian",
    "scalar-4b+rotation": "scalar-4b-rot-gaussian",
    "scalar-1b no-rot": "scalar-1b-no-rot",
    "scalar-2b no-rot": "scalar-2b-no-rot",
    "scalar-3b no-rot": "scalar-3b-no-rot",
    "scalar-4b no-rot": "scalar-4b-no-rot",
    "uniform-1b": "uniform-1b",
    "uniform-2b": "uniform-2b",
    "uniform-3b": "uniform-3b",
    "uniform-4b": "uniform-4b",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", default=str(ROOT / "results" / "ratio_at_k" / "ratio_at_k.csv"))
    ap.add_argument("--semantic-dir", default=str(ROOT / "results" / "semantic"))
    ap.add_argument("--out", default=str(ROOT / "results" / "ratio_at_k"))
    args = ap.parse_args()

    geo: dict[tuple[str, str], dict] = {}
    for r in csv.DictReader(Path(args.ratio).open(newline="", encoding="utf-8")):
        if r.get("recall@10"):
            geo[(r["model"], r["method"])] = {
                "recall@10": float(r["recall@10"]),
                "inv_ratio@10": float(r["inv_ratio@10"]),
            }

    rows = []
    for model in MODELS:
        p = Path(args.semantic_dir) / f"semantic_{model}.csv"
        if not p.exists():
            print(f"skip {model}: no semantic file")
            continue
        # average retention over the tasks present (speaker, noise_type)
        ret: dict[str, list[float]] = defaultdict(list)
        for r in csv.DictReader(p.open(newline="", encoding="utf-8")):
            if not r.get("retained_frac"):
                continue
            ret[r["method"]].append(float(r["retained_frac"]))
        for sem_method, vals in ret.items():
            key = ALIAS.get(sem_method)
            if key is None or (model, key) not in geo:
                continue
            g = geo[(model, key)]
            rows.append({
                "model": model, "method": key,
                "semantic_retention": float(np.mean(vals)),
                "n_tasks": len(vals),
                "recall@10": g["recall@10"],
                "inv_ratio@10": g["inv_ratio@10"],
                "recall_error": g["recall@10"] - float(np.mean(vals)),
                "inv_ratio_error": g["inv_ratio@10"] - float(np.mean(vals)),
            })

    if not rows:
        raise SystemExit("no overlapping methods between semantic and ratio results")

    sem = np.array([r["semantic_retention"] for r in rows])
    rec = np.array([r["recall@10"] for r in rows])
    inv = np.array([r["inv_ratio@10"] for r in rows])

    stats_out = {
        "n_points": len(rows),
        "recall": {
            "mean_abs_error": float(np.mean(np.abs(rec - sem))),
            "pearson_r": float(stats.pearsonr(rec, sem)[0]),
            "spearman_rho": float(stats.spearmanr(rec, sem)[0]),
        },
        "inv_ratio": {
            "mean_abs_error": float(np.mean(np.abs(inv - sem))),
            "pearson_r": float(stats.pearsonr(inv, sem)[0]),
            "spearman_rho": float(stats.spearmanr(inv, sem)[0]),
        },
    }
    # discrimination check: the broken quantizers must score low
    broken = [r for r in rows if r["method"].startswith("uniform-1b")]
    stats_out["discrimination_uniform_1b"] = {
        r["model"]: {"recall@10": r["recall@10"], "inv_ratio@10": r["inv_ratio@10"],
                     "semantic_retention": r["semantic_retention"]} for r in broken}

    out = Path(args.out)
    with (out / "metric_predicts_task.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "metric_predicts_task.json").write_text(
        json.dumps({"rows": rows, "stats": stats_out}, indent=2), encoding="utf-8")

    lines = ["# Does a label-free geometric metric predict task retention?", "",
             "Semantic retention is the measured ground truth: downstream retrieval precision "
             "after quantization divided by its float32 value, averaged over the speaker and "
             "noise-type tasks. Both candidate metrics are purely geometric and need no labels.",
             "",
             "| model | method | semantic | Recall@10 | err | 1/Ratio@10 | err |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["model"], r["method"])):
        lines.append(f"| {r['model']} | {r['method']} | {r['semantic_retention']:.4f} | "
                     f"{r['recall@10']:.4f} | {r['recall_error']:+.4f} | "
                     f"{r['inv_ratio@10']:.4f} | {r['inv_ratio_error']:+.4f} |")
    lines += ["", "## Summary", "",
              "| metric | mean abs error vs semantic | Pearson r | Spearman rho |",
              "|---|---|---|---|",
              f"| Recall@10 | {stats_out['recall']['mean_abs_error']:.4f} | "
              f"{stats_out['recall']['pearson_r']:+.3f} | "
              f"{stats_out['recall']['spearman_rho']:+.3f} |",
              f"| **1/Ratio@10** | **{stats_out['inv_ratio']['mean_abs_error']:.4f}** | "
              f"{stats_out['inv_ratio']['pearson_r']:+.3f} | "
              f"{stats_out['inv_ratio']['spearman_rho']:+.3f} |",
              "",
              "## Discrimination check", "",
              "A metric pinned near 1.0 everywhere would be useless. Uniform 1-bit destroys "
              "retrieval and is the case that decides whether 1/Ratio@10 discriminates:", "",
              "| model | Recall@10 | 1/Ratio@10 | semantic |", "|---|---|---|---|"]
    for m, v in stats_out["discrimination_uniform_1b"].items():
        lines.append(f"| {m} | {v['recall@10']:.4f} | {v['inv_ratio@10']:.4f} | "
                     f"{v['semantic_retention']:.4f} |")
    (out / "metric_predicts_task.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"n = {len(rows)} (model, method) points\n")
    print(f"{'metric':14s} {'MAE vs semantic':>16s} {'Pearson':>9s} {'Spearman':>9s}")
    for k, lab in (("recall", "Recall@10"), ("inv_ratio", "1/Ratio@10")):
        s = stats_out[k]
        print(f"{lab:14s} {s['mean_abs_error']:16.4f} {s['pearson_r']:+9.3f} "
              f"{s['spearman_rho']:+9.3f}")
    print("\ndiscrimination on uniform-1b (must be low):")
    for m, v in stats_out["discrimination_uniform_1b"].items():
        print(f"  {m:16s} recall={v['recall@10']:.4f} inv_ratio={v['inv_ratio@10']:.4f} "
              f"semantic={v['semantic_retention']:.4f}")
    print(f"\nwrote {out / 'metric_predicts_task.md'}")


if __name__ == "__main__":
    main()
