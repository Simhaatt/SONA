"""R3 -- the dither diagnostic, formalised.

The problem it solves. On MS-SNSD, uniform INT8 scores *better* than the stem-exempt mixed
allocation on STOI, ESTOI and LSD. Read naively that says uniform is the better allocation. It is
not: uniform INT8 also beats **FP32 itself** on those metrics, and a lossy approximation of a model
cannot be better than the model it approximates. The extra quantization noise is acting as a mild
dither that flatters intelligibility metrics under distribution shift while degrading waveform
fidelity.

The diagnostic. Post-training quantization approximates a source model, so FP32 is a *ceiling*,
not a competitor. Any metric on which a quantized variant beats FP32 is, on that corpus, not
measuring approximation quality:

    if quantized_score is better than fp32_score  ->  the metric is compromised here;
                                                      rank on |deviation from FP32| instead

This script applies that test to every (corpus, method, metric) cell and separates the metrics
that behave (quantization only ever hurts) from the ones that do not. The output is the evidence
for reporting *fidelity to FP32* rather than absolute metric scores.

Usage:
    python PERCEPTUAL_MP/scripts/dither_diagnostic.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# metric -> True if higher is better
METRICS = {"si_sdr": True, "stoi": True, "estoi": True, "lsd": False}
CORPORA = {
    "voicebank_824": HERE.parent / "results" / "phase3" / "voicebank_test824_per_clip.csv",
    "ms_snsd_1100": HERE.parent / "results" / "phase3" / "ms_snsd_per_clip.csv",
}
REFERENCE = "fp32"
N_BOOT = 10000
SEED = 2026


def load(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """method -> uid -> metric -> value"""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            vals = {}
            for m in METRICS:
                try:
                    vals[m] = float(r[m])
                except (KeyError, ValueError):
                    pass
            out[r["method"]][r["uid"]] = vals
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "dither"))
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for corpus, path in CORPORA.items():
        data = load(path)
        if REFERENCE not in data:
            raise SystemExit(f"{corpus}: no `{REFERENCE}` rows in {path}")
        ref = data[REFERENCE]

        for method, per_uid in sorted(data.items()):
            if method == REFERENCE:
                continue
            uids = sorted(set(per_uid) & set(ref))
            for metric, higher_better in METRICS.items():
                a = np.array([per_uid[u][metric] for u in uids if metric in per_uid[u]])
                b = np.array([ref[u][metric] for u in uids if metric in ref[u]])
                if len(a) != len(b) or len(a) == 0:
                    continue

                diff = a - b                       # quantized minus FP32
                signed = diff if higher_better else -diff   # >0 means "beats FP32"
                mean_signed = float(np.mean(signed))
                idx = rng.integers(0, len(signed), size=(N_BOOT, len(signed)))
                boot = signed[idx].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                p = float(stats.wilcoxon(a, b).pvalue) if np.any(diff != 0) else float("nan")

                beats_fp32 = bool(lo > 0)          # significantly better than the source model
                rows.append({
                    "corpus": corpus, "method": method, "metric": metric,
                    "n": len(a),
                    "mean_quantized": float(np.mean(a)), "mean_fp32": float(np.mean(b)),
                    "mean_signed_advantage_over_fp32": mean_signed,
                    "ci95_low": float(lo), "ci95_high": float(hi),
                    "wilcoxon_p": p,
                    "beats_fp32": beats_fp32,
                    "metric_compromised": beats_fp32,
                    "mean_abs_deviation_from_fp32": float(np.mean(np.abs(diff))),
                })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "dither_diagnostic.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "dither_diagnostic.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ report
    compromised = [r for r in rows if r["metric_compromised"]]
    by_metric: dict[tuple[str, str], int] = defaultdict(int)
    for r in compromised:
        by_metric[(r["corpus"], r["metric"])] += 1

    lines = [
        "# Dither diagnostic", "",
        "Post-training quantization approximates a source model, so **FP32 is a ceiling, not a "
        "competitor**. Any metric on which a quantized variant significantly beats FP32 is not "
        "measuring approximation quality on that corpus -- the extra quantization noise is acting "
        "as a dither. On such a metric, rank by **|deviation from FP32|** instead of by the "
        "absolute score.", "",
        f"Advantage is signed so that **positive means the quantized model beat FP32**. "
        f"Bootstrap 95 % CI over {N_BOOT} resamples; `compromised` is flagged when the CI lies "
        "entirely above zero.", "",
        "| corpus | method | metric | quantized | FP32 | advantage | 95% CI | compromised |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['corpus']} | {r['method']} | {r['metric']} | {r['mean_quantized']:.4f} | "
            f"{r['mean_fp32']:.4f} | {r['mean_signed_advantage_over_fp32']:+.4f} | "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | "
            f"{'**YES**' if r['metric_compromised'] else 'no'} |"
        )
    lines += ["", "## Summary", ""]
    if compromised:
        lines.append("Metrics on which at least one quantized variant significantly beat FP32:")
        lines.append("")
        for (corpus, metric), n in sorted(by_metric.items()):
            lines.append(f"- **{corpus} / {metric}** — {n} variant(s)")
        lines += ["",
                  "On these cells, absolute scores cannot be used to rank allocations. This is "
                  "the evidence for the paper's claim of *fidelity to FP32* rather than "
                  "\"better on every metric everywhere\"."]
    else:
        lines.append("No metric was compromised: quantization only ever hurt. Absolute scores are "
                     "safe to rank on.")
    (out / "dither_diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out / 'dither_diagnostic.md'}\n")
    print(f"{len(compromised)}/{len(rows)} (corpus, method, metric) cells beat FP32:")
    for r in compromised:
        print(f"  {r['corpus']:14s} {r['method']:18s} {r['metric']:6s} "
              f"advantage={r['mean_signed_advantage_over_fp32']:+.4f} "
              f"CI=[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}]")


if __name__ == "__main__":
    main()
