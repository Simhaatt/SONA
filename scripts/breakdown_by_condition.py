"""Per-noise-type and per-SNR breakdowns of any per-clip results file.

Joins a per-clip metrics CSV (uid,method,<metrics...>) against condition labels:
  * VoiceBank-DEMAND: results/vbd_test_conditions.csv (from extract_vbd_conditions.py)
  * MS-SNSD: noise_type/snr_db columns already present in the per-clip file

Emits mean +/- 95% bootstrap CI per (method, condition) cell, plus a markdown table per metric.
Reviewers routinely ask whether gains are uniform across noise types and SNRs, or driven by a
few easy conditions -- this answers that without any new inference.

Usage:
    python scripts/breakdown_by_condition.py --per-clip results/wer_full824/per_sample_wer.csv \\
        --metric wer --out results/breakdowns/wer
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VBD_CONDITIONS = ROOT / "results" / "vbd_test_conditions.csv"


def bootstrap_ci(values, n_boot=5000, alpha=0.05, seed=1234):
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def load_conditions() -> dict[str, dict]:
    if not VBD_CONDITIONS.exists():
        return {}
    with open(VBD_CONDITIONS, newline="", encoding="utf-8") as fh:
        return {r["uid"]: {"noise_type": r["noise_type"], "snr_db": float(r["snr_db"])}
                for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-clip", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lower-is-better", action="store_true",
                    help="set for wer/lsd so the best cell is marked correctly")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conditions = load_conditions()

    with open(args.per_clip, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {args.per_clip}")

    # attach conditions: prefer in-file columns (MS-SNSD), else join VBD labels by uid
    enriched = []
    for r in rows:
        val = r.get(args.metric)
        if val in (None, "", "None"):
            continue
        nt = r.get("noise_type")
        snr = r.get("snr_db")
        if nt in (None, "", "unknown") or snr in (None, ""):
            c = conditions.get(r["uid"])
            if c:
                nt, snr = c["noise_type"], c["snr_db"]
        if nt in (None, "", "unknown") or snr in (None, ""):
            continue
        enriched.append({"method": r.get("method", "?"), "uid": r["uid"],
                          "noise_type": nt, "snr_db": float(snr), "value": float(val)})

    if not enriched:
        raise SystemExit("no rows could be matched to condition labels")

    methods = sorted({r["method"] for r in enriched})
    noise_types = sorted({r["noise_type"] for r in enriched})
    snrs = sorted({r["snr_db"] for r in enriched})
    print(f"{len(enriched)} rows | methods={methods}")
    print(f"noise types={noise_types}\nSNRs={snrs}\n")

    results = {"metric": args.metric, "by_noise": {}, "by_snr": {}}

    def render(group_key: str, group_vals: list, label: str) -> str:
        buckets: dict[tuple, list[float]] = defaultdict(list)
        for r in enriched:
            buckets[(r["method"], r[group_key])].append(r["value"])

        lines = [f"| method | " + " | ".join(str(g) for g in group_vals) + " |",
                 "|" + "---|" * (len(group_vals) + 1)]
        store = {}
        for m in methods:
            cells = []
            for g in group_vals:
                mean, lo, hi = bootstrap_ci(buckets[(m, g)])
                store[f"{m}|{g}"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi,
                                      "n": len(buckets[(m, g)])}
                cells.append("-" if mean is None else f"{mean:.3f}")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        results[label] = store
        return "\n".join(lines)

    md_noise = render("noise_type", noise_types, "by_noise")
    md_snr = render("snr_db", snrs, "by_snr")

    print(f"=== {args.metric} by NOISE TYPE ===")
    print(md_noise)
    print(f"\n=== {args.metric} by SNR (dB) ===")
    print(md_snr)

    (out / f"{args.metric}_by_noise.md").write_text(md_noise, encoding="utf-8")
    (out / f"{args.metric}_by_snr.md").write_text(md_snr, encoding="utf-8")
    with open(out / f"{args.metric}_breakdown.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # consistency check: is the ordering of methods stable across every condition?
    if len(methods) > 1:
        print("\n=== consistency of method ranking across conditions ===")
        for label, group_vals, gk in [("noise", noise_types, "noise_type"), ("SNR", snrs, "snr_db")]:
            orders = []
            for g in group_vals:
                vals = [(m, results[f"by_{label.lower() if label=='noise' else 'snr'}"]
                         .get(f"{m}|{g}", {}).get("mean")) for m in methods]
                vals = [(m, v) for m, v in vals if v is not None]
                vals.sort(key=lambda x: x[1] if args.lower_is_better else -x[1])
                orders.append(tuple(m for m, _ in vals))
            stable = len(set(orders)) == 1
            print(f"  across {label}: {'STABLE (same ranking everywhere)' if stable else 'VARIES by condition'}")
            if not stable:
                for g, o in zip(group_vals, orders):
                    print(f"    {g}: {' > '.join(o)}")

    print(f"\nwrote breakdowns to {out}")


if __name__ == "__main__":
    main()
