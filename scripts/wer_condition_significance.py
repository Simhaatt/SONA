"""E3 -- paired significance for the WER condition split.

The aggregate WER table says enhancement slightly hurts (0.047 vs 0.043). The condition
breakdown says it cuts WER sharply in some noise types and raises it in others. That claim
currently rests on point estimates only, while every other leg of the paper carries paired
Wilcoxon tests and bootstrap CIs. This closes that gap.

Note on the join: results/wer_full824/per_sample_wer.csv carries a `noise_type` column, but it
is "unknown" on every row -- the condition join never landed in that file. Conditions come from
results/vbd_test_conditions.csv instead, keyed by uid (all 824 clips are covered).

Note on the test: per-clip WER is heavily zero-inflated (many clips transcribe perfectly under
both methods), so a large share of pairs are exact ties. Wilcoxon discards ties, so the effective
n is much smaller than the clip count; we report both so the power is visible rather than implied.

Usage:
    python scripts/wer_condition_significance.py
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

N_BOOT = 10000
SEED = 2026


def load_wer(path: Path) -> dict[tuple[str, str], float]:
    """(method, uid) -> wer"""
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[(r["method"], r["uid"])] = float(r["wer"])
    return out


def load_conditions(path: Path) -> dict[str, tuple[str, str]]:
    """uid -> (noise_type, snr_db)"""
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["uid"]] = (r["noise_type"], r["snr_db"])
    return out


def paired_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """a - b, with bootstrap CI on the mean difference and a Wilcoxon signed-rank p."""
    d = a - b
    n = len(d)
    n_tied = int(np.sum(d == 0))
    mean_diff = float(np.mean(d))

    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    if n_tied == n:
        p = float("nan")  # every pair identical; no test is defined
    else:
        p = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)

    return {
        "n_pairs": n,
        "n_tied": n_tied,
        "n_effective": n - n_tied,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": mean_diff,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "wilcoxon_p": p,
        "significant_0.05": bool(p == p and p < 0.05),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wer", default=str(ROOT / "results" / "wer_full824" / "per_sample_wer.csv"))
    ap.add_argument("--conditions", default=str(ROOT / "results" / "vbd_test_conditions.csv"))
    ap.add_argument("--baseline", default="passthrough")
    ap.add_argument("--methods", default="neural_gtcrn,neural_complex_gtcrn,"
                                         "spectral_subtraction,wiener")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "breakdowns" / "wer_significance"))
    args = ap.parse_args()

    wer = load_wer(Path(args.wer))
    cond = load_conditions(Path(args.conditions))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    base = args.baseline

    uids = sorted({u for (_, u) in wer})
    covered = [u for u in uids if u in cond]
    print(f"clips with WER: {len(uids)}   with condition labels: {len(covered)}")

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for method in methods:
        # group uids by condition, plus an "overall" bucket
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for u in covered:
            if (method, u) not in wer or (base, u) not in wer:
                continue
            noise, snr = cond[u]
            groups[("overall", "all")].append(u)
            groups[("noise", noise)].append(u)
            groups[("snr", snr)].append(u)

        for (kind, label), us in sorted(groups.items()):
            a = np.array([wer[(method, u)] for u in us], dtype=float)
            b = np.array([wer[(base, u)] for u in us], dtype=float)
            st = paired_stats(a, b, rng)
            st.update({"method": method, "baseline": base, "split": kind, "condition": label})
            # relative change is what the paper quotes ("41 % cut in bus")
            st["relative_pct"] = (
                100.0 * st["mean_diff"] / st["mean_b"] if st["mean_b"] > 0 else float("nan")
            )
            rows.append(st)

    # ---- Benjamini-Hochberg FDR within each (method, split) family -------------------
    # Corrected across the tests a reader interprets together: the 5 noise types for one
    # method, or its 4 SNR bins. "overall" is a single test and needs no correction.
    fams: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        fams[(r["method"], r["split"])].append(r)
    for (_, split), fam in fams.items():
        tested = [r for r in fam if r["wilcoxon_p"] == r["wilcoxon_p"]]
        m = len(tested)
        for r in fam:
            r["bh_family_size"] = m
        if m <= 1:
            for r in tested:
                r["q_value"] = r["wilcoxon_p"]
                r["significant_fdr_0.05"] = bool(r["wilcoxon_p"] < 0.05)
            continue
        tested.sort(key=lambda r: r["wilcoxon_p"])
        # step-up: q_i = min over j>=i of (p_j * m / j), enforcing monotonicity
        running = 1.0
        for i in range(m, 0, -1):
            r = tested[i - 1]
            running = min(running, r["wilcoxon_p"] * m / i)
            r["q_value"] = float(running)
        for r in tested:
            r["significant_fdr_0.05"] = bool(r["q_value"] < 0.05)
    for r in rows:
        r.setdefault("q_value", float("nan"))
        r.setdefault("significant_fdr_0.05", False)
        r.setdefault("bh_family_size", 1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    field_order = ["method", "baseline", "split", "condition", "n_pairs", "n_tied", "n_effective",
                   "mean_a", "mean_b", "mean_diff", "relative_pct", "ci95_low", "ci95_high",
                   "wilcoxon_p", "significant_0.05", "bh_family_size", "q_value",
                   "significant_fdr_0.05"]
    with (out / "wer_significance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=field_order)
        w.writeheader()
        w.writerows(rows)

    lines = ["# WER condition split -- paired significance", "",
             f"Baseline: `{base}`. Mean difference is method minus baseline, so **negative is "
             f"better** (lower WER). Bootstrap 95 % CI over {N_BOOT} resamples; two-sided "
             f"Wilcoxon signed-rank.", "",
             "`n_eff` is the number of clips whose WER actually differs between the two methods "
             "-- per-clip WER is zero-inflated, so most pairs are exact ties and the effective "
             "sample is far smaller than 824.", "",
             "Multiplicity: Benjamini-Hochberg FDR is applied **within each (method, split) "
             "family** -- the tests a reader interprets together (the 5 noise types for one "
             "method, or its 4 SNR bins). `q` is the FDR-adjusted value; judge significance on "
             "`q`, not `p`.", "",
             "| method | split | condition | n_eff/n | mean WER | baseline | diff | rel % | 95% CI | p | q | sig(FDR) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["wilcoxon_p"]
        p_s = "--" if p != p else (f"{p:.2e}" if p < 1e-3 else f"{p:.3f}")
        q = r["q_value"]
        q_s = "--" if q != q else (f"{q:.2e}" if q < 1e-3 else f"{q:.3f}")
        lines.append(
            f"| {r['method']} | {r['split']} | {r['condition']} | "
            f"{r['n_effective']}/{r['n_pairs']} | {r['mean_a']:.4f} | {r['mean_b']:.4f} | "
            f"{r['mean_diff']:+.4f} | {r['relative_pct']:+.1f} | "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {p_s} | {q_s} | "
            f"{'**yes**' if r['significant_fdr_0.05'] else 'no'} |"
        )
    (out / "wer_significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "wer_significance.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nwrote {out}")
    sig = [r for r in rows if r["significant_0.05"]]
    sig_q = [r for r in rows if r["significant_fdr_0.05"]]
    print(f"{len(sig)}/{len(rows)} significant at nominal p<0.05")
    print(f"{len(sig_q)}/{len(rows)} still significant after BH-FDR within family")
    for r in sig_q:
        print(f"    {r['method']:22s} {r['split']:8s} {r['condition']:8s} "
              f"diff={r['mean_diff']:+.4f} ({r['relative_pct']:+.1f}%) q={r['q_value']:.4f}")


if __name__ == "__main__":
    main()
