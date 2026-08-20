#!/usr/bin/env python3
"""Paired significance tests between two methods on per-clip metrics.

Aligns two per_sample.csv method groups on `uid`, then for each metric reports
the paired mean difference (A - B), a 95% bootstrap confidence interval, and a
two-sided Wilcoxon signed-rank p-value. Works whether the two method rows live
in the same per_sample.csv or in two different files.

Single comparison:
    python scripts/significance.py \
        --a-file "<runA>/per_sample.csv" --a-method neural_gtcrn_... \
        --b-file "<runB>/per_sample.csv" --b-method neural_attention \
        --label "GTCRN vs Attention (VoiceBank)" --out-dir SIGNIFICANCE

Batch (uses the real run paths; pass --preset all):
    python scripts/significance.py --preset all \
        --results-dir "../results"
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS_DEFAULT = ["si_sdr_i", "pesq", "stoi", "estoi", "lsd"]
LOWER_IS_BETTER = {"lsd"}


def load_group(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    g = df[df["method"] == method].copy()
    if g.empty:
        raise SystemExit(f"method '{method}' not found in {path}")
    if "uid" not in g.columns:
        raise SystemExit(f"no 'uid' column in {path}")
    return g.set_index("uid")


def bootstrap_ci(diff: np.ndarray, n_boot: int, seed: int = 0):
    if len(diff) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def compare(a_path, a_method, b_path, b_method, label, metrics, n_boot):
    A = load_group(Path(a_path), a_method)
    B = load_group(Path(b_path), b_method)
    common = A.index.intersection(B.index)
    rows = []
    for m in metrics:
        if m not in A.columns or m not in B.columns:
            continue
        a = pd.to_numeric(A.loc[common, m], errors="coerce")
        b = pd.to_numeric(B.loc[common, m], errors="coerce")
        mask = a.notna() & b.notna()
        a, b = a[mask].to_numpy(), b[mask].to_numpy()
        diff = a - b
        if len(diff) == 0:
            continue
        try:
            _, p = wilcoxon(a, b) if np.any(diff != 0) else (np.nan, float("nan"))
        except ValueError:
            p = float("nan")
        lo, hi = bootstrap_ci(diff, n_boot)
        better = "lower" if m in LOWER_IS_BETTER else "higher"
        rows.append({
            "comparison": label, "metric": m, "better_is": better,
            "n_pairs": len(diff),
            "mean_A": round(float(a.mean()), 6),
            "mean_B": round(float(b.mean()), 6),
            "mean_diff_A_minus_B": round(float(diff.mean()), 6),
            "ci95_low": round(lo, 6), "ci95_high": round(hi, 6),
            "wilcoxon_p": p if np.isnan(p) else round(float(p), 6),
            "significant_0.05": bool(p < 0.05) if not np.isnan(p) else False,
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body]) + "\n"


def presets(results_dir: Path):
    R = results_dir
    vb_gtcrn = R / "kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/extracted/per_sample.csv"
    vb_attn = R / "kaggle_voicebank_strict_guard_full824_2026-07-05/extracted/per_sample.csv"
    vb_mamba = R / "kaggle_voicebank_mamba_gtcrn_full824_2026-07-07/extracted/per_sample.csv"
    vb_gov = R / "kaggle_voicebank_governed_gtcrn_full824_2026-07-07/per_sample.csv"
    vb_res = R / "kaggle_voicebank_residual_gtcrn_full824_2026-07-07/per_sample.csv"
    G = "neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25"
    return [
        ("GTCRN vs Attention (VoiceBank)", vb_gtcrn, G, vb_attn, "neural_attention"),
        ("GTCRN vs Mamba-GTCRN (VoiceBank)", vb_gtcrn, G, vb_mamba,
         "neural_mamba_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
        ("GTCRN vs SONA-G governed (VoiceBank)", vb_gtcrn, G, vb_gov,
         "neural_governed_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
        ("GTCRN vs SONA-R residual (VoiceBank)", vb_gtcrn, G, vb_res,
         "neural_residual_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
        ("Strict-guard online vs frozen (VoiceBank)", vb_attn,
         "attention_online_lowrank_s1_a2_d0p01_go0p0001", vb_attn, "neural_attention"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-file"); ap.add_argument("--a-method")
    ap.add_argument("--b-file"); ap.add_argument("--b-method")
    ap.add_argument("--label", default="A vs B")
    ap.add_argument("--metrics", default=",".join(METRICS_DEFAULT))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--preset", choices=["all"], default=None)
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "results")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "SIGNIFICANCE")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.preset == "all":
        for label, af, am, bf, bm in presets(args.results_dir.resolve()):
            if not Path(af).exists() or not Path(bf).exists():
                print(f"  ! skip '{label}': missing per_sample.csv")
                continue
            frames.append(compare(af, am, bf, bm, label, metrics, args.n_boot))
    else:
        for req in ("a_file", "a_method", "b_file", "b_method"):
            if getattr(args, req) is None:
                raise SystemExit(f"--{req.replace('_','-')} is required without --preset")
        frames.append(compare(args.a_file, args.a_method, args.b_file,
                              args.b_method, args.label, metrics, args.n_boot))

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        if args.preset:
            print("  (no preset comparisons available — per_sample.csv files not found)")
            return
        raise SystemExit("no comparisons produced")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(args.out_dir / "significance.csv", index=False)
    (args.out_dir / "significance.md").write_text(
        "# SONA — paired significance tests\n\n"
        "Mean difference is A - B on clips shared by both methods. "
        "Bootstrap 95% CI (10k resamples) and two-sided Wilcoxon signed-rank.\n\n"
        + to_markdown(out), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nwrote {args.out_dir/'significance.csv'} and .md")


if __name__ == "__main__":
    main()
