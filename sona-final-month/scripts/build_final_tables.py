#!/usr/bin/env python3
"""Aggregate every results/**/summary.csv into one master table plus the
curated tables the thesis / paper / viva actually cite.

Usage:
    python scripts/build_final_tables.py \
        --results-dir "../results" \
        --out-dir     "FINAL_TABLES"

Outputs (csv + markdown):
    master_all_runs.{csv,md}      every method row from every run
    voicebank_full824.{csv,md}    backbone search on the full VoiceBank test set
    ms_snsd_external.{csv,md}     external generalisation (MS-SNSD, 1100 clips)
    online_adaptation_story.{csv,md}  frozen vs test-time-adaptation, both datasets
    classical_synthetic.{csv,md}  classical DSP baselines (synthetic dev set)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

METRICS = ["si_sdr_i", "si_sdr", "snr_i", "snr", "pesq", "stoi", "estoi", "lsd", "rtf"]


def default_results_dir() -> Path:
    # scripts/ -> sona-final-month/ -> FYP/  then  FYP/noise cancellation/results
    return Path(__file__).resolve().parents[2] / "results"


def run_name_for(summary_path: Path, results_dir: Path) -> str:
    """Human-readable run id derived from the summary.csv location."""
    rel = summary_path.parent.relative_to(results_dir)
    parts = [p for p in rel.parts if p != "extracted"]
    return "/".join(parts) if parts else "results_root"


def load_master(results_dir: Path) -> pd.DataFrame:
    rows = []
    for sp in sorted(results_dir.rglob("summary.csv")):
        try:
            df = pd.read_csv(sp)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! skipping {sp}: {exc}", file=sys.stderr)
            continue
        if "method" not in df.columns:
            continue
        df.insert(0, "run", run_name_for(sp, results_dir))
        rows.append(df)
    if not rows:
        raise SystemExit(f"No summary.csv found under {results_dir}")
    master = pd.concat(rows, ignore_index=True)
    for m in METRICS + ["n"]:
        if m in master.columns:
            master[m] = pd.to_numeric(master[m], errors="coerce")
    return master


def pick(master: pd.DataFrame, run_substr: str, method: str):
    sel = master[(master["run"].str.contains(run_substr, regex=False))
                 & (master["method"] == method)]
    if sel.empty:
        print(f"  ! not found: run~'{run_substr}' method='{method}'", file=sys.stderr)
        return None
    return sel.iloc[0]


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return "" if pd.isna(v) else str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(fmt(v) for v in row) + " |"
                     for row in df.itertuples(index=False))
    return "\n".join([head, sep, body]) + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, name: str, title: str) -> None:
    df.to_csv(out_dir / f"{name}.csv", index=False)
    md = f"# {title}\n\n" + to_markdown(df)
    (out_dir / f"{name}.md").write_text(md, encoding="utf-8")
    print(f"  wrote {name}.csv / {name}.md ({len(df)} rows)")


def curated_table(master, spec, keep):
    out = []
    for label, run_substr, method in spec:
        s = pick(master, run_substr, method)
        if s is None:
            continue
        row = {"label": label, "method": s["method"]}
        row.update({k: s.get(k) for k in keep})
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=default_results_dir())
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "FINAL_TABLES")
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"results-dir: {results_dir}")
    print(f"out-dir:     {out_dir}")

    master = load_master(results_dir)
    write_table(master[["run", "method"] + [m for m in METRICS if m in master] + ["n"]],
                out_dir, "master_all_runs", "SONA — all runs (master table)")

    keep = ["si_sdr_i", "pesq", "stoi", "estoi", "lsd", "rtf"]

    vb = curated_table(master, [
        ("Attention (baseline)", "strict_guard_full824", "neural_attention"),
        ("GTCRN + MR loss (WINNER)", "gtcrn_loss_focus_full824_rerun",
         "neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
        ("Mamba-GTCRN", "mamba_gtcrn_full824",
         "neural_mamba_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
        ("SONA-G governed", "governed_gtcrn_full824",
         "neural_governed_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
        ("SONA-R residual", "residual_gtcrn_full824",
         "neural_residual_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
    ], keep)
    write_table(vb, out_dir, "voicebank_full824",
                "VoiceBank-DEMAND full 824-clip test set")

    ext = curated_table(master, [
        ("Attention", "ms_snsd_external_2026-07-04_recovered", "neural_attention"),
        ("GTCRN + MR loss", "ms_snsd_gtcrn_external_eval",
         "neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25"),
    ], keep)
    write_table(ext, out_dir, "ms_snsd_external",
                "MS-SNSD external validation (1100 clips, unseen dataset)")

    keep2 = ["pesq", "si_sdr_i", "stoi", "estoi", "lsd"]
    story = curated_table(master, [
        ("VoiceBank frozen", "strict_guard_full824", "neural_attention"),
        ("VoiceBank strict-guard online", "strict_guard_full824",
         "attention_online_lowrank_s1_a2_d0p01_go0p0001"),
        ("MS-SNSD frozen", "ms_snsd_external_2026-07-04_recovered", "neural_attention"),
        ("MS-SNSD unguarded online d0.01", "ms_snsd_external_2026-07-04_recovered",
         "attention_online_lowrank_s1_a2_d0p01"),
        ("MS-SNSD unguarded online d0.02", "ms_snsd_external_2026-07-04_recovered",
         "attention_online_lowrank_s1_a2_d0p02"),
    ], keep2)
    write_table(story, out_dir, "online_adaptation_story",
                "Test-time adaptation: frozen vs online (VoiceBank vs MS-SNSD)")

    classical = master[master["run"] == "results_root"][
        ["method"] + keep].reset_index(drop=True)
    if not classical.empty:
        write_table(classical, out_dir, "classical_synthetic",
                    "Classical DSP baselines (synthetic dev set, illustrative)")

    print("done.")


if __name__ == "__main__":
    main()
