"""E3b -- corpus-level WER with paired clip bootstrap.

Why this exists. The per-clip analysis (wer_condition_significance.py) averages a *rate* over
clips, and per-clip WER on this corpus is severely zero-inflated: utterances are short (median 7
reference words, 6183 words over 824 clips) and Whisper transcribes most of them perfectly under
both methods, so only ~129/824 pairs differ at all. Wilcoxon discards the ties, the effective
sample collapses to 16-37 clips inside a single noise condition, and the test ends up underpowered
against effects that are visibly large.

Corpus-level WER -- total errors over total reference words -- is the standard ASR metric and does
not throw away that information: a clip with 36 words counts twelve times as much as a clip with
3, which is exactly the weighting the aggregate is supposed to have.

    corpus_WER = sum_i(wer_i * words_i) / sum_i(words_i)

That is exact given the stored per-clip rates, so no error counts have to be reconstructed and no
rounding is involved. Uncertainty comes from a *paired* clip bootstrap: resample clip indices once
per replicate and score both methods on the same resampled clips, preserving the pairing.

Usage:
    python scripts/wer_corpus_level.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

N_BOOT = 10000
SEED = 2026


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wer", default=str(ROOT / "results" / "wer_full824" / "per_sample_wer.csv"))
    ap.add_argument("--transcripts",
                    default=str(ROOT / "results" / "wer_full824" / "clean_transcripts.csv"))
    ap.add_argument("--conditions", default=str(ROOT / "results" / "vbd_test_conditions.csv"))
    ap.add_argument("--baseline", default="passthrough")
    ap.add_argument("--methods", default="neural_gtcrn,neural_complex_gtcrn,"
                                         "spectral_subtraction,wiener")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "breakdowns" / "wer_significance"))
    args = ap.parse_args()

    words = {r["uid"]: len(r["text"].split())
             for r in csv.DictReader(Path(args.transcripts).open(newline="", encoding="utf-8"))}
    wer: dict[tuple[str, str], float] = {}
    for r in csv.DictReader(Path(args.wer).open(newline="", encoding="utf-8")):
        wer[(r["method"], r["uid"])] = float(r["wer"])
    cond = {r["uid"]: (r["noise_type"], r["snr_db"])
            for r in csv.DictReader(Path(args.conditions).open(newline="", encoding="utf-8"))}

    base = args.baseline
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for method in methods:
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for u, (noise, snr) in cond.items():
            if (method, u) not in wer or (base, u) not in wer or u not in words:
                continue
            groups[("overall", "all")].append(u)
            groups[("noise", noise)].append(u)
            groups[("snr", snr)].append(u)

        for (kind, label), us in sorted(groups.items()):
            w = np.array([words[u] for u in us], dtype=float)
            ea = np.array([wer[(method, u)] for u in us], dtype=float) * w   # errors, method
            eb = np.array([wer[(base, u)] for u in us], dtype=float) * w     # errors, baseline

            corpus_a = ea.sum() / w.sum()
            corpus_b = eb.sum() / w.sum()
            diff = corpus_a - corpus_b

            # paired clip bootstrap: one index draw per replicate, applied to both arms
            idx = rng.integers(0, len(us), size=(N_BOOT, len(us)))
            wb = w[idx].sum(axis=1)
            boot_a = ea[idx].sum(axis=1) / wb
            boot_b = eb[idx].sum(axis=1) / wb
            boot_d = boot_a - boot_b
            lo, hi = np.percentile(boot_d, [2.5, 97.5])

            # bootstrap two-sided p: how often the resampled difference crosses zero
            p_boot = 2.0 * min((boot_d <= 0).mean(), (boot_d >= 0).mean())
            p_boot = float(min(1.0, max(p_boot, 1.0 / N_BOOT)))

            rows.append({
                "method": method, "baseline": base, "split": kind, "condition": label,
                "n_clips": len(us), "n_words": int(w.sum()),
                "corpus_wer": float(corpus_a), "corpus_wer_baseline": float(corpus_b),
                "diff": float(diff),
                "relative_pct": float(100.0 * diff / corpus_b) if corpus_b > 0 else float("nan"),
                "ci95_low": float(lo), "ci95_high": float(hi),
                "boot_p": p_boot,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "wer_corpus_level.csv").open("w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    (out / "wer_corpus_level.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# Corpus-level WER (total errors / total reference words)", "",
        "Standard ASR aggregation, weighting each clip by its reference length instead of "
        "averaging per-clip rates. Avoids the zero-inflation that collapses the effective sample "
        "in the per-clip analysis. Uncertainty from a **paired** clip bootstrap "
        f"({N_BOOT} replicates): each replicate resamples clip indices once and scores both "
        "methods on the same clips.", "",
        "Difference is method minus baseline, so **negative is better**.", "",
        "| method | split | condition | clips | words | corpus WER | baseline | diff | rel % | 95% CI | boot p |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['split']} | {r['condition']} | {r['n_clips']} | "
            f"{r['n_words']} | {r['corpus_wer']:.4f} | {r['corpus_wer_baseline']:.4f} | "
            f"{r['diff']:+.4f} | {r['relative_pct']:+.1f} | "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['boot_p']:.4f} |"
        )
    (out / "wer_corpus_level.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out / 'wer_corpus_level.md'}\n")
    hdr = f"{'method':22s} {'split':8s} {'cond':8s} {'diff':>9s} {'rel%':>8s} {'CI excl 0':>10s}"
    print(hdr)
    for r in rows:
        if r["method"].startswith("neural"):
            print(f"{r['method']:22s} {r['split']:8s} {r['condition']:8s} "
                  f"{r['diff']:+9.4f} {r['relative_pct']:+8.1f} "
                  f"{'YES' if r['ci_excludes_zero'] else 'no':>10s}")


if __name__ == "__main__":
    main()
