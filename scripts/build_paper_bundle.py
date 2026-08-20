"""Assemble the paper bundle: collect every result table in the tree into one document.

Transcribing numbers by hand is how papers acquire figures that no longer match their source, so
this walks the results directories and inlines whatever it finds. Re-run it after any experiment
and the compendium is current by construction.

Usage:
    python scripts/build_paper_bundle.py
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "paper_bundle"

# (heading, path, kind) -- kind is "md" to inline verbatim, "csv" to render as a table.
SECTIONS: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- Study 1: quantization
    ("Study 1 — Quantization", "", "h1"),
    ("Main result: VoiceBank-DEMAND 824-clip test set",
     "PERCEPTUAL_MP/results/phase4/table1_main_result.tex", "raw"),
    ("Condition breakdown", "PERCEPTUAL_MP/results/phase4/table2_conditions.tex", "raw"),
    ("Phase 0 — structural probe", "PERCEPTUAL_MP/results/phase0/phase0_findings.json", "json"),
    ("Phase 1 — sensitivity rankings",
     "PERCEPTUAL_MP/results/phase1/phase1_final_analysis.json", "json"),
    ("Phase 2 — allocation search", "PERCEPTUAL_MP/results/phase2/phase2_pareto.json", "json"),
    ("Phase 2b — stem verification",
     "PERCEPTUAL_MP/results/phase2/phase2b_stem_verification.json", "json"),
    ("Phase 3 — frozen allocation",
     "PERCEPTUAL_MP/results/phase3/frozen_allocation.json", "json"),
    ("Phase 3b — MS-SNSD external", "PERCEPTUAL_MP/results/phase3/phase3b_ms_snsd.json", "json"),
    ("Phase 4 — summary", "PERCEPTUAL_MP/results/phase4/phase4_summary.json", "json"),
    ("E2 — Hessian-trace ranking (HAWQ-V2 proxy)",
     "PERCEPTUAL_MP/results/hessian/hessian_ranking.md", "md"),
    ("R3 — dither diagnostic", "PERCEPTUAL_MP/results/dither/dither_diagnostic.md", "md"),

    # ---------------------------------------------------------------- Study 2: evaluation
    ("Study 2 — Evaluation metrics", "", "h1"),
    ("Benchmark, 824-clip VoiceBank test", "sona-final-month/FYP_BENCHMARK/summary.md", "md"),
    ("Significance battery", "sona-final-month/SIGNIFICANCE/significance.md", "md"),
    ("Final tables — VoiceBank full 824",
     "sona-final-month/FINAL_TABLES/voicebank_full824.md", "md"),
    ("Final tables — MS-SNSD external",
     "sona-final-month/FINAL_TABLES/ms_snsd_external.md", "md"),
    ("Final tables — online adaptation",
     "sona-final-month/FINAL_TABLES/online_adaptation_story.md", "md"),
    ("Final tables — classical baselines (synthetic dev)",
     "sona-final-month/FINAL_TABLES/classical_synthetic.md", "md"),
    ("DeepFilterNet2 reproduction", "results/deepfilternet2_scored/summary.md", "md"),
    ("WER — aggregate", "results/wer_full824/summary_wer.md", "md"),
    ("WER — by noise type", "results/breakdowns/wer/wer_by_noise.md", "md"),
    ("WER — by SNR", "results/breakdowns/wer/wer_by_snr.md", "md"),
    ("E3 — per-clip WER significance with FDR",
     "results/breakdowns/wer_significance/wer_significance.md", "md"),
    ("E3b — corpus-level WER",
     "results/breakdowns/wer_significance/wer_corpus_level.md", "md"),
    ("E3 — findings", "results/breakdowns/wer_significance/FINDINGS_WER.md", "md"),
    ("Complex-mask A/B", "sona-final-month/COMPLEX_MASK_FINDINGS.md", "md"),
    ("DeepFilterNet2 SI-SDRi by noise",
     "results/breakdowns/dfn2_sisdri/si_sdr_i_by_noise.md", "md"),
    ("DeepFilterNet2 SI-SDRi by SNR", "results/breakdowns/dfn2_sisdri/si_sdr_i_by_snr.md", "md"),

    # ---------------------------------------------------------------- Study 3: embeddings
    ("Study 3 — Embedding quantization", "", "h1"),
    ("E1 — 1/Ratio@k vs Recall@k", "AudioVQ/results/ratio_at_k/ratio_at_k.md", "md"),

    # ---------------------------------------------------------------- Remedies
    ("Remedies", "", "h1"),
    ("R1 — condition-gated enhancement",
     "results/condition_gating/condition_gating.md", "md"),
    ("R1 — findings", "results/condition_gating/FINDINGS_R1.md", "md"),

    # ---------------------------------------------------------------- Deployment
    ("Deployment", "", "h1"),
    ("ONNX edge benchmark (idle machine, 2026-08-20)",
     "results/onnx/edge_bench_idle_2026-08-20/summary.csv", "csv"),
    ("R2 — mixed vs uniform INT8 timing",
     "results/onnx/edge_bench_mixed_2026-08-20/summary.csv", "csv"),
    ("Scene classifier", "sona-final-month/SCENE_CLASSIFIER_FINDINGS.md", "md"),
]


def render_csv(path: Path, max_rows: int = 60) -> str:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return "_(empty)_"
    buf = io.StringIO()
    head, body = rows[0], rows[1:max_rows]
    buf.write("| " + " | ".join(head) + " |\n")
    buf.write("|" + "---|" * len(head) + "\n")
    for r in body:
        r = r + [""] * (len(head) - len(r))
        buf.write("| " + " | ".join(x[:40] for x in r[:len(head)]) + " |\n")
    if len(rows) - 1 > max_rows:
        buf.write(f"\n_({len(rows)-1} rows total, first {max_rows} shown)_\n")
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BUNDLE / "02_ALL_RESULTS.md"))
    args = ap.parse_args()
    BUNDLE.mkdir(parents=True, exist_ok=True)

    parts = [
        "# Every result, in one place", "",
        "Generated by `scripts/build_paper_bundle.py` — do not hand-edit; re-run it instead. "
        "Each section names the file it came from so any number can be traced back to the "
        "artefact that produced it.", "",
    ]
    missing: list[str] = []

    for heading, rel, kind in SECTIONS:
        if kind == "h1":
            parts += ["", "---", "", f"# {heading}", ""]
            continue
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            parts += [f"## {heading}", "", f"> _Not yet generated: `{rel}`_", ""]
            continue
        parts += [f"## {heading}", "", f"`{rel}`", ""]
        try:
            if kind == "md":
                text = path.read_text(encoding="utf-8")
                # demote headings so the bundle keeps one hierarchy
                text = "\n".join(("##" + ln) if ln.startswith("#") else ln
                                 for ln in text.splitlines())
                parts += [text, ""]
            elif kind == "csv":
                parts += [render_csv(path), ""]
            elif kind == "json":
                parts += ["```json", path.read_text(encoding="utf-8").strip(), "```", ""]
            else:
                parts += ["```", path.read_text(encoding="utf-8").strip(), "```", ""]
        except Exception as e:
            parts += [f"> _Could not read: {e}_", ""]

    out = Path(args.out)
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}  ({len(out.read_text(encoding='utf-8').splitlines())} lines)")
    if missing:
        print(f"\n{len(missing)} source(s) not yet generated:")
        for m in missing:
            print("   ", m)


if __name__ == "__main__":
    main()
