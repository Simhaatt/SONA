"""Differential WER: how much does each method's output hurt/help downstream ASR accuracy?

VoiceBank-DEMAND ships no ground-truth transcripts, so this uses a pseudo-reference: an ASR
transcript of the *clean* signal, cached once per clip and reused as the reference for every
method (see `src/eval/asr.py` for the rationale). Each method's enhanced output is transcribed
and scored against that same pseudo-reference with `jiwer`.

Reuses the existing shared harness (`evaluate_method` / `compute_metrics`) unmodified: setting
`Sample.text` to the cached pseudo-reference and requesting only the `wer` metric makes
`compute_metrics` skip every reference-audio metric and just run ASR + WER.

Usage:
    python scripts/eval_wer.py --manifest "E:\\Rohit\\FYP\\vbd_local\\test_manifest.csv" --limit 20
    python scripts/eval_wer.py --manifest "E:\\Rohit\\FYP\\vbd_local\\test_manifest.csv" \\
        --gtcrn-checkpoint "...\\ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25\\last.pt" \\
        --complex-checkpoint "...\\ckpt_complex_gtcrn\\last.pt"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_transcript_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["uid"]: row["text"] for row in csv.DictReader(fh)}


def _save_transcript_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["uid", "text"])
        w.writeheader()
        for uid, text in cache.items():
            w.writerow({"uid": uid, "text": text})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="clean,noisy manifest CSV")
    ap.add_argument("--limit", type=int, default=0, help="0 means full manifest")
    ap.add_argument("--out", default=str(ROOT / "results" / "wer"))
    ap.add_argument("--gtcrn-checkpoint", default=None)
    ap.add_argument("--complex-checkpoint", default=None)
    ap.add_argument("--whisper-model", default="small.en")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--methods",
        default="passthrough,spectral_subtraction,wiener",
        help="comma-separated classical baseline names from BASELINES",
    )
    args = ap.parse_args()

    import os

    os.environ["SONA_WHISPER_MODEL"] = args.whisper_model

    import torch

    from src.baselines import BASELINES
    from src.data import PairDataset
    from src.eval.asr import transcribe
    from src.eval.evaluate import aggregate, evaluate_method
    from src.eval.report import markdown_table, save_rows_csv
    from src.frontend import SpectralFrontend
    from src.models import NeuralEnhancer
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.config import Config
    from src.utils.logging import get_logger
    from src.utils.seed import set_seed

    log = get_logger("eval_wer")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1234)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    samples = list(PairDataset(manifest=args.manifest, limit=(args.limit or None)))
    if not samples:
        raise SystemExit(f"No samples found in {args.manifest}")
    log.info("device=%s | whisper=%s | samples=%d", device, args.whisper_model, len(samples))

    # --- 1. clean pseudo-reference transcripts (cached; shared across all methods) ---
    cache_path = out / "clean_transcripts.csv"
    cache = _load_transcript_cache(cache_path)
    missing = [s for s in samples if s.uid not in cache]
    log.info("pseudo-reference transcripts: %d cached, %d to compute", len(cache), len(missing))
    for i, s in enumerate(missing):
        cache[s.uid] = transcribe(s.clean, s.sr, model_size=args.whisper_model)
        if (i + 1) % 50 == 0:
            log.info("  transcribed %d/%d clean references", i + 1, len(missing))
            _save_transcript_cache(cache_path, cache)
    _save_transcript_cache(cache_path, cache)

    for s in samples:
        s.text = cache[s.uid]

    # --- 2. build enhancers: classical baselines + optional learned checkpoints ---
    frontend = SpectralFrontend()
    enhancers = []
    for name in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if name not in BASELINES:
            log.warning("unknown baseline '%s' (skipping)", name)
            continue
        enhancers.append(BASELINES[name](frontend))

    def _load_neural(ckpt_path: str, label: str):
        checkpoint = torch.load(ckpt_path, map_location=device)
        cfg = Config(checkpoint["cfg"])
        tf = build_frontend(cfg).to(device)
        model = build_model(cfg, tf).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return NeuralEnhancer(model, tf, device=device, name=label)

    if args.gtcrn_checkpoint:
        enhancers.append(_load_neural(args.gtcrn_checkpoint, "neural_gtcrn"))
    if args.complex_checkpoint:
        enhancers.append(_load_neural(args.complex_checkpoint, "neural_complex_gtcrn"))

    # --- 3. transcribe each method's output, score WER against the cached reference ---
    all_rows: list[dict] = []
    for enh in enhancers:
        log.info("scoring WER for %s over %d samples", enh.name, len(samples))
        rows = evaluate_method(enh, samples, include_metrics=["wer"], method_name=enh.name)
        all_rows += rows

    save_rows_csv(all_rows, out / "per_sample_wer.csv")
    summary = aggregate(all_rows, group_keys=("method",))
    save_rows_csv(summary, out / "summary_wer.csv")
    md = markdown_table(summary)
    (out / "summary_wer.md").write_text(md, encoding="utf-8")

    print("\n=== WER summary (lower is better) ===\n")
    print(md)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
