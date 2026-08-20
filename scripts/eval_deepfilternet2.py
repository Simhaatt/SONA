"""Score pre-generated DeepFilterNet2 outputs (from run_deepfilternet2_inference.py) through the
same shared harness every other method in the benchmark uses, so Table 3 gets a measured row
instead of a cited one.

DeepFilterNet2's public pretrained checkpoint (epoch 96, `DeepFilterNet2` model settings) was
NOT trained on VoiceBank-DEMAND -- it ships trained on the DNS4-Challenge-family corpora used in
the original DeepFilterNet paper. This is a zero-shot cross-corpus evaluation, not matched-training-
data comparison; state that explicitly in any table footnote.

Usage:
    python scripts/eval_deepfilternet2.py --manifest "E:\\...\\test_manifest.csv" \\
        --enhanced-dir results/deepfilternet2_voicebank --out results/deepfilternet2_scored
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--enhanced-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from src.data import PairDataset
    from src.eval.evaluate import aggregate
    from src.eval.metrics import compute_metrics
    from src.eval.report import markdown_table, save_rows_csv
    from src.utils.audio_io import load_audio

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enhanced_dir = Path(args.enhanced_dir)

    samples = list(PairDataset(manifest=args.manifest, limit=(args.limit or None)))
    metrics = ["si_sdr", "si_sdr_i", "snr", "snr_i", "stoi", "estoi", "lsd"]

    rows = []
    missing = 0
    for s in samples:
        enh_path = enhanced_dir / f"{s.uid}.wav"
        if not enh_path.exists():
            missing += 1
            continue
        enhanced, _ = load_audio(enh_path, sr=s.sr, mono=True)
        m = compute_metrics(clean=s.clean, enhanced=enhanced, noisy=s.noisy, sr=s.sr, include=metrics)
        rows.append({"method": "deepfilternet2", "uid": s.uid, **m})

    if missing:
        print(f"WARNING: {missing} clips had no enhanced wav (inference incomplete?)")

    save_rows_csv(rows, out / "per_sample.csv")
    summary = aggregate(rows, group_keys=("method",))
    save_rows_csv(summary, out / "summary.csv")
    md = markdown_table(summary)
    (out / "summary.md").write_text(md, encoding="utf-8")

    print(f"\nscored {len(rows)}/{len(samples)} clips\n")
    print(md)
    print(f"\nNOTE: DeepFilterNet2's pretrained checkpoint was trained on DNS4-family data, not "
          f"VoiceBank-DEMAND -- this is a zero-shot cross-corpus evaluation.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
