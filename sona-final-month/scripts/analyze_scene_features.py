#!/usr/bin/env python3
"""Dump per-class scene-classifier feature statistics for real audio.

Diagnostic behind the Layer B threshold retune: the classifier is a hand-tuned
deterministic scorer (`src/controller/scene.py`), so when it underperforms the
fix is to re-seat its `_tri()` windows on real feature distributions rather than
to guess. This script reports, per true label, the median / IQR of every feature
so the confusions have a measurable explanation for the thesis.

    python scripts/analyze_scene_features.py --manifest recordings/manifest_msnsd.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURES = ["centroid_hz", "flatness", "rolloff_hz", "zcr",
            "low_ratio", "mid_ratio", "high_ratio", "modulation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "SCENE_EVAL_REAL"
                    / "feature_stats.json")
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo.resolve()))
    from src.controller import SceneClassifier  # type: ignore
    from src.utils.audio_io import load_audio, DEFAULT_SR  # type: ignore

    clf = SceneClassifier(sr=DEFAULT_SR)

    by_label: dict[str, list[dict]] = defaultdict(list)
    preds: dict[str, list[str]] = defaultdict(list)
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wav, _ = load_audio(row["path"], sr=DEFAULT_SR, mono=True)
            feats = clf.features(np.asarray(wav, dtype=np.float32))
            by_label[row["label"]].append(feats)
            preds[row["label"]].append(clf.predict(wav, sr=DEFAULT_SR).label)

    stats: dict[str, dict] = {}
    print(f"{'label':8s} {'n':>4s}  " + "  ".join(f"{k[:11]:>11s}" for k in FEATURES))
    for label in sorted(by_label):
        rows = by_label[label]
        stats[label] = {"n": len(rows)}
        med = []
        for k in FEATURES:
            vals = np.array([r[k] for r in rows], dtype=float)
            q1, q2, q3 = (float(np.percentile(vals, p)) for p in (25, 50, 75))
            stats[label][k] = {"p25": q1, "median": q2, "p75": q3}
            med.append(q2)
        print(f"{label:8s} {len(rows):4d}  " +
              "  ".join(f"{v:11.3f}" for v in med))

    print("\nprediction spread per true label:")
    for label in sorted(preds):
        c: dict[str, int] = {}
        for p in preds[label]:
            c[p] = c.get(p, 0) + 1
        top = sorted(c.items(), key=lambda kv: -kv[1])
        print(f"  {label:8s} -> " + ", ".join(f"{k}:{v}" for k, v in top))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
