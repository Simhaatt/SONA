#!/usr/bin/env python3
"""Synthesise speech+noise mixtures for deployment-realistic scene evaluation.

Why this exists
---------------
The first real-audio retune was fitted on *pure* noise clips and *clean* speech.
That is not what the classifier sees in production: the web demo and the realtime
path both feed it **noisy speech**, whose features sit between the two (noise
fills the pauses, so modulation drops and the clip stops looking like either
pure source). Tuning on pure sources and deploying on mixtures is a train/serve
mismatch — it made the bundled demo sample (speech + babble @ 5 dB) classify as
`fan`, which would route it to `aggressive_speech` instead of `hearing`.

`SCENE_TO_MODE` routes on the *noise environment*, not on whether speech is
present, so a mixture is labelled by its noise category:

    fan     AirConditioner, VacuumCleaner, WasherDryer, Washing
    crowd   Babble, CafeTeria, Cafe, Restaurant, Square, Park
    traffic Traffic, Car, Bus, Metro, Station
    speech  clean speech (no added noise)

Mixtures are generated across several SNRs so the eval can also report accuracy
as a function of SNR.

    python scripts/build_scene_mixtures.py --snrs 0 5 10 15 --per-combo 2
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

NOISE_LABELS = {
    "AirConditioner": "fan", "VacuumCleaner": "fan",
    "WasherDryer": "fan", "Washing": "fan",
    "Babble": "crowd", "CafeTeria": "crowd", "Cafe": "crowd",
    "Restaurant": "crowd", "Square": "crowd", "Park": "crowd",
    "Traffic": "traffic", "Car": "traffic", "Bus": "traffic",
    "Metro": "traffic", "Station": "traffic",
}
SR = 16000
EPS = 1e-10


def category_of(p: Path) -> str:
    return p.stem.rstrip("0123456789").rstrip("_")


def load_mono(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:  # MS-SNSD is already 16 kHz; guard anyway
        idx = np.linspace(0, len(x) - 1, int(len(x) * SR / sr))
        x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return x


def fit_length(noise: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if len(noise) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(noise) < n:
        reps = int(np.ceil(n / len(noise)))
        noise = np.tile(noise, reps)
    start = rng.randrange(0, max(1, len(noise) - n + 1))
    return noise[start:start + n]


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = float(np.mean(speech ** 2)) + EPS
    pn = float(np.mean(noise ** 2)) + EPS
    scale = np.sqrt(ps / (pn * (10.0 ** (snr_db / 10.0))))
    y = speech + noise * scale
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 0.99:
        y = y / peak * 0.99
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msnsd", type=Path,
                    default=Path(__file__).resolve().parents[2] / "external_data" / "MS-SNSD")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "recordings" / "mixtures")
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).resolve().parents[1] / "recordings"
                    / "manifest_mixtures.csv")
    ap.add_argument("--snrs", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0])
    ap.add_argument("--per-combo", type=int, default=2,
                    help="mixtures per (noise file, SNR)")
    ap.add_argument("--max-noise-per-cat", type=int, default=6,
                    help="cap noise files per category to keep classes balanced")
    ap.add_argument("--n-clean", type=int, default=40)
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    clean_files = sorted((args.msnsd / "clean_test").glob("*.wav"))
    if not clean_files:
        raise SystemExit(f"no clean speech under {args.msnsd/'clean_test'}")

    # gather noise files grouped by SONA label
    by_cat: dict[str, list[Path]] = {}
    for sub in ("noise_test", "noise_train"):
        for wav in sorted((args.msnsd / sub).glob("*.wav")):
            lab = NOISE_LABELS.get(category_of(wav))
            if lab:
                by_cat.setdefault(category_of(wav), []).append(wav)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_max = int(args.max_seconds * SR)

    for cat, files in sorted(by_cat.items()):
        label = NOISE_LABELS[cat]
        for nf in files[: args.max_noise_per_cat]:
            noise = load_mono(nf)
            for snr in args.snrs:
                for k in range(args.per_combo):
                    sp = load_mono(rng.choice(clean_files))[:n_max]
                    if len(sp) < SR:  # skip <1 s speech
                        continue
                    y = mix_at_snr(sp, fit_length(noise, len(sp), rng), snr)
                    name = f"{cat}_{nf.stem}_snr{int(snr)}_{k}.wav"
                    out = args.out_dir / name
                    sf.write(out, y, SR)
                    # source_file identifies the exact noise texture, so grouped
                    # CV can keep all mixtures of one texture on the same side
                    rows.append((str(out), label, cat, nf.stem, snr))

    for cf in clean_files[: args.n_clean]:
        rows.append((str(cf), "speech", "CleanSpeech", cf.stem, ""))

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source_category", "source_file", "snr_db"])
        w.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row[1]] = counts.get(row[1], 0) + 1
    print(f"wrote {args.manifest} ({len(rows)} clips)")
    for lab in sorted(counts):
        print(f"  {lab:8s} {counts[lab]:4d}")


if __name__ == "__main__":
    main()
