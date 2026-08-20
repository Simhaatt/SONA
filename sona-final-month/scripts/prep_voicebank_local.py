#!/usr/bin/env python3
"""Extract a local VoiceBank-DEMAND subset for the complex-mask A/B experiment.

The full corpus lives in `DS_10283_2791.zip` as nested per-split zips (paired
clean/noisy). This pulls the full 824-clip test set (matches the existing
`full824` benchmark) plus a training subset, into a flat directory with a
manifest, so `train_supervised` can consume it locally without touching Kaggle.

    python scripts/prep_voicebank_local.py --n-train 1000
"""
from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path


def extract_inner(outer: zipfile.ZipFile, inner_name: str, dest: Path,
                  limit: int | None = None) -> list[str]:
    """Extract wavs from a nested zip (member of the outer zip) into dest/."""
    dest.mkdir(parents=True, exist_ok=True)
    raw = outer.read(inner_name)  # inner zip bytes
    stems = []
    with zipfile.ZipFile(io.BytesIO(raw)) as zin:
        wavs = sorted(n for n in zin.namelist() if n.lower().endswith(".wav"))
        if limit:
            wavs = wavs[:limit]
        for n in wavs:
            stem = Path(n).name
            (dest / stem).write_bytes(zin.read(n))
            stems.append(stem)
    return stems


# The SONA repo root: scripts/ -> sona-final-month/ -> <repo>.
REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path,
                    default=REPO / "DS_10283_2791.zip")
    ap.add_argument("--out", type=Path, default=REPO / "vbd_local")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=None, help="None = all 824")
    args = ap.parse_args()

    outer = zipfile.ZipFile(args.zip)

    print("extracting test set (clean+noisy) ...")
    tc = extract_inner(outer, "clean_testset_wav.zip", args.out / "test" / "clean", args.n_test)
    tn = extract_inner(outer, "noisy_testset_wav.zip", args.out / "test" / "noisy", args.n_test)
    print(f"  test: {len(tc)} clean, {len(tn)} noisy")

    print(f"extracting train subset (n={args.n_train}) ...")
    rc = extract_inner(outer, "clean_trainset_28spk_wav.zip", args.out / "train" / "clean", args.n_train)
    rn = extract_inner(outer, "noisy_trainset_28spk_wav.zip", args.out / "train" / "noisy", args.n_train)
    print(f"  train: {len(rc)} clean, {len(rn)} noisy")

    for split, clean_stems in (("train", rc), ("test", tc)):
        rows = []
        for stem in clean_stems:
            noisy = args.out / split / "noisy" / stem
            clean = args.out / split / "clean" / stem
            if noisy.exists() and clean.exists():
                rows.append((str(noisy), str(clean)))
        man = args.out / f"{split}_manifest.csv"
        with open(man, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["noisy", "clean"])
            w.writerows(rows)
        print(f"  wrote {man} ({len(rows)} pairs)")

    print("done.")


if __name__ == "__main__":
    main()
