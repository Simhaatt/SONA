"""Preprocess VoiceBank-DEMAND (28-speaker) into a compact 16 kHz mono dataset for Kaggle.

Reads directly from the full DataShare bundle `DS_10283_2791.zip` (or a folder of the individual
inner zips / extracted folders), resamples every wav 48 kHz -> 16 kHz mono, and writes the four
folders our pipeline expects:

    <out>/clean_trainset_wav/  noisy_trainset_wav/  clean_testset_wav/  noisy_testset_wav/

Result is ~3x smaller than the 48 kHz originals -> a much lighter Kaggle upload, and matches the
16 kHz pipeline exactly (no per-run resampling).

Usage:
    python scripts/preprocess_vbd.py --bundle "D:/noise cancellation/DS_10283_2791.zip" --out D:/vbd16k
    python scripts/preprocess_vbd.py --bundle ... --out ... --sets test   # smoke on the test set only
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from math import gcd
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf

TARGET_SR = 16000

# inner-zip name in the bundle  ->  output folder name our pipeline expects
SET_MAP = {
    "train": [
        ("clean_trainset_28spk_wav.zip", "clean_trainset_wav"),
        ("noisy_trainset_28spk_wav.zip", "noisy_trainset_wav"),
    ],
    "test": [
        ("clean_testset_wav.zip", "clean_testset_wav"),
        ("noisy_testset_wav.zip", "noisy_testset_wav"),
    ],
}


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        x = sps.resample_poly(x, TARGET_SR // g, sr // g)
    return np.ascontiguousarray(x, dtype=np.float32)


def _iter_wavs(inner_zip: zipfile.ZipFile):
    for n in inner_zip.namelist():
        if n.lower().endswith(".wav"):
            yield n


def process_inner(bundle: zipfile.ZipFile, inner_name: str, out_dir: Path, tmp: Path) -> int:
    """Extract one inner zip from the bundle and resample all its wavs into `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    tmp_zip = tmp / inner_name
    print(f"  extracting {inner_name} from bundle ...", flush=True)
    with bundle.open(inner_name) as src, open(tmp_zip, "wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)

    n = 0
    t0 = time.time()
    with zipfile.ZipFile(tmp_zip) as iz:
        names = list(_iter_wavs(iz))
        for name in names:
            data = iz.read(name)
            x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            y = to_16k_mono(x, sr)
            sf.write(str(out_dir / Path(name).name), y, TARGET_SR, subtype="PCM_16")
            n += 1
            if n % 1000 == 0:
                print(f"    {n}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    tmp_zip.unlink(missing_ok=True)
    print(f"  [ok] {out_dir.name}: {n} files @16k ({time.time()-t0:.0f}s)", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="path to DS_10283_2791.zip")
    ap.add_argument("--out", required=True, help="output dir for the 16k dataset")
    ap.add_argument("--sets", nargs="+", default=["train", "test"], choices=["train", "test"])
    ap.add_argument("--tmp", default=None, help="scratch dir for inner zips (default: <out>/_tmp)")
    args = ap.parse_args()

    out = Path(args.out)
    tmp = Path(args.tmp) if args.tmp else out / "_tmp"
    total = 0
    with zipfile.ZipFile(args.bundle) as bundle:
        present = set(bundle.namelist())
        for split in args.sets:
            for inner_name, out_name in SET_MAP[split]:
                if inner_name not in present:
                    raise SystemExit(f"{inner_name} not found in bundle")
                total += process_inner(bundle, inner_name, out / out_name, tmp)
    try:
        tmp.rmdir()
    except OSError:
        pass
    print(f"\nDone. {total} files written under {out}")
    print("Next: zip the four folders in", out, "and upload as a Kaggle dataset.")


if __name__ == "__main__":
    main()
