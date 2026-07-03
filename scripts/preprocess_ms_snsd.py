"""Create SONA clean/noisy manifests from MS-SNSD.

MS-SNSD already ships 16 kHz mono clean speech and noise clips. This script creates deterministic
noisy mixtures and a manifest that `PairDataset` can read, while keeping the original clean files
in place instead of copying them.

Example:
    python scripts/preprocess_ms_snsd.py ^
        --root external_data/MS-SNSD ^
        --out external_data/ms_snsd_sona ^
        --splits test --limit-test 1100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.noise_bank import mix_at_snr
from src.utils.audio_io import load_audio, save_audio


def _noise_type(path: Path) -> str:
    stem = path.stem
    return stem.split("_")[0] if "_" in stem else stem


def _fit_length(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / len(noise))))
    if len(noise) > length:
        off = int(rng.integers(0, len(noise) - length + 1))
        noise = noise[off : off + length]
    return noise.astype(np.float32)


def make_split(
    root: Path,
    out: Path,
    split: str,
    limit: int | None,
    snrs: list[float],
    seed: int,
) -> int:
    clean_dir = root / f"clean_{split}"
    noise_dir = root / f"noise_{split}"
    if not clean_dir.is_dir() or not noise_dir.is_dir():
        raise SystemExit(f"missing MS-SNSD split folders: {clean_dir} / {noise_dir}")

    clean_files = sorted(clean_dir.glob("*.wav"))
    noise_files = sorted(noise_dir.glob("*.wav"))
    if limit:
        clean_files = clean_files[:limit]
    if not clean_files or not noise_files:
        raise SystemExit(f"no wav files found for split={split}")

    noisy_out = out / f"noisy_{split}"
    noisy_out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = np.random.default_rng(seed + (0 if split == "train" else 10000))

    for i, clean_path in enumerate(clean_files):
        clean, sr = load_audio(clean_path, sr=16000)
        noise_path = noise_files[int(rng.integers(0, len(noise_files)))]
        noise, _ = load_audio(noise_path, sr=16000)
        noise = _fit_length(noise, len(clean), rng)
        snr = float(snrs[i % len(snrs)])
        noisy, _ = mix_at_snr(clean, noise, snr, rng)
        uid = f"ms_snsd_{split}_{i:05d}_snr{snr:g}".replace("-", "m").replace(".", "p")
        noisy_path = noisy_out / f"{uid}.wav"
        save_audio(noisy_path, noisy, sr)
        rows.append({
            "clean": str(clean_path),
            "noisy": str(noisy_path),
            "noise_type": _noise_type(noise_path),
            "snr_db": snr,
        })

    manifest = out / f"{split}.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clean", "noisy", "noise_type", "snr_db"])
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {split}: {len(rows)} pairs -> {manifest}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="external_data/MS-SNSD")
    ap.add_argument("--out", default="external_data/ms_snsd_sona")
    ap.add_argument("--splits", default="test", help="comma list: train,test")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--snrs", default="-5,0,5,10",
                    help="comma list of SNRs in dB, cycled across clean clips")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    snrs = [float(x.strip()) for x in args.snrs.split(",") if x.strip()]
    total = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in {"train", "test"}:
            raise SystemExit("--splits entries must be 'train' and/or 'test'")
        limit = args.limit_train if split == "train" else args.limit_test
        total += make_split(root, out, split, limit, snrs, args.seed)
    print(f"Done. {total} pairs written under {out}")


if __name__ == "__main__":
    main()
