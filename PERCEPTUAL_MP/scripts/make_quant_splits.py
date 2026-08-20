"""Build speaker-disjoint calibration / dev manifests from the VoiceBank-DEMAND TRAIN split.

Bit allocation is a model-selection decision, so it must not be chosen on the test set. This
carves a dev set out of TRAIN by speaker, leaving the 824-clip test set (p232, p257) untouched
until the allocation is frozen.

Guarantees:
  * calibration speakers and dev speakers are disjoint
  * neither overlaps the test speakers (test is a separate manifest entirely)

Usage:
    python PERCEPTUAL_MP/scripts/make_quant_splits.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "splits"
# <repo>/vbd_local holds the local 16 kHz VoiceBank-DEMAND copy the manifests index.
REPO = HERE.parents[1]

# Held out from calibration and used only for sensitivity / allocation search.
DEV_SPEAKERS = ("p254", "p256")
SEED = 2026


def speaker_of(path: str) -> str:
    m = re.match(r"(p\d+)", Path(path).stem)
    return m.group(1) if m else "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest",
                    default=os.environ.get("SONA_VBD_TRAIN_MANIFEST",
                                           str(REPO / "vbd_local" / "train_manifest.csv")))
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.train_manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())

    dev_pool = [r for r in rows if speaker_of(r["noisy"]) in DEV_SPEAKERS]
    calib_pool = [r for r in rows if speaker_of(r["noisy"]) not in DEV_SPEAKERS]

    rng = random.Random(SEED)
    calib = rng.sample(calib_pool, k=min(args.n_calib, len(calib_pool)))
    dev = rng.sample(dev_pool, k=min(args.n_dev, len(dev_pool)))

    def write(path: Path, sel):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sel)

    write(out_dir / "calib_manifest.csv", calib)
    write(out_dir / "dev_manifest.csv", dev)

    calib_spk = sorted({speaker_of(r["noisy"]) for r in calib})
    dev_spk = sorted({speaker_of(r["noisy"]) for r in dev})
    overlap = set(calib_spk) & set(dev_spk)

    print(f"calibration : {len(calib):4d} clips from {len(calib_spk)} speakers {calib_spk}")
    print(f"dev         : {len(dev):4d} clips from {len(dev_spk)} speakers {dev_spk}")
    print(f"speaker overlap calib/dev: {sorted(overlap) if overlap else 'NONE (correct)'}")
    print(f"test speakers (untouched) : ['p232', 'p257']")
    print(f"\nwrote {out_dir / 'calib_manifest.csv'}")
    print(f"wrote {out_dir / 'dev_manifest.csv'}")

    if overlap:
        raise SystemExit("ERROR: calibration and dev share speakers -- split is leaky")


if __name__ == "__main__":
    main()
