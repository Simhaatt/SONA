#!/usr/bin/env python3
"""Score PESQ (+ STOI/ESTOI) on saved wav pairs. RUN INSIDE WSL (Linux) where the
`pesq` wheel installs without a compiler.

    wsl -d Ubuntu-24.04 -u root -- /opt/pesqenv/bin/python \
        /mnt/e/Rohit/FYP/sona-final-month/scripts/score_pesq_wsl.py

Reads pesq_wavs/{clean,noisy,magnitude_gtcrn,complex_gtcrn}/<uid>.wav, computes
wideband PESQ + STOI + ESTOI for each method vs the clean reference, and writes
per-clip + summary CSVs next to the wavs.
"""
import csv
import os

import numpy as np
import soundfile as sf
from pesq import pesq
from pystoi import stoi

WAVS = os.environ.get("PESQ_WAVS",
                      "/mnt/e/Rohit/FYP/sona-final-month/COMPLEX_MASK_FULL5K/pesq_wavs")
SR = 16000
METHODS = ["noisy", "magnitude_gtcrn", "complex_gtcrn", "metricgan_plus"]


def load(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64)


def main():
    clean_dir = os.path.join(WAVS, "clean")
    uids = sorted(os.path.splitext(f)[0] for f in os.listdir(clean_dir)
                  if f.endswith(".wav"))
    print(f"scoring {len(uids)} clips x {len(METHODS)} methods")

    per_clip = []
    for i, uid in enumerate(uids):
        clean = load(os.path.join(clean_dir, uid + ".wav"))
        for m in METHODS:
            p = os.path.join(WAVS, m, uid + ".wav")
            if not os.path.exists(p):
                continue
            deg = load(p)
            n = min(len(clean), len(deg))
            c, d = clean[:n], deg[:n]
            row = {"uid": uid, "method": m}
            try:
                row["pesq"] = float(pesq(SR, c, d, "wb"))
            except Exception as e:
                row["pesq"] = None
            try:
                row["stoi"] = float(stoi(c, d, SR, extended=False))
                row["estoi"] = float(stoi(c, d, SR, extended=True))
            except Exception:
                row["stoi"] = row["estoi"] = None
            per_clip.append(row)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(uids)}")

    # write per-clip
    pc_path = os.path.join(WAVS, "pesq_per_clip.csv")
    with open(pc_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "method", "pesq", "stoi", "estoi"])
        w.writeheader()
        w.writerows(per_clip)

    # summary means per method
    summ_path = os.path.join(WAVS, "pesq_summary.csv")
    print(f"\n{'method':18s} {'PESQ':>7s} {'STOI':>7s} {'ESTOI':>7s}  (n)")
    with open(summ_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "pesq", "stoi", "estoi", "n"])
        for m in METHODS:
            vals = [r for r in per_clip if r["method"] == m]
            def mean(k):
                xs = [r[k] for r in vals if r.get(k) is not None]
                return sum(xs) / len(xs) if xs else float("nan")
            pe, st, es = mean("pesq"), mean("stoi"), mean("estoi")
            w.writerow([m, pe, st, es, len(vals)])
            print(f"{m:18s} {pe:7.4f} {st:7.4f} {es:7.4f}  ({len(vals)})")

    # deltas
    def mmean(method, k):
        xs = [r[k] for r in per_clip if r["method"] == method and r.get(k) is not None]
        return sum(xs) / len(xs) if xs else float("nan")
    print("\ndeltas (complex - magnitude):")
    for k in ("pesq", "stoi", "estoi"):
        print(f"  {k:6s} {mmean('complex_gtcrn', k) - mmean('magnitude_gtcrn', k):+.4f}")
    print(f"\nwrote {pc_path}\nwrote {summ_path}")


if __name__ == "__main__":
    main()
