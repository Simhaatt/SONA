#!/usr/bin/env python3
"""Score PESQ for DeepFilterNet2 outputs vs clean, inside WSL, reading from the native (ext4)
filesystem to avoid the WSL2<->NTFS /mnt bridge hang found during Phase 1.

    MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -u root -- /opt/pesqenv/bin/python \
        /root/dfn2_pesq/score_dfn2_pesq_wsl.py
"""
import csv
import os

import numpy as np
import soundfile as sf
from pesq import pesq

CLEAN_DIR = os.environ.get("CLEAN_DIR", "/root/dfn2_pesq/clean")
DEG_DIR = os.environ.get("DEG_DIR", "/root/dfn2_pesq/deepfilternet2")
OUT = os.environ.get("OUT", "/root/dfn2_pesq/pesq_per_clip.csv")
SR = 16000


def load(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float64)


def main():
    uids = sorted(os.path.splitext(f)[0] for f in os.listdir(DEG_DIR) if f.endswith(".wav"))
    print(f"scoring {len(uids)} clips")
    rows = []
    for i, uid in enumerate(uids):
        cp, dp = os.path.join(CLEAN_DIR, uid + ".wav"), os.path.join(DEG_DIR, uid + ".wav")
        if not (os.path.exists(cp) and os.path.exists(dp)):
            continue
        c, d = load(cp), load(dp)
        n = min(len(c), len(d))
        try:
            p = float(pesq(SR, c[:n], d[:n], "wb"))
        except Exception as e:
            p = None
        rows.append({"uid": uid, "pesq": p})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(uids)}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "pesq"])
        w.writeheader()
        w.writerows(rows)

    vals = [r["pesq"] for r in rows if r["pesq"] is not None]
    print(f"\nmean PESQ = {sum(vals)/len(vals):.4f}  (n={len(vals)}/{len(rows)})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
