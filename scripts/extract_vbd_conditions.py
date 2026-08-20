"""Extract the official VoiceBank-DEMAND test-set condition labels (noise type + SNR).

The local `vbd_local/test_manifest.csv` carries only clean/noisy paths, so per-noise and per-SNR
breakdowns were impossible. The official DOI archive ships `logfiles.zip -> log_testset.txt`,
which labels every one of the 824 test clips with its noise type and SNR. This pulls that out
into a CSV keyed by uid so any per-clip results file can be joined against it.

Usage:
    python scripts/extract_vbd_conditions.py
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "DS_10283_2791.zip"
OUT = ROOT / "results" / "vbd_test_conditions.csv"


def main() -> None:
    z = zipfile.ZipFile(ARCHIVE)
    inner = zipfile.ZipFile(io.BytesIO(z.read("logfiles.zip")))
    text = inner.read("log_testset.txt").decode("utf-8", errors="replace")

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        uid, noise_type, snr = parts[0], parts[1], float(parts[2])
        rows.append({"uid": uid, "noise_type": noise_type, "snr_db": snr})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["uid", "noise_type", "snr_db"])
        w.writeheader()
        w.writerows(rows)

    noise_types = sorted({r["noise_type"] for r in rows})
    snrs = sorted({r["snr_db"] for r in rows})
    print(f"wrote {len(rows)} clips -> {OUT}")
    print(f"  noise types ({len(noise_types)}): {noise_types}")
    print(f"  SNRs ({len(snrs)}): {snrs}")
    from collections import Counter
    print(f"  per noise type: {dict(Counter(r['noise_type'] for r in rows))}")
    print(f"  per SNR: {dict(Counter(r['snr_db'] for r in rows))}")


if __name__ == "__main__":
    main()
