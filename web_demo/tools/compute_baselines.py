"""Score the ORIGINAL (unprocessed) noisy audio on both test sets, for the app's baseline rows.

Writes `web_demo/data/baselines.json` with the mean SI-SDR, SNR, STOI, ESTOI and LSD of the
noisy input against its clean reference — the "before cleaning" reference every method is
compared with. PESQ is added only when the `pesq` package is installed.

    python web_demo/tools/compute_baselines.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

WEB_DEMO = Path(__file__).resolve().parents[1]
REPO = WEB_DEMO.parent
sys.path.insert(0, str(REPO))

from src.eval.metrics import compute_metrics  # noqa: E402
from src.utils.audio_io import DEFAULT_SR, load_audio  # noqa: E402

OUT = WEB_DEMO / "data" / "baselines.json"
SETS = {
    "voicebank": (REPO / "vbd_local" / "test_manifest.csv", "VoiceBank-DEMAND test (824 clips)"),
    "ms_snsd": (REPO / "external_data" / "ms_snsd_sona" / "test.csv", "MS-SNSD external test (1,100 clips)"),
}
METRICS = ["si_sdr", "snr", "stoi", "estoi", "lsd", "pesq"]


def _local(path_str: str) -> Path:
    tail = path_str.split("noise cancellation", 1)[-1].lstrip("\\/")
    return REPO / Path(tail.replace("\\", "/"))


def main() -> None:
    out: dict = {}
    for key, (manifest, title) in SETS.items():
        rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
        scores: dict[str, list[float]] = {m: [] for m in METRICS}
        for i, row in enumerate(rows, 1):
            noisy, _ = load_audio(_local(row["noisy"]), sr=DEFAULT_SR, mono=True)
            clean, _ = load_audio(_local(row["clean"]), sr=DEFAULT_SR, mono=True)
            n = min(noisy.size, clean.size)
            m = compute_metrics(clean[:n], noisy[:n], noisy[:n], sr=DEFAULT_SR, include=METRICS)
            for k, v in m.items():
                if v is not None and np.isfinite(v):
                    scores[k].append(float(v))
            if i % 100 == 0:
                print(f"  {key}: {i}/{len(rows)}", flush=True)
        out[key] = {
            "title": title,
            "n": len(rows),
            **{k: round(float(np.mean(v)), 4) for k, v in scores.items() if v},
        }
        print(key, out[key], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
