"""Build the web demo's sample library: real noisy/clean pairs with metadata.

Picks a small, diverse set of test-set mixtures (never training data) so the demo
can score enhancement against a true clean reference:

* MS-SNSD SONA-mixed test set — one clip per noise type, labelled noise + SNR.
* VoiceBank-DEMAND test set — two clips (the paper's main benchmark corpus).

Writes `samples/<id>_noisy.wav`, `samples/<id>_clean.wav` and `samples/library.json`.

    python web_demo/tools/build_sample_library.py
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

from src.utils.audio_io import DEFAULT_SR, load_audio, save_audio  # noqa: E402

SAMPLES = WEB_DEMO / "samples"
MS_SNSD_CSV = REPO / "external_data" / "ms_snsd_sona" / "test.csv"
VBD_DIR = REPO / "vbd_local" / "test"

# (noise_type, snr_db, display title) — one clip each, chosen for audible variety
MS_SNSD_PICKS = [
    ("AirConditioner", 0.0, "Air conditioner hum"),
    ("Babble", 5.0, "Cafe babble"),
    ("Typing", 0.0, "Keyboard typing"),
    ("VacuumCleaner", 0.0, "Vacuum cleaner"),
    ("ShuttingDoor", 5.0, "Door slams"),
    ("AirportAnnouncements", 5.0, "Airport announcements"),
    ("CopyMachine", -5.0, "Copy machine (very noisy)"),
    ("Munching", 0.0, "Munching / eating"),
]
VBD_PICKS = [("p232_005", "VoiceBank-DEMAND test clip A"), ("p257_010", "VoiceBank-DEMAND test clip B")]
MIN_S, CROP_S = 4.0, 8.0  # long MS-SNSD clips are cropped to the first 8 s


def _local(path_str: str) -> Path:
    """Manifests store absolute paths from the machine that built them; re-root them here."""
    marker = "noise cancellation"
    tail = path_str.split(marker, 1)[-1].lstrip("\\/")
    return REPO / Path(tail.replace("\\", "/"))


def _write_pair(sample_id: str, noisy: np.ndarray, clean: np.ndarray) -> None:
    n = min(noisy.size, clean.size)
    save_audio(SAMPLES / f"{sample_id}_noisy.wav", noisy[:n], DEFAULT_SR)
    save_audio(SAMPLES / f"{sample_id}_clean.wav", clean[:n], DEFAULT_SR)


def main() -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    library: list[dict] = []

    rows = list(csv.DictReader(MS_SNSD_CSV.open(encoding="utf-8")))
    for noise, snr, title in MS_SNSD_PICKS:
        for row in rows:
            if row["noise_type"] != noise or float(row["snr_db"]) != snr:
                continue
            noisy_p, clean_p = _local(row["noisy"]), _local(row["clean"])
            if not (noisy_p.is_file() and clean_p.is_file()):
                continue
            noisy, _ = load_audio(noisy_p, sr=DEFAULT_SR, mono=True)
            if noisy.size / DEFAULT_SR < MIN_S:
                continue
            clean, _ = load_audio(clean_p, sr=DEFAULT_SR, mono=True)
            crop = int(CROP_S * DEFAULT_SR)
            noisy, clean = noisy[:crop], clean[:crop]
            sample_id = f"mssnsd_{noise.lower()}_{'m' if snr < 0 else ''}{abs(int(snr))}db"
            _write_pair(sample_id, noisy, clean)
            library.append({
                "id": sample_id, "title": title, "dataset": "MS-SNSD (unseen corpus)",
                "noise": noise, "snr_db": snr, "duration_s": round(noisy.size / DEFAULT_SR, 2),
            })
            print(f"  {sample_id:34s} {noisy_p.name}")
            break
        else:
            print(f"  (no {noise} @ {snr} dB clip found)")

    for stem, title in VBD_PICKS:
        noisy_p, clean_p = VBD_DIR / "noisy" / f"{stem}.wav", VBD_DIR / "clean" / f"{stem}.wav"
        if not (noisy_p.is_file() and clean_p.is_file()):
            print(f"  (missing VoiceBank {stem})")
            continue
        noisy, _ = load_audio(noisy_p, sr=DEFAULT_SR, mono=True)
        clean, _ = load_audio(clean_p, sr=DEFAULT_SR, mono=True)
        sample_id = f"vbd_{stem}"
        _write_pair(sample_id, noisy, clean)
        library.append({
            "id": sample_id, "title": title, "dataset": "VoiceBank-DEMAND test",
            "noise": "DEMAND (mixed)", "snr_db": None, "duration_s": round(noisy.size / DEFAULT_SR, 2),
        })
        print(f"  {sample_id:34s} {noisy_p.name}")

    (SAMPLES / "library.json").write_text(json.dumps(library, indent=2), encoding="utf-8")
    print(f"\n{len(library)} samples -> {SAMPLES / 'library.json'}")


if __name__ == "__main__":
    main()
