#!/usr/bin/env python3
"""Build a real-audio manifest for the scene classifier from MS-SNSD.

Plan item (Week 2, item 5) / SONA_STATUS open item 7: the scene classifier has
only ever been scored on synthetic self-test data. MS-SNSD encodes the true
noise type in each filename, so we can assemble a genuinely labelled real-audio
set without waiting on hand-recorded clips.

Label policy — only categories whose SONA scene label is unambiguous are used
for the headline accuracy number:

    fan     steady mechanical / HVAC hum   (AirConditioner, VacuumCleaner,
                                            WasherDryer, Washing)
    crowd   multi-talker babble            (Babble, CafeTeria, Cafe,
                                            Restaurant, Square, Park)
    traffic vehicular / transit            (Traffic, Car, Bus, Metro, Station)
    speech  clean speech                   (clean_test/clnsp*.wav)

Deliberately EXCLUDED from the scored set (kept via --include-ambiguous for a
separate confusion analysis) because their ground-truth SONA label is genuinely
arguable, not because they score badly:

    NeighborSpeaking, AirportAnnouncements   speech-like but contextually noise
    Typing, ShuttingDoor, CopyMachine,       transient / impulsive events with
    SqueakyChair, Munching                   no clean SONA scene label
    Office, Hallway, Kitchen, LivingRoom,    room tone, multi-source
    Field

`music` has no MS-SNSD source, so it is absent from the scored set — reported
as a coverage gap rather than silently counted as a miss.

    python scripts/build_scene_manifest.py --out recordings/manifest_msnsd.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# category stem -> SONA scene label (unambiguous only)
UNAMBIGUOUS = {
    "AirConditioner": "fan",
    "VacuumCleaner": "fan",
    "WasherDryer": "fan",
    "Washing": "fan",
    "Babble": "crowd",
    "CafeTeria": "crowd",
    "Cafe": "crowd",
    "Restaurant": "crowd",
    "Square": "crowd",
    "Park": "crowd",
    "Traffic": "traffic",
    "Car": "traffic",
    "Bus": "traffic",
    "Metro": "traffic",
    "Station": "traffic",
}

# kept out of the scored set; ground truth is arguable
AMBIGUOUS = {
    "NeighborSpeaking": "speech",
    "Neighbor": "speech",
    "AirportAnnouncements": "speech",
    "AirportAnnouncement": "speech",
    "Typing": "unknown",
    "ShuttingDoor": "unknown",
    "CopyMachine": "unknown",
    "SqueakyChair": "unknown",
    "Munching": "unknown",
    "Office": "unknown",
    "Hallway": "unknown",
    "Kitchen": "unknown",
    "LivingRoom": "unknown",
    "Field": "unknown",
}

STEM_RE = re.compile(r"^(.*?)_?\d*$")


def category_of(path: Path) -> str:
    m = STEM_RE.match(path.stem)
    return (m.group(1) if m else path.stem).rstrip("_")


def collect(msnsd: Path, n_speech: int, include_ambiguous: bool):
    rows = []
    mapping = dict(UNAMBIGUOUS)
    if include_ambiguous:
        mapping.update(AMBIGUOUS)

    for sub in ("noise_test", "noise_train"):
        d = msnsd / sub
        if not d.is_dir():
            continue
        for wav in sorted(d.glob("*.wav")):
            label = mapping.get(category_of(wav))
            if label:
                rows.append((str(wav), label, category_of(wav)))

    clean = msnsd / "clean_test"
    if clean.is_dir():
        for wav in sorted(clean.glob("*.wav"))[:n_speech]:
            rows.append((str(wav), "speech", "CleanSpeech"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msnsd", type=Path,
                    default=Path(__file__).resolve().parents[2] / "external_data" / "MS-SNSD")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "recordings"
                    / "manifest_msnsd.csv")
    ap.add_argument("--n-speech", type=int, default=30,
                    help="clean-speech clips to include (they are plentiful; "
                         "cap them so speech does not dominate the set)")
    ap.add_argument("--include-ambiguous", action="store_true",
                    help="also emit the arguable categories (confusion analysis)")
    args = ap.parse_args()

    if not args.msnsd.is_dir():
        raise SystemExit(f"MS-SNSD not found: {args.msnsd}")

    rows = collect(args.msnsd, args.n_speech, args.include_ambiguous)
    if not rows:
        raise SystemExit("no clips collected — check --msnsd path")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source_category"])
        w.writerows(rows)

    counts: dict[str, int] = {}
    for _, label, _ in rows:
        counts[label] = counts.get(label, 0) + 1
    print(f"wrote {args.out} ({len(rows)} clips)")
    for label in sorted(counts):
        print(f"  {label:8s} {counts[label]:4d}")
    if "music" not in counts:
        print("  note: 'music' has no MS-SNSD source — coverage gap, not a miss")


if __name__ == "__main__":
    main()
