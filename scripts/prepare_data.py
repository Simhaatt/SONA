"""Prepare / verify the speech-enhancement dataset and emit a CSV manifest.

Primary corpus: **VoiceBank-DEMAND** (Valentini 2017), the standard benchmark. It ships as four
folders: clean_trainset_wav, noisy_trainset_wav, clean_testset_wav, noisy_testset_wav.

This script does NOT auto-download (licences / large files). Instead it:
  * On Kaggle: point --root at the attached dataset directory.
  * Locally: download once and unzip, then point --root at it.
It verifies clean/noisy filename pairing and writes train/test manifests consumed by
`PairDataset(manifest=...)`.

Kaggle dataset slug (attach in a notebook): `VoiceBank-DEMAND` community mirrors exist, e.g.
search "Valentini noisy speech" on Kaggle Datasets, or use the Edinburgh DataShare originals.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PAIRS = [
    ("clean_trainset_wav", "noisy_trainset_wav", "train"),
    ("clean_testset_wav", "noisy_testset_wav", "test"),
    # 28-speaker variants use the same names; some mirrors add _28spk suffixes.
]


def _find(root: Path, name: str) -> Path | None:
    if (root / name).is_dir():
        return root / name
    matches = [p for p in root.rglob(name) if p.is_dir()]
    return matches[0] if matches else None


def build_manifest(root: Path, clean_name: str, noisy_name: str, out_csv: Path) -> int:
    clean_dir = _find(root, clean_name)
    noisy_dir = _find(root, noisy_name)
    if not clean_dir or not noisy_dir:
        print(f"  [skip] {clean_name}/{noisy_name} not found under {root}")
        return 0

    rows = []
    for noisy_path in sorted(noisy_dir.glob("*.wav")):
        clean_path = clean_dir / noisy_path.name
        if clean_path.exists():
            rows.append({"clean": str(clean_path), "noisy": str(noisy_path)})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clean", "noisy"])
        w.writeheader()
        w.writerows(rows)
    print(f"  [ok] {out_csv.name}: {len(rows)} pairs")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the VoiceBank-DEMAND folder")
    ap.add_argument("--out", default="data/manifests")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"root not found: {root}")
    out = ROOT / args.out

    total = 0
    for clean_name, noisy_name, split in PAIRS:
        total += build_manifest(root, clean_name, noisy_name, out / f"{split}.csv")
    if total == 0:
        raise SystemExit(
            "No pairs found. Check --root points at the VoiceBank-DEMAND folder containing "
            "clean_*/noisy_* subdirectories."
        )
    print(f"\nDone. Manifests in {out}. Use with: PairDataset(manifest='{out}/test.csv')")


if __name__ == "__main__":
    main()
