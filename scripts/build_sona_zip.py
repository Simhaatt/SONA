#!/usr/bin/env python3
"""Package the SONA repo into sona.zip for upload as a Kaggle dataset.

Includes the code the Kaggle notebooks need (src/, scripts/, configs/, tests/,
notebooks/, top-level docs) and EXCLUDES heavy artifacts (datasets, the 16 GB
VoiceBank zip, kaggle result folders, checkpoints, caches). Run after changing
any src/ or scripts/ code that a Kaggle run depends on.

    python scripts/build_sona_zip.py
"""
from __future__ import annotations

import fnmatch
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sona.zip"

INCLUDE_DIRS = ["src", "scripts", "configs", "tests", "notebooks"]
INCLUDE_ROOT_FILES = ["README.md", "SONA_STATUS.md", "LAYER_A_RESULTS.md",
                      "NEXT_STEPS_PLAN.md", "requirements.txt", "pyproject.toml",
                      ".gitignore"]
# root notebooks the Kaggle workflow uses
INCLUDE_ROOT_GLOBS = ["sona_*.ipynb"]

EXCLUDE_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints", "external_data",
                 "results", "checkpoints", ".pytest_cache", "vbd_local"}
EXCLUDE_SUFFIX = {".pyc", ".zip", ".wav", ".flac", ".pt", ".onnx", ".tflite"}


def keep(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIX:
        return False
    return True


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name in INCLUDE_ROOT_FILES:
            p = ROOT / name
            if p.is_file():
                z.write(p, p.relative_to(ROOT).as_posix())
                n += 1
        for pattern in INCLUDE_ROOT_GLOBS:
            for p in sorted(ROOT.glob(pattern)):
                if p.is_file():
                    z.write(p, p.relative_to(ROOT).as_posix())
                    n += 1
        for d in INCLUDE_DIRS:
            base = ROOT / d
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*")):
                if p.is_file() and keep(p):
                    z.write(p, p.relative_to(ROOT).as_posix())
                    n += 1
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT} ({size_mb:.1f} MB, {n} files)")
    # sanity: confirm the key new pieces are present
    with zipfile.ZipFile(OUT) as z:
        names = z.namelist()
    for must in ("scripts/complex_mask_full.py", "src/models/gtcrn.py",
                 "src/train/losses.py", "src/models/torch_frontend.py"):
        print(f"  {'OK' if must in names else 'MISSING'}  {must}")


if __name__ == "__main__":
    main()
