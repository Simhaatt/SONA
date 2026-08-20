"""Corpus locations, resolved once.

Scripts previously hardcoded absolute paths to the machine this work was developed on, which made
the pipeline unreproducible elsewhere. Paths now come from environment variables when set, and
fall back to the development locations so existing invocations keep working.

Set before running on another machine:

    AUDIOVQ_MS_SNSD        root of the MS-SNSD corpus (contains clean_train/, clean_test/, ...)
    AUDIOVQ_VBD_TEST_CLEAN VoiceBank-DEMAND test clean directory (speaker-labelled filenames)
    AUDIOVQ_NOISY_MANIFEST CSV with a `noisy` column and optional `noise_type` / `snr_db`

`describe()` prints what resolved and whether it exists, so a misconfigured run fails with a clear
message instead of an empty file list.
"""

from __future__ import annotations

import os
from pathlib import Path

# AudioVQ lives at <SONA_ROOT>/AudioVQ, so the corpora sit alongside it in the same tree.
# Deriving the defaults from this file's location keeps them correct when the tree is moved
# or copied to another drive; the environment variables still override.
_ROOT = Path(__file__).resolve().parents[2]

_DEFAULTS = {
    "MS_SNSD": _ROOT / "external_data" / "MS-SNSD",
    "VBD_TEST_CLEAN": _ROOT / "vbd_local" / "test" / "clean",
    "NOISY_MANIFEST": _ROOT / "PERCEPTUAL_MP" / "results" / "phase3" / "ms_snsd_resolved.csv",
}


def _resolve(key: str) -> Path:
    return Path(os.environ.get(f"AUDIOVQ_{key}", _DEFAULTS[key]))


MS_SNSD = _resolve("MS_SNSD")
VBD_TEST_CLEAN = _resolve("VBD_TEST_CLEAN")
NOISY_MANIFEST = _resolve("NOISY_MANIFEST")


def describe() -> None:
    print("corpus paths:")
    for name, p in (("MS-SNSD", MS_SNSD), ("VoiceBank test clean", VBD_TEST_CLEAN),
                    ("noisy manifest", NOISY_MANIFEST)):
        print(f"  {name:22s} {p}  {'[ok]' if p.exists() else '[MISSING]'}")


def require(*paths: Path) -> None:
    """Fail early and legibly if a corpus a script needs is not where it was told to look."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        lines = "\n".join(f"    {p}" for p in missing)
        raise SystemExit(
            "Required corpus path(s) not found:\n" + lines +
            "\n\nSet AUDIOVQ_MS_SNSD / AUDIOVQ_VBD_TEST_CLEAN / AUDIOVQ_NOISY_MANIFEST "
            "to point at your copies (see src/config.py)."
        )
