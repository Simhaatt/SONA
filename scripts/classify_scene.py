"""Classify one audio file into a SONA Layer B scene label."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_repo_to_path(repo: str | None) -> Path:
    root = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="repo root; defaults to this script's repo")
    ap.add_argument("--input", required=True, help="input WAV/audio file")
    args = ap.parse_args()

    _add_repo_to_path(args.repo)
    from src.controller.scene import SceneClassifier
    from src.utils.audio_io import load_audio

    wav, sr = load_audio(args.input, sr=16000, mono=True)
    pred = SceneClassifier(sr=sr).predict(wav, sr=sr)
    print(json.dumps(pred.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
