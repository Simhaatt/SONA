"""Enhance one audio file with the locked Layer A checkpoint plus Layer B controls."""

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
    ap.add_argument("--checkpoint", required=True, help="supervised checkpoint .pt")
    ap.add_argument("--input", required=True, help="input WAV/audio file")
    ap.add_argument("--output", required=True, help="output WAV file")
    ap.add_argument("--mode", default=None,
                    help="speech | music | podcast | gaming | hearing | studio")
    ap.add_argument("--scene", default=None,
                    help="speech | music | traffic | fan | crowd | mixed | unknown")
    ap.add_argument("--auto-scene", action="store_true",
                    help="infer scene from input audio when --scene is not provided")
    ap.add_argument("--denoise-strength", type=float, default=None,
                    help="override mode denoise blend, usually 0.0..1.0")
    ap.add_argument("--safety-floor", type=float, default=None,
                    help="recorded controller value for future mask-floor control")
    ap.add_argument("--memory", default=None,
                    help="preference JSON path; enables scene preference lookup")
    ap.add_argument("--remember", action="store_true",
                    help="save the chosen mode/settings for --scene after enhancement")
    ap.add_argument("--ignore-memory", action="store_true",
                    help="do not apply stored scene preferences")
    ap.add_argument("--device", default=None, help="cpu | cuda; default auto")
    args = ap.parse_args()

    _add_repo_to_path(args.repo)
    from src.controller import enhance_file

    info = enhance_file(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        scene=args.scene,
        denoise_strength=args.denoise_strength,
        safety_floor=args.safety_floor,
        memory_path=args.memory,
        remember=args.remember,
        use_memory=not args.ignore_memory,
        auto_scene=args.auto_scene,
        device=args.device,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
