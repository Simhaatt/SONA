"""View and edit SONA Layer B scene preference memory."""

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
    ap.add_argument("--memory", default=None, help="preference JSON path; default is ~/.sona/preferences.json")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="print all remembered scene preferences")

    set_ap = sub.add_parser("set", help="remember a mode/settings for one scene")
    set_ap.add_argument("--scene", required=True)
    set_ap.add_argument("--mode", required=True)
    set_ap.add_argument("--denoise-strength", type=float, default=None)
    set_ap.add_argument("--safety-floor", type=float, default=None)
    set_ap.add_argument("--bypass-eq", action="store_true")
    set_ap.add_argument("--bypass-compressor", action="store_true")

    del_ap = sub.add_parser("delete", help="delete one remembered scene")
    del_ap.add_argument("--scene", required=True)

    sub.add_parser("clear", help="delete all remembered preferences")
    args = ap.parse_args()

    _add_repo_to_path(args.repo)
    from src.controller.memory import DEFAULT_MEMORY_PATH, PreferenceMemory
    from src.controller.modes import SonaController

    memory = PreferenceMemory(args.memory or DEFAULT_MEMORY_PATH)
    controller = SonaController(memory=memory)

    if args.cmd == "list":
        print(json.dumps(memory.to_dict(), indent=2, sort_keys=True))
        return

    if args.cmd == "set":
        settings = controller.settings(
            mode=args.mode,
            denoise_strength=args.denoise_strength,
            safety_floor=args.safety_floor,
            bypass_eq=args.bypass_eq if args.bypass_eq else None,
            bypass_compressor=args.bypass_compressor if args.bypass_compressor else None,
            use_memory=False,
        )
        record = memory.set(args.scene, args.mode, settings)
        print(json.dumps(memory.to_dict()[record.scene], indent=2, sort_keys=True))
        return

    if args.cmd == "delete":
        print(json.dumps({"scene": args.scene, "deleted": memory.delete(args.scene)}, indent=2))
        return

    if args.cmd == "clear":
        memory.clear()
        print(json.dumps({"cleared": True}, indent=2))


if __name__ == "__main__":
    main()
