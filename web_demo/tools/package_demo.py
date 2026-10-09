"""Package the web demo as a self-contained zip a friend can run on their own laptop.

Copies only what the website needs, keeping the repo-relative layout `engine.py`
expects: the `src/` package, the 5 model checkpoints, the sample library, the
streaming processor and the frozen result tables — plus one-click launch scripts.

    python web_demo/tools/package_demo.py            # -> SONA_web_demo.zip in the repo root
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

WEB_DEMO = Path(__file__).resolve().parents[1]
REPO = WEB_DEMO.parent
sys.path.insert(0, str(WEB_DEMO))

from engine import ENGINES  # noqa: E402

ROOT = "SONA_web_demo"
OUT = REPO / f"{ROOT}.zip"
SKIP_PARTS = {"__pycache__", ".pytest_cache"}

REQUIREMENTS = """\
# Core (torch is installed separately by the launch script — CPU build)
numpy>=1.24
scipy>=1.10
soundfile>=0.12
librosa>=0.10
PyYAML>=6.0
flask>=3.0
pystoi>=0.4
# Optional: PESQ metric (needs a C compiler on some systems):  pip install pesq
"""

RUN_WINDOWS = r"""@echo off
setlocal
cd /d "%~dp0"
title SONA web demo

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python was not found. Install Python 3.10 - 3.13 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during setup, then run this file again.
  pause & exit /b 1
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || (
  echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
  pause & exit /b 1
)

if not exist ".venv\installed.ok" (
  echo.
  echo  First run: setting up a private Python environment. This downloads ~300 MB
  echo  and takes a few minutes. Later runs start in seconds.
  echo.
  if not exist ".venv" %PY% -m venv .venv || goto :fail
  call ".venv\Scripts\activate.bat"
  python -m pip install --upgrade pip || goto :fail
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :fail
  python -m pip install -r requirements.txt || goto :fail
  echo ok> ".venv\installed.ok"
) else (
  call ".venv\Scripts\activate.bat"
)

echo.
echo  Starting SONA... the website opens in your browser at http://localhost:7860
echo  Keep this window open while using it. Close it (or press Ctrl+C) to stop.
python web_demo\app.py --open
pause
exit /b 0

:fail
echo.
echo  Setup failed - check your internet connection and run this file again.
echo  (Delete the .venv folder first if it keeps failing.)
pause
exit /b 1
"""

RUN_UNIX = """#!/usr/bin/env bash
# SONA web demo launcher for macOS / Linux
set -e
cd "$(dirname "$0")"

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ is required: https://www.python.org/downloads/"; exit 1
fi

if [ ! -f .venv/installed.ok ]; then
  echo
  echo " First run: setting up a private Python environment (~300 MB download, a few minutes)."
  echo
  [ -d .venv ] || "$PY" -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  if [ "$(uname)" = "Darwin" ]; then
    python -m pip install torch
  else
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  python -m pip install -r requirements.txt
  echo ok > .venv/installed.ok
else
  . .venv/bin/activate
fi

echo
echo " Starting SONA... open http://localhost:7860 (Ctrl+C to stop)"
python web_demo/app.py --open
"""

README = """SONA - speech enhancement studio (final-year project demo)
===========================================================

Removes background noise (fans, traffic, babble, typing...) from speech using
small neural networks trained for our final-year project. Everything runs
locally on your laptop - no internet needed after the first setup, nothing is
uploaded anywhere.

HOW TO RUN
----------
1. Unzip this folder anywhere (e.g. Desktop). Don't run it from inside the zip.
2. You need Python 3.10 - 3.13:  https://www.python.org/downloads/
   (Windows: tick "Add python.exe to PATH" in the installer.)
3. Windows:      double-click  run_windows.bat
   macOS/Linux:  open a terminal in this folder and run   bash run_mac_linux.sh
4. The FIRST run installs the libraries it needs (~300 MB download, a few
   minutes). After that it starts in seconds.
5. The website opens automatically at  http://localhost:7860
   Keep the black terminal window open while you use it; close it to stop.

WHAT TO TRY
-----------
* Enhance: pick a clip from the library (e.g. "Vacuum cleaner") and press
  "Enhance". On the Results page, switch Original / Enhanced / Removed / Clean
  while it plays. You can also upload your own audio file.
* Compare: runs all 9 methods (5 neural networks, 4 classic DSP filters) on the
  same clip and ranks them.
* Real-time: simulates running live in small blocks and shows latency.
* About: the project's benchmark results.

TROUBLESHOOTING
---------------
* "Python was not found": install Python (step 2), then run again.
* Setup failed halfway: delete the ".venv" folder and run again.
* Port already in use: another copy is running - close it first.
"""


def _add_file(zf: zipfile.ZipFile, src: Path, arc: str) -> int:
    zf.write(src, f"{ROOT}/{arc}")
    return src.stat().st_size


def _add_tree(zf: zipfile.ZipFile, base: Path, rel: str, suffixes: set[str] | None = None) -> int:
    total = 0
    for path in sorted((base / rel).rglob("*")):
        if not path.is_file() or SKIP_PARTS & set(path.parts):
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        total += _add_file(zf, path, path.relative_to(base).as_posix())
    return total


def main() -> None:
    total = 0
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in ("app.py", "engine.py", "README.md"):
            total += _add_file(zf, WEB_DEMO / name, f"web_demo/{name}")
        total += _add_tree(zf, REPO, "web_demo/static")
        total += _add_tree(zf, REPO, "web_demo/samples", {".wav", ".json"})
        total += _add_tree(zf, REPO, "src", {".py"})
        for spec in ENGINES.values():
            if spec.path is not None:
                total += _add_file(zf, spec.path, spec.path.relative_to(REPO).as_posix())
        total += _add_file(zf, REPO / "sona-final-month/realtime/stream_processor.py",
                           "sona-final-month/realtime/stream_processor.py")
        total += _add_tree(zf, REPO, "sona-final-month/FINAL_TABLES", {".csv"})
        total += _add_file(zf, REPO / "sona-final-month/COMPLEX_MASK_FULL5K/summary.csv",
                           "sona-final-month/COMPLEX_MASK_FULL5K/summary.csv")
        # original (unprocessed) baselines: frozen VoiceBank row + computed MS-SNSD scores
        total += _add_file(zf, REPO / "sona-final-month/FYP_BENCHMARK/summary.csv",
                           "sona-final-month/FYP_BENCHMARK/summary.csv")
        total += _add_tree(zf, REPO, "web_demo/data", {".json"})
        for arc, text in (("requirements.txt", REQUIREMENTS), ("README.txt", README)):
            zf.writestr(f"{ROOT}/{arc}", text)
        # .bat needs CRLF line endings for cmd.exe; the shell script must stay LF + executable
        zf.writestr(f"{ROOT}/run_windows.bat", RUN_WINDOWS.replace("\n", "\r\n"))
        info = zipfile.ZipInfo(f"{ROOT}/run_mac_linux.sh")
        info.external_attr = 0o755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, RUN_UNIX)
        n = len(zf.namelist())
    print(f"{n} files, {total / 1e6:.1f} MB uncompressed -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
