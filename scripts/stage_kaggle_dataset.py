"""Build a single self-contained folder to upload to Kaggle as one dataset.

Why this exists. The manifests carry absolute Windows paths from this machine, and the earlier
notebook tried to fix that by globbing the Kaggle input tree once per file. With 3 848 audio files
each glob walks the entire tree, which is quadratic and would dominate the session.

Instead the audio is copied into a known layout and the manifests are rewritten to paths *relative*
to the dataset root. On Kaggle a single string prefix turns them into absolute paths -- one pass,
no globbing.

Layout produced:

    kaggle_upload/
      repo/                     the SONA source tree (code only)
      ckpt/last.pt              locked GTCRN checkpoint
      onnx/model_{fp32,int8,mixed}.onnx
      vbd/test/{noisy,clean}/   824 x 2 VoiceBank test clips
      vbd/test_manifest.csv     relative paths
      ms_snsd/{clean,noisy}/    1100 x 2 MS-SNSD clips
      ms_snsd/test.csv          relative paths

Usage:
    python scripts/stage_kaggle_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO_INCLUDE = ["src", "scripts", "configs"]
REPO_EXCLUDE = {"__pycache__", ".pytest_cache"}


def copy_repo(dst: Path) -> int:
    n = 0
    for top in REPO_INCLUDE:
        srcdir = ROOT / top
        if not srcdir.exists():
            continue
        for p in srcdir.rglob("*"):
            if p.is_dir() or any(x in p.parts for x in REPO_EXCLUDE):
                continue
            if p.suffix in {".pyc", ".pyo"}:
                continue
            rel = p.relative_to(ROOT)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            n += 1
    return n


def stage_manifest(rows: list[dict], cols: list[str], dst_root: Path, subdir: str,
                   fields: dict[str, str], out_csv: Path) -> tuple[int, int]:
    """Copy the audio named in `fields` and rewrite those columns to dataset-relative paths."""
    copied = missing = 0
    for r in rows:
        for col, bucket in fields.items():
            src = Path(r[col])
            if not src.exists():
                missing += 1
                continue
            rel = Path(subdir) / bucket / src.name
            out = dst_root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                shutil.copy2(src, out)
                copied += 1
            r[col] = rel.as_posix()          # relative, forward slashes
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return copied, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "kaggle_upload"))
    ap.add_argument("--limit-vbd", type=int, default=0, help="0 = all 824")
    ap.add_argument("--limit-ms", type=int, default=0, help="0 = all 1100")
    args = ap.parse_args()

    dst = Path(args.out)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    print("copying repo source ...")
    n = copy_repo(dst / "repo")
    print(f"  {n} files")

    print("copying checkpoint + onnx ...")
    (dst / "ckpt").mkdir()
    shutil.copy2(ROOT / "results" / "kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07" /
                 "extracted" / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25" / "last.pt",
                 dst / "ckpt" / "last.pt")
    (dst / "onnx").mkdir()
    for name in ("model_fp32.onnx", "model_int8.onnx", "model_mixed.onnx"):
        shutil.copy2(ROOT / "results" / "onnx" / "gtcrn_variants" / name, dst / "onnx" / name)

    print("staging VoiceBank test set ...")
    rows = list(csv.DictReader((ROOT / "vbd_local" / "test_manifest.csv")
                               .open(newline="", encoding="utf-8")))
    if args.limit_vbd:
        rows = rows[:args.limit_vbd]
    cols = list(rows[0].keys())
    c, m = stage_manifest(rows, cols, dst, "vbd/test",
                          {"noisy": "noisy", "clean": "clean"},
                          dst / "vbd" / "test_manifest.csv")
    print(f"  {len(rows)} rows, {c} files copied, {m} missing")

    print("staging MS-SNSD test set ...")
    rows = list(csv.DictReader((ROOT / "external_data" / "ms_snsd_sona" / "test.csv")
                               .open(newline="", encoding="utf-8")))
    if args.limit_ms:
        rows = rows[:args.limit_ms]
    cols = list(rows[0].keys())
    c, m = stage_manifest(rows, cols, dst, "ms_snsd",
                          {"clean": "clean", "noisy": "noisy"},
                          dst / "ms_snsd" / "test.csv")
    print(f"  {len(rows)} rows, {c} files copied, {m} missing")

    total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    files = sum(1 for p in dst.rglob("*") if p.is_file())
    print(f"\nstaged {files} files, {total/1e6:.0f} MB -> {dst}")
    print("\nUpload that folder to Kaggle as ONE dataset, then run sona_phase_a_kaggle.ipynb.")


if __name__ == "__main__":
    main()
