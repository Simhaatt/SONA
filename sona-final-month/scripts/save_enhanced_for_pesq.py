#!/usr/bin/env python3
"""Enhance the VoiceBank test set with the trained A/B checkpoints and save wavs.

PESQ cannot be built on this Windows box (no C compiler), but it installs cleanly
in WSL. So we save (clean, enhanced) wav pairs here (CUDA), then score PESQ in WSL
on the saved files. Also copies the clean references once and the raw noisy input
(so PESQ-of-noisy gives the unprocessed baseline).

    python scripts/save_enhanced_for_pesq.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# The SONA repo root: scripts/ -> sona-final-month/ -> <repo>.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.dataset import PairDataset  # noqa: E402
from src.models import NeuralEnhancer  # noqa: E402


def load_model(ckpt_path: Path, device: str):
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.config import Config
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(ckpt["cfg"])
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tf


def main():
    import argparse
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--run-dir", type=Path, default=root / "COMPLEX_MASK_FULL5K")
    ap.add_argument("--data", type=Path, default=REPO / "vbd_local")
    ap.add_argument("--out", type=Path, default=root / "COMPLEX_MASK_FULL5K" / "pesq_wavs")
    ap.add_argument("--limit", type=int, default=824)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = list(PairDataset(clean_dir=args.data / "test" / "clean",
                            noisy_dir=args.data / "test" / "noisy", limit=args.limit))
    print(f"device={device}  test clips={len(test)}")

    ckpts = {
        "magnitude_gtcrn": args.run_dir / "ckpt_gtcrn" / "last.pt",
        "complex_gtcrn": args.run_dir / "ckpt_complex_gtcrn" / "last.pt",
    }
    (args.out / "clean").mkdir(parents=True, exist_ok=True)
    (args.out / "noisy").mkdir(parents=True, exist_ok=True)

    # save clean + noisy once (resumable: skip existing)
    for i, s in enumerate(test):
        uid = s.uid or f"clip{i}"
        for sub, wav in (("clean", s.clean), ("noisy", s.noisy)):
            p = args.out / sub / f"{uid}.wav"
            if not p.exists():
                sf.write(p, np.asarray(wav, dtype=np.float32), s.sr)
    print("clean/noisy references ready", flush=True)

    for name, ckpt in ckpts.items():
        d = args.out / name
        d.mkdir(parents=True, exist_ok=True)
        pending = [(i, s) for i, s in enumerate(test)
                   if not (d / f"{(s.uid or f'clip{i}')}.wav").exists()]
        if not pending:
            print(f"{name}: already complete ({len(test)} wavs)", flush=True)
            continue
        model, tf = load_model(ckpt, device)
        enh = NeuralEnhancer(model, tf, device=device, name=name)
        print(f"{name}: enhancing {len(pending)} remaining clips ...", flush=True)
        for n, (i, s) in enumerate(pending):
            uid = s.uid or f"clip{i}"
            y = enh.enhance(s.noisy, s.sr)
            sf.write(d / f"{uid}.wav", np.asarray(y, dtype=np.float32), s.sr)
            if (n + 1) % 100 == 0:
                print(f"  {name}: {n+1}/{len(pending)}", flush=True)
        print(f"saved enhanced wavs for {name} -> {d}", flush=True)

    print(f"\ndone. clean/noisy/enhanced wavs under {args.out}")


if __name__ == "__main__":
    main()
