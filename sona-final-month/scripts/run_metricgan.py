#!/usr/bin/env python3
"""Enhance the VoiceBank test set with SpeechBrain's pretrained MetricGAN+ .

MetricGAN+ (Fu et al., 2021) is trained ON VoiceBank-DEMAND and is a standard
published baseline for that benchmark, so running it through the same 824-clip
test set gives a real published-method comparison (not just our own variants).

Reads the already-saved noisy wavs, writes enhanced wavs alongside the A/B outputs
so the WSL PESQ scorer picks it up as another method. Resumable.

    python scripts/run_metricgan.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--wavs", type=Path, default=root / "COMPLEX_MASK_FULL5K" / "pesq_wavs")
    ap.add_argument("--name", default="metricgan_plus")
    args = ap.parse_args()

    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}; loading MetricGAN+ (downloads on first run) ...", flush=True)
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir=str(root / "pretrained" / "metricgan-plus-voicebank"),
        run_opts={"device": device},
    )

    noisy_dir = args.wavs / "noisy"
    out_dir = args.wavs / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(noisy_dir.glob("*.wav"))
    pending = [f for f in files if not (out_dir / f.name).exists()]
    print(f"{len(files)} clips, {len(pending)} pending", flush=True)

    for n, f in enumerate(pending):
        wav, sr = sf.read(f, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = torch.from_numpy(wav).unsqueeze(0).to(device)
        with torch.no_grad():
            y = enhancer.enhance_batch(x, lengths=torch.tensor([1.0]))
        y = y.squeeze(0).cpu().numpy().astype(np.float32)
        sf.write(out_dir / f.name, y, sr)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(pending)}", flush=True)
    print(f"saved MetricGAN+ enhanced wavs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
