#!/usr/bin/env python3
"""Run SpeechBrain MetricGAN+ on the saved noisy wavs. RUN INSIDE WSL.

    wsl -d Ubuntu-24.04 -u root -- /opt/sbenv/bin/python \
        /mnt/e/Rohit/FYP/sona-final-month/scripts/run_metricgan_wsl.py

Reads pesq_wavs/noisy/*.wav, enhances with MetricGAN+ (trained on VoiceBank-DEMAND),
writes pesq_wavs/metricgan_plus/*.wav. Model is cached on the Linux side
(/root/mgan) so the SpeechBrain symlink/copy step does not touch the Windows fs.
"""
import os

import soundfile as sf
import torch
from speechbrain.inference.enhancement import SpectralMaskEnhancement
from speechbrain.utils.fetching import LocalStrategy

WAVS = os.environ.get("PESQ_WAVS",
                      "/mnt/e/Rohit/FYP/sona-final-month/COMPLEX_MASK_FULL5K/pesq_wavs")


def main():
    print("loading MetricGAN+ ...", flush=True)
    enh = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="/root/mgan",
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )
    noisy_dir = os.path.join(WAVS, "noisy")
    out_dir = os.path.join(WAVS, "metricgan_plus")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(noisy_dir) if f.endswith(".wav"))
    pending = [f for f in files if not os.path.exists(os.path.join(out_dir, f))]
    print(f"{len(files)} clips, {len(pending)} pending", flush=True)

    for n, fname in enumerate(pending):
        wav, sr = sf.read(os.path.join(noisy_dir, fname), dtype="float32")
        x = torch.from_numpy(wav).unsqueeze(0)
        with torch.no_grad():
            y = enh.enhance_batch(x, lengths=torch.tensor([1.0]))
        y = y.squeeze(0).cpu().numpy()
        sf.write(os.path.join(out_dir, fname), y, sr)
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(pending)}", flush=True)
    print(f"saved MetricGAN+ wavs -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
