#!/usr/bin/env python3
"""Headline figures for the FYP results deck.

  1. benchmark_bar.png   — SI-SDR improvement by method (classical vs learned)
  2. before_after.png    — noisy input / learned output / clean target spectrograms
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_fyp"
OUT.mkdir(parents=True, exist_ok=True)
WAVS = ROOT / "COMPLEX_MASK_FULL5K" / "pesq_wavs"

DEEP, TEAL, AQUA, CORAL, NAVY, INK, GREY = "#0B5563", "#1C7293", "#2A9D8F", "#E07A5F", "#1B2A4A", "#22333B", "#9AA9B1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold",
                     "axes.edgecolor": "#B8C6CE", "figure.dpi": 150, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": "#5A6B73", "ytick.color": "#5A6B73"})

# ---------------- FIG 1: benchmark bar ----------------
rows = list(csv.DictReader(open(ROOT / "FYP_BENCHMARK" / "summary.csv")))
order = [("hard_threshold", "Hard\nthreshold"), ("soft_threshold", "Soft\nthreshold"),
         ("spectral_subtraction", "Spectral\nsubtraction"), ("wiener", "Wiener\nfilter"),
         ("learned_gtcrn", "Learned\nmodel (ours)")]
vals = {r["method"]: float(r["si_sdr_i"]) for r in rows}
labels = [l for _, l in order]
y = [vals[m] for m, _ in order]
colors = [CORAL if v < 0 else AQUA for v in y]
colors[-1] = AQUA

fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
bars = ax.bar(labels, y, color=colors, width=0.62, edgecolor="white", linewidth=1)
ax.axhline(0, color=INK, lw=1.2)
ax.set_ylabel("SI-SDR improvement (dB)  — higher is better")
ax.set_title("Only the learned model improves the signal on real noise\n"
             "(824 VoiceBank-DEMAND test clips)", fontsize=13, color=INK)
for b, v in zip(bars, y):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.35 if v >= 0 else -0.75),
            f"{v:+.1f}", ha="center", fontweight="bold",
            color=(AQUA if v >= 0 else CORAL), fontsize=13)
ax.text(1.5, -4.8, "classical DSP baselines degrade the signal\n(strip weak speech on non-stationary noise)",
        ha="center", fontsize=10, color=CORAL, style="italic")
ax.set_ylim(-7.5, 11.5)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "benchmark_bar.png", bbox_inches="tight")
plt.close(fig)

# ---------------- FIG 2: before / after spectrograms ----------------
CLIP = "p257_255"
SR, NFFT, HOP = 16000, 512, 128


def load(m):
    x, _ = sf.read(WAVS / m / f"{CLIP}.wav", dtype="float64")
    return x.mean(1) if x.ndim > 1 else x


def spec(x):
    f, t, Z = stft(x, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")
    return f, t, 20 * np.log10(np.abs(Z) + 1e-6)


sig = {k: load(v) for k, v in [("noisy", "noisy"), ("learned", "magnitude_gtcrn"), ("clean", "clean")]}
n = min(len(v) for v in sig.values())
S = {k: spec(v[:n]) for k, v in sig.items()}
vmax = max(Z.max() for _, _, Z in S.values())
vmin = vmax - 70

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
for ax, (key, ttl) in zip(axes, [("noisy", "Noisy input"),
                                  ("learned", "Learned model output"),
                                  ("clean", "Clean target")]):
    f, t, Z = S[key]
    im = ax.pcolormesh(t, f / 1000, Z, cmap="magma", vmin=vmin, vmax=vmax, shading="auto")
    ax.set_title(ttl); ax.set_ylim(0, 8); ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (kHz)")
fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, label="Magnitude (dB)")
fig.suptitle("Before and after: broadband noise removed, speech harmonics preserved",
             fontsize=13, fontweight="bold", color=INK)
fig.savefig(OUT / "before_after.png", bbox_inches="tight")
plt.close(fig)

print("wrote 2 FYP figures to", OUT)
