#!/usr/bin/env python3
"""Generate the four analysis figures for the complex-mask study.

  1. spectrogram_comparison.png  — noisy / clean / magnitude / complex
  2. error_spectrogram.png       — |output - clean| for magnitude vs complex
  3. phase_analysis.png          — the phase correction the complex mask applies
                                   vs the ideal correction, + phase-error summary
  4. metric_correlation.png      — per-clip SI-SDR change vs PESQ change (the
                                   trade-off) + SI-SDR vs PESQ weak correlation

Figures 1-3 use one representative clip; figure 4 uses all 824 clips.

    python scripts/make_analysis_figures.py
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
WAVS = ROOT / "COMPLEX_MASK_FULL5K" / "pesq_wavs"
OUT = ROOT / "figures_analysis"
OUT.mkdir(parents=True, exist_ok=True)
CLIP = "p257_255"           # +3.9 dB SI-SDR advantage for complex, moderate SNR
SR = 16000
NFFT, HOP = 512, 128

DEEP, TEAL, AQUA, CORAL, NAVY, INK = "#0B5563", "#1C7293", "#2A9D8F", "#E07A5F", "#1B2A4A", "#22333B"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.edgecolor": "#B8C6CE", "figure.dpi": 150,
    "axes.titlecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": "#5A6B73", "ytick.color": "#5A6B73",
})


def load(method):
    x, _ = sf.read(WAVS / method / f"{CLIP}.wav", dtype="float64")
    return x.mean(1) if x.ndim > 1 else x


def load_uid(method, u):
    x, _ = sf.read(WAVS / method / f"{u}.wav", dtype="float64")
    return x.mean(1) if x.ndim > 1 else x


def spec(x):
    f, t, Z = stft(x, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, window="hann")
    return f, t, Z


def db(Z):
    return 20 * np.log10(np.abs(Z) + 1e-6)


# ---- align all four signals to common length ----
sigs = {m: load(m) for m in ("noisy", "clean", "magnitude_gtcrn", "complex_gtcrn")}
n = min(len(v) for v in sigs.values())
sigs = {k: v[:n] for k, v in sigs.items()}
specs = {k: spec(v) for k, v in sigs.items()}
f = specs["clean"][0]
t = specs["clean"][1]

# common dB scale for comparability
allmax = max(db(Z).max() for _, _, Z in specs.values())
VMIN, VMAX = allmax - 70, allmax


# ============================================ FIG 1: spectrogram comparison
fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), constrained_layout=True)
panels = [("noisy", "Noisy input", axes[0, 0]),
          ("clean", "Clean reference (target)", axes[0, 1]),
          ("magnitude_gtcrn", "Magnitude mask (ours)", axes[1, 0]),
          ("complex_gtcrn", "Complex mask (ours)", axes[1, 1])]
im = None
for key, ttl, ax in panels:
    _, _, Z = specs[key]
    im = ax.pcolormesh(t, f / 1000, db(Z), cmap="magma", vmin=VMIN, vmax=VMAX, shading="auto")
    ax.set_title(ttl)
    ax.set_ylim(0, 8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (kHz)")
cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
cbar.set_label("Magnitude (dB)")
fig.suptitle(f"Spectrogram comparison — clip {CLIP} (SI-SDR 11.2 → 15.1 dB)",
             fontsize=14, fontweight="bold", color=INK)
fig.savefig(OUT / "spectrogram_comparison.png", bbox_inches="tight")
plt.close(fig)

# ============================================ FIG 2: error spectrograms
C = specs["clean"][2]
errs = {"magnitude_gtcrn": np.abs(specs["magnitude_gtcrn"][2] - C),
        "complex_gtcrn": np.abs(specs["complex_gtcrn"][2] - C)}
emax = max(20 * np.log10(e + 1e-6).max() for e in errs.values())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for ax, (key, ttl) in zip(axes, [("magnitude_gtcrn", "Magnitude mask — error vs clean"),
                                  ("complex_gtcrn", "Complex mask — error vs clean")]):
    E = 20 * np.log10(errs[key] + 1e-6)
    im = ax.pcolormesh(t, f / 1000, E, cmap="inferno", vmin=emax - 60, vmax=emax, shading="auto")
    tot = 10 * np.log10((errs[key] ** 2).sum())
    ax.set_title(f"{ttl}\n(total error energy {tot:.1f} dB)")
    ax.set_ylim(0, 8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (kHz)")
fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, label="|output − clean| (dB)")
fig.suptitle("Reconstruction error — the complex mask leaves less residual",
             fontsize=14, fontweight="bold", color=INK)
fig.savefig(OUT / "error_spectrogram.png", bbox_inches="tight")
plt.close(fig)

# ============================================ FIG 3: phase analysis (honest)
import glob
Yn = specs["noisy"][2]
X = specs["complex_gtcrn"][2]
M = specs["magnitude_gtcrn"][2]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# (a) representative clip: does the applied phase shift point toward clean?
mask = np.abs(C) > (np.abs(C).max() * 0.05)
ideal_v = wrap(np.angle(C) - np.angle(Yn))[mask]      # ideal correction
applied_v = wrap(np.angle(X) - np.angle(Yn))[mask]    # what complex applies
r_phase = np.corrcoef(applied_v, ideal_v)[0, 1]

# (b) aggregate phase error to clean over many clips
uids_agg = [Path(p).stem for p in glob.glob(str(WAVS / "clean" / "*.wav"))][:80]
em, ec = [], []
for u in uids_agg:
    try:
        cc = spec(load_uid("clean", u))[2]; yy = spec(load_uid("noisy", u))[2]
        mm = spec(load_uid("magnitude_gtcrn", u))[2]; xx = spec(load_uid("complex_gtcrn", u))[2]
    except Exception:
        continue
    k = min(cc.shape[1], yy.shape[1], mm.shape[1], xx.shape[1])
    cc, yy, mm, xx = cc[:, :k], yy[:, :k], mm[:, :k], xx[:, :k]
    w = np.abs(cc)
    em.append(np.sum(w * np.abs(wrap(np.angle(mm) - np.angle(cc)))) / w.sum())
    ec.append(np.sum(w * np.abs(wrap(np.angle(xx) - np.angle(cc)))) / w.sum())
em, ec = np.array(em), np.array(ec)
pct_lower = 100 * np.mean(ec < em)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
# panel A: applied vs ideal density
hb = ax1.hexbin(ideal_v, applied_v, gridsize=45, cmap="viridis", mincnt=1, bins="log")
lim = np.pi
ax1.plot([-lim, lim], [-lim, lim], "--", color=CORAL, lw=1.5, label="perfect correction")
ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
ax1.set_xlabel("Ideal phase shift toward clean (rad)")
ax1.set_ylabel("Phase shift the complex mask applies (rad)")
ax1.set_title(f"Corrections point the right way (r = {r_phase:.2f})")
ax1.legend(loc="upper left", frameon=False, fontsize=9)
fig.colorbar(hb, ax=ax1, shrink=0.8, label="T-F bins (log)")
# panel B: aggregate phase error, small but consistent
ax2.bar(["Magnitude\nmask", "Complex\nmask"], [em.mean(), ec.mean()], color=[TEAL, AQUA], width=0.55)
ax2.set_ylabel("Mean phase error to clean (rad)")
ax2.set_title(f"Small but consistent\n(complex lower on {pct_lower:.0f}% of {len(em)} clips)")
for i, v in enumerate([em.mean(), ec.mean()]):
    ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", fontweight="bold", color=INK)
ax2.set_ylim(0, max(em.mean(), ec.mean()) * 1.18)
fig.suptitle("The complex mask makes genuine — but small — phase corrections",
             fontsize=14, fontweight="bold", color=INK)
fig.savefig(OUT / "phase_analysis.png", bbox_inches="tight")
plt.close(fig)
print(f"phase: applied-vs-ideal r={r_phase:.2f}; error mag={em.mean():.3f} cx={ec.mean():.3f} "
      f"({pct_lower:.0f}% of clips lower)")

# ============================================ FIG 4: metric correlation / trade-off
ps = list(csv.DictReader(open(ROOT / "COMPLEX_MASK_FULL5K" / "per_sample.csv")))
pq = list(csv.DictReader(open(WAVS / "pesq_per_clip.csv")))
sisdr = {m: {r["uid"]: float(r["si_sdr_i"]) for r in ps if r["method"] == m}
         for m in ("magnitude_gtcrn", "complex_gtcrn")}
pesq = {m: {r["uid"]: float(r["pesq"]) for r in pq if r["method"] == m and r["pesq"] not in ("", "None")}
        for m in ("magnitude_gtcrn", "complex_gtcrn")}
uids = sorted(set(sisdr["complex_gtcrn"]) & set(pesq["complex_gtcrn"])
              & set(sisdr["magnitude_gtcrn"]) & set(pesq["magnitude_gtcrn"]))
d_sisdr = np.array([sisdr["complex_gtcrn"][u] - sisdr["magnitude_gtcrn"][u] for u in uids])
d_pesq = np.array([pesq["complex_gtcrn"][u] - pesq["magnitude_gtcrn"][u] for u in uids])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
# (a) trade-off scatter: per-clip Δ
ax1.axhline(0, color="#B8C6CE", lw=1); ax1.axvline(0, color="#B8C6CE", lw=1)
ax1.scatter(d_sisdr, d_pesq, s=14, alpha=0.5, color=DEEP, edgecolors="none")
ax1.scatter([d_sisdr.mean()], [d_pesq.mean()], s=160, color=CORAL, marker="X",
            edgecolors="white", linewidths=1.5, zorder=5, label="mean")
ax1.set_xlabel("SI-SDRi change  (complex − magnitude, dB)")
ax1.set_ylabel("PESQ change  (complex − magnitude)")
ax1.set_title("Per-clip trade-off")
ax1.legend(loc="upper right", frameon=False)
ax1.text(0.03, 0.05, f"mean: +{d_sisdr.mean():.2f} dB SI-SDR,  {d_pesq.mean():+.3f} PESQ",
         transform=ax1.transAxes, fontsize=9, color="#5A6B73")
# (b) SI-SDR vs PESQ absolute (both methods) — weak correlation explains disagreement
for m, col, lab in [("magnitude_gtcrn", TEAL, "magnitude"), ("complex_gtcrn", AQUA, "complex")]:
    xs = np.array([sisdr[m][u] for u in uids]); ys = np.array([pesq[m][u] for u in uids])
    ax2.scatter(xs, ys, s=12, alpha=0.4, color=col, edgecolors="none", label=lab)
allx = np.array([sisdr["complex_gtcrn"][u] for u in uids])
ally = np.array([pesq["complex_gtcrn"][u] for u in uids])
r = np.corrcoef(allx, ally)[0, 1]
ax2.set_xlabel("SI-SDRi (dB)")
ax2.set_ylabel("PESQ")
ax2.set_title(f"SI-SDR vs PESQ are only weakly correlated (r = {r:.2f})")
ax2.legend(loc="lower right", frameon=False)
fig.suptitle("Why the metrics disagree: fidelity gains do not track perceptual score",
             fontsize=14, fontweight="bold", color=INK)
fig.savefig(OUT / "metric_correlation.png", bbox_inches="tight")
plt.close(fig)

print("wrote 4 figures to", OUT)
print(f"per-clip trade-off mean: +{d_sisdr.mean():.2f} dB SI-SDR, {d_pesq.mean():+.3f} PESQ")
print(f"SI-SDR vs PESQ correlation r={r:.2f}")
