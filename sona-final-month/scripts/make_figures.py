#!/usr/bin/env python3
"""Generate thesis/paper figures from the curated tables in FINAL_TABLES/.

    python scripts/make_figures.py --tables-dir FINAL_TABLES --out-dir figures
    # optional before/after spectrogram (16 kHz mono wavs):
    python scripts/make_figures.py --sample-wav noisy.wav --enhanced-wav enhanced.wav

Figures: voicebank_metrics, ms_snsd_external, online_adaptation, rtf_comparison,
architecture_progression, and (if wavs given) spectrogram_before_after.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(tables_dir: Path, name: str):
    p = tables_dir / f"{name}.csv"
    if not p.exists():
        print(f"  ! missing {p}, skipping related figure")
        return None
    return pd.read_csv(p)


def small_multiples(df, metrics, title, out_png, label_col="label"):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    if n == 1:
        axes = [axes]
    labels = df[label_col].tolist()
    x = np.arange(len(labels))
    for ax, m in zip(axes, metrics):
        vals = pd.to_numeric(df[m], errors="coerce").to_numpy()
        bars = ax.bar(x, vals, color="#3b6ea5")
        best = np.nanargmax(vals) if m != "lsd" else np.nanargmin(vals)
        bars[best].set_color("#e07b39")
        ax.set_title(m)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.margins(y=0.15)
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_png.name}")


def grouped_pesq(story, out_png):
    # frozen vs online PESQ, per dataset (shows the VB gain / MS-SNSD reversal)
    data = {"VoiceBank": {}, "MS-SNSD": {}}
    for _, r in story.iterrows():
        lab = r["label"]
        ds = "VoiceBank" if "VoiceBank" in lab else "MS-SNSD"
        kind = "frozen" if "frozen" in lab else "online"
        data[ds].setdefault(kind, r["pesq"])
    fig, ax = plt.subplots(figsize=(6, 4))
    dsets = list(data.keys())
    x = np.arange(len(dsets))
    w = 0.35
    frozen = [data[d].get("frozen", np.nan) for d in dsets]
    online = [data[d].get("online", np.nan) for d in dsets]
    ax.bar(x - w/2, frozen, w, label="frozen", color="#3b6ea5")
    ax.bar(x + w/2, online, w, label="online (TTA)", color="#e07b39")
    ax.set_xticks(x); ax.set_xticklabels(dsets)
    ax.set_ylabel("PESQ"); ax.set_title("Test-time adaptation: PESQ by dataset")
    ax.legend()
    for xi, (f, o) in enumerate(zip(frozen, online)):
        ax.text(xi - w/2, f, f"{f:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w/2, o, f"{o:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


def rtf_fig(vb, out_png):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = vb["label"].tolist()
    vals = pd.to_numeric(vb["rtf"], errors="coerce").to_numpy()
    ax.barh(labels, vals, color="#3b6ea5")
    ax.axvline(1.0, color="red", ls="--", lw=1, label="realtime (RTF=1)")
    ax.set_xlabel("RTF (lower = faster than realtime)")
    ax.set_title("Real-time factor (CPU)")
    ax.legend()
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


def progression_fig(vb, out_png):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    labels = vb["label"].tolist()
    x = np.arange(len(labels))
    si = pd.to_numeric(vb["si_sdr_i"], errors="coerce")
    pq = pd.to_numeric(vb["pesq"], errors="coerce")
    ax1.plot(x, si, "-o", color="#3b6ea5", label="SI-SDRi")
    ax1.set_ylabel("SI-SDRi (dB)", color="#3b6ea5")
    ax2 = ax1.twinx()
    ax2.plot(x, pq, "-s", color="#e07b39", label="PESQ")
    ax2.set_ylabel("PESQ", color="#e07b39")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax1.set_title("Backbone search progression (VoiceBank full-824)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


def spectrogram_fig(noisy, enhanced, out_png):
    from scipy.io import wavfile
    def load_mono(p):
        sr, x = wavfile.read(p)
        x = x.astype(np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        m = np.max(np.abs(x)) or 1.0
        return sr, x / m
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, path) in zip(axes, [("noisy", noisy), ("enhanced", enhanced)]):
        sr, x = load_mono(path)
        ax.specgram(x, NFFT=512, Fs=sr, noverlap=256, cmap="magma")
        ax.set_title(title); ax.set_xlabel("time (s)"); ax.set_ylabel("Hz")
    fig.suptitle("Before / after spectrogram")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_png, dpi=150); plt.close(fig)
    print(f"  wrote {out_png.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "FINAL_TABLES")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
    ap.add_argument("--sample-wav", type=Path, default=None)
    ap.add_argument("--enhanced-wav", type=Path, default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vb = load(args.tables_dir, "voicebank_full824")
    ext = load(args.tables_dir, "ms_snsd_external")
    story = load(args.tables_dir, "online_adaptation_story")

    if vb is not None:
        small_multiples(vb, ["si_sdr_i", "pesq", "lsd"],
                        "VoiceBank full-824: backbone comparison",
                        args.out_dir / "voicebank_metrics.png")
        rtf_fig(vb, args.out_dir / "rtf_comparison.png")
        progression_fig(vb, args.out_dir / "architecture_progression.png")
    if ext is not None:
        small_multiples(ext, ["si_sdr_i", "pesq", "stoi", "lsd"],
                        "MS-SNSD external validation (1100 clips)",
                        args.out_dir / "ms_snsd_external.png")
    if story is not None:
        grouped_pesq(story, args.out_dir / "online_adaptation.png")
    if args.sample_wav and args.enhanced_wav:
        spectrogram_fig(args.sample_wav, args.enhanced_wav,
                        args.out_dir / "spectrogram_before_after.png")
    else:
        print("  (spectrogram skipped: pass --sample-wav and --enhanced-wav)")
    print("done.")


if __name__ == "__main__":
    main()
