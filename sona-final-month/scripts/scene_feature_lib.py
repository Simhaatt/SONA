"""Richer scene features + shared loaders for the scene-classifier search.

The shipped `SceneClassifier.features` returns 8 scalars, each a *mean over
frames* — so all temporal dynamics (how steady vs bursty the noise is over time)
are discarded. Steady fan hum vs bursty babble vs passing traffic differ mostly
in their dynamics, so this module adds:

  * temporal spread of each per-frame descriptor (std, IQR)
  * delta features (mean absolute frame-to-frame change)
  * finer band structure + spectral bandwidth, contrast, crest factor
  * modulation percentiles (not just the std/mean ratio)

`base_features` reproduces the shipped 8 exactly, so a model trained on `base`
can be compared apples-to-apples against one trained on the richer set.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

EPS = 1e-8
DEFAULT_SR = 16000

BASE_KEYS = ["centroid_hz", "flatness", "rolloff_hz", "zcr",
             "low_ratio", "mid_ratio", "high_ratio", "modulation"]


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n = 1 + (x.size - n_fft) // hop
    if n <= 0:
        return np.empty((0, n_fft), dtype=np.float32)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=(n, n_fft), strides=strides).copy()


def _band_energy(power, freqs, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    total = power.sum(axis=1) + EPS
    return power[:, mask].sum(axis=1) / total


def _rolloff(power, freqs, ratio):
    cumulative = np.cumsum(power, axis=1)
    target = cumulative[:, -1:] * ratio
    idx = np.argmax(cumulative >= target, axis=1)
    return freqs[idx]


def _zcr(frames):
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=1)


def _stats(v):
    """(mean, std, iqr) of a per-frame series."""
    return (float(np.mean(v)),
            float(np.std(v)),
            float(np.percentile(v, 75) - np.percentile(v, 25)))


def compute_all(wav: np.ndarray, sr: int = DEFAULT_SR,
                n_fft: int = 1024, hop: int = 256) -> dict:
    """Return the union of base + rich features for one clip."""
    x = np.asarray(wav, dtype=np.float32)
    if x.size == 0:
        return {k: 0.0 for k in BASE_KEYS}
    x = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(x**2) + EPS))
    frames = _frames(x, n_fft, hop)
    if frames.size == 0:
        return {k: 0.0 for k in BASE_KEYS} | {"rms": rms}

    win = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(frames * win[None, :], axis=1)
    mag = np.abs(spec).astype(np.float32) + EPS
    power = mag**2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
    total = power.sum(axis=1) + EPS

    centroid = (power * freqs[None, :]).sum(axis=1) / total
    flatness = np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + EPS)
    rolloff = _rolloff(power, freqs, 0.85)
    zcr = _zcr(frames)
    frame_rms = np.sqrt(np.mean(frames**2, axis=1) + EPS)

    low = _band_energy(power, freqs, 20.0, 250.0)
    mid = _band_energy(power, freqs, 250.0, 3000.0)
    high = _band_energy(power, freqs, 3000.0, sr / 2.0)

    # ---- shipped 8 (means) ----
    f = {
        "rms": rms,
        "centroid_hz": float(np.mean(centroid)),
        "flatness": float(np.mean(flatness)),
        "rolloff_hz": float(np.mean(rolloff)),
        "zcr": float(np.mean(zcr)),
        "low_ratio": float(np.mean(low)),
        "mid_ratio": float(np.mean(mid)),
        "high_ratio": float(np.mean(high)),
        "modulation": float(np.std(frame_rms) / (np.mean(frame_rms) + EPS)),
    }

    # ---- rich additions ----
    # temporal spread of key descriptors
    for name, series in (("centroid", centroid), ("flatness", flatness),
                         ("rolloff", rolloff), ("zcr", zcr),
                         ("low", low), ("mid", mid), ("high", high)):
        _, s, iqr = _stats(series)
        f[f"{name}_std"] = s
        f[f"{name}_iqr"] = iqr

    # delta (frame-to-frame) dynamics
    logrms = np.log(frame_rms + EPS)
    f["dyn_range_db"] = float(20.0 * (np.percentile(logrms, 95) - np.percentile(logrms, 5)) / np.log(10))
    f["delta_centroid"] = float(np.mean(np.abs(np.diff(centroid))))
    f["delta_flux"] = float(np.mean(np.abs(np.diff(mag, axis=0)))) if mag.shape[0] > 1 else 0.0

    # finer band structure
    f["sub_ratio"] = float(np.mean(_band_energy(power, freqs, 20.0, 120.0)))
    f["lowmid_ratio"] = float(np.mean(_band_energy(power, freqs, 120.0, 500.0)))
    f["presence_ratio"] = float(np.mean(_band_energy(power, freqs, 2000.0, 4000.0)))
    f["brilliance_ratio"] = float(np.mean(_band_energy(power, freqs, 6000.0, sr / 2.0)))

    # spectral shape
    bandwidth = np.sqrt((power * (freqs[None, :] - centroid[:, None])**2).sum(axis=1) / total)
    f["bandwidth_hz"] = float(np.mean(bandwidth))
    f["crest"] = float(np.mean(mag.max(axis=1) / (mag.mean(axis=1) + EPS)))
    # spectral contrast: peak-to-valley in log power per frame
    sort_p = np.sort(np.log(power + EPS), axis=1)
    k = max(1, sort_p.shape[1] // 10)
    f["contrast"] = float(np.mean(sort_p[:, -k:].mean(axis=1) - sort_p[:, :k].mean(axis=1)))

    # modulation percentiles
    f["mod_p90"] = float(np.percentile(frame_rms, 90) / (np.mean(frame_rms) + EPS))
    f["mod_p10"] = float(np.percentile(frame_rms, 10) / (np.mean(frame_rms) + EPS))
    return f


RICH_KEYS = None  # filled on first compute (kept stable/sorted below)


def rich_key_order(sample: dict) -> list:
    # deterministic: base 8 first, then the rest sorted
    rest = sorted(k for k in sample if k not in BASE_KEYS and k != "rms")
    return BASE_KEYS + rest


def load_manifest_rows(manifest: Path):
    """Yield (path, label, group) where group identifies the noise SOURCE.

    Grouping by source category (+ file stem for the many-file categories) is
    what makes CV honest: mixtures built from the same noise texture must not be
    split across train and test, or the model can memorise the texture.
    """
    rows = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            # prefer the exact source file; fall back to category, then path stem
            src = (r.get("source_file") or r.get("source_category")
                   or Path(r["path"]).stem)
            if r["label"] == "speech":
                group = "speech:" + Path(r["path"]).stem
            else:
                group = "noise:" + src
            rows.append((r["path"], r["label"], group))
    return rows
