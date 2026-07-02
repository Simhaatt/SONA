"""Noise mixing utilities: SNR-controlled mixing, burst/impulsive injection, a noise sampler.

The burst/impulsive augmentation directly supports the paper's core claim (adaptivity to
non-stationary noise) by creating a "hard" evaluation slice the classical/frozen baselines
should struggle with.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EPS = 1e-8


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + EPS))


def mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix `clean` + scaled `noise` at the target SNR. Returns (noisy, scaled_noise).

    Noise is tiled/cropped to the clean length (with a random offset) before scaling.
    """
    rng = rng or np.random.default_rng()
    clean = np.asarray(clean, dtype=np.float32)
    noise = _fit_length(noise, len(clean), rng)

    target = _rms(clean) / (10.0 ** (snr_db / 20.0))
    scaled = noise * (target / _rms(noise))
    noisy = clean + scaled
    peak = np.max(np.abs(noisy)) + EPS
    if peak > 1.0:  # prevent clipping while preserving the SNR
        noisy /= peak
        scaled /= peak
    return noisy.astype(np.float32), scaled.astype(np.float32)


def add_burst_noise(
    signal: np.ndarray,
    sr: int = 16000,
    n_bursts: int = 3,
    burst_ms: tuple[float, float] = (40.0, 200.0),
    gain_db: tuple[float, float] = (0.0, 12.0),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inject short, loud impulsive bursts (non-stationary) into a signal."""
    rng = rng or np.random.default_rng()
    out = np.asarray(signal, dtype=np.float32).copy()
    n = len(out)
    base = _rms(out)
    for _ in range(n_bursts):
        dur = int(sr * rng.uniform(*burst_ms) / 1000.0)
        if dur >= n:
            continue
        start = rng.integers(0, n - dur)
        amp = base * (10.0 ** (rng.uniform(*gain_db) / 20.0))
        burst = amp * rng.standard_normal(dur).astype(np.float32)
        # Hann fade to avoid clicks at burst edges.
        fade = np.hanning(dur).astype(np.float32)
        out[start : start + dur] += burst * fade
    return np.clip(out, -1.0, 1.0)


def _fit_length(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float32)
    if len(noise) < length:
        reps = int(np.ceil(length / len(noise)))
        noise = np.tile(noise, reps)
    if len(noise) > length:
        off = int(rng.integers(0, len(noise) - length + 1))
        noise = noise[off : off + length]
    return noise


class NoiseBank:
    """Samples noise clips from a directory of wav files (for on-the-fly mixing)."""

    def __init__(self, root: str | Path, sr: int = 16000) -> None:
        from ..utils.audio_io import load_audio

        self.root = Path(root)
        self.sr = sr
        self._load = load_audio
        self.files = sorted(self.root.rglob("*.wav"))
        if not self.files:
            raise FileNotFoundError(f"no wav files under {self.root}")

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        path = self.files[int(rng.integers(0, len(self.files)))]
        wav, _ = self._load(path, sr=self.sr)
        return wav
