"""Audio load/save helpers. All internal audio is float32, mono, in a known sample rate.

We standardise on 16 kHz mono float32 throughout the pipeline (VoiceBank-DEMAND and most
speech-enhancement benchmarks are 16 kHz), resampling on load when needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SR = 16000


def load_audio(path: str | Path, sr: int | None = DEFAULT_SR, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio as float32. Resamples to `sr` (if given) and downmixes to mono.

    Returns (samples, sample_rate).
    """
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, channels)
    if mono and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    data = data[:, 0] if mono else data

    if sr is not None and file_sr != sr:
        data = _resample(data, file_sr, sr)
        file_sr = sr
    return np.ascontiguousarray(data, dtype=np.float32), file_sr


def save_audio(path: str | Path, samples: np.ndarray, sr: int = DEFAULT_SR) -> None:
    """Write float32 samples to `path`, clipping to [-1, 1] to avoid wrap-around."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), samples, sr)


def _resample(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1-D or 2-D (n, ch) audio. Uses librosa if available, else scipy poly-phase."""
    if orig_sr == target_sr:
        return x
    try:
        import librosa

        if x.ndim == 1:
            return librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)
        return np.stack(
            [librosa.resample(x[:, c], orig_sr=orig_sr, target_sr=target_sr) for c in range(x.shape[1])],
            axis=1,
        )
    except ImportError:  # pragma: no cover
        from math import gcd
        from scipy.signal import resample_poly

        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        return resample_poly(x, up, down, axis=0).astype(np.float32)
