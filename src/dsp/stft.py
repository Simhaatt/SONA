"""STFT / inverse-STFT front-end shared by every method (classical + learned).

Design goals:
  * **Perfect reconstruction** when the mask is all-ones (WOLA / weighted overlap-add).
    This is asserted by the unit tests and is the correctness anchor for the whole pipeline.
  * Conventions match ``torch.stft`` (periodic Hann, ``center=True`` reflect padding,
    ``F = n_fft // 2 + 1`` one-sided bins) so the numpy front-end used by classical baselines
    and the torch front-end used during training stay consistent.

Everything here is numpy so it runs without torch (classical baselines, evaluation, streaming).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann


def stft_params_from_config(cfg) -> dict:
    """Extract STFT kwargs from a config's ``stft`` block, with sane defaults."""
    s = cfg.get_path("stft", {}) if hasattr(cfg, "get_path") else (cfg or {})
    n_fft = int(s.get("n_fft", 512))
    return {
        "n_fft": n_fft,
        "hop": int(s.get("hop", n_fft // 4)),
        "win_length": int(s.get("win_length", n_fft)),
        "center": bool(s.get("center", True)),
    }


@dataclass
class STFT:
    """Block STFT with weighted overlap-add synthesis.

    Attributes
    ----------
    n_fft : FFT size (number of frequency bins is ``n_fft // 2 + 1``).
    hop : hop size in samples (75% overlap => hop = n_fft // 4).
    win_length : analysis window length (<= n_fft; zero-padded to n_fft).
    center : if True, reflect-pad by ``n_fft // 2`` so frame t is centred at ``t * hop``.
    """

    n_fft: int = 512
    hop: int = 128
    win_length: int | None = None
    center: bool = True

    def __post_init__(self) -> None:
        self.win_length = self.win_length or self.n_fft
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft")
        # Periodic Hann (sym=False) matches torch/librosa and satisfies COLA at 75% overlap.
        self._window = hann(self.win_length, sym=False).astype(np.float64)
        # Zero-pad the window to n_fft, centred.
        pad = self.n_fft - self.win_length
        left = pad // 2
        self._win_padded = np.pad(self._window, (left, pad - left))

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def window(self) -> np.ndarray:
        return self._win_padded.copy()

    # ------------------------------------------------------------------ analysis
    def stft(self, x: np.ndarray) -> np.ndarray:
        """Return the one-sided complex spectrogram of shape ``(n_bins, n_frames)``."""
        x = np.asarray(x, dtype=np.float64)
        if self.center:
            x = np.pad(x, self.n_fft // 2, mode="reflect")

        n = len(x)
        if n < self.n_fft:
            x = np.pad(x, (0, self.n_fft - n))
            n = len(x)
        n_frames = 1 + (n - self.n_fft) // self.hop

        # Build framed matrix (n_fft, n_frames) via stride tricks, then window + rfft.
        idx = np.arange(self.n_fft)[:, None] + self.hop * np.arange(n_frames)[None, :]
        frames = x[idx] * self._win_padded[:, None]
        spec = np.fft.rfft(frames, n=self.n_fft, axis=0)
        return spec.astype(np.complex64)

    # ----------------------------------------------------------------- synthesis
    def istft(self, spec: np.ndarray, length: int | None = None) -> np.ndarray:
        """Inverse STFT via weighted overlap-add. Recovers ``x`` (perfect when spec is unmodified)."""
        spec = np.asarray(spec, dtype=np.complex128)
        n_frames = spec.shape[1]
        frames = np.fft.irfft(spec, n=self.n_fft, axis=0)  # (n_fft, n_frames)

        expected_len = self.n_fft + self.hop * (n_frames - 1)
        out = np.zeros(expected_len, dtype=np.float64)
        wsum = np.zeros(expected_len, dtype=np.float64)  # accumulated window^2 for normalisation

        win = self._win_padded
        win_sq = win**2
        for t in range(n_frames):
            start = t * self.hop
            out[start : start + self.n_fft] += frames[:, t] * win
            wsum[start : start + self.n_fft] += win_sq

        nonzero = wsum > 1e-8
        out[nonzero] /= wsum[nonzero]

        if self.center:
            out = out[self.n_fft // 2 : -(self.n_fft // 2)] if expected_len > self.n_fft else out
        if length is not None:
            out = out[:length] if len(out) >= length else np.pad(out, (0, length - len(out)))
        return out.astype(np.float32)

    # --------------------------------------------------------------- convenience
    def magphase(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (magnitude, phase) spectrograms."""
        spec = self.stft(x)
        return np.abs(spec), np.angle(spec)

    def reconstruct(self, mag: np.ndarray, phase: np.ndarray, length: int | None = None) -> np.ndarray:
        """Rebuild a waveform from magnitude + phase (e.g. after applying a mask to `mag`)."""
        spec = mag * np.exp(1j * phase)
        return self.istft(spec, length=length)
