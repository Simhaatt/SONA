"""Bin <-> band mapping.

The learned models and classical baselines predict one attenuation value per *perceptual band*
(~24-48 mel/bark bands) rather than per FFT bin (257/513 bins). This:
  * matches the deck's "attenuation level for each band",
  * shrinks the RL action space and the model output dimensionality,
  * is edge-friendly, and
  * ties suppression to perceptually meaningful regions.

`BandMapper` exposes two matrices:
  * ``analysis`` (n_bands x n_bins): aggregates per-bin magnitude/energy into per-band values
    (rows sum to 1 => band value is a weighted average over its bins).
  * ``synthesis`` (n_bins x n_bands): expands a per-band gain back to per-bin gains as a
    partition of unity (each bin's rows sum to 1 => smooth interpolation, no gain blow-up).

Both are plain numpy arrays; wrap with ``torch.from_numpy`` for use inside models.
"""

from __future__ import annotations

import numpy as np


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _hz_to_bark(f: np.ndarray) -> np.ndarray:
    # Traunmüller (1990) bark scale.
    return (26.81 * f) / (1960.0 + f) - 0.53


def _bark_to_hz(b: np.ndarray) -> np.ndarray:
    b = (b + 0.53) / 26.81
    return 1960.0 * b / (1.0 - b)


_SCALES = {
    "mel": (_hz_to_mel, _mel_to_hz),
    "bark": (_hz_to_bark, _bark_to_hz),
}


class BandMapper:
    def __init__(
        self,
        n_fft: int,
        sr: int = 16000,
        n_bands: int = 32,
        scale: str = "bark",
        fmin: float = 0.0,
        fmax: float | None = None,
    ) -> None:
        self.n_fft = n_fft
        self.sr = sr
        self.n_bands = n_bands
        self.scale = scale
        self.n_bins = n_fft // 2 + 1
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr / 2.0

        self.bin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        self.analysis, self.synthesis = self._build()

    # ------------------------------------------------------------------
    def _build(self) -> tuple[np.ndarray, np.ndarray]:
        if self.scale == "linear":
            edges = np.linspace(self.fmin, self.fmax, self.n_bands + 2)
        else:
            to_scale, from_scale = _SCALES[self.scale]
            s_edges = np.linspace(
                to_scale(np.array([self.fmin]))[0],
                to_scale(np.array([self.fmax]))[0],
                self.n_bands + 2,
            )
            edges = from_scale(s_edges)

        # Triangular filters over the bin-frequency axis.
        fb = np.zeros((self.n_bands, self.n_bins), dtype=np.float64)
        f = self.bin_freqs
        for b in range(self.n_bands):
            lo, ctr, hi = edges[b], edges[b + 1], edges[b + 2]
            left = (f - lo) / max(ctr - lo, 1e-9)
            right = (hi - f) / max(hi - ctr, 1e-9)
            tri = np.maximum(0.0, np.minimum(left, right))
            fb[b] = tri

        # Analysis: normalise each band (row) to sum 1 -> weighted average magnitude in band.
        row_sums = fb.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-9] = 1.0
        analysis = fb / row_sums

        # Synthesis: partition of unity across bands for each bin (columns of fb, normalised).
        col_sums = fb.sum(axis=0, keepdims=True)  # (1, n_bins)
        col_sums[col_sums < 1e-9] = 1.0
        synthesis = (fb / col_sums).T  # (n_bins, n_bands)
        # Bins outside all triangles (e.g. DC / Nyquist edges) get pass-through (gain via nearest band).
        uncovered = fb.sum(axis=0) < 1e-9
        if uncovered.any():
            nearest = np.argmax(fb, axis=0)  # 0 for uncovered; fix below
            for bin_idx in np.where(uncovered)[0]:
                # assign to the band whose centre is closest in frequency
                centers = 0.5 * (edges[:-2] + edges[2:])
                nb = int(np.argmin(np.abs(centers - f[bin_idx])))
                synthesis[bin_idx, :] = 0.0
                synthesis[bin_idx, nb] = 1.0
        return analysis.astype(np.float32), synthesis.astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def to_bands(self, per_bin: np.ndarray) -> np.ndarray:
        """Aggregate per-bin values (n_bins,) or (n_bins, T) into per-band (n_bands, ...)."""
        return self.analysis @ per_bin

    def expand_gain(self, per_band: np.ndarray) -> np.ndarray:
        """Expand per-band gain (n_bands,) or (n_bands, T) to per-bin (n_bins, ...)."""
        return self.synthesis @ per_band

    def band_energy(self, mag: np.ndarray) -> np.ndarray:
        """Per-band energy from a magnitude spectrogram (uses power = mag**2)."""
        return self.analysis @ (mag ** 2)
