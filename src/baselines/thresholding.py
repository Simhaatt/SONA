"""Hard and soft spectral thresholding.

These are the deck's primary straw-men: a per-bin threshold derived from the estimated noise
level, applied with fixed rules and no learning. The threshold uses the *universal threshold*
``sigma * sqrt(2 ln N)`` (Donoho & Johnstone) scaled by a tunable factor `k`, where sigma is the
per-bin noise std from the noise-PSD estimate.
"""

from __future__ import annotations

import numpy as np

from ..frontend import Enhancer, SpectralFrontend
from ._noise import noise_power

EPS = 1e-8


class _ThresholdBase(Enhancer):
    def __init__(self, frontend: SpectralFrontend, k: float = 1.0, noise_method: str = "minstat") -> None:
        super().__init__(frontend)
        self.k = k
        self.noise_method = noise_method

    def _threshold(self, analysis: dict) -> np.ndarray:
        sigma = np.sqrt(noise_power(analysis, self.noise_method))  # (n_bins, T)
        n = analysis["mag"].shape[0]
        return self.k * sigma * np.sqrt(2.0 * np.log(max(n, 2)))


class HardThreshold(_ThresholdBase):
    name = "hard_threshold"

    def compute_gain(self, analysis: dict) -> np.ndarray:
        mag = analysis["mag"]
        thr = self._threshold(analysis)
        return (mag > thr).astype(np.float32)  # keep bin (gain 1) or kill it (gain 0)


class SoftThreshold(_ThresholdBase):
    name = "soft_threshold"

    def compute_gain(self, analysis: dict) -> np.ndarray:
        mag = analysis["mag"]
        thr = self._threshold(analysis)
        shrunk = np.maximum(mag - thr, 0.0)
        return (shrunk / (mag + EPS)).astype(np.float32)
