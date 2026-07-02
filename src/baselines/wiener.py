"""Wiener filtering with decision-directed a-priori SNR estimation (Ephraim & Malah, 1984).

Gain = xi / (1 + xi), where xi is the a-priori SNR estimated by the decision-directed method:
    xi_t = w * G_{t-1}^2 * gamma_{t-1}  +  (1 - w) * max(gamma_t - 1, 0)
with gamma = posterior SNR = |X|^2 / noise_power. This is the strongest classical anchor and a
non-trivial baseline for the paper.
"""

from __future__ import annotations

import numpy as np

from ..frontend import Enhancer, SpectralFrontend
from ._noise import noise_power

EPS = 1e-8


class WienerFilter(Enhancer):
    name = "wiener"

    def __init__(
        self,
        frontend: SpectralFrontend,
        weight: float = 0.98,
        min_gain: float = 0.05,
        noise_method: str = "minstat",
    ) -> None:
        super().__init__(frontend)
        self.weight = weight
        self.min_gain = min_gain
        self.noise_method = noise_method

    def compute_gain(self, analysis: dict) -> np.ndarray:
        power = analysis["mag"].astype(np.float64) ** 2  # (n_bins, T)
        npow = noise_power(analysis, self.noise_method) + EPS
        gamma = power / npow  # posterior SNR

        n_bins, T = power.shape
        gain = np.empty((n_bins, T), dtype=np.float64)
        prev_gain = np.ones(n_bins)
        prev_gamma = np.maximum(gamma[:, 0] - 1.0, 0.0)
        for t in range(T):
            ml = np.maximum(gamma[:, t] - 1.0, 0.0)  # maximum-likelihood a-priori SNR
            xi = self.weight * (prev_gain**2) * prev_gamma + (1.0 - self.weight) * ml
            xi = np.maximum(xi, EPS)
            g = xi / (1.0 + xi)
            gain[:, t] = g
            prev_gain, prev_gamma = g, gamma[:, t]

        gain = np.clip(gain, self.min_gain, 1.0)
        return gain.astype(np.float32)
