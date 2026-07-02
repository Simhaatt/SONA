"""Spectral subtraction (Boll, 1979) with over-subtraction and a spectral floor.

Enhanced power = max(|X|^2 - alpha * noise_power, beta * noise_power). The gain mask is the
square root of the ratio to the observed power. `alpha` over-subtracts to reduce residual noise;
`beta` sets a floor to limit musical noise.
"""

from __future__ import annotations

import numpy as np

from ..frontend import Enhancer, SpectralFrontend
from ._noise import noise_power

EPS = 1e-8


class SpectralSubtraction(Enhancer):
    name = "spectral_subtraction"

    def __init__(
        self,
        frontend: SpectralFrontend,
        alpha: float = 2.0,
        beta: float = 0.01,
        noise_method: str = "minstat",
    ) -> None:
        super().__init__(frontend)
        self.alpha = alpha
        self.beta = beta
        self.noise_method = noise_method

    def compute_gain(self, analysis: dict) -> np.ndarray:
        power = analysis["mag"].astype(np.float64) ** 2
        npow = noise_power(analysis, self.noise_method)
        enh_power = np.maximum(power - self.alpha * npow, self.beta * npow)
        gain = np.sqrt(enh_power / (power + EPS))
        return np.clip(gain, 0.0, 1.0).astype(np.float32)
