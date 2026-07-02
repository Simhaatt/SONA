"""Shared noise-PSD helper for the SNR-based baselines."""

from __future__ import annotations

import numpy as np

from ..dsp import estimate_noise_psd


def noise_power(analysis: dict, method: str = "minstat") -> np.ndarray:
    """Per-bin noise power ``(n_bins, T)`` from the analysis dict's magnitude."""
    power = analysis["mag"].astype(np.float64) ** 2
    return estimate_noise_psd(power, method=method)
