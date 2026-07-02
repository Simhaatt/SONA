"""Per-frame / per-band features and noise-PSD estimation.

Features feed the learned models (and the SNR-based classical baselines). We keep them cheap
and causal-friendly so the same code can run frame-by-frame in the real-time streamer:
  * log band energy
  * spectral flux (positive temporal change of log band energy)
  * temporal delta of log band energy (previous-frame context)
  * estimated a-priori SNR per band (from a noise-PSD estimate)

`estimate_noise_psd` provides the noise floor used by spectral subtraction / Wiener and by the
SNR feature. Two estimators: assume-first-frames-are-noise (simple, good for fixed clips) and a
minimum-statistics tracker (works on non-stationary noise, closer to real use).
"""

from __future__ import annotations

import numpy as np

from .bands import BandMapper

EPS = 1e-8


def estimate_noise_psd(
    power: np.ndarray,
    method: str = "minstat",
    init_frames: int = 6,
    smooth: float = 0.9,
    window: int = 30,
) -> np.ndarray:
    """Estimate per-bin noise power.

    Parameters
    ----------
    power : (n_bins, T) power spectrogram (|STFT|**2).
    method : 'first_frames' -> mean of the first `init_frames` (assumed noise-only), broadcast
             over time; 'minstat' -> minimum-statistics tracking over a sliding `window`.
    Returns (n_bins, T) noise power (for 'first_frames' the value is constant over time).
    """
    power = np.asarray(power, dtype=np.float64)
    n_bins, T = power.shape

    if method == "first_frames":
        k = min(init_frames, T)
        noise = power[:, :k].mean(axis=1, keepdims=True)
        return np.repeat(noise, T, axis=1)

    if method == "minstat":
        # Recursive smoothing, then running minimum over a trailing window (Martin-style, simplified).
        smoothed = np.empty_like(power)
        smoothed[:, 0] = power[:, 0]
        for t in range(1, T):
            smoothed[:, t] = smooth * smoothed[:, t - 1] + (1.0 - smooth) * power[:, t]
        noise = np.empty_like(power)
        for t in range(T):
            lo = max(0, t - window + 1)
            noise[:, t] = smoothed[:, lo : t + 1].min(axis=1)
        # Small bias compensation (min-stat underestimates); scale up modestly.
        return noise * 1.5

    raise ValueError(f"unknown noise estimator: {method}")


class FrameFeatures:
    """Extract a (T, n_bands, C) feature tensor from a magnitude spectrogram."""

    #: channel order produced by __call__, exposed for model input-dim wiring.
    CHANNELS = ("log_band_energy", "spectral_flux", "delta", "snr_prior_db")

    def __init__(self, band_mapper: BandMapper, noise_method: str = "minstat") -> None:
        self.bands = band_mapper
        self.noise_method = noise_method

    @property
    def n_channels(self) -> int:
        return len(self.CHANNELS)

    def __call__(self, mag: np.ndarray) -> np.ndarray:
        mag = np.asarray(mag, dtype=np.float64)
        power = mag**2  # (n_bins, T)

        band_pow = self.bands.band_energy(mag)  # (n_bands, T)
        log_band = np.log(band_pow + EPS)

        # Spectral flux: positive part of temporal difference (onset/burst sensitivity).
        d = np.diff(log_band, axis=1, prepend=log_band[:, :1])
        flux = np.maximum(d, 0.0)

        # A-priori SNR per band from an estimated noise floor.
        noise_pow = estimate_noise_psd(power, method=self.noise_method)  # (n_bins, T)
        band_noise = self.bands.analysis @ noise_pow  # (n_bands, T)
        snr = band_pow / (band_noise + EPS)
        snr_db = 10.0 * np.log10(snr + EPS)

        feats = np.stack([log_band, flux, d, snr_db], axis=-1)  # (n_bands, T, C)
        feats = np.transpose(feats, (1, 0, 2))  # (T, n_bands, C)
        return feats.astype(np.float32)
