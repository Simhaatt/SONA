"""Shared spectral front-end and the common `Enhancer` interface.

`SpectralFrontend` bundles the STFT, band mapper and feature extractor so *every* method —
classical baseline or learned model — analyses, masks and reconstructs audio identically. Only
the gain-estimation step differs between methods. This is the crux of a fair comparison.

`Enhancer` is the one interface the evaluation harness knows about: ``enhance(wav, sr) -> wav``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .dsp import STFT, BandMapper, FrameFeatures


class SpectralFrontend:
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop: int | None = None,
        n_bands: int = 32,
        band_scale: str = "bark",
        noise_method: str = "minstat",
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        self.n_bands = n_bands
        self.stft = STFT(n_fft=n_fft, hop=self.hop)
        self.bands = BandMapper(n_fft=n_fft, sr=sr, n_bands=n_bands, scale=band_scale)
        self.features = FrameFeatures(self.bands, noise_method=noise_method)

    @classmethod
    def from_config(cls, cfg) -> "SpectralFrontend":
        s = cfg.get_path("stft", {})
        b = cfg.get_path("bands", {})
        return cls(
            sr=int(cfg.get_path("sr", 16000)),
            n_fft=int(s.get("n_fft", 512)),
            hop=s.get("hop"),
            n_bands=int(b.get("n_bands", 32)),
            band_scale=b.get("scale", "bark"),
            noise_method=cfg.get_path("noise.method", "minstat"),
        )

    # ------------------------------------------------------------------ analysis
    def analyze(self, wav: np.ndarray) -> dict:
        """Return dict with magnitude, phase, per-band features and original length."""
        mag, phase = self.stft.magphase(wav)
        feats = self.features(mag)  # (T, n_bands, C)
        return {"mag": mag, "phase": phase, "feats": feats, "length": len(wav)}

    # ------------------------------------------------------------- apply + rebuild
    def apply_gain(self, mag: np.ndarray, gain: np.ndarray) -> np.ndarray:
        """Multiply magnitude by a gain mask.

        `gain` may be per-bin ``(n_bins, T)`` or per-band ``(n_bands, T)`` (auto-expanded).
        """
        gain = np.asarray(gain)
        if gain.shape[0] == self.n_bands and self.n_bands != self.stft.n_bins:
            gain = self.bands.expand_gain(gain)  # -> (n_bins, T)
        return mag * gain

    def synthesize(self, mag: np.ndarray, phase: np.ndarray, length: int | None = None) -> np.ndarray:
        return self.stft.reconstruct(mag, phase, length=length)

    def enhance_with_gain(self, wav: np.ndarray, gain: np.ndarray) -> np.ndarray:
        a = self.analyze(wav)
        enh_mag = self.apply_gain(a["mag"], gain)
        return self.synthesize(enh_mag, a["phase"], length=a["length"])


class Enhancer(ABC):
    """Base class for every denoising method (classical or learned)."""

    name: str = "enhancer"

    def __init__(self, frontend: SpectralFrontend) -> None:
        self.frontend = frontend

    @abstractmethod
    def compute_gain(self, analysis: dict) -> np.ndarray:
        """Return a gain mask (per-bin ``(n_bins,T)`` or per-band ``(n_bands,T)``)."""

    def enhance(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        analysis = self.frontend.analyze(np.asarray(wav, dtype=np.float32))
        gain = self.compute_gain(analysis)
        enh_mag = self.frontend.apply_gain(analysis["mag"], gain)
        return self.frontend.synthesize(enh_mag, analysis["phase"], length=analysis["length"])
