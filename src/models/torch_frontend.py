"""Differentiable torch mirror of the numpy STFT front-end (for training the learned models).

Uses ``torch.stft``/``torch.istft`` with a periodic Hann window and ``center=True`` so it matches
`src.dsp.stft.STFT` conventions (one-sided bins, 75% overlap). Band aggregation / gain expansion
reuse the exact matrices from `BandMapper`, so the learned models operate on the *same* per-band
gain space as the classical baselines.

The mask is applied to the noisy magnitude with the noisy phase preserved, then inverted — so the
whole path (features -> gains -> masked reconstruction) is differentiable end-to-end and an
SI-SDR loss can backprop into the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..dsp.bands import BandMapper

EPS = 1e-8


class TorchFrontend(nn.Module):
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 512,
        hop: int | None = None,
        n_bands: int = 32,
        band_scale: str = "bark",
    ) -> None:
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        self.n_bands = n_bands
        self.n_bins = n_fft // 2 + 1

        bm = BandMapper(n_fft=n_fft, sr=sr, n_bands=n_bands, scale=band_scale)
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True))
        self.register_buffer("analysis", torch.from_numpy(bm.analysis))   # (n_bands, n_bins)
        self.register_buffer("synthesis", torch.from_numpy(bm.synthesis))  # (n_bins, n_bands)
        self.n_feat_channels = 3  # log_band_energy, delta, flux

    # ------------------------------------------------------------------ analysis
    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, N) -> complex spectrogram (B, n_bins, T)."""
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )

    def features(self, mag: torch.Tensor) -> torch.Tensor:
        """mag (B, n_bins, T) -> per-band feature tensor (B, T, n_bands, C)."""
        power = mag**2
        band_pow = torch.einsum("bf t, k f -> b k t", power, self.analysis)  # (B, n_bands, T)
        log_band = torch.log(band_pow + EPS)
        delta = torch.diff(log_band, dim=-1, prepend=log_band[..., :1])
        flux = torch.clamp(delta, min=0.0)
        feats = torch.stack([log_band, delta, flux], dim=-1)  # (B, n_bands, T, C)
        return feats.permute(0, 2, 1, 3).contiguous()  # (B, T, n_bands, C)

    # ------------------------------------------------------------- reconstruction
    def apply_band_gain(self, spec: torch.Tensor, band_gain: torch.Tensor, length: int) -> torch.Tensor:
        """Apply per-band gain to a complex spectrogram and invert to a waveform.

        spec (B, n_bins, T) complex; band_gain (B, T, n_bands) in [0, 1].
        Returns wav (B, length).
        """
        # Expand per-band gain to per-bin: (B, T, n_bands) x (n_bins, n_bands)^T -> (B, T, n_bins)
        bin_gain = torch.einsum("b t k, f k -> b t f", band_gain, self.synthesis)
        bin_gain = bin_gain.permute(0, 2, 1)  # (B, n_bins, T)
        enh = spec * bin_gain  # complex * real, phase preserved
        wav = torch.istft(
            enh,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=length,
        )
        return wav

    def apply_band_complex(self, spec: torch.Tensor, mask_ri: torch.Tensor, length: int) -> torch.Tensor:
        """Apply a per-band COMPLEX ratio mask to a complex spectrogram and invert.

        Unlike `apply_band_gain` (real gain on magnitude, noisy phase preserved), a
        complex ratio mask M = M_r + jM_i multiplies the complex spectrum, so BOTH
        magnitude and phase are modified: enhanced = M * noisy. All bins within a
        band share the mask value (same band resolution as the gain baseline, so the
        only thing that differs between the two is real-gain vs complex-mask).

        spec (B, n_bins, T) complex; mask_ri (B, T, n_bands, 2) real; -> wav (B, length).
        """
        mask_r = mask_ri[..., 0]  # (B, T, n_bands)
        mask_i = mask_ri[..., 1]
        bin_r = torch.einsum("b t k, f k -> b t f", mask_r, self.synthesis).permute(0, 2, 1)
        bin_i = torch.einsum("b t k, f k -> b t f", mask_i, self.synthesis).permute(0, 2, 1)
        mask = torch.complex(bin_r, bin_i)  # (B, n_bins, T)
        enh = spec * mask
        wav = torch.istft(
            enh,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=length,
        )
        return wav

    def analyze(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (complex spectrogram, feature tensor) for a batch of waveforms."""
        spec = self.stft(wav)
        feats = self.features(spec.abs())
        return spec, feats
