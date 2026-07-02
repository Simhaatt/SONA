"""Differentiable, frequency-domain parametric EQ + dynamic-range processing (torch).

Because the whole pipeline already lives in the STFT domain, we realise EQ as a per-bin
magnitude multiplier equal to the magnitude response of a cascade of RBJ peaking biquads — this
is exact, cheap, and differentiable w.r.t. the band gains (so a learned controller can *set*
these gains, and they can be adapted online). Dynamic-range control is applied as a per-frame
broadband gain from the frame level. Hum removal is a fixed narrow notch at the mains frequency
and its harmonics.

Everything is unity-at-zero: gains of 0 dB / bypassed DRC reproduce the input exactly, which the
tests assert. This block is the action space of the unified controller (Layer B) and the
smart-EQ demo.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

EPS = 1e-8

# 10-band graphic EQ centres (Hz). At 16 kHz the top band sits below Nyquist (8 kHz); centres
# above Nyquist are dropped by ParametricEQ.
GRAPHIC_EQ_FREQS = (31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


def _peaking_response_mag(freqs: torch.Tensor, f0: float, q: float, gain_db: torch.Tensor,
                          sr: int) -> torch.Tensor:
    """Magnitude response |H(e^jw)| of one RBJ peaking biquad at bin frequencies `freqs` (Hz).

    `gain_db` is a scalar tensor (differentiable). Returns a (len(freqs),) magnitude curve.
    """
    a_amp = torch.pow(10.0, gain_db / 40.0)          # sqrt of linear gain
    w0 = 2.0 * np.pi * f0 / sr
    cw0 = float(np.cos(w0))
    alpha = float(np.sin(w0) / (2.0 * q))

    b0 = 1.0 + alpha * a_amp
    b1 = -2.0 * cw0
    b2 = 1.0 - alpha * a_amp
    a0 = 1.0 + alpha / a_amp
    a1 = -2.0 * cw0
    a2 = 1.0 - alpha / a_amp

    w = 2.0 * np.pi * freqs / sr                      # (F,)
    c1 = torch.cos(w); c2 = torch.cos(2 * w)
    s1 = torch.sin(w); s2 = torch.sin(2 * w)
    # Numerator/denominator real & imag parts of H(e^{-jw}).
    num_re = b0 + b1 * c1 + b2 * c2
    num_im = -(b1 * s1 + b2 * s2)
    den_re = a0 + a1 * c1 + a2 * c2
    den_im = -(a1 * s1 + a2 * s2)
    num_mag = torch.sqrt(num_re**2 + num_im**2 + EPS)
    den_mag = torch.sqrt(den_re**2 + den_im**2 + EPS)
    return num_mag / den_mag


class ParametricEQ(nn.Module):
    """Cascade of peaking biquads -> per-bin magnitude gain curve from per-band dB gains."""

    def __init__(self, sr: int = 16000, n_fft: int = 512,
                 center_freqs: tuple[float, ...] = GRAPHIC_EQ_FREQS, q: float = 1.41) -> None:
        super().__init__()
        self.sr, self.n_fft, self.q = sr, n_fft, q
        self.center_freqs = [f for f in center_freqs if f < sr / 2.0]
        self.n_bands = len(self.center_freqs)
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr)
        self.register_buffer("bin_freqs", freqs)

    def curve(self, gains_db: torch.Tensor) -> torch.Tensor:
        """Return the per-bin linear gain (n_bins,) for a (n_bands,) dB-gain vector."""
        gains_db = torch.as_tensor(gains_db, dtype=torch.float32, device=self.bin_freqs.device)
        resp = torch.ones_like(self.bin_freqs)
        for i, f0 in enumerate(self.center_freqs):
            resp = resp * _peaking_response_mag(self.bin_freqs, f0, self.q, gains_db[i], self.sr)
        return resp

    def forward(self, mag: torch.Tensor, gains_db: torch.Tensor) -> torch.Tensor:
        """Apply EQ to a magnitude spectrogram `mag` (..., n_bins, T)."""
        curve = self.curve(gains_db).view(*([1] * (mag.dim() - 2)), -1, 1)
        return mag * curve


def hum_notch_curve(sr: int = 16000, n_fft: int = 512, base_hz: float = 50.0,
                    n_harmonics: int = 4, depth_db: float = -50.0,
                    width_hz: float = 8.0) -> torch.Tensor:
    """Per-bin gain that notches out mains hum at `base_hz` and its harmonics."""
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr)
    gain = torch.ones_like(freqs)
    depth = 10.0 ** (depth_db / 20.0)
    for h in range(1, n_harmonics + 1):
        f0 = base_hz * h
        if f0 >= sr / 2.0:
            break
        notch = 1.0 - (1.0 - depth) * torch.exp(-((freqs - f0) ** 2) / (2.0 * width_hz**2))
        gain = gain * notch
    return gain


class Compressor(nn.Module):
    """Simple per-frame broadband compressor/limiter on the magnitude spectrogram.

    Computes each frame's level (dB) from its band energy, applies a static compression curve
    above `threshold_db` with `ratio`, plus `makeup_db`, and an optional hard `ceiling_db` limiter.
    Bypassed (identity) when ratio == 1 and makeup == 0 and ceiling is None.
    """

    def __init__(self, threshold_db: float = -24.0, ratio: float = 2.0, makeup_db: float = 0.0,
                 ceiling_db: float | None = None) -> None:
        super().__init__()
        self.threshold_db = threshold_db
        self.ratio = ratio
        self.makeup_db = makeup_db
        self.ceiling_db = ceiling_db

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        # Frame level from mean power across frequency -> dB. mag: (..., F, T)
        level = 10.0 * torch.log10(mag.pow(2).mean(dim=-2, keepdim=True) + EPS)  # (...,1,T)
        over = torch.clamp(level - self.threshold_db, min=0.0)
        gain_db = -over * (1.0 - 1.0 / self.ratio) + self.makeup_db
        if self.ceiling_db is not None:
            out_level = level + gain_db
            gain_db = gain_db - torch.clamp(out_level - self.ceiling_db, min=0.0)
        gain = torch.pow(10.0, gain_db / 20.0)
        return mag * gain
