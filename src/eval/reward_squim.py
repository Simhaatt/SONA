"""Reference-free (non-intrusive) reward signals for on-device online adaptation.

The core idea of the project: adapt the enhancer at inference by maximizing a quality signal that
needs **no clean reference**. The preferred signal is a pretrained non-intrusive estimator:

  * `SquimReward` — TorchAudio ``SQUIM_OBJECTIVE`` predicts STOI / PESQ / SI-SDR from the signal
    alone. Differentiable w.r.t. the input waveform, so it backprops into the adaptation module.
    (Requires torchaudio; available on Kaggle. This is the primary reward for real experiments.)

  * `SpectralContrastReward` — a lightweight, dependency-free fallback (works on Python 3.14 with
    no torchaudio): rewards speech-like spectral peakiness (low spectral flatness) in the speech
    band. Coarser than SQUIM but keeps the whole pipeline runnable and testable locally.

`make_reward(kind, ...)` returns a reward callable (or ``None`` to signal "use the adapter's
built-in heuristic"). All rewards return a scalar tensor where **higher is better**.
"""

from __future__ import annotations

import torch
import torch.nn as nn

EPS = 1e-8


def squim_available() -> bool:
    try:
        import torchaudio  # noqa: F401
        from torchaudio.pipelines import SQUIM_OBJECTIVE  # noqa: F401
        return True
    except Exception:
        return False


class SquimReward(nn.Module):
    """Non-intrusive reward from TorchAudio SQUIM (predicted STOI/PESQ/SI-SDR, no reference)."""

    def __init__(self, device: str = "cpu", sr: int = 16000,
                 weights: tuple[float, float, float] = (0.4, 0.4, 0.2)) -> None:
        super().__init__()
        from torchaudio.pipelines import SQUIM_OBJECTIVE

        self.bundle = SQUIM_OBJECTIVE
        self.model = SQUIM_OBJECTIVE.get_model().to(device).eval()
        self.model_sr = SQUIM_OBJECTIVE.sample_rate  # 16000
        self.sr = sr
        self.w_stoi, self.w_pesq, self.w_sisdr = weights
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, enhanced: torch.Tensor, noisy: torch.Tensor | None = None) -> torch.Tensor:
        if enhanced.dim() == 1:
            enhanced = enhanced.unsqueeze(0)
        if self.sr != self.model_sr:
            import torchaudio.functional as AF

            enhanced = AF.resample(enhanced, self.sr, self.model_sr)
        stoi_hat, pesq_hat, sisdr_hat = self.model(enhanced)  # each (B,)
        # Normalize to comparable [0,1]-ish ranges; higher is better.
        stoi_n = stoi_hat.clamp(0, 1)
        pesq_n = ((pesq_hat - 1.0) / 3.5).clamp(0, 1)
        sisdr_n = torch.sigmoid(sisdr_hat / 10.0)
        reward = self.w_stoi * stoi_n + self.w_pesq * pesq_n + self.w_sisdr * sisdr_n
        return reward.mean()


class SpectralContrastReward(nn.Module):
    """Dependency-free fallback: reward speech-like spectral peakiness (low flatness) in-band.

    Speech has a peaky harmonic spectrum; broadband noise is spectrally flat. Rewarding low
    spectral flatness in the 300-3400 Hz speech band is a crude but reference-free, differentiable
    quality proxy. Used when torchaudio/SQUIM is unavailable.
    """

    def __init__(self, sr: int = 16000, n_fft: int = 512, hop: int = 128,
                 fmin: float = 300.0, fmax: float = 3400.0) -> None:
        super().__init__()
        self.sr, self.n_fft, self.hop = sr, n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr)
        band = (freqs >= fmin) & (freqs <= fmax)
        self.register_buffer("band_idx", torch.nonzero(band, as_tuple=False).squeeze(-1))

    def forward(self, enhanced: torch.Tensor, noisy: torch.Tensor | None = None) -> torch.Tensor:
        if enhanced.dim() == 1:
            enhanced = enhanced.unsqueeze(0)
        spec = torch.stft(enhanced, n_fft=self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                          window=self.window, center=True, return_complex=True)
        power = spec.abs()[:, self.band_idx, :] ** 2 + EPS  # (B, Fband, T)
        arith = power.mean(dim=1)                            # (B, T)
        geo = torch.exp(torch.log(power).mean(dim=1))        # (B, T)
        flatness = (geo / arith).clamp(0, 1)                 # low => tonal/speech-like
        return (1.0 - flatness).mean()


def make_reward(kind: str = "auto", device: str = "cpu", sr: int = 16000, **kw):
    """Return a reward callable, or ``None`` to fall back to the adapter's built-in heuristic.

    kind: 'auto' (SQUIM if available else spectral), 'squim', 'spectral', 'none'.
    """
    if kind == "none":
        return None
    if kind in ("auto", "squim") and squim_available():
        return SquimReward(device=device, sr=sr, **kw).to(device)
    if kind == "squim" and not squim_available():
        return None  # caller falls back to heuristic; SQUIM only on Kaggle
    return SpectralContrastReward(sr=sr, **kw).to(device)
