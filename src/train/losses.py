"""Differentiable training losses.

SI-SDR is the primary objective (it correlates with the SI-SDR(i) eval metric and is robust to
scale). A magnitude-spectrogram term stabilises early training and discourages over-suppression.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def si_sdr_loss(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Negative mean SI-SDR (so minimising improves SI-SDR). Shapes (B, N)."""
    est = estimate - estimate.mean(dim=-1, keepdim=True)
    ref = reference - reference.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(dim=-1, keepdim=True) / (ref.pow(2).sum(dim=-1, keepdim=True) + EPS)
    target = alpha * ref
    noise = est - target
    ratio = (target.pow(2).sum(dim=-1) + EPS) / (noise.pow(2).sum(dim=-1) + EPS)
    si_sdr = 10.0 * torch.log10(ratio)
    return -si_sdr.mean()


def magnitude_loss(estimate: torch.Tensor, reference: torch.Tensor, n_fft: int = 512,
                   hop: int = 128) -> torch.Tensor:
    """L1 distance between log-magnitude spectrograms of estimate and reference."""
    window = torch.hann_window(n_fft, periodic=True, device=estimate.device)
    kw = dict(n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
              center=True, return_complex=True)
    est_mag = torch.stft(estimate, **kw).abs()
    ref_mag = torch.stft(reference, **kw).abs()
    return torch.log1p(est_mag).sub(torch.log1p(ref_mag)).abs().mean()


def multi_resolution_magnitude_loss(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    resolutions: tuple[tuple[int, int], ...] = ((256, 64), (512, 128), (1024, 256)),
) -> torch.Tensor:
    """Average log-magnitude loss across several STFT resolutions."""
    losses = [
        magnitude_loss(estimate, reference, n_fft=n_fft, hop=hop)
        for n_fft, hop in resolutions
    ]
    return torch.stack(losses).mean()


def combined_loss(estimate: torch.Tensor, reference: torch.Tensor, mag_weight: float = 0.5,
                  n_fft: int = 512, hop: int = 128,
                  mr_mag_weight: float = 0.0) -> torch.Tensor:
    loss = si_sdr_loss(estimate, reference)
    if mag_weight > 0:
        loss = loss + mag_weight * magnitude_loss(estimate, reference, n_fft, hop)
    if mr_mag_weight > 0:
        loss = loss + mr_mag_weight * multi_resolution_magnitude_loss(estimate, reference)
    return loss


def mask_polarity_loss(gain: torch.Tensor, target: float = 0.0) -> torch.Tensor:
    """Encourage confident masks by penalizing low average distance from 0.5.

    `target` is the desired mean polarity in [0, 1], where polarity is
    ``2 * abs(gain - 0.5)``. A target of 0 disables the pressure, while larger values encourage
    gains to move away from the uncertain middle without specifying speech/noise labels.
    """
    if target <= 0:
        return torch.zeros((), device=gain.device)
    polarity = (2.0 * (gain - 0.5).abs()).clamp(0.0, 1.0).mean()
    return torch.relu(torch.as_tensor(target, device=gain.device) - polarity)
