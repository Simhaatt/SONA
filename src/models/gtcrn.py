"""Bark-band GTCRN-style per-band mask predictor.

The repo's frontend already maps spectra to Bark/ERB-like bands, so this backbone keeps that
representation and learns a compact gated time-frequency encoder followed by a causal recurrent
decoder. It is intentionally smaller than paper-scale GTCRN variants so it remains Kaggle/edge
friendly while giving Layer B a real convolutional-recurrent architecture to test.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseGainModel


class _GatedTFBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        dilation: int,
        kernel_t: int = 3,
        kernel_f: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pad_t = (kernel_t - 1) * dilation
        pad_f = kernel_f // 2
        self.conv = nn.Conv2d(
            ch,
            2 * ch,
            kernel_size=(kernel_t, kernel_f),
            padding=(self.pad_t, pad_f),
            dilation=(dilation, 1),
        )
        self.norm = nn.GroupNorm(1, ch)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.pad_t:
            y = y[:, :, : -self.pad_t, :]
        value, gate = y.chunk(2, dim=1)
        y = value * torch.sigmoid(gate)
        y = self.drop(self.norm(y))
        return x + y


class GTCRNMask(BaseGainModel):
    """Compact causal gated temporal convolutional recurrent network.

    Input and output match the existing learned mask models:
    ``(B, T, n_bands, n_feat_channels) -> (B, T, n_bands)`` logits.
    """

    def __init__(
        self,
        n_bands: int,
        n_feat_channels: int = 3,
        ch: int = 64,
        hidden: int = 96,
        n_blocks: int = 4,
        kernel_t: int = 3,
        kernel_f: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(n_bands, n_feat_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(n_feat_channels, ch, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(1, ch),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                _GatedTFBlock(
                    ch,
                    dilation=2**i,
                    kernel_t=kernel_t,
                    kernel_f=kernel_f,
                    dropout=dropout,
                )
                for i in range(n_blocks)
            ]
        )
        self.band_gru = nn.GRU(ch, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def predict_gain(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, k, _ = feats.shape
        x = feats.permute(0, 3, 1, 2)  # (B, C, T, K)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 3, 2, 1).reshape(b * k, t, -1)  # (B*K, T, C)
        h, _ = self.band_gru(x)
        logits = self.head(h).reshape(b, k, t).transpose(1, 2)
        return logits


class _MambaLiteBlock(nn.Module):
    """Small causal selective state-space block.

    This is an implementation-friendly Mamba-style memory block: input-dependent gates modulate a
    diagonal stable state update over time. It avoids custom CUDA dependencies while preserving the
    key behavior we need to test next: long causal temporal memory beyond the GRU baseline.
    """

    def __init__(self, dim: int, state: int = 16, expand: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        inner = dim * expand
        self.in_proj = nn.Linear(dim, 2 * inner)
        self.x_proj = nn.Linear(inner, state)
        self.dt_proj = nn.Linear(inner, state)
        self.out_proj = nn.Linear(inner, dim)
        self.skip = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.log_decay = nn.Parameter(torch.linspace(-3.0, -1.0, state))
        self.readout = nn.Parameter(torch.randn(inner, state) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = torch.nn.functional.silu(value)
        gate = torch.sigmoid(gate)

        drive = self.x_proj(value)
        dt = torch.sigmoid(self.dt_proj(value))
        decay = torch.exp(-torch.exp(self.log_decay)).view(1, -1)
        state = torch.zeros(x.size(0), decay.size(1), device=x.device, dtype=x.dtype)
        outs = []
        for step in range(x.size(1)):
            state = decay * state + dt[:, step, :] * drive[:, step, :]
            outs.append(state)
        memory = torch.stack(outs, dim=1)
        y = value + torch.einsum("bts,hs->bth", memory, self.readout)
        y = self.out_proj(y * gate)
        return residual + self.drop(y) + 0.1 * self.skip(residual)


class MambaGTCRNMask(BaseGainModel):
    """GTCRN encoder with Mamba-style causal state-space memory before mask decoding."""

    def __init__(
        self,
        n_bands: int,
        n_feat_channels: int = 3,
        ch: int = 96,
        hidden: int = 96,
        n_blocks: int = 4,
        ssm_state: int = 16,
        ssm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(n_bands, n_feat_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(n_feat_channels, ch, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(1, ch),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_GatedTFBlock(ch, dilation=2**i, dropout=dropout) for i in range(n_blocks)]
        )
        self.memory = nn.Sequential(
            *[_MambaLiteBlock(ch, state=ssm_state, dropout=dropout) for _ in range(ssm_layers)]
        )
        self.band_gru = nn.GRU(ch, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def predict_gain(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, k, _ = feats.shape
        x = feats.permute(0, 3, 1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 3, 2, 1).reshape(b * k, t, -1)
        x = self.memory(x)
        h, _ = self.band_gru(x)
        return self.head(h).reshape(b, k, t).transpose(1, 2)


class GovernedGTCRNMask(BaseGainModel):
    """Dual-mask GTCRN with a tiny reference-free blend governor.

    The backbone stays close to the validated GTCRN. Instead of a single decoder head, it predicts
    conservative and aggressive gains and lets a learned per-band governor blend them. A learned
    safe floor prevents very low gains in uncertain speech-like regions.
    """

    def __init__(
        self,
        n_bands: int,
        n_feat_channels: int = 3,
        ch: int = 96,
        hidden: int = 96,
        n_blocks: int = 4,
        floor_max: float = 0.20,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(n_bands, n_feat_channels)
        self.floor_max = floor_max
        self.stem = nn.Sequential(
            nn.Conv2d(n_feat_channels, ch, kernel_size=(1, 3), padding=(0, 1)),
            nn.GroupNorm(1, ch),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_GatedTFBlock(ch, dilation=2**i, dropout=dropout) for i in range(n_blocks)]
        )
        self.band_gru = nn.GRU(ch, hidden, num_layers=1, batch_first=True)
        self.cons_head = nn.Linear(hidden, 1)
        self.aggr_head = nn.Linear(hidden, 1)
        self.alpha_head = nn.Linear(hidden, 1)
        self.floor_head = nn.Linear(hidden, 1)

    @staticmethod
    def _to_logits(gain: torch.Tensor) -> torch.Tensor:
        gain = gain.clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(gain)

    def _heads(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, k, _ = feats.shape
        x = feats.permute(0, 3, 1, 2)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 3, 2, 1).reshape(b * k, t, -1)
        h, _ = self.band_gru(x)

        cons = self.cons_head(h).reshape(b, k, t).transpose(1, 2)
        aggr = self.aggr_head(h).reshape(b, k, t).transpose(1, 2)
        alpha = self.alpha_head(h).reshape(b, k, t).transpose(1, 2)
        floor = self.floor_head(h).reshape(b, k, t).transpose(1, 2)
        return cons, aggr, alpha, floor

    def predict_gain(self, feats: torch.Tensor) -> torch.Tensor:
        cons_logits, aggr_logits, alpha_logits, floor_logits = self._heads(feats)
        cons_gain = torch.sigmoid(cons_logits)
        aggr_gain = torch.sigmoid(aggr_logits)
        # Keep branch semantics stable even if heads swap during early training.
        conservative = torch.maximum(cons_gain, aggr_gain)
        aggressive = torch.minimum(cons_gain, aggr_gain)
        alpha = torch.sigmoid(alpha_logits)
        floor = self.floor_max * torch.sigmoid(floor_logits)
        final = alpha * aggressive + (1.0 - alpha) * conservative
        final = torch.maximum(final, floor)
        return self._to_logits(final)
