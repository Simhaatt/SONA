"""Supervised per-band mask predictors: a recurrent (GRU) and a convolutional (TCN) variant.

These are the "frozen DNN" reference points the deck critiques — strong, trainable, but fixed at
inference. Both are deliberately small (Kaggle-T4 / RTX-4050 friendly, edge-plausible).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseGainModel


class MaskGRU(BaseGainModel):
    def __init__(self, n_bands: int, n_feat_channels: int = 3, hidden: int = 128, layers: int = 2,
                 bidirectional: bool = False, dropout: float = 0.1) -> None:
        super().__init__(n_bands, n_feat_channels)
        self.gru = nn.GRU(
            input_size=self.in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        out_in = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(out_in, n_bands)

    def predict_gain(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, k, c = feats.shape
        x = feats.reshape(b, t, k * c)
        h, _ = self.gru(x)
        return self.head(h)  # (B, T, n_bands) logits


class _TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation  # causal padding
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.pad = pad
        self.norm = nn.GroupNorm(1, ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x (B, C, T)
        y = self.conv(x)
        if self.pad:
            y = y[..., : -self.pad]  # keep causality (no future leakage)
        y = self.drop(self.act(self.norm(y)))
        return x + y


class MaskTCN(BaseGainModel):
    """Causal temporal convolutional network — low-latency-friendly for the real-time target."""

    def __init__(self, n_bands: int, n_feat_channels: int = 3, ch: int = 128,
                 n_blocks: int = 5, kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__(n_bands, n_feat_channels)
        self.inp = nn.Conv1d(self.in_dim, ch, 1)
        self.blocks = nn.ModuleList(
            [_TCNBlock(ch, dilation=2**i, kernel=kernel, dropout=dropout) for i in range(n_blocks)]
        )
        self.head = nn.Conv1d(ch, n_bands, 1)

    def predict_gain(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, k, c = feats.shape
        x = feats.reshape(b, t, k * c).transpose(1, 2)  # (B, in_dim, T)
        x = self.inp(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).transpose(1, 2)  # (B, T, n_bands) logits
