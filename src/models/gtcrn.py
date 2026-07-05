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
