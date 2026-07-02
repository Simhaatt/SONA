"""Attention-based per-band mask predictor.

A compact Transformer encoder over time (with per-frame band+feature embedding). Attention lets
the model weigh temporal context when deciding each frame's per-band suppression — a step toward
the RL+attention hybrid, and a learned baseline in its own right.

Causal masking is optional so the same block can be used in the low-latency streaming setting.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseGainModel


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionMask(BaseGainModel):
    def __init__(self, n_bands: int, n_feat_channels: int = 3, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, ff: int = 256, dropout: float = 0.1, causal: bool = True) -> None:
        super().__init__(n_bands, n_feat_channels)
        self.causal = causal
        self.embed = nn.Linear(self.in_dim, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_bands)

    def predict_gain(self, feats: torch.Tensor) -> torch.Tensor:
        b, t, k, c = feats.shape
        x = self.embed(feats.reshape(b, t, k * c))
        x = self.pos(x)
        mask = None
        if self.causal:
            mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        h = self.encoder(x, mask=mask)
        return self.head(h)  # (B, T, n_bands) logits
