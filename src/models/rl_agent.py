"""Reinforcement-learning gain policy (actor-critic) over per-band attenuation.

Framing: each STFT frame is a contextual-bandit step. The policy observes the per-band feature
vector (with temporal context via its recurrent/attention trunk) and outputs, for every band, a
**Beta distribution** over the gain in (0, 1). Actions (gains) are sampled during training and set
to the distribution mean at inference.

Why RL rather than supervised regression (the paper/patent argument):
  * the reward can be an arbitrary, possibly non-differentiable perceptual signal, and
  * the same policy supports **reference-free online adaptation** at inference (see rl_attention).

A per-band critic estimates the expected reward, giving a low-variance advantage for the
contextual-bandit policy-gradient update (see `src/train/train_rl.py`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from .attention import _PositionalEncoding

GAIN_EPS = 1e-4  # keep sampled gains inside (0,1) so Beta.log_prob stays finite


def _make_trunk(kind: str, in_dim: int, hidden: int, **kw) -> tuple[nn.Module, int]:
    """Return (trunk, out_dim). Trunk maps (B, T, in_dim) -> (B, T, out_dim)."""
    if kind == "gru":
        layers = int(kw.get("layers", 2))
        return _GRUTrunk(in_dim, hidden, layers, kw.get("dropout", 0.1)), hidden
    if kind == "attention":
        return (
            _AttnTrunk(
                in_dim, hidden,
                n_heads=int(kw.get("n_heads", 4)),
                n_layers=int(kw.get("n_layers", 3)),
                ff=int(kw.get("ff", 256)),
                dropout=float(kw.get("dropout", 0.1)),
                causal=bool(kw.get("causal", True)),
            ),
            hidden,
        )
    raise ValueError(f"unknown trunk kind '{kind}'")


class _GRUTrunk(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return h


class _AttnTrunk(nn.Module):
    def __init__(self, in_dim: int, d_model: int, n_heads: int, n_layers: int, ff: int,
                 dropout: float, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.embed = nn.Linear(in_dim, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout,
                                           batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.embed(x))
        mask = None
        if self.causal:
            t = x.size(1)
            mask = torch.triu(torch.full((t, t), float("-inf"), device=x.device), diagonal=1)
        return self.encoder(x, mask=mask)


class RLGainPolicy(nn.Module):
    """Actor-critic policy: Beta(alpha, beta) per band + per-band value head."""

    def __init__(self, n_bands: int, n_feat_channels: int = 3, trunk: str = "gru",
                 hidden: int = 128, max_conc: float = 40.0, **trunk_kw) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.n_feat_channels = n_feat_channels
        self.in_dim = n_bands * n_feat_channels
        self.max_conc = max_conc  # cap Beta concentration for numerical stability
        self.trunk, h = _make_trunk(trunk, self.in_dim, hidden, **trunk_kw)
        self.actor = nn.Linear(h, 2 * n_bands)   # -> (alpha, beta) params
        self.critic = nn.Linear(h, n_bands)      # per-band value baseline

    # ------------------------------------------------------------------ core
    def _params(self, feats: torch.Tensor):
        b, t, k, c = feats.shape
        x = feats.reshape(b, t, k * c)
        h = self.trunk(x)
        raw = self.actor(h).reshape(b, t, self.n_bands, 2)
        # softplus -> (0, inf); +1 keeps distributions unimodal; cap for stability.
        conc = 1.0 + torch.clamp(F.softplus(raw), max=self.max_conc)
        alpha, beta = conc[..., 0], conc[..., 1]  # (B, T, n_bands)
        value = self.critic(h)  # (B, T, n_bands)
        return alpha, beta, value

    def dist(self, feats: torch.Tensor) -> tuple[Beta, torch.Tensor]:
        alpha, beta, value = self._params(feats)
        return Beta(alpha, beta), value

    def act(self, feats: torch.Tensor, deterministic: bool = False):
        """Return (gain, log_prob, value, entropy), each (B, T, n_bands)."""
        dist, value = self.dist(feats)
        alpha, beta = dist.concentration1, dist.concentration0
        if deterministic:
            gain = alpha / (alpha + beta)
        else:
            gain = dist.sample()
        gain = gain.clamp(GAIN_EPS, 1.0 - GAIN_EPS)
        return gain, dist.log_prob(gain), value, dist.entropy()

    # -------------------------------------------------- Enhancer-compatible API
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Deterministic per-band gain (distribution mean), so this plugs into NeuralEnhancer."""
        alpha, beta, _ = self._params(feats)
        return alpha / (alpha + beta)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
