"""Online-adaptation modules — the low-dimensional parameters updated at inference.

The core thesis is that adapting a *small* set of parameters over the (interpretable) gain output
is what makes reference-free online adaptation stable and edge-cheap. This module provides
swappable adapters of increasing dimensionality so we can ablate exactly that:

  ScalarBias   -- 1 param  (a single global gain bias)
  LowRankBias  -- r params (smooth band correction via a fixed DCT-ish basis; r << n_bands)
  BandBias     -- n_bands params (independent per-band additive bias) [default; original behaviour]
  FiLM         -- 2*n_bands params (per-band scale + shift of the policy's gains)

Each adapter creates fresh parameters per utterance (`reset`), exposes them to the optimizer
(`params`), applies its correction to the base gains (`apply`), and reports an L2 regulariser
(`reg`) that keeps the online correction a *refinement* of the trained policy.
"""

from __future__ import annotations

import math

import torch


class BaseAdapter:
    dim: int = 0

    def __init__(self, n_bands: int) -> None:
        self.n_bands = n_bands
        self._params: list[torch.Tensor] = []

    def reset(self, device: str = "cpu") -> list[torch.Tensor]:
        raise NotImplementedError

    def params(self) -> list[torch.Tensor]:
        return self._params

    def apply(self, base_gain: torch.Tensor) -> torch.Tensor:
        """base_gain: (1, T, n_bands) -> adapted gain (pre-clamp), same shape."""
        raise NotImplementedError

    def reg(self) -> torch.Tensor:
        return sum((p.pow(2).mean() for p in self._params), start=torch.zeros((), device=self._dev()))

    def correction_norm(self) -> float:
        with torch.no_grad():
            return float(sum(p.abs().mean() for p in self._params) / max(len(self._params), 1))

    def _dev(self):
        return self._params[0].device if self._params else torch.device("cpu")


class ScalarBias(BaseAdapter):
    dim = 1

    def reset(self, device: str = "cpu") -> list[torch.Tensor]:
        self._params = [torch.zeros(1, device=device, requires_grad=True)]
        return self._params

    def apply(self, base_gain: torch.Tensor) -> torch.Tensor:
        return base_gain + self._params[0]


class BandBias(BaseAdapter):
    """Independent per-band additive bias (the original OnlineAdapter behaviour)."""

    def __init__(self, n_bands: int) -> None:
        super().__init__(n_bands)
        self.dim = n_bands

    def reset(self, device: str = "cpu") -> list[torch.Tensor]:
        self._params = [torch.zeros(self.n_bands, device=device, requires_grad=True)]
        return self._params

    def apply(self, base_gain: torch.Tensor) -> torch.Tensor:
        return base_gain + self._params[0].view(1, 1, -1)


class LowRankBias(BaseAdapter):
    """Smooth band correction: delta = basis @ coeff, with a fixed low-freq DCT basis.

    Only `rank` coefficients are learned, so adaptation is low-dimensional but still shapes the
    correction across bands smoothly. This is the knob for the dimensionality ablation.
    """

    def __init__(self, n_bands: int, rank: int = 4) -> None:
        super().__init__(n_bands)
        self.rank = rank
        self.dim = rank
        self._basis = self._dct_basis(n_bands, rank)  # (n_bands, rank)

    @staticmethod
    def _dct_basis(n: int, r: int) -> torch.Tensor:
        idx = torch.arange(n).float()
        cols = []
        for k in range(r):
            cols.append(torch.cos(math.pi * (idx + 0.5) * k / n))
        return torch.stack(cols, dim=1)  # (n, r)

    def reset(self, device: str = "cpu") -> list[torch.Tensor]:
        self._basis = self._basis.to(device)
        self._params = [torch.zeros(self.rank, device=device, requires_grad=True)]
        return self._params

    def apply(self, base_gain: torch.Tensor) -> torch.Tensor:
        delta = (self._basis @ self._params[0])  # (n_bands,)
        return base_gain + delta.view(1, 1, -1)


class FiLM(BaseAdapter):
    """Per-band affine (scale gamma + shift beta) of the policy gains. 2*n_bands params."""

    def __init__(self, n_bands: int) -> None:
        super().__init__(n_bands)
        self.dim = 2 * n_bands

    def reset(self, device: str = "cpu") -> list[torch.Tensor]:
        gamma = torch.zeros(self.n_bands, device=device, requires_grad=True)  # exp(0)=1 -> identity
        beta = torch.zeros(self.n_bands, device=device, requires_grad=True)
        self._params = [gamma, beta]
        return self._params

    def apply(self, base_gain: torch.Tensor) -> torch.Tensor:
        gamma, beta = self._params
        return base_gain * torch.exp(gamma).view(1, 1, -1) + beta.view(1, 1, -1)


ADAPTERS = {
    "scalar": ScalarBias,
    "lowrank": LowRankBias,
    "band_bias": BandBias,
    "film": FiLM,
}


def make_adapter(kind: str, n_bands: int, **kw) -> BaseAdapter:
    if kind not in ADAPTERS:
        raise ValueError(f"unknown adapter '{kind}' (choose from {list(ADAPTERS)})")
    return ADAPTERS[kind](n_bands, **kw)
