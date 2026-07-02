"""Adaptation modules + OnlineAdapter with swappable adapters (dimensionality ablation)."""

import numpy as np
import torch

from src.models import (
    TorchFrontend, RLAttentionPolicy, OnlineAdapter, make_adapter, ADAPTERS,
)
from src.models.adapt import ScalarBias, BandBias, LowRankBias, FiLM
from src.eval.reward_squim import SpectralContrastReward
from src.data import SyntheticNoisyDataset


def test_adapter_dims():
    n = 32
    assert ScalarBias(n).dim == 1
    assert LowRankBias(n, rank=4).dim == 4
    assert BandBias(n).dim == n
    assert FiLM(n).dim == 2 * n


def test_adapter_identity_at_reset():
    # Fresh (zero) params must leave the base gain unchanged (adaptation is a *refinement*).
    base = torch.rand(1, 10, 32)
    for kind in ADAPTERS:
        a = make_adapter(kind, 32)
        a.reset("cpu")
        out = a.apply(base)
        assert torch.allclose(out, base, atol=1e-6), kind


def test_adapter_params_are_learnable():
    for kind in ADAPTERS:
        a = make_adapter(kind, 16)
        params = a.reset("cpu")
        assert params and all(p.requires_grad for p in params)
        # A gradient step should change the correction norm away from zero.
        loss = a.apply(torch.rand(1, 5, 16)).sum()
        loss.backward()
        assert all(p.grad is not None for p in params), kind


def test_online_adapter_with_each_adapter_runs():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    pol = RLAttentionPolicy(n_bands=32, hidden=64, n_layers=2)
    reward = SpectralContrastReward(sr=16000)
    s = next(iter(SyntheticNoisyDataset(n=1, duration=1.0, seed=5)))
    for kind in ("scalar", "lowrank", "band_bias", "film"):
        adapter = make_adapter(kind, 32)
        adp = OnlineAdapter(pol, tf, steps=4, lr=0.1, reward_fn=reward, adapter=adapter,
                            name=f"online_{kind}")
        y = adp.enhance(s.noisy, s.sr)
        assert y.shape[0] == len(s.noisy) and np.isfinite(y).all()
