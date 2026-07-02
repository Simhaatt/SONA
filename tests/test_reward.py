"""Reference-free reward + pluggable online-adaptation + noise-split correctness."""

import numpy as np
import torch

from src.eval.reward_squim import (
    SpectralContrastReward, make_reward, squim_available,
)
from src.models import TorchFrontend, RLAttentionPolicy, OnlineAdapter
from src.data import SyntheticNoisyDataset


def test_spectral_contrast_reward_prefers_tonal():
    sr, n = 16000, 16000
    t = np.arange(n) / sr
    tonal = torch.tensor(0.5 * np.sin(2 * np.pi * 500 * t), dtype=torch.float32)
    rng = np.random.default_rng(0)
    noise = torch.tensor(0.5 * rng.standard_normal(n), dtype=torch.float32)
    r = SpectralContrastReward(sr=sr)
    r_tonal, r_noise = float(r(tonal)), float(r(noise))
    assert np.isfinite(r_tonal) and np.isfinite(r_noise)
    assert r_tonal > r_noise, (r_tonal, r_noise)


def test_make_reward_fallback_local():
    # Locally (no torchaudio) 'squim' -> None (caller falls back), 'auto' -> spectral heuristic.
    if not squim_available():
        assert make_reward("squim") is None
        assert isinstance(make_reward("auto"), SpectralContrastReward)
    assert make_reward("none") is None
    assert isinstance(make_reward("spectral"), SpectralContrastReward)


def test_online_adapter_with_waveform_reward():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    pol = RLAttentionPolicy(n_bands=32, hidden=64, n_layers=2)
    reward = SpectralContrastReward(sr=16000)
    adapter = OnlineAdapter(pol, tf, steps=6, lr=0.1, reward_fn=reward, name="rl_online_squim")
    s = next(iter(SyntheticNoisyDataset(n=1, duration=1.0, seed=3)))
    y = adapter.enhance(s.noisy, s.sr)
    assert y.shape[0] == len(s.noisy) and np.isfinite(y).all()
    assert adapter.adaptation_delta_norm(s.noisy) > 1e-5


def test_noise_type_split_disjoint():
    train = list(SyntheticNoisyDataset(n=6, duration=0.5, seed=1, noise_types=("white", "pink")))
    test = list(SyntheticNoisyDataset(n=4, duration=0.5, seed=2, noise_types=("burst",)))
    assert {s.noise_type for s in train} == {"white", "pink"}
    assert {s.noise_type for s in test} == {"burst"}
