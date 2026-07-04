"""RL policy + online-adaptation correctness."""

import numpy as np
import torch

from src.utils.config import Config
from src.models import TorchFrontend, RLGainPolicy, RLAttentionPolicy, OnlineAdapter, NeuralEnhancer, MaskGRU
from src.models.rl_attention import speech_presence, reference_free_reward
from src.data import SyntheticNoisyDataset
from src.train.train_rl import improve_reward_on_batch


def _cfg(kind):
    return Config({
        "sr": 16000,
        "stft": {"n_fft": 512, "hop": 128},
        "bands": {"n_bands": 32, "scale": "bark"},
        "model": {"kind": kind, "hidden": 96},
    })


def test_policy_act_shapes_and_bounds():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    x = torch.randn(2, 8000)
    _, feats = tf.analyze(x)
    pol = RLGainPolicy(n_bands=32, trunk="gru", hidden=64)
    gain, logp, value, ent = pol.act(feats)
    T = feats.shape[1]
    assert gain.shape == (2, T, 32) == logp.shape == value.shape == ent.shape
    assert float(gain.min()) > 0.0 and float(gain.max()) < 1.0
    # Deterministic forward() (mean gain) also in [0,1] for NeuralEnhancer compatibility.
    det = pol(feats)
    assert float(det.min().detach()) >= 0.0 and float(det.max().detach()) <= 1.0


def test_rl_policy_gradient_improves_reward():
    samples = list(SyntheticNoisyDataset(n=2, duration=1.0, seed=11))
    noisy = torch.stack([torch.from_numpy(s.noisy) for s in samples])
    clean = torch.stack([torch.from_numpy(s.clean) for s in samples])
    start, end = improve_reward_on_batch(_cfg("rl_gru"), noisy, clean, steps=120)
    assert end > start, f"reward did not improve: start={start:.4f} end={end:.4f}"


def test_reference_free_reward_prefers_matching_gain():
    presence = torch.tensor([[[1.0, 0.0]]]).permute(0, 2, 1)  # (1, n_bands=2, T=1)
    good = reference_free_reward(torch.tensor([[[1.0, 0.0]]]).permute(0, 2, 1), presence)
    bad = reference_free_reward(torch.tensor([[[0.0, 1.0]]]).permute(0, 2, 1), presence)
    assert float(good) > float(bad)


def test_speech_presence_range():
    p = speech_presence(torch.rand(1, 32, 50) + 0.1)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


def test_online_adapter_runs_and_adapts():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    pol = RLAttentionPolicy(n_bands=32, hidden=64, n_layers=2)
    adapter = OnlineAdapter(pol, tf, steps=6, lr=0.1)
    s = next(iter(SyntheticNoisyDataset(n=1, duration=1.0, seed=13)))
    y = adapter.enhance(s.noisy, s.sr)
    assert y.shape[0] == len(s.noisy) and np.isfinite(y).all()
    # Online adaptation should produce a non-trivial per-band bias.
    assert adapter.adaptation_delta_norm(s.noisy) > 1e-4


def test_online_adapter_wraps_supervised_model_with_anchor():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    model = MaskGRU(n_bands=32, hidden=32, layers=1)

    def reward_fn(enhanced, noisy):
        return -((enhanced - noisy) ** 2).mean()

    adapter = OnlineAdapter(model, tf, steps=2, lr=0.05, anchor_weight=0.1, reward_fn=reward_fn)
    s = next(iter(SyntheticNoisyDataset(n=1, duration=1.0, seed=14)))
    y = adapter.enhance(s.noisy, s.sr)
    assert y.shape[0] == len(s.noisy) and np.isfinite(y).all()


def test_online_adapter_trust_region_bounds_gain_delta():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    model = MaskGRU(n_bands=32, hidden=32, layers=1)
    adapter = OnlineAdapter(model, tf, steps=1, max_gain_delta=0.05)
    base = torch.full((1, 4, 32), 0.5)
    adapter.adapter.reset("cpu")
    with torch.no_grad():
        adapter.adapter.params()[0].fill_(1.0)
    gain = adapter._bounded_gain(base)
    assert torch.all((gain - base).abs() <= 0.0501)


def test_online_adapter_guard_rejects_large_output_drift():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    model = MaskGRU(n_bands=32, hidden=32, layers=1)
    adapter = OnlineAdapter(model, tf, steps=1, guard_max_output_delta=0.0)
    spec = torch.ones(1, 257, 4, dtype=torch.complex64)
    x = torch.zeros(1, 512)
    base_gain = torch.full((1, 4, 32), 0.5)
    adapted_gain = torch.full((1, 4, 32), 0.75)
    base_enhanced = tf.apply_band_gain(spec, base_gain, length=x.shape[-1])
    presence = torch.ones(1, 32, 4)
    guarded = adapter._guarded_gain(spec, x, base_gain, base_enhanced, adapted_gain, presence, x.shape[-1])
    assert torch.equal(guarded, base_gain)


def test_rl_policy_plugs_into_neural_enhancer():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    pol = RLGainPolicy(n_bands=32, trunk="gru", hidden=64)
    enh = NeuralEnhancer(pol, tf, name="rl_gru")
    s = next(iter(SyntheticNoisyDataset(n=1, duration=1.0, seed=15)))
    y = enh.enhance(s.noisy, s.sr)
    assert y.shape[0] == len(s.noisy) and np.isfinite(y).all()
