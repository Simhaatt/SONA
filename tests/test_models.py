"""Learned-model correctness: front-end consistency, shapes, and single-batch overfitting."""

import numpy as np
import torch

from src.utils.config import Config
from src.models import TorchFrontend, MaskGRU, MaskTCN, AttentionMask, NeuralEnhancer
from src.data import SyntheticNoisyDataset
from src.train.train_supervised import overfit_batch


def _cfg(kind, **model_kw):
    return Config({
        "sr": 16000,
        "stft": {"n_fft": 512, "hop": 128},
        "bands": {"n_bands": 32, "scale": "bark"},
        "model": {"kind": kind, **model_kw},
    })


def test_torch_frontend_perfect_reconstruction():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    x = torch.randn(2, 16000)
    spec = tf.stft(x)
    ones = torch.ones(spec.shape[0], spec.shape[2], tf.n_bands)  # (B, T, n_bands) all-pass
    y = tf.apply_band_gain(spec, ones, length=x.shape[-1])
    err = (x[:, 512:-512] - y[:, 512:-512]).abs().max().item()
    assert err < 1e-4, err


def test_feature_and_gain_shapes():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=24)
    x = torch.randn(3, 8000)
    spec, feats = tf.analyze(x)
    assert feats.shape[0] == 3 and feats.shape[2] == 24 and feats.shape[3] == 3
    for Model in (MaskGRU, MaskTCN, AttentionMask):
        model = Model(n_bands=24)
        gains = model(feats)
        assert gains.shape == (3, feats.shape[1], 24)
        assert float(gains.min().detach()) >= 0.0 and float(gains.max().detach()) <= 1.0


def test_gru_overfits_single_batch():
    # A model with enough capacity should drive a single-batch loss well below its start value.
    samples = list(SyntheticNoisyDataset(n=2, duration=1.0, seed=5))
    noisy = torch.stack([torch.from_numpy(s.noisy) for s in samples])
    clean = torch.stack([torch.from_numpy(s.clean) for s in samples])
    start, end = overfit_batch(_cfg("mask_gru", hidden=96, layers=2), noisy, clean, steps=120)
    assert end < start - 1.0, f"no meaningful learning: start={start:.3f} end={end:.3f}"


def test_neural_enhancer_roundtrip():
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    model = MaskTCN(n_bands=32, ch=32, n_blocks=3)
    enh = NeuralEnhancer(model, tf, name="mask_tcn")
    s = next(iter(SyntheticNoisyDataset(n=1, duration=1.0, seed=7)))
    y = enh.enhance(s.noisy, s.sr)
    assert y.shape[0] == len(s.noisy) and np.isfinite(y).all()
