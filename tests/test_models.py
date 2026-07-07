"""Learned-model correctness: front-end consistency, shapes, and single-batch overfitting."""

import numpy as np
import torch

from src.utils.config import Config
from src.models import (
    TorchFrontend,
    MaskGRU,
    MaskTCN,
    AttentionMask,
    GTCRNMask,
    MambaGTCRNMask,
    GovernedGTCRNMask,
    ResidualGTCRNMask,
    NeuralEnhancer,
)
from src.data import SyntheticNoisyDataset
from src.train.train_supervised import overfit_batch
from src.train.losses import mask_polarity_loss, multi_resolution_magnitude_loss


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
    models = (
        MaskGRU(n_bands=24),
        MaskTCN(n_bands=24),
        AttentionMask(n_bands=24),
        GTCRNMask(n_bands=24, ch=16, hidden=24, n_blocks=2),
        MambaGTCRNMask(n_bands=24, ch=16, hidden=24, n_blocks=2, ssm_state=8),
        GovernedGTCRNMask(n_bands=24, ch=16, hidden=24, n_blocks=2),
        ResidualGTCRNMask(n_bands=24, ch=16, hidden=24, n_blocks=2),
    )
    for model in models:
        gains = model(feats)
        assert gains.shape == (3, feats.shape[1], 24)
        assert float(gains.min().detach()) >= 0.0 and float(gains.max().detach()) <= 1.0


def test_mask_polarity_loss_prefers_confident_masks():
    uncertain = torch.full((2, 5, 8), 0.5)
    confident = torch.cat([torch.zeros(2, 5, 4), torch.ones(2, 5, 4)], dim=2)
    assert float(mask_polarity_loss(uncertain, target=0.5)) > 0.0
    assert float(mask_polarity_loss(confident, target=0.5)) == 0.0


def test_multi_resolution_magnitude_loss_is_finite():
    x = torch.randn(2, 2048)
    y = x + 0.01 * torch.randn_like(x)
    loss = multi_resolution_magnitude_loss(x, y, resolutions=((128, 32), (256, 64)))
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


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
