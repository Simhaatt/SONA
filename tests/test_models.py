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
from src.train.losses import (
    compressed_complex_stft_loss,
    mask_polarity_loss,
    multi_resolution_magnitude_loss,
)


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


def test_complex_mask_identity_reconstructs():
    # a complex ratio mask of (1 + 0j) must reconstruct the unmasked signal exactly
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    x = torch.randn(2, 16000)
    spec = tf.stft(x)
    ident = torch.zeros(spec.shape[0], spec.shape[2], tf.n_bands, 2)
    ident[..., 0] = 1.0  # real=1, imag=0
    y = tf.apply_band_complex(spec, ident, length=x.shape[-1])
    ref = torch.istft(spec, n_fft=512, hop_length=128, win_length=512,
                      window=tf.window, center=True, length=x.shape[-1])
    assert (y - ref).abs().max().item() < 1e-4


def test_complex_gtcrn_shapes_and_flag():
    from src.models import ComplexGTCRNMask
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=24)
    x = torch.randn(3, 8000)
    spec, feats = tf.analyze(x)
    model = ComplexGTCRNMask(n_bands=24, ch=16, hidden=24, n_blocks=2)
    assert model.is_complex
    out = model(feats)
    assert out.shape == (3, feats.shape[1], 24, 2)          # real+imag per band
    assert out.abs().max().item() <= model.bound + 1e-5     # bounded mask
    est = tf.apply_band_complex(spec, out, length=x.shape[-1])
    assert est.shape == x.shape and torch.isfinite(est).all()


def test_complex_gtcrn_overfits_single_batch():
    # the complex path must be differentiable end-to-end and able to learn
    from src.models import ComplexGTCRNMask
    from src.train.losses import si_sdr_loss
    torch.manual_seed(0)
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    x = torch.randn(2, 12000) * 0.1
    spec, feats = tf.analyze(x)
    clean = torch.tanh(x * 3.0) * 0.1
    model = ComplexGTCRNMask(n_bands=32, ch=32, hidden=48, n_blocks=3)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    start = None
    for i in range(100):
        opt.zero_grad()
        est = tf.apply_band_complex(spec, model(feats), length=x.shape[-1])
        loss = si_sdr_loss(est, clean)
        loss.backward()
        opt.step()
        if i == 0:
            start = loss.item()
    assert loss.item() < start - 1.0, f"no learning: start={start:.3f} end={loss.item():.3f}"


def test_complex_neural_enhancer_roundtrip():
    from src.models import ComplexGTCRNMask
    tf = TorchFrontend(n_fft=512, hop=128, n_bands=32)
    model = ComplexGTCRNMask(n_bands=32, ch=16, hidden=24, n_blocks=2)
    enh = NeuralEnhancer(model, tf, name="complex_gtcrn")
    s = next(iter(SyntheticNoisyDataset(n=1, duration=1.0, seed=7)))
    y = enh.enhance(s.noisy, s.sr)
    assert y.shape[0] == len(s.noisy) and np.isfinite(y).all()


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


def test_compressed_complex_stft_loss_is_phase_aware_and_differentiable():
    reference = torch.randn(2, 2048)
    estimate = reference.clone().requires_grad_(True)
    identical = compressed_complex_stft_loss(estimate, reference, n_fft=256, hop=64)
    assert float(identical.detach()) < 1e-6
    changed = compressed_complex_stft_loss(-estimate, reference, n_fft=256, hop=64)
    assert float(changed.detach()) > float(identical.detach())
    changed.backward()
    assert estimate.grad is not None and torch.isfinite(estimate.grad).all()


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
