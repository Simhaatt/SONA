"""Differentiable EQ / dynamic-range correctness: unity=identity, boosts/cuts behave, grads flow."""

import numpy as np
import torch

from src.models import ParametricEQ, Compressor, hum_notch_curve


def _nearest_bin(sr, n_fft, hz):
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return int(np.argmin(np.abs(freqs - hz)))


def test_eq_unity_is_identity():
    eq = ParametricEQ(sr=16000, n_fft=512)
    curve = eq.curve(torch.zeros(eq.n_bands))
    assert torch.allclose(curve, torch.ones_like(curve), atol=1e-4), curve.min()


def test_eq_boost_and_cut():
    sr, n_fft = 16000, 512
    eq = ParametricEQ(sr=sr, n_fft=n_fft)
    # Boost the band centred at 1000 Hz.
    b = eq.center_freqs.index(1000.0)
    gains = torch.zeros(eq.n_bands)
    gains[b] = 12.0
    curve = eq.curve(gains)
    bin1k = _nearest_bin(sr, n_fft, 1000.0)
    bin_far = _nearest_bin(sr, n_fft, 100.0)
    assert curve[bin1k] > 1.5, curve[bin1k]
    assert abs(curve[bin_far] - 1.0) < 0.2  # distant band roughly untouched
    # Cut.
    gains[b] = -12.0
    curve_cut = eq.curve(gains)
    assert curve_cut[bin1k] < 0.7, curve_cut[bin1k]


def test_eq_forward_shapes():
    eq = ParametricEQ(sr=16000, n_fft=512)
    mag = torch.rand(1, 257, 40)
    out = eq(mag, torch.zeros(eq.n_bands))
    assert out.shape == mag.shape
    assert torch.allclose(out, mag, atol=1e-4)


def test_eq_differentiable():
    eq = ParametricEQ(sr=16000, n_fft=512)
    g = torch.zeros(eq.n_bands, requires_grad=True)
    loss = eq.curve(g).sum()
    loss.backward()
    assert g.grad is not None and torch.isfinite(g.grad).all()


def test_hum_notch_attenuates_mains():
    sr, n_fft = 16000, 1024
    curve = hum_notch_curve(sr=sr, n_fft=n_fft, base_hz=50.0)
    bin50 = _nearest_bin(sr, n_fft, 50.0)
    bin1k = _nearest_bin(sr, n_fft, 1000.0)
    assert curve[bin50] < 0.1
    assert curve[bin1k] > 0.95


def test_compressor_identity_and_reduces_loud():
    comp_id = Compressor(ratio=1.0, makeup_db=0.0, ceiling_db=None)
    mag = torch.rand(1, 257, 30) + 0.1
    assert torch.allclose(comp_id(mag), mag, atol=1e-5)

    comp = Compressor(threshold_db=-24.0, ratio=4.0)
    loud = torch.full((1, 257, 5), 1.0)
    quiet = torch.full((1, 257, 5), 0.01)
    # Gain reduction (out/in) should be smaller for the loud frame than the quiet one.
    r_loud = (comp(loud) / loud).mean()
    r_quiet = (comp(quiet) / quiet).mean()
    assert r_loud < r_quiet
