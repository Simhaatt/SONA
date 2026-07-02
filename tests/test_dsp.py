"""Correctness anchors for the shared DSP front-end.

The most important guarantee: STFT->ISTFT with an unmodified spectrum reconstructs the input
(perfect reconstruction). If this drifts, every downstream metric is wrong.
"""

import numpy as np
import pytest

from src.dsp import STFT, BandMapper, FrameFeatures, estimate_noise_psd


def _sig(n=16000, sr=16000):
    rng = np.random.default_rng(0)
    t = np.arange(n) / sr
    x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
    x += 0.05 * rng.standard_normal(n)
    return x.astype(np.float32)


@pytest.mark.parametrize("n_fft,hop", [(512, 128), (1024, 256), (512, 256)])
def test_perfect_reconstruction(n_fft, hop):
    x = _sig()
    st = STFT(n_fft=n_fft, hop=hop)
    spec = st.stft(x)
    y = st.istft(spec, length=len(x))
    assert y.shape[0] == len(x)
    # Interior region should reconstruct to near machine precision.
    err = np.abs(x[n_fft:-n_fft] - y[n_fft:-n_fft]).max()
    assert err < 1e-5, f"max reconstruction error {err:.2e}"


def test_magphase_roundtrip():
    x = _sig()
    st = STFT(n_fft=512, hop=128)
    mag, phase = st.magphase(x)
    y = st.reconstruct(mag, phase, length=len(x))
    err = np.abs(x[512:-512] - y[512:-512]).max()
    assert err < 1e-5


def test_bin_count():
    st = STFT(n_fft=1024, hop=256)
    spec = st.stft(_sig())
    assert spec.shape[0] == 1024 // 2 + 1


def test_band_mapper_shapes_and_partition():
    n_fft = 512
    bm = BandMapper(n_fft=n_fft, sr=16000, n_bands=32, scale="bark")
    assert bm.analysis.shape == (32, n_fft // 2 + 1)
    assert bm.synthesis.shape == (n_fft // 2 + 1, 32)
    # Partition of unity: expanding an all-ones band gain gives all-ones per-bin gain.
    ones = np.ones(32)
    per_bin = bm.expand_gain(ones)
    assert np.allclose(per_bin, 1.0, atol=1e-5), per_bin.min()


def test_band_gain_bounds():
    # Expanding gains in [0,1] must stay in [0,1] (no overshoot from interpolation).
    bm = BandMapper(n_fft=512, n_bands=24, scale="mel")
    rng = np.random.default_rng(1)
    g = rng.uniform(0, 1, size=24)
    per_bin = bm.expand_gain(g)
    assert per_bin.min() >= -1e-6 and per_bin.max() <= 1 + 1e-6


def test_features_shape():
    st = STFT(n_fft=512, hop=128)
    bm = BandMapper(n_fft=512, n_bands=32)
    ff = FrameFeatures(bm)
    mag, _ = st.magphase(_sig())
    feats = ff(mag)
    assert feats.shape[1:] == (32, ff.n_channels)
    assert feats.shape[0] == mag.shape[1]
    assert np.isfinite(feats).all()


def test_noise_psd_estimators():
    st = STFT(n_fft=512, hop=128)
    mag, _ = st.magphase(_sig())
    power = mag**2
    for method in ("first_frames", "minstat"):
        noise = estimate_noise_psd(power, method=method)
        assert noise.shape == power.shape
        assert (noise >= 0).all()
