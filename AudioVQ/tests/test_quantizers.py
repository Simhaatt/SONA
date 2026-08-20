"""Correctness tests for the quantizers, including validation against the paper's theory.

The load-bearing test is `test_mse_distortion_matches_paper_bounds`: TurboQuant states specific
MSE distortion values for b = 1..4 bits/coordinate on unit vectors (~0.36, 0.117, 0.03, 0.009).
If our implementation reproduces those, the rotation + Lloyd-Max codebook is behaving as the paper
describes and every downstream audio result rests on a correct implementation. If it does not, no
audio experiment is worth running yet.

Run:  python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.quantizers import (  # noqa: E402
    RotatedScalarQuantizer,
    UniformScalarQuantizer,
    lloyd_max_codebook,
    sphere_coordinate_samples,
)

# Paper's stated MSE distortion for b = 1,2,3,4 bits (Theorem 1 refinement).
PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


def random_unit_vectors(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------- rotation properties
def test_hadamard_is_orthonormal():
    from src.quantizers import _hadamard

    H = _hadamard(64)
    np.testing.assert_allclose(H @ H.T, np.eye(64), atol=1e-10)


@pytest.mark.parametrize("rotation", ["hadamard", "gaussian"])
def test_rotation_is_invertible(rotation):
    d = 64
    q = RotatedScalarQuantizer(d=d, bits=8, rotate=True, rotation=rotation, seed=0)
    X = random_unit_vectors(50, d, seed=1).astype(np.float64)
    round_trip = q._inverse(q._forward(X))
    np.testing.assert_allclose(round_trip, X, atol=1e-8)


@pytest.mark.parametrize("rotation", ["hadamard", "gaussian"])
def test_rotation_preserves_norm(rotation):
    d = 64
    q = RotatedScalarQuantizer(d=d, bits=8, rotate=True, rotation=rotation, seed=0)
    X = random_unit_vectors(50, d, seed=2).astype(np.float64)
    np.testing.assert_allclose(
        np.linalg.norm(q._forward(X), axis=1), np.linalg.norm(X, axis=1), atol=1e-8
    )


# --------------------------------------------------------------------- codebook properties
def test_lloyd_max_levels_are_sorted_and_centroidal():
    x = sphere_coordinate_samples(64, n=100_000, seed=0)
    levels = lloyd_max_codebook(x, n_levels=8)
    assert np.all(np.diff(levels) > 0), "levels must be strictly increasing"
    # centroid condition: each level is the mean of the samples assigned to it
    bounds = (levels[:-1] + levels[1:]) / 2.0
    idx = np.searchsorted(bounds, np.sort(x))
    xs = np.sort(x)
    for k in range(len(levels)):
        sel = xs[idx == k]
        if sel.size > 100:
            assert abs(sel.mean() - levels[k]) < 1e-3


def test_lloyd_max_beats_uniform_on_gaussian():
    """The optimal quantizer must beat a uniform one on the same data, or it isn't optimal."""
    x = sphere_coordinate_samples(128, n=200_000, seed=0)
    levels = lloyd_max_codebook(x, n_levels=8)
    bounds = (levels[:-1] + levels[1:]) / 2.0
    mse_opt = np.mean((x - levels[np.searchsorted(bounds, x)]) ** 2)

    grid = np.linspace(x.min(), x.max(), 8)
    gb = (grid[:-1] + grid[1:]) / 2.0
    mse_uni = np.mean((x - grid[np.searchsorted(gb, x)]) ** 2)
    assert mse_opt < mse_uni


# --------------------------------------------------------------------- the key theory check
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_mse_distortion_matches_paper_bounds(bits):
    """Measured MSE on unit vectors should match the paper's stated distortion within tolerance.

    Distortion is E[||x - Q^-1(Q(x))||^2] for ||x|| = 1. We allow a generous relative tolerance
    because the paper quotes rounded values and our reconstruction renormalises, but an
    implementation error would miss by far more than this.
    """
    d = 256
    q = RotatedScalarQuantizer(d=d, bits=bits, rotate=True, rotation="gaussian", seed=0)
    X = random_unit_vectors(2000, d, seed=3)
    codes, norms = q.encode(X)
    Xh = q.decode(codes, norms)
    mse = float(np.mean(np.sum((X - Xh) ** 2, axis=1)))
    expected = PAPER_MSE[bits]
    assert mse == pytest.approx(expected, rel=0.45), (
        f"b={bits}: measured MSE {mse:.4f} vs paper {expected:.4f}"
    )


def test_distortion_decreases_with_bits():
    d = 256
    X = random_unit_vectors(1000, d, seed=4)
    prev = np.inf
    for bits in (1, 2, 3, 4, 5):
        q = RotatedScalarQuantizer(d=d, bits=bits, rotate=True, seed=0)
        codes, norms = q.encode(X)
        mse = float(np.mean(np.sum((X - q.decode(codes, norms)) ** 2, axis=1)))
        assert mse < prev, f"MSE must fall as bits rise (b={bits})"
        prev = mse


def test_inner_product_error_shrinks_with_bits():
    d = 256
    X = random_unit_vectors(500, d, seed=5)
    Y = random_unit_vectors(500, d, seed=6)
    prev = np.inf
    for bits in (2, 4, 6):
        q = RotatedScalarQuantizer(d=d, bits=bits, rotate=True, seed=0)
        codes, norms = q.encode(X)
        Xh = q.decode(codes, norms)
        err = float(np.mean((np.sum(X * Y, axis=1) - np.sum(Xh * Y, axis=1)) ** 2))
        assert err < prev
        prev = err


# --------------------------------------------------------------------- round-trip sanity
def test_reconstruction_preserves_direction_reasonably():
    d, bits = 256, 6
    q = RotatedScalarQuantizer(d=d, bits=bits, rotate=True, seed=0)
    X = random_unit_vectors(200, d, seed=7)
    Xh = q.decode(*q.encode(X))
    cos = np.sum(X * Xh, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Xh, axis=1))
    assert cos.mean() > 0.99


def test_norms_are_recovered_exactly():
    d = 128
    q = RotatedScalarQuantizer(d=d, bits=4, rotate=True, seed=0)
    rng = np.random.default_rng(8)
    X = (rng.standard_normal((100, d)) * rng.uniform(0.1, 10, (100, 1))).astype(np.float32)
    Xh = q.decode(*q.encode(X))
    np.testing.assert_allclose(
        np.linalg.norm(Xh, axis=1), np.linalg.norm(X, axis=1), rtol=1e-4
    )


def test_uniform_quantizer_round_trip():
    d = 64
    rng = np.random.default_rng(9)
    X = rng.standard_normal((200, d)).astype(np.float32)
    q = UniformScalarQuantizer(d=d, bits=8).fit(X)
    Xh = q.decode(*q.encode(X))
    assert np.mean((X - Xh) ** 2) < 1e-3


def test_bits_per_vector_accounting():
    q = RotatedScalarQuantizer(d=256, bits=4, rotate=True, rotation="gaussian", seed=0)
    assert q.bits_per_vector(include_norm=False) == 256 * 4
    assert q.bits_per_vector(include_norm=True) == 256 * 4 + 32
