"""Correctness tests for the retrieval evaluation code.

Every headline number in this project flows through `sorted_topk` and `recall_at_k`. A bug in
either would silently corrupt all of them, so they are tested against brute-force references and
hand-checkable cases rather than trusted.

Run:  python -m pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval import (  # noqa: E402
    Float32Index,
    ScalarVQIndex,
    distortion_stats,
    normalize,
    recall_at_k,
    sorted_topk,
)
from src.quantizers import RotatedScalarQuantizer  # noqa: E402


# ------------------------------------------------------------------ normalize
def test_normalize_gives_unit_norm():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 16)).astype(np.float32) * 7.0
    np.testing.assert_allclose(np.linalg.norm(normalize(X), axis=1), 1.0, atol=1e-5)


def test_normalize_handles_zero_vector():
    X = np.zeros((3, 8), dtype=np.float32)
    out = normalize(X)
    assert np.all(np.isfinite(out)), "zero vectors must not produce NaN/inf"


# ------------------------------------------------------------------ sorted_topk
def test_sorted_topk_matches_bruteforce_ip():
    rng = np.random.default_rng(1)
    D = rng.standard_normal((200, 12)).astype(np.float32)
    Q = rng.standard_normal((17, 12)).astype(np.float32)
    k = 5
    got = sorted_topk(Q, D, k, metric="ip")
    want = np.argsort(-(Q @ D.T), axis=1)[:, :k]
    np.testing.assert_array_equal(got, want)


def test_sorted_topk_is_actually_sorted():
    rng = np.random.default_rng(2)
    D = rng.standard_normal((150, 10)).astype(np.float32)
    Q = rng.standard_normal((9, 10)).astype(np.float32)
    idx = sorted_topk(Q, D, 8, metric="ip")
    sims = Q @ D.T
    for i in range(len(Q)):
        vals = sims[i, idx[i]]
        assert np.all(np.diff(vals) <= 1e-6), "top-k must be in descending similarity order"


def test_sorted_topk_l2_matches_bruteforce():
    rng = np.random.default_rng(3)
    D = rng.standard_normal((120, 9)).astype(np.float32)
    Q = rng.standard_normal((6, 9)).astype(np.float32)
    k = 4
    got = sorted_topk(Q, D, k, metric="l2")
    d2 = ((Q[:, None, :] - D[None, :, :]) ** 2).sum(-1)
    want = np.argsort(d2, axis=1)[:, :k]
    np.testing.assert_array_equal(got, want)


def test_query_retrieves_itself_when_in_database():
    rng = np.random.default_rng(4)
    D = normalize(rng.standard_normal((80, 14)).astype(np.float32))
    idx = sorted_topk(D, D, 1, metric="ip")
    np.testing.assert_array_equal(idx[:, 0], np.arange(len(D)))


# ------------------------------------------------------------------ recall_at_k
def test_recall_is_one_for_identical_lists():
    gt = np.arange(50).reshape(5, 10)
    assert recall_at_k(gt.copy(), gt, 10) == pytest.approx(1.0)


def test_recall_is_zero_for_disjoint_lists():
    gt = np.arange(20).reshape(2, 10)
    ret = np.arange(100, 120).reshape(2, 10)
    assert recall_at_k(ret, gt, 10) == pytest.approx(0.0)


def test_recall_counts_set_overlap_not_order():
    gt = np.array([[1, 2, 3, 4]])
    ret = np.array([[4, 3, 2, 1]])  # same set, reversed order
    assert recall_at_k(ret, gt, 4) == pytest.approx(1.0)


def test_recall_half_overlap():
    gt = np.array([[1, 2, 3, 4]])
    ret = np.array([[1, 2, 99, 98]])
    assert recall_at_k(ret, gt, 4) == pytest.approx(0.5)


def test_recall_respects_k_truncation():
    gt = np.array([[1, 2, 3, 4, 5]])
    ret = np.array([[1, 2, 3, 99, 98]])
    assert recall_at_k(ret, gt, 3) == pytest.approx(1.0)   # first 3 all match
    assert recall_at_k(ret, gt, 5) == pytest.approx(0.6)   # 3 of 5


# ------------------------------------------------------------------ index wrappers
def test_float32_index_recall_is_perfect():
    rng = np.random.default_rng(5)
    D = normalize(rng.standard_normal((300, 20)).astype(np.float32))
    Q = normalize(rng.standard_normal((40, 20)).astype(np.float32))
    gt = sorted_topk(Q, D, 10, metric="ip")
    ret = Float32Index(20).build(D).search(Q, 10, metric="ip")
    assert recall_at_k(ret, gt, 10) == pytest.approx(1.0)


def test_high_bit_quantizer_nearly_matches_exact():
    """A generous bit budget must recover almost all neighbours -- catches broken encode/decode."""
    rng = np.random.default_rng(6)
    d = 64
    D = normalize(rng.standard_normal((400, d)).astype(np.float32))
    Q = normalize(rng.standard_normal((50, d)).astype(np.float32))
    gt = sorted_topk(Q, D, 10, metric="ip")
    q = RotatedScalarQuantizer(d=d, bits=8, rotate=True, rotation="gaussian", seed=0)
    ret = ScalarVQIndex(q).build(D).search(Q, 10, metric="ip")
    assert recall_at_k(ret, gt, 10) > 0.9


def test_more_bits_never_hurts_recall_much():
    rng = np.random.default_rng(7)
    d = 64
    D = normalize(rng.standard_normal((400, d)).astype(np.float32))
    Q = normalize(rng.standard_normal((50, d)).astype(np.float32))
    gt = sorted_topk(Q, D, 10, metric="ip")
    prev = -1.0
    for bits in (1, 2, 4, 6):
        q = RotatedScalarQuantizer(d=d, bits=bits, rotate=True, rotation="gaussian", seed=0)
        r = recall_at_k(ScalarVQIndex(q).build(D).search(Q, 10, metric="ip"), gt, 10)
        assert r >= prev - 0.02, f"recall fell going to {bits} bits ({r:.3f} < {prev:.3f})"
        prev = r


def test_bits_per_vector_reported_by_index_matches_quantizer():
    q = RotatedScalarQuantizer(d=64, bits=3, rotate=True, rotation="gaussian", seed=0)
    assert ScalarVQIndex(q).bits_per_vector == q.bits_per_vector(include_norm=True)


# ------------------------------------------------------------------ distortion diagnostics
def test_distortion_zero_for_identical_input():
    rng = np.random.default_rng(8)
    X = normalize(rng.standard_normal((100, 16)).astype(np.float32))
    s = distortion_stats(X, X.copy())
    assert s["mse"] == pytest.approx(0.0, abs=1e-8)
    assert s["cosine_to_original"] == pytest.approx(1.0, abs=1e-5)


def test_distortion_increases_with_noise():
    rng = np.random.default_rng(9)
    X = normalize(rng.standard_normal((200, 16)).astype(np.float32))
    prev = -1.0
    for scale in (0.01, 0.1, 0.5):
        Y = X + rng.standard_normal(X.shape).astype(np.float32) * scale
        mse = distortion_stats(X, Y)["mse"]
        assert mse > prev
        prev = mse
