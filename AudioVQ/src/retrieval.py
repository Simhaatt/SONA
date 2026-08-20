"""Retrieval evaluation: recall against exhaustive float32 ground truth, at matched bit budgets.

Protocol follows the standard ANN benchmark methodology. Ground truth for a query is its true
top-k nearest neighbours in the UNCOMPRESSED database; a quantizer is scored by how many of those
it still returns after compression. This measures exactly what a quantizer is for -- preserving
the geometry -- and needs no semantic labels.

Baselines come from faiss (PQ, OPQ) rather than being reimplemented here, so the data-dependent
side of the comparison is the reference implementation practitioners actually deploy.

Bit accounting is made explicit and comparable: every method reports bits/vector for its codes,
and the scalar quantizer's 32-bit stored norm is counted rather than quietly ignored.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)


def exact_topk(queries: np.ndarray, database: np.ndarray, k: int,
               metric: str = "ip") -> np.ndarray:
    """Exhaustive top-k indices. metric: 'ip' (inner product) or 'l2'."""
    Q = np.asarray(queries, dtype=np.float32)
    D = np.asarray(database, dtype=np.float32)
    if metric == "ip":
        sims = Q @ D.T
        return np.argpartition(-sims, kth=k - 1, axis=1)[:, :k], sims
    d2 = (Q**2).sum(1)[:, None] - 2 * (Q @ D.T) + (D**2).sum(1)[None, :]
    return np.argpartition(d2, kth=k - 1, axis=1)[:, :k], -d2


def sorted_topk(queries: np.ndarray, database: np.ndarray, k: int,
                metric: str = "ip") -> np.ndarray:
    idx, sims = exact_topk(queries, database, k, metric)
    rows = np.arange(len(idx))[:, None]
    order = np.argsort(-sims[rows, idx], axis=1)
    return idx[rows, order]


def recall_at_k(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Fraction of the true top-k that appear in the retrieved top-k, averaged over queries."""
    gt = ground_truth[:, :k]
    ret = retrieved[:, :k]
    hits = [len(set(r.tolist()) & set(g.tolist())) for r, g in zip(ret, gt)]
    return float(np.mean(hits) / k)


# ------------------------------------------------------------------ method wrappers
class ScalarVQIndex:
    """Wraps a quantizer from src.quantizers into the retrieval interface.

    Search is performed by reconstructing the database from codes and doing exact search in the
    reconstructed space -- i.e. the retrieval loss measured is purely the quantization loss, with
    no additional approximation from an ANN search structure. That isolates the variable under
    study.
    """

    def __init__(self, quantizer) -> None:
        self.q = quantizer
        self._recon = None

    def build(self, database: np.ndarray) -> "ScalarVQIndex":
        if hasattr(self.q, "fit"):
            self.q.fit(database)
        codes, norms = self.q.encode(database)
        self._recon = self.q.decode(codes, norms)
        return self

    def search(self, queries: np.ndarray, k: int, metric: str = "ip") -> np.ndarray:
        return sorted_topk(queries, self._recon, k, metric)

    @property
    def bits_per_vector(self) -> float:
        return self.q.bits_per_vector(include_norm=True)

    @property
    def name(self) -> str:
        return self.q.name


class FaissPQIndex:
    """Product Quantization / Optimized PQ baseline (data-dependent, trained on the database)."""

    def __init__(self, d: int, m: int, nbits: int = 8, opq: bool = False, seed: int = 0) -> None:
        self.d, self.m, self.nbits, self.opq = d, m, nbits, opq
        self._index = None

    def build(self, database: np.ndarray) -> "FaissPQIndex":
        import faiss

        X = np.ascontiguousarray(database, dtype=np.float32)
        if self.opq:
            opq = faiss.OPQMatrix(self.d, self.m)
            pq = faiss.IndexPQ(self.d, self.m, self.nbits, faiss.METRIC_INNER_PRODUCT)
            index = faiss.IndexPreTransform(opq, pq)
        else:
            index = faiss.IndexPQ(self.d, self.m, self.nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(X)
        index.add(X)
        self._index = index
        return self

    def search(self, queries: np.ndarray, k: int, metric: str = "ip") -> np.ndarray:
        Q = np.ascontiguousarray(queries, dtype=np.float32)
        _, idx = self._index.search(Q, k)
        return idx

    @property
    def bits_per_vector(self) -> float:
        return self.m * self.nbits

    @property
    def name(self) -> str:
        return f"{'OPQ' if self.opq else 'PQ'}-m{self.m}-{self.nbits}b"


class Float32Index:
    """Uncompressed upper bound."""

    def __init__(self, d: int) -> None:
        self.d = d
        self._db = None

    def build(self, database: np.ndarray) -> "Float32Index":
        self._db = np.asarray(database, dtype=np.float32)
        return self

    def search(self, queries: np.ndarray, k: int, metric: str = "ip") -> np.ndarray:
        return sorted_topk(queries, self._db, k, metric)

    @property
    def bits_per_vector(self) -> float:
        return self.d * 32

    @property
    def name(self) -> str:
        return "float32-exact"


# ------------------------------------------------------------------ distortion diagnostics
def distortion_stats(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """MSE on unit-normalised vectors plus inner-product error, for comparison with theory."""
    A = normalize(original).astype(np.float64)
    B = normalize(reconstructed).astype(np.float64)
    mse = float(np.mean(np.sum((A - B) ** 2, axis=1)))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(A), size=min(len(A), 2000), replace=False)
    jdx = rng.choice(len(A), size=min(len(A), 2000), replace=False)
    ip_true = np.sum(A[idx] * A[jdx], axis=1)
    ip_hat = np.sum(B[idx] * B[jdx], axis=1)
    return {
        "mse": mse,
        "ip_mse": float(np.mean((ip_true - ip_hat) ** 2)),
        "ip_bias": float(np.mean(ip_hat - ip_true)),
        "cosine_to_original": float(np.mean(np.sum(A * B, axis=1))),
    }
