"""Vector quantizers under test.

The method under test is a TurboQuant-style data-oblivious scheme (Zandieh et al.,
arXiv:2504.19874): randomly rotate a vector so its coordinates become Beta-distributed and
near-independent, then quantize each coordinate with a scalar quantizer that is optimal for that
induced distribution. Because the rotation makes the coordinate distribution known a priori, the
codebook is computed once from theory and never fitted to data -- this is what makes the scheme
data-oblivious, in contrast to PQ/OPQ which learn k-means codebooks per dataset.

Implemented here:
  * `RotatedScalarQuantizer` -- the method under test, and (with `rotate=False`) the ablation that
    isolates how much the rotation itself contributes.
  * `UniformScalarQuantizer` -- naive per-dimension min/max quantizer, the floor.
PQ/OPQ baselines come from faiss (see `src/baselines.py`) rather than being reimplemented, so the
comparison is against the reference implementations practitioners actually use.

Conventions: vectors are rows of a float32 array `(n, d)`. Norms are stored separately in float32
and reapplied at reconstruction, as the paper prescribes -- so the quantizer only has to encode
direction, and the reported bit budget counts the codes only (the norm overhead is reported
separately and honestly by `bits_per_vector`).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- helpers
def _hadamard(n: int) -> np.ndarray:
    """Normalised Hadamard matrix of order n (n must be a power of two)."""
    if n & (n - 1):
        raise ValueError(f"Hadamard order must be a power of two, got {n}")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def lloyd_max_codebook(samples: np.ndarray, n_levels: int, iters: int = 100,
                       seed: int = 0) -> np.ndarray:
    """1-D Lloyd-Max quantizer levels for the distribution the samples are drawn from.

    Used to build the *theoretical* codebook for the post-rotation coordinate distribution. The
    samples come from that analytic distribution, not from the dataset, which is what keeps the
    quantizer data-oblivious.
    """
    x = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    # initialise levels at evenly spaced quantiles so the search starts inside the support
    qs = (np.arange(n_levels) + 0.5) / n_levels
    levels = np.quantile(x, qs)
    for _ in range(iters):
        # boundaries midway between adjacent levels; assign then recentre
        bounds = (levels[:-1] + levels[1:]) / 2.0
        idx = np.searchsorted(bounds, x)
        new = levels.copy()
        for k in range(n_levels):
            sel = x[idx == k]
            if sel.size:
                new[k] = sel.mean()
        if np.allclose(new, levels, rtol=1e-10, atol=1e-12):
            levels = new
            break
        levels = new
    return levels.astype(np.float64)


def sphere_coordinate_samples(d: int, n: int = 400_000, seed: int = 0) -> np.ndarray:
    """Samples of a single coordinate of a random unit vector in R^d.

    After a random rotation any input direction is uniform on the unit sphere, so each coordinate
    follows the (scaled) Beta distribution of Lemma 1 in the paper, converging to N(0, 1/d). We
    sample it directly rather than assuming the Gaussian limit, so the codebook stays correct at
    the small `d` where the limit is loose.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True) + EPS
    return g[:, 0]


# --------------------------------------------------------------------------- quantizers
class RotatedScalarQuantizer:
    """TurboQuant-style data-oblivious quantizer: random rotation + per-coordinate optimal scalar.

    Parameters
    ----------
    d : ambient dimension.
    bits : bits per coordinate (so 2**bits levels).
    rotate : if False, skip the rotation. This is the RQ2 ablation -- everything else identical,
        which isolates the rotation's contribution rather than confounding it with the codebook.
    rotation : 'hadamard' (fast, structured, O(d log d) in principle and what a deployment would
        use) or 'gaussian' (a dense random orthogonal matrix, closer to the analysis).
    """

    def __init__(self, d: int, bits: int = 4, rotate: bool = True,
                 rotation: str = "hadamard", seed: int = 0) -> None:
        self.d, self.bits, self.rotate, self.rotation_kind = d, bits, rotate, rotation
        self.n_levels = 2 ** bits
        rng = np.random.default_rng(seed)

        if rotate:
            if rotation == "hadamard":
                # pad to the next power of two, then apply random sign flips so the transform is
                # a random rotation rather than a fixed one
                self.d_pad = 1 << (d - 1).bit_length()
                self.signs = rng.choice([-1.0, 1.0], size=self.d_pad)
                self._H = _hadamard(self.d_pad)
            elif rotation == "gaussian":
                self.d_pad = d
                q, _ = np.linalg.qr(rng.standard_normal((d, d)))
                self._R = q
            else:
                raise ValueError(f"unknown rotation '{rotation}'")
        else:
            self.d_pad = d

        # Codebook from the analytic post-rotation distribution -- no dataset involved.
        self._levels = lloyd_max_codebook(
            sphere_coordinate_samples(self.d_pad, seed=seed), self.n_levels
        )
        self._bounds = (self._levels[:-1] + self._levels[1:]) / 2.0

    # ------------------------------------------------------------------ transform
    def _forward(self, x: np.ndarray) -> np.ndarray:
        if not self.rotate:
            return x
        if self.rotation_kind == "hadamard":
            if x.shape[1] < self.d_pad:
                x = np.pad(x, ((0, 0), (0, self.d_pad - x.shape[1])))
            return (x * self.signs) @ self._H.T
        return x @ self._R

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        if not self.rotate:
            return y
        if self.rotation_kind == "hadamard":
            x = (y @ self._H) * self.signs
            return x[:, : self.d]
        return y @ self._R.T

    # ------------------------------------------------------------------ API
    def encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (codes uint8 (n, d_pad), norms float32 (n,))."""
        X = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1)
        Xn = X / (norms[:, None] + EPS)
        Y = self._forward(Xn)
        codes = np.searchsorted(self._bounds, Y).astype(np.uint8)
        return codes, norms.astype(np.float32)

    def decode(self, codes: np.ndarray, norms: np.ndarray) -> np.ndarray:
        Y = self._levels[codes]
        Xn = self._inverse(Y)
        # renormalise: quantization perturbs the norm, and the true norm is stored exactly
        Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + EPS)
        return (Xn * norms[:, None]).astype(np.float32)

    def bits_per_vector(self, include_norm: bool = True) -> float:
        """Code bits per vector. `include_norm` adds the 32-bit stored norm, reported honestly."""
        return self.d_pad * self.bits + (32 if include_norm else 0)

    @property
    def name(self) -> str:
        rot = f"rot-{self.rotation_kind}" if self.rotate else "no-rot"
        return f"scalar-{self.bits}b-{rot}"


class UniformScalarQuantizer:
    """Naive per-dimension uniform quantizer over the observed min/max. The floor baseline.

    Mildly data-dependent (it reads per-dimension ranges), which is noted rather than hidden --
    it makes it a *generous* floor, not a strawman.
    """

    def __init__(self, d: int, bits: int = 4) -> None:
        self.d, self.bits, self.n_levels = d, bits, 2 ** bits
        self._lo = None
        self._hi = None

    def fit(self, X: np.ndarray) -> "UniformScalarQuantizer":
        X = np.asarray(X, dtype=np.float64)
        self._lo, self._hi = X.min(axis=0), X.max(axis=0)
        return self

    def encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if self._lo is None:
            self.fit(X)
        span = np.maximum(self._hi - self._lo, EPS)
        t = (X - self._lo) / span
        codes = np.clip(np.round(t * (self.n_levels - 1)), 0, self.n_levels - 1).astype(np.uint8)
        return codes, np.zeros(len(X), dtype=np.float32)

    def decode(self, codes: np.ndarray, norms: np.ndarray) -> np.ndarray:
        span = np.maximum(self._hi - self._lo, EPS)
        t = codes.astype(np.float64) / (self.n_levels - 1)
        return (t * span + self._lo).astype(np.float32)

    def bits_per_vector(self, include_norm: bool = False) -> float:
        return self.d * self.bits

    @property
    def name(self) -> str:
        return f"uniform-{self.bits}b"
