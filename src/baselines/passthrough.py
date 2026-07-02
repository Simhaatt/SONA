"""Identity method: returns the noisy input unchanged.

Serves as the zero-processing reference row in every results table (its *_i improvement
metrics are ~0 by construction) and as an end-to-end sanity check on the harness.
"""

from __future__ import annotations

import numpy as np

from ..frontend import Enhancer


class PassThrough(Enhancer):
    name = "passthrough"

    def compute_gain(self, analysis: dict) -> np.ndarray:
        return np.ones_like(analysis["mag"])
