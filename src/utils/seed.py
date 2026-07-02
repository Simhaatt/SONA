"""Deterministic seeding across random / numpy / torch for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """Seed all RNGs. `deterministic=True` also forces deterministic cuDNN kernels.

    Reproducibility is a hard requirement for the paper, so this is called at the top of
    every training/eval entry point.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch always present in practice
        pass
