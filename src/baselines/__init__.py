"""Classical (training-free) denoising baselines, all sharing the spectral front-end."""

from .thresholding import HardThreshold, SoftThreshold
from .spectral_sub import SpectralSubtraction
from .wiener import WienerFilter
from .passthrough import PassThrough

BASELINES = {
    "passthrough": PassThrough,
    "hard_threshold": HardThreshold,
    "soft_threshold": SoftThreshold,
    "spectral_subtraction": SpectralSubtraction,
    "wiener": WienerFilter,
}

__all__ = [
    "HardThreshold",
    "SoftThreshold",
    "SpectralSubtraction",
    "WienerFilter",
    "PassThrough",
    "BASELINES",
]
