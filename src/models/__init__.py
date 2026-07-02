"""Learned gain-estimation models and their Enhancer wrapper."""

from .torch_frontend import TorchFrontend
from .base import BaseGainModel, NeuralEnhancer
from .mask_net import MaskGRU, MaskTCN
from .attention import AttentionMask

MODELS = {
    "mask_gru": MaskGRU,
    "mask_tcn": MaskTCN,
    "attention": AttentionMask,
}

__all__ = [
    "TorchFrontend",
    "BaseGainModel",
    "NeuralEnhancer",
    "MaskGRU",
    "MaskTCN",
    "AttentionMask",
    "MODELS",
]
