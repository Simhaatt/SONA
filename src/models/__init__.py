"""Learned gain-estimation models and their Enhancer wrapper."""

from .torch_frontend import TorchFrontend
from .base import BaseGainModel, NeuralEnhancer
from .mask_net import MaskGRU, MaskTCN
from .attention import AttentionMask
from .gtcrn import GTCRNMask
from .rl_agent import RLGainPolicy
from .rl_attention import RLAttentionPolicy, OnlineAdapter
from .eq import ParametricEQ, Compressor, hum_notch_curve, GRAPHIC_EQ_FREQS
from .adapt import make_adapter, ADAPTERS, ScalarBias, BandBias, LowRankBias, FiLM

MODELS = {
    "mask_gru": MaskGRU,
    "mask_tcn": MaskTCN,
    "attention": AttentionMask,
    "gtcrn": GTCRNMask,
    "bark_gtcrn": GTCRNMask,
}

# RL policies use a separate training loop (train_rl) but reuse NeuralEnhancer for deterministic eval.
RL_MODELS = {
    "rl_gru": RLGainPolicy,
    "rl_attention": RLAttentionPolicy,
}

__all__ = [
    "TorchFrontend",
    "BaseGainModel",
    "NeuralEnhancer",
    "MaskGRU",
    "MaskTCN",
    "AttentionMask",
    "GTCRNMask",
    "RLGainPolicy",
    "RLAttentionPolicy",
    "OnlineAdapter",
    "ParametricEQ",
    "Compressor",
    "hum_notch_curve",
    "GRAPHIC_EQ_FREQS",
    "make_adapter",
    "ADAPTERS",
    "ScalarBias",
    "BandBias",
    "LowRankBias",
    "FiLM",
    "MODELS",
    "RL_MODELS",
]
