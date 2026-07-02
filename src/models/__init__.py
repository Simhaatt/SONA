"""Learned gain-estimation models and their Enhancer wrapper."""

from .torch_frontend import TorchFrontend
from .base import BaseGainModel, NeuralEnhancer
from .mask_net import MaskGRU, MaskTCN
from .attention import AttentionMask
from .rl_agent import RLGainPolicy
from .rl_attention import RLAttentionPolicy, OnlineAdapter

MODELS = {
    "mask_gru": MaskGRU,
    "mask_tcn": MaskTCN,
    "attention": AttentionMask,
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
    "RLGainPolicy",
    "RLAttentionPolicy",
    "OnlineAdapter",
    "MODELS",
    "RL_MODELS",
]
