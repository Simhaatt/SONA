"""Common base for learned gain models + an Enhancer wrapper for evaluation.

`BaseGainModel`: maps per-band features (B, T, n_bands, C) -> per-band gains (B, T, n_bands) in
[0, 1]. Subclasses implement `predict_gain`; the sigmoid squashing lives here so every learned
model shares the exact same output space as the classical baselines.

`NeuralEnhancer`: adapts a trained model to the `Enhancer` interface so it drops straight into the
existing evaluation harness (single-sample numpy in / numpy out).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..frontend import Enhancer, SpectralFrontend
from .torch_frontend import TorchFrontend


class BaseGainModel(nn.Module):
    def __init__(self, n_bands: int, n_feat_channels: int = 3) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.n_feat_channels = n_feat_channels
        self.in_dim = n_bands * n_feat_channels

    def predict_gain(self, feats: torch.Tensor) -> torch.Tensor:  # -> logits (B, T, n_bands)
        raise NotImplementedError

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.predict_gain(feats)
        return torch.sigmoid(logits)  # gains in [0, 1]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class NeuralEnhancer(Enhancer):
    """Wrap a trained `BaseGainModel` + `TorchFrontend` as an Enhancer for evaluation."""

    def __init__(
        self,
        model: BaseGainModel,
        torch_frontend: TorchFrontend,
        frontend: SpectralFrontend | None = None,
        device: str = "cpu",
        name: str = "neural",
    ) -> None:
        # `frontend` (numpy) is only needed to satisfy Enhancer.__init__; inference uses torch.
        fe = frontend or SpectralFrontend(
            sr=torch_frontend.sr,
            n_fft=torch_frontend.n_fft,
            hop=torch_frontend.hop,
            n_bands=torch_frontend.n_bands,
        )
        super().__init__(fe)
        self.name = name
        self.model = model.to(device).eval()
        self.tf = torch_frontend.to(device)
        self.device = device

    def compute_gain(self, analysis: dict) -> np.ndarray:  # not used; enhance is overridden
        raise NotImplementedError("NeuralEnhancer overrides enhance()")

    @torch.no_grad()
    def enhance(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(self.device)
        spec, feats = self.tf.analyze(x)
        gains = self.model(feats)  # (1, T, n_bands)
        out = self.tf.apply_band_gain(spec, gains, length=x.shape[-1])
        return out.squeeze(0).cpu().numpy().astype(np.float32)
