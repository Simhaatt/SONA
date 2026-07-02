"""Supervised training loop for the learned mask models (MaskGRU / MaskTCN / AttentionMask).

Runs on CPU (local smoke tests) or GPU (Kaggle T4 / RTX 4050) unchanged. Includes:
  * `build_model` / `build_frontend` from a config,
  * a standard train/validate loop with checkpointing,
  * `overfit_batch` — a fast correctness probe that a model can drive the loss toward zero on a
    single batch (the M3 sanity gate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import MODELS, NeuralEnhancer, TorchFrontend
from ..models.base import BaseGainModel
from ..utils.logging import get_logger
from .losses import combined_loss

log = get_logger("train")


def build_frontend(cfg) -> TorchFrontend:
    s = cfg.get_path("stft", {})
    b = cfg.get_path("bands", {})
    return TorchFrontend(
        sr=int(cfg.get_path("sr", 16000)),
        n_fft=int(s.get("n_fft", 512)),
        hop=s.get("hop"),
        n_bands=int(b.get("n_bands", 32)),
        band_scale=b.get("scale", "bark"),
    )


def build_model(cfg, tf: TorchFrontend) -> BaseGainModel:
    m = cfg.get_path("model", {})
    kind = m.get("kind", "mask_gru")
    if kind not in MODELS:
        raise ValueError(f"unknown model kind '{kind}'. options: {list(MODELS)}")
    kwargs = {k: v for k, v in (m or {}).items() if k != "kind"}
    return MODELS[kind](n_bands=tf.n_bands, n_feat_channels=tf.n_feat_channels, **kwargs)


def _step(model, tf, noisy, clean, mag_weight, n_fft, hop):
    spec, feats = tf.analyze(noisy)
    gains = model(feats)
    est = tf.apply_band_gain(spec, gains, length=noisy.shape[-1])
    return combined_loss(est, clean, mag_weight=mag_weight, n_fft=n_fft, hop=hop)


def train(
    cfg,
    train_dataset,
    val_dataset=None,
    device: str | None = None,
    ckpt_dir: str | Path = "checkpoints",
) -> BaseGainModel:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    log.info("Model '%s' with %d params on %s", cfg.get_path("model.kind", "?"),
             model.num_params(), device)

    t = cfg.get_path("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(t.get("lr", 3e-4)),
                            weight_decay=float(t.get("weight_decay", 1e-5)))
    epochs = int(t.get("epochs", 20))
    mag_weight = float(t.get("mag_weight", 0.5))
    n_fft = int(cfg.get_path("stft.n_fft", 512))
    hop = int(cfg.get_path("stft.hop", n_fft // 4))

    loader = DataLoader(train_dataset, batch_size=int(t.get("batch_size", 8)),
                        shuffle=True, drop_last=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=int(t.get("batch_size", 8))) if val_dataset else None
    )
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        losses = []
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            opt.zero_grad()
            loss = _step(model, tf, noisy, clean, mag_weight, n_fft, hop)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        msg = f"epoch {epoch+1}/{epochs} train_loss={np.mean(losses):.4f}"
        if val_loader:
            val = _validate(model, tf, val_loader, device, mag_weight, n_fft, hop)
            msg += f" val_loss={val:.4f}"
            if val < best_val:
                best_val = val
                torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()},
                           ckpt_dir / "best.pt")
        log.info(msg)

    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict()}, ckpt_dir / "last.pt")
    return model


@torch.no_grad()
def _validate(model, tf, loader, device, mag_weight, n_fft, hop) -> float:
    model.eval()
    losses = []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        losses.append(_step(model, tf, noisy, clean, mag_weight, n_fft, hop).item())
    return float(np.mean(losses)) if losses else float("nan")


def make_enhancer(model: BaseGainModel, tf: TorchFrontend, device: str = "cpu",
                  name: str = "neural") -> NeuralEnhancer:
    return NeuralEnhancer(model, tf, device=device, name=name)


def overfit_batch(cfg, noisy: torch.Tensor, clean: torch.Tensor, steps: int = 150,
                  device: str = "cpu") -> tuple[float, float]:
    """Drive a single batch's loss down as a correctness probe. Returns (start_loss, end_loss)."""
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    noisy, clean = noisy.to(device), clean.to(device)
    n_fft = int(cfg.get_path("stft.n_fft", 512))
    hop = int(cfg.get_path("stft.hop", n_fft // 4))

    start = None
    model.train()
    for i in range(steps):
        opt.zero_grad()
        loss = _step(model, tf, noisy, clean, mag_weight=0.0, n_fft=n_fft, hop=hop)
        loss.backward()
        opt.step()
        if i == 0:
            start = loss.item()
    return float(start), float(loss.item())
