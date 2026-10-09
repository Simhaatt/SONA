"""Supervised training loop for the learned mask models (MaskGRU / MaskTCN / AttentionMask).

Runs on CPU (local smoke tests) or GPU (Kaggle T4 / RTX 4050) unchanged. Includes:
  * `build_model` / `build_frontend` from a config,
  * a standard train/validate loop with checkpointing,
  * `overfit_batch` — a fast correctness probe that a model can drive the loss toward zero on a
    single batch (the M3 sanity gate).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models import MODELS, NeuralEnhancer, TorchFrontend
from ..models.base import BaseGainModel
from ..utils.logging import get_logger
from .losses import combined_loss, mask_polarity_loss

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


def _step(model, tf, noisy, clean, mag_weight, n_fft, hop, mr_mag_weight=0.0,
          mask_polarity_weight=0.0, mask_polarity_target=0.0, snr_weight=0.0,
          complex_stft_weight=0.0):
    spec, feats = tf.analyze(noisy)
    gains = model(feats)
    if getattr(model, "is_complex", False):
        est = tf.apply_band_complex(spec, gains, length=noisy.shape[-1])
    else:
        est = tf.apply_band_gain(spec, gains, length=noisy.shape[-1])
    loss = combined_loss(
        est, clean, mag_weight=mag_weight, n_fft=n_fft, hop=hop,
        mr_mag_weight=mr_mag_weight, snr_weight=snr_weight,
        complex_stft_weight=complex_stft_weight,
    )
    if mask_polarity_weight > 0.0 and not getattr(model, "is_complex", False):
        loss = loss + mask_polarity_weight * mask_polarity_loss(gains, mask_polarity_target)
    return loss


def train(
    cfg,
    train_dataset,
    val_dataset=None,
    device: str | None = None,
    ckpt_dir: str | Path = "checkpoints",
    init_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    max_hours: float | None = None,
    init_compatible_only: bool = False,
    save_every_epoch: bool = False,
) -> BaseGainModel:
    if init_checkpoint is not None and resume_checkpoint is not None:
        raise ValueError("use either init_checkpoint or resume_checkpoint, not both")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    resume_state = None
    if init_checkpoint is not None:
        init_checkpoint = Path(init_checkpoint)
        checkpoint = torch.load(init_checkpoint, map_location=device, weights_only=False)
        state = checkpoint.get("model", checkpoint)
        if init_compatible_only:
            current = model.state_dict()
            compatible = {
                key: value for key, value in state.items()
                if key in current and current[key].shape == value.shape
            }
            model.load_state_dict(compatible, strict=False)
            log.info(
                "Warm-started %d/%d compatible tensors from %s",
                len(compatible), len(current), init_checkpoint,
            )
        else:
            model.load_state_dict(state, strict=True)
            log.info("Warm-started model from %s", init_checkpoint)
    elif resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
        resume_state = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_state.get("model", resume_state), strict=True)
        log.info("Loaded resumable model from %s", resume_checkpoint)
    log.info("Model '%s' with %d params on %s", cfg.get_path("model.kind", "?"),
             model.num_params(), device)

    t = cfg.get_path("train", {})
    opt = torch.optim.AdamW(model.parameters(), lr=float(t.get("lr", 3e-4)),
                            weight_decay=float(t.get("weight_decay", 1e-5)))
    if resume_state is not None and "optimizer" in resume_state:
        opt.load_state_dict(resume_state["optimizer"])
    epochs = int(t.get("epochs", 20))
    mag_weight = float(t.get("mag_weight", 0.5))
    mr_mag_weight = float(t.get("mr_mag_weight", 0.0))
    snr_weight = float(t.get("snr_weight", 0.0))
    complex_stft_weight = float(t.get("complex_stft_weight", 0.0))
    mask_polarity_weight = float(t.get("mask_polarity_weight", 0.0))
    mask_polarity_target = float(t.get("mask_polarity_target", 0.0))
    n_fft = int(cfg.get_path("stft.n_fft", 512))
    hop = int(cfg.get_path("stft.hop", n_fft // 4))

    loader = DataLoader(train_dataset, batch_size=int(t.get("batch_size", 8)),
                        shuffle=True, drop_last=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=int(t.get("batch_size", 8))) if val_dataset else None
    )
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float(resume_state.get("best_val", float("inf"))) if resume_state else float("inf")
    start_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    started = time.monotonic()

    def checkpoint_payload(completed_epochs: int) -> dict:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "cfg": cfg.to_dict(),
            "epoch": completed_epochs,
            "best_val": best_val,
        }

    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            opt.zero_grad()
            loss = _step(
                model, tf, noisy, clean, mag_weight, n_fft, hop,
                mr_mag_weight=mr_mag_weight,
                mask_polarity_weight=mask_polarity_weight,
                mask_polarity_target=mask_polarity_target,
                snr_weight=snr_weight,
                complex_stft_weight=complex_stft_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())

        msg = f"epoch {epoch+1}/{epochs} train_loss={np.mean(losses):.4f}"
        if val_loader:
            val = _validate(
                model, tf, val_loader, device, mag_weight, n_fft, hop,
                mr_mag_weight, mask_polarity_weight, mask_polarity_target, snr_weight,
                complex_stft_weight,
            )
            msg += f" val_loss={val:.4f}"
            if val < best_val:
                best_val = val
                torch.save(checkpoint_payload(epoch + 1), ckpt_dir / "best.pt")
        log.info(msg)
        torch.save(checkpoint_payload(epoch + 1), ckpt_dir / "last.pt")
        if save_every_epoch:
            torch.save(checkpoint_payload(epoch + 1), ckpt_dir / f"epoch_{epoch + 1:02d}.pt")
        if max_hours is not None and time.monotonic() - started >= max_hours * 3600:
            log.warning("Reached %.2f-hour session guard after epoch %d; stopping safely",
                        max_hours, epoch + 1)
            break

    if not (ckpt_dir / "last.pt").exists():
        torch.save(checkpoint_payload(start_epoch), ckpt_dir / "last.pt")
    return model


@torch.no_grad()
def _validate(model, tf, loader, device, mag_weight, n_fft, hop, mr_mag_weight=0.0,
              mask_polarity_weight=0.0, mask_polarity_target=0.0,
              snr_weight=0.0, complex_stft_weight=0.0) -> float:
    model.eval()
    losses = []
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        losses.append(_step(
            model, tf, noisy, clean, mag_weight, n_fft, hop,
            mr_mag_weight=mr_mag_weight,
            mask_polarity_weight=mask_polarity_weight,
            mask_polarity_target=mask_polarity_target,
            snr_weight=snr_weight,
            complex_stft_weight=complex_stft_weight,
        ).item())
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
