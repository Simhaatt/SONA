"""Actor-critic (contextual-bandit) training for the RL gain policy.

Each STFT frame is a bandit step: the policy samples a per-band gain, receives a per-band reward,
and is updated by policy gradient with a per-band value baseline (advantage). Training reward is
computed in the band-magnitude domain (dense, cheap, no waveform reconstruction per step):

    reward = -(enhanced_band_mag - clean_band_mag)^2  / (clean band energy)

This objective is deliberately modular — swap in a perceptual or reference-free reward without
touching the update. Inference-time reference-free adaptation lives in `models/rl_attention.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.rl_agent import RLGainPolicy
from ..models.rl_attention import RLAttentionPolicy
from ..models.torch_frontend import TorchFrontend
from ..utils.logging import get_logger
from .train_supervised import build_frontend

log = get_logger("train_rl")

RL_MODELS = {"rl_gru": RLGainPolicy, "rl_attention": RLAttentionPolicy}


def build_policy(cfg, tf: TorchFrontend) -> RLGainPolicy:
    m = cfg.get_path("model", {})
    kind = m.get("kind", "rl_attention")
    if kind not in RL_MODELS:
        raise ValueError(f"unknown RL model '{kind}'. options: {list(RL_MODELS)}")
    kwargs = {k: v for k, v in (m or {}).items() if k != "kind"}
    if kind == "rl_gru":
        kwargs.setdefault("trunk", "gru")
    return RL_MODELS[kind](n_bands=tf.n_bands, n_feat_channels=tf.n_feat_channels, **kwargs)


def band_mag(tf: TorchFrontend, wav: torch.Tensor) -> torch.Tensor:
    """Per-band magnitude (B, T, n_bands) for the reward."""
    spec = tf.stft(wav)
    bm = torch.einsum("bft,kf->bkt", spec.abs(), tf.analysis)  # (B, n_bands, T)
    return bm.permute(0, 2, 1)  # (B, T, n_bands)


def reference_reward(gain: torch.Tensor, noisy_bm: torch.Tensor, clean_bm: torch.Tensor) -> torch.Tensor:
    """Per-band, per-frame reward (B, T, n_bands): closeness of masked band mag to clean."""
    enh = gain * noisy_bm
    err = (enh - clean_bm) ** 2
    scale = clean_bm.pow(2).mean() + 1e-6
    return -err / scale


def _rl_step(policy, tf, noisy, clean, coefs):
    spec = tf.stft(noisy)
    feats = tf.features(spec.abs())
    noisy_bm = band_mag(tf, noisy)
    clean_bm = band_mag(tf, clean)

    gain, log_prob, value, entropy = policy.act(feats, deterministic=False)  # (B,T,n_bands)
    reward = reference_reward(gain, noisy_bm, clean_bm)

    advantage = (reward - value).detach()
    policy_loss = -(log_prob * advantage).mean()
    value_loss = F.mse_loss(value, reward.detach())
    entropy_loss = -entropy.mean()
    loss = policy_loss + coefs["value"] * value_loss + coefs["entropy"] * entropy_loss
    return loss, reward.mean().item()


def train_rl(cfg, train_dataset, val_dataset=None, device: str | None = None,
             ckpt_dir: str | Path = "checkpoints") -> RLGainPolicy:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tf = build_frontend(cfg).to(device)
    policy = build_policy(cfg, tf).to(device)
    log.info("RL policy '%s' with %d params on %s", cfg.get_path("model.kind", "?"),
             policy.num_params(), device)

    t = cfg.get_path("train", {})
    opt = torch.optim.AdamW(policy.parameters(), lr=float(t.get("lr", 3e-4)))
    epochs = int(t.get("epochs", 30))
    coefs = {"value": float(t.get("value_coef", 0.5)), "entropy": float(t.get("entropy_coef", 0.01))}
    loader = DataLoader(train_dataset, batch_size=int(t.get("batch_size", 8)),
                        shuffle=True, drop_last=True)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        policy.train()
        losses, rewards = [], []
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            opt.zero_grad()
            loss, r = _rl_step(policy, tf, noisy, clean, coefs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            losses.append(loss.item())
            rewards.append(r)
        log.info("epoch %d/%d loss=%.4f mean_reward=%.4f",
                 epoch + 1, epochs, np.mean(losses), np.mean(rewards))

    torch.save({"model": policy.state_dict(), "cfg": cfg.to_dict()}, ckpt_dir / "rl_last.pt")
    return policy


def improve_reward_on_batch(cfg, noisy: torch.Tensor, clean: torch.Tensor, steps: int = 120,
                            device: str = "cpu") -> tuple[float, float]:
    """Probe: policy-gradient should raise mean reward on a single batch. Returns (start, end)."""
    tf = build_frontend(cfg).to(device)
    policy = build_policy(cfg, tf).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    noisy, clean = noisy.to(device), clean.to(device)
    coefs = {"value": 0.5, "entropy": 0.0}
    start = None
    policy.train()
    for i in range(steps):
        opt.zero_grad()
        loss, r = _rl_step(policy, tf, noisy, clean, coefs)
        loss.backward()
        opt.step()
        if i == 0:
            start = r
    return float(start), float(r)
