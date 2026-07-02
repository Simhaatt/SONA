"""Flagship: RL + attention hybrid with reference-free online adaptation.

`RLAttentionPolicy` is `RLGainPolicy` with an attention trunk — an attention encoder builds the
per-band temporal context, and an actor-critic head chooses per-band attenuation.

`OnlineAdapter` is the novelty for the paper/patent: at **inference**, with **no clean reference**,
it adapts the policy's output to the current noise by optimizing a small per-band gain bias against
a self-supervised reward (agreement between the applied gain and a speech-presence estimate derived
from the noisy signal itself). This is what lets the system track non-stationary / burst noise that
a frozen model cannot — the deck's "adapts from the signal itself" claim, made concrete.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..frontend import Enhancer, SpectralFrontend
from .rl_agent import RLGainPolicy
from .torch_frontend import TorchFrontend

EPS = 1e-8


class RLAttentionPolicy(RLGainPolicy):
    def __init__(self, n_bands: int, n_feat_channels: int = 3, hidden: int = 128, **kw) -> None:
        super().__init__(n_bands, n_feat_channels, trunk="attention", hidden=hidden, **kw)


def speech_presence(noisy_band_power: torch.Tensor, floor_quantile: float = 0.1,
                    sharpness: float = 2.0) -> torch.Tensor:
    """Reference-free per-band speech-presence probability in [0,1] from noisy band power.

    Noise floor is a low temporal quantile of each band's power; the a-priori SNR above that floor
    is squashed to a probability. No clean signal is used.
    """
    floor = torch.quantile(noisy_band_power, floor_quantile, dim=-1, keepdim=True).clamp_min(EPS)
    snr = noisy_band_power / floor
    return torch.sigmoid(sharpness * (torch.log(snr + EPS)))


def reference_free_reward(gain: torch.Tensor, presence: torch.Tensor,
                          suppress_weight: float = 1.5) -> torch.Tensor:
    """Scalar reward (higher is better): agreement between applied gain and speech presence.

    gain, presence: (B, n_bands, T). Reward is maximal when the gain preserves speech-present bands
    (gain->1) and suppresses speech-absent bands (gain->0). `suppress_weight` > 1 tilts the balance
    toward suppression so the objective does not collapse to "pass everything" (a preservation bias
    that otherwise degrades SNR — see OnlineAdapter).
    """
    preserve = presence * gain
    suppress = (1.0 - presence) * (1.0 - gain)
    return (preserve + suppress_weight * suppress).mean()


class OnlineAdapter(Enhancer):
    """Wrap a trained RL policy and adapt per-band gains online (reference-free) per utterance."""

    def __init__(
        self,
        policy: RLGainPolicy,
        torch_frontend: TorchFrontend,
        frontend: SpectralFrontend | None = None,
        device: str = "cpu",
        steps: int = 8,
        lr: float = 0.05,
        suppress_weight: float = 1.5,
        reg: float = 0.5,
        reward_fn=None,
        adapter=None,
        name: str = "rl_attention_online",
    ) -> None:
        fe = frontend or SpectralFrontend(
            sr=torch_frontend.sr, n_fft=torch_frontend.n_fft,
            hop=torch_frontend.hop, n_bands=torch_frontend.n_bands,
        )
        super().__init__(fe)
        self.name = name
        self.policy = policy.to(device).eval()
        self.tf = torch_frontend.to(device)
        self.device = device
        self.steps = steps
        self.lr = lr
        self.suppress_weight = suppress_weight
        self.reg = reg  # keeps the online bias a *refinement* of the trained policy, not a takeover
        # Optional waveform-domain reference-free reward (e.g. SQUIM). If None, use the built-in
        # band-domain speech-presence heuristic. This is the primary reward for real experiments.
        self.reward_fn = reward_fn
        # Pluggable low-dimensional adaptation module (ablation knob). Default = per-band bias.
        from .adapt import BandBias

        self.adapter = adapter if adapter is not None else BandBias(self.tf.n_bands)

    def compute_gain(self, analysis: dict) -> np.ndarray:
        raise NotImplementedError("OnlineAdapter overrides enhance()")

    def _adapt(self, wav: np.ndarray):
        """Run reference-free online adaptation. Returns (spec, adapted_gain, length)."""
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(self.device)
        spec, feats = self.tf.analyze(x)
        noisy_band_power = torch.einsum("bft,kf->bkt", spec.abs() ** 2, self.tf.analysis)
        presence = speech_presence(noisy_band_power).detach()  # (1, n_bands, T)

        with torch.no_grad():
            base_gain = self.policy(feats)  # (1, T, n_bands), deterministic mean

        # Adapt a low-dimensional correction per-utterance. The L2 reg keeps it a *refinement* of
        # the trained policy so adaptation corrects for the current noise without discarding what
        # the policy learned. No clean reference is used.
        params = self.adapter.reset(self.device)
        opt = torch.optim.Adam(params, lr=self.lr)
        length = x.shape[-1]
        for _ in range(self.steps):
            opt.zero_grad()
            gain_bt = self.adapter.apply(base_gain).clamp(0.0, 1.0)  # (1, T, n_bands)
            if self.reward_fn is not None:
                # Waveform-domain reward (SQUIM etc.): reconstruct and score, no clean reference.
                enhanced = self.tf.apply_band_gain(spec, gain_bt, length=length)
                reward = self.reward_fn(enhanced, x)
            else:
                reward = reference_free_reward(gain_bt.permute(0, 2, 1), presence, self.suppress_weight)
            loss = -reward + self.reg * self.adapter.reg()
            loss.backward()
            opt.step()
        with torch.no_grad():
            adapted = self.adapter.apply(base_gain).clamp(0.0, 1.0)
        return spec, adapted.detach(), length

    def enhance(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        spec, gain, length = self._adapt(wav)
        with torch.no_grad():
            out = self.tf.apply_band_gain(spec, gain, length=length)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def adaptation_delta_norm(self, wav: np.ndarray) -> float:
        """Diagnostic: magnitude of the online adaptation applied to `wav` (for the ablation)."""
        self._adapt(wav)
        return self.adapter.correction_norm()
