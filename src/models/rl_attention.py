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
    """Wrap a trained gain model and adapt per-band gains online per utterance.

    The wrapped model can be an RL policy or any supervised mask predictor because all learned
    models share the same ``model(feats) -> per-band gains`` interface.
    """

    def __init__(
        self,
        policy: nn.Module,
        torch_frontend: TorchFrontend,
        frontend: SpectralFrontend | None = None,
        device: str = "cpu",
        steps: int = 8,
        lr: float = 0.05,
        suppress_weight: float = 1.5,
        reg: float = 0.5,
        anchor_weight: float = 0.0,
        max_gain_delta: float | None = None,
        smooth_time_weight: float = 0.0,
        smooth_band_weight: float = 0.0,
        guard_min_reward_delta: float | None = None,
        guard_max_output_delta: float | None = None,
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
        # Penalize deviation from the frozen model's enhanced waveform. This keeps a SQUIM-driven
        # adaptation from improving perceptual score by drifting too far from the strong base output.
        self.anchor_weight = anchor_weight
        # Trust-region and smoothness controls: keep online adaptation a bounded correction to the
        # frozen gain curve, which is the key guard against PESQ reward hacking.
        self.max_gain_delta = max_gain_delta
        self.smooth_time_weight = smooth_time_weight
        self.smooth_band_weight = smooth_band_weight
        # Optional post-adaptation accept/reject guard. If enabled, the online correction is applied
        # only when it improves the reference-free reward enough and stays close to the frozen
        # waveform. This catches dataset-shift cases where adaptation is "active" but risky.
        self.guard_min_reward_delta = guard_min_reward_delta
        self.guard_max_output_delta = guard_max_output_delta
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
            base_enhanced = self.tf.apply_band_gain(spec, base_gain, length=x.shape[-1])

        # Adapt a low-dimensional correction per-utterance. The L2 reg keeps it a *refinement* of
        # the trained policy so adaptation corrects for the current noise without discarding what
        # the policy learned. No clean reference is used.
        params = self.adapter.reset(self.device)
        opt = torch.optim.Adam(params, lr=self.lr)
        length = x.shape[-1]
        # Disable cuDNN inside the loop: the reward net (SQUIM) is a frozen eval-mode RNN, and
        # cuDNN refuses RNN backward in eval mode. The native RNN kernels allow it (a bit slower).
        with torch.backends.cudnn.flags(enabled=False):
            for _ in range(self.steps):
                opt.zero_grad()
                gain_bt = self._bounded_gain(base_gain)  # (1, T, n_bands)
                if self.reward_fn is not None:
                    # Waveform-domain reward (SQUIM etc.): reconstruct + score, no clean reference.
                    enhanced = self.tf.apply_band_gain(spec, gain_bt, length=length)
                    reward = self.reward_fn(enhanced, x)
                else:
                    enhanced = None
                    reward = reference_free_reward(gain_bt.permute(0, 2, 1), presence, self.suppress_weight)
                anchor = torch.zeros((), device=self.device)
                if self.anchor_weight > 0.0:
                    if enhanced is None:
                        enhanced = self.tf.apply_band_gain(spec, gain_bt, length=length)
                    denom = base_enhanced.pow(2).mean().clamp_min(EPS)
                    anchor = (enhanced - base_enhanced).pow(2).mean() / denom
                smooth = self._correction_smoothness(gain_bt, base_gain)
                loss = (
                    -reward
                    + self.reg * self.adapter.reg()
                    + self.anchor_weight * anchor
                    + smooth
                )
                loss.backward()
                opt.step()
        with torch.no_grad():
            adapted = self._bounded_gain(base_gain)
            adapted = self._guarded_gain(spec, x, base_gain, base_enhanced, adapted, presence, length)
        return spec, adapted.detach(), length

    def _bounded_gain(self, base_gain: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter.apply(base_gain)
        if self.max_gain_delta is not None:
            delta = (adapted - base_gain).clamp(-self.max_gain_delta, self.max_gain_delta)
            adapted = base_gain + delta
        return adapted.clamp(0.0, 1.0)

    def _correction_smoothness(self, gain: torch.Tensor, base_gain: torch.Tensor) -> torch.Tensor:
        correction = gain - base_gain
        loss = torch.zeros((), device=gain.device)
        if self.smooth_time_weight > 0.0 and correction.shape[1] > 1:
            loss = loss + self.smooth_time_weight * torch.diff(correction, dim=1).abs().mean()
        if self.smooth_band_weight > 0.0 and correction.shape[2] > 1:
            loss = loss + self.smooth_band_weight * torch.diff(correction, dim=2).abs().mean()
        return loss

    def _score_gain(
        self,
        spec: torch.Tensor,
        x: torch.Tensor,
        gain_bt: torch.Tensor,
        presence: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        if self.reward_fn is not None:
            enhanced = self.tf.apply_band_gain(spec, gain_bt, length=length)
            return self.reward_fn(enhanced, x)
        return reference_free_reward(gain_bt.permute(0, 2, 1), presence, self.suppress_weight)

    def _guarded_gain(
        self,
        spec: torch.Tensor,
        x: torch.Tensor,
        base_gain: torch.Tensor,
        base_enhanced: torch.Tensor,
        adapted_gain: torch.Tensor,
        presence: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        if self.guard_min_reward_delta is None and self.guard_max_output_delta is None:
            return adapted_gain

        accept = True
        if self.guard_min_reward_delta is not None:
            base_reward = self._score_gain(spec, x, base_gain, presence, length)
            adapted_reward = self._score_gain(spec, x, adapted_gain, presence, length)
            accept = accept and bool((adapted_reward - base_reward) >= self.guard_min_reward_delta)

        if self.guard_max_output_delta is not None:
            adapted_enhanced = self.tf.apply_band_gain(spec, adapted_gain, length=length)
            denom = base_enhanced.pow(2).mean().clamp_min(EPS)
            output_delta = (adapted_enhanced - base_enhanced).pow(2).mean() / denom
            accept = accept and bool(output_delta <= self.guard_max_output_delta)

        return adapted_gain if accept else base_gain

    def enhance(self, wav: np.ndarray, sr: int | None = None) -> np.ndarray:
        spec, gain, length = self._adapt(wav)
        with torch.no_grad():
            out = self.tf.apply_band_gain(spec, gain, length=length)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def adaptation_delta_norm(self, wav: np.ndarray) -> float:
        """Diagnostic: magnitude of the online adaptation applied to `wav` (for the ablation)."""
        self._adapt(wav)
        return self.adapter.correction_norm()
