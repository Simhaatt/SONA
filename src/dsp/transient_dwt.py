"""Localized Haar-DWT post-filter for short residual acoustic impulses.

The detector deliberately uses only signals available at deployment time: the
microphone waveform and the enhancer output.  Reference-clean audio is never an
input to :func:`apply_transient_dwt`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TransientDWTConfig:
    threshold: float = 3.0
    transition: float = 1.5
    dilation_frames: int = 2
    detail_attenuation: float = 0.55
    soft_threshold: float = 0.35
    local_gain: float = 0.95
    levels: int = 3
    frame_ms: float = 20.0
    hop_ms: float = 10.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _robust_positive_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    # A small standard-deviation fallback prevents a constant trace from being
    # marked transient due to floating-point noise.
    scale = max(1.4826 * mad, 0.10 * float(np.std(values)), 1e-4)
    return np.maximum((values - center) / scale, 0.0)


def transient_frame_scores(noisy: np.ndarray, enhanced: np.ndarray, sr: int,
                           frame_ms: float = 20.0,
                           hop_ms: float = 10.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute threshold-independent observable transient scores once per clip."""
    x = np.asarray(noisy, dtype=np.float64).reshape(-1)
    y = np.asarray(enhanced, dtype=np.float64).reshape(-1)
    n = min(len(x), len(y))
    if n == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64), 0
    x, y = x[:n], y[:n]
    frame = max(32, int(round(sr * frame_ms / 1000.0)))
    hop = max(16, int(round(sr * hop_ms / 1000.0)))
    starts = np.arange(0, n, hop)
    padded_length = int(starts[-1] + frame)
    x_pad = np.pad(x, (0, padded_length - n))
    y_pad = np.pad(y, (0, padded_length - n))
    x_frames = np.lib.stride_tricks.sliding_window_view(x_pad, frame)[::hop][:len(starts)]
    y_frames = np.lib.stride_tricks.sliding_window_view(y_pad, frame)[::hop][:len(starts)]
    x_rms = np.sqrt(np.mean(x_frames * x_frames, axis=1) + 1e-12)
    y_rms = np.sqrt(np.mean(y_frames * y_frames, axis=1) + 1e-12)
    log_rms = np.log(x_rms + 1e-8)
    diff_rms = np.log(np.sqrt(np.mean(np.diff(x_frames, axis=1, prepend=x_frames[:, :1]) ** 2,
                                           axis=1) + 1e-12) + 1e-8)
    crest = np.log(np.max(np.abs(x_frames), axis=1) / (x_rms + 1e-8) + 1e-8)
    output_diff = np.log(np.sqrt(np.mean(np.diff(y_frames, axis=1, prepend=y_frames[:, :1]) ** 2,
                                             axis=1) + 1e-12) / (y_rms + 1e-8) + 1e-8)
    # Sudden onset relative to the recent past is useful for gunshots while the
    # robust global scores keep siren/engine energy from triggering every frame.
    previous = np.empty_like(log_rms)
    for index in range(len(log_rms)):
        previous[index] = np.median(log_rms[max(0, index - 5):max(1, index)])
    onset = np.maximum(log_rms - previous, 0.0)
    broad = (0.38 * _robust_positive_z(log_rms)
             + 0.32 * _robust_positive_z(diff_rms)
             + 0.18 * _robust_positive_z(crest)
             + 0.12 * _robust_positive_z(output_diff))
    sharp = 0.65 * _robust_positive_z(onset) + 0.35 * _robust_positive_z(crest)
    score = np.maximum(broad, sharp)
    centers = np.minimum(starts + frame // 2, n - 1)
    return score.astype(np.float32), centers.astype(np.int64), n


def transient_confidence(noisy: np.ndarray, enhanced: np.ndarray, sr: int,
                         config: TransientDWTConfig,
                         prepared_scores: tuple[np.ndarray, np.ndarray, int] | None = None) -> np.ndarray:
    """Return a sample-aligned [0, 1] transient confidence trace."""
    score, centers, n = prepared_scores or transient_frame_scores(
        noisy, enhanced, sr, config.frame_ms, config.hop_ms)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    confidence = np.clip((score - config.threshold) / max(config.transition, 1e-6), 0.0, 1.0)
    radius = max(0, int(config.dilation_frames))
    if radius:
        expanded = confidence.copy()
        for offset in range(1, radius + 1):
            expanded[offset:] = np.maximum(expanded[offset:], confidence[:-offset])
            expanded[:-offset] = np.maximum(expanded[:-offset], confidence[offset:])
        confidence = expanded
    sample_positions = np.arange(n)
    trace = np.interp(sample_positions, centers, confidence, left=confidence[0], right=confidence[-1])
    return trace.astype(np.float32)


def _haar_decompose(signal: np.ndarray, levels: int) -> tuple[np.ndarray, list[np.ndarray]]:
    approx = np.asarray(signal, dtype=np.float64)
    details = []
    root_two = np.sqrt(2.0)
    for _ in range(levels):
        even, odd = approx[0::2], approx[1::2]
        details.append((even - odd) / root_two)
        approx = (even + odd) / root_two
    return approx, details


def _haar_reconstruct(approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    root_two = np.sqrt(2.0)
    value = np.asarray(approx, dtype=np.float64)
    for detail in reversed(details):
        output = np.empty(detail.size * 2, dtype=np.float64)
        output[0::2] = (value + detail) / root_two
        output[1::2] = (value - detail) / root_two
        value = output
    return value


def apply_transient_dwt(noisy: np.ndarray, enhanced: np.ndarray, sr: int,
                        config: TransientDWTConfig,
                        prepared_scores: tuple[np.ndarray, np.ndarray, int] | None = None
                        ) -> tuple[np.ndarray, dict[str, float]]:
    """Shrink localized Haar details and return output plus deployment diagnostics."""
    original = np.asarray(enhanced, dtype=np.float64).reshape(-1)
    n = min(len(np.asarray(noisy).reshape(-1)), len(original))
    if n == 0:
        return original.astype(np.float32), {"active_fraction": 0.0, "mean_confidence": 0.0}
    confidence = transient_confidence(noisy, original, sr, config, prepared_scores)[:n]
    block = 2 ** max(1, int(config.levels))
    padded_n = int(np.ceil(n / block) * block)
    padded = np.pad(original[:n], (0, padded_n - n), mode="reflect" if n > 1 else "edge")
    mask = np.pad(confidence, (0, padded_n - n), mode="edge")
    approx, details = _haar_decompose(padded, int(config.levels))
    filtered_details = []
    for level, detail in enumerate(details, 1):
        width = 2 ** level
        local_mask = mask.reshape(-1, width).max(axis=1)
        sigma = float(np.median(np.abs(detail - np.median(detail))) / 0.6745 + 1e-10)
        threshold = config.soft_threshold * sigma * local_mask
        shrunk = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        shrunk *= 1.0 - config.detail_attenuation * local_mask
        filtered_details.append(shrunk)
    output = _haar_reconstruct(approx, filtered_details)[:n]
    if config.local_gain < 1.0:
        output *= 1.0 - (1.0 - config.local_gain) * confidence
    if len(original) > n:
        output = np.concatenate((output, original[n:]))
    diagnostics = {"active_fraction": float(np.mean(confidence > 1e-3)),
                   "strong_fraction": float(np.mean(confidence >= .5)),
                   "mean_confidence": float(np.mean(confidence)),
                   "max_confidence": float(np.max(confidence))}
    return output.astype(np.float32), diagnostics
