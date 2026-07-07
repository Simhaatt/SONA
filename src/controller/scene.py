"""Lightweight feature-based scene classifier for SONA Layer B."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..utils.audio_io import DEFAULT_SR

EPS = 1e-8


@dataclass(frozen=True)
class ScenePrediction:
    """Scene label plus confidence and interpretable audio features."""

    label: str
    confidence: float
    features: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SceneClassifier:
    """Simple deterministic classifier for speech/music/traffic/fan/crowd/unknown."""

    labels = ("speech", "music", "traffic", "fan", "crowd", "mixed", "unknown")

    def __init__(self, sr: int = DEFAULT_SR, n_fft: int = 1024, hop: int = 256) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop

    def predict(self, wav: np.ndarray, sr: int | None = None) -> ScenePrediction:
        if sr is not None and sr != self.sr:
            raise ValueError(f"expected {self.sr} Hz audio, got {sr} Hz")
        x = np.asarray(wav, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        feats = self.features(x)
        scores = self._scores(feats)
        label, score = max(scores.items(), key=lambda item: item[1])
        confidence = _soft_confidence(scores, label)
        if feats["rms"] < 1e-4:
            label, confidence = "unknown", 0.2
        return ScenePrediction(label=label, confidence=confidence, features=feats)

    def features(self, wav: np.ndarray) -> dict[str, float]:
        x = np.asarray(wav, dtype=np.float32)
        if x.size == 0:
            return _empty_features()
        x = x - float(np.mean(x))
        rms = float(np.sqrt(np.mean(x**2) + EPS))
        frames = _frames(x, self.n_fft, self.hop)
        if frames.size == 0:
            return _empty_features(rms=rms)

        win = np.hanning(self.n_fft).astype(np.float32)
        spec = np.fft.rfft(frames * win[None, :], axis=1)
        mag = np.abs(spec).astype(np.float32) + EPS
        power = mag**2
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr).astype(np.float32)
        total = power.sum(axis=1) + EPS

        centroid = (power * freqs[None, :]).sum(axis=1) / total
        flatness = np.exp(np.mean(np.log(mag), axis=1)) / (np.mean(mag, axis=1) + EPS)
        rolloff = _rolloff(power, freqs, ratio=0.85)
        low = _band_energy(power, freqs, 20.0, 250.0)
        mid = _band_energy(power, freqs, 250.0, 3000.0)
        high = _band_energy(power, freqs, 3000.0, self.sr / 2.0)
        zcr = _zero_crossing_rate(frames)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1) + EPS)

        return {
            "rms": rms,
            "centroid_hz": float(np.mean(centroid)),
            "flatness": float(np.mean(flatness)),
            "rolloff_hz": float(np.mean(rolloff)),
            "zcr": float(np.mean(zcr)),
            "low_ratio": float(np.mean(low)),
            "mid_ratio": float(np.mean(mid)),
            "high_ratio": float(np.mean(high)),
            "modulation": float(np.std(frame_rms) / (np.mean(frame_rms) + EPS)),
        }

    def _scores(self, f: dict[str, float]) -> dict[str, float]:
        centroid = f["centroid_hz"]
        flatness = f["flatness"]
        zcr = f["zcr"]
        low = f["low_ratio"]
        mid = f["mid_ratio"]
        high = f["high_ratio"]
        mod = f["modulation"]
        rolloff = f["rolloff_hz"]

        speech = (
            1.5 * _tri(mid, 0.30, 0.75)
            + 1.1 * _tri(mod, 0.12, 1.3)
            + 1.0 * _tri(centroid, 450.0, 2400.0)
            + 0.5 * _tri(zcr, 0.03, 0.18)
            + 0.4 * _tri(low, 0.10, 0.45)
            - 0.5 * high
        )
        music = (
            1.1 * _tri(low + mid, 0.65, 0.98)
            + 0.8 * _tri(rolloff, 1800.0, 6500.0)
            + 0.7 * _tri(mod, 0.03, 0.5)
            + 0.4 * _tri(high, 0.05, 0.35)
            - 0.4 * flatness
        )
        fan = (
            1.4 * _tri(flatness, 0.35, 0.95)
            + 1.0 * _tri(mod, 0.0, 0.2)
            + 0.8 * _tri(centroid, 200.0, 1600.0)
            + 0.4 * _tri(low, 0.25, 0.8)
        )
        traffic = (
            1.2 * _tri(low, 0.35, 0.9)
            + 0.8 * _tri(flatness, 0.15, 0.75)
            + 0.8 * _tri(mod, 0.18, 1.5)
            + 0.5 * _tri(centroid, 300.0, 2200.0)
            - 0.5 * _tri(mid, 0.35, 0.75)
        )
        crowd = (
            1.1 * _tri(mid, 0.35, 0.8)
            + 0.9 * _tri(high, 0.12, 0.55)
            + 0.8 * _tri(flatness, 0.2, 0.75)
            + 0.6 * _tri(zcr, 0.08, 0.32)
            + 0.4 * _tri(mod, 0.10, 1.0)
        )
        mixed = 0.5 + 0.4 * _tri(flatness, 0.1, 0.8) + 0.4 * _tri(mod, 0.08, 1.2)
        unknown = 0.1
        return {
            "speech": speech,
            "music": music,
            "traffic": traffic,
            "fan": fan,
            "crowd": crowd,
            "mixed": mixed,
            "unknown": unknown,
        }


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    n = 1 + (x.size - n_fft) // hop
    if n <= 0:
        return np.empty((0, n_fft), dtype=np.float32)
    shape = (n, n_fft)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def _rolloff(power: np.ndarray, freqs: np.ndarray, ratio: float) -> np.ndarray:
    cumulative = np.cumsum(power, axis=1)
    target = cumulative[:, -1:] * ratio
    idx = np.argmax(cumulative >= target, axis=1)
    return freqs[idx]


def _band_energy(power: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    total = power.sum(axis=1) + EPS
    return power[:, mask].sum(axis=1) / total


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=1)


def _tri(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    center = (lo + hi) * 0.5
    half = (hi - lo) * 0.5
    return max(0.0, 1.0 - abs(float(x) - center) / half)


def _soft_confidence(scores: dict[str, float], label: str) -> float:
    values = np.array(list(scores.values()), dtype=np.float32)
    values = values - float(values.max())
    probs = np.exp(values)
    probs = probs / (float(probs.sum()) + EPS)
    idx = list(scores).index(label)
    return float(np.clip(probs[idx], 0.0, 1.0))


def _empty_features(rms: float = 0.0) -> dict[str, float]:
    return {
        "rms": float(rms),
        "centroid_hz": 0.0,
        "flatness": 0.0,
        "rolloff_hz": 0.0,
        "zcr": 0.0,
        "low_ratio": 0.0,
        "mid_ratio": 0.0,
        "high_ratio": 0.0,
        "modulation": 0.0,
    }
