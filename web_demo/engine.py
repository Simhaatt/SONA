"""SONA web-demo engine: caches the Layer A checkpoint and drives the Layer B controller.

Thin wrapper over `src.controller` — the model loads once and is reused across requests,
unlike `scripts/enhance_controlled.py` which reloads the checkpoint per file.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

WEB_DEMO = Path(__file__).resolve().parent
REPO = WEB_DEMO.parent

import sys  # noqa: E402

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.controller import (  # noqa: E402
    MODE_PRESETS,
    SCENE_TO_MODE,
    ControllerSettings,
    PreferenceMemory,
    SceneClassifier,
    SonaController,
    apply_post_processing,
    load_neural_enhancer,
)
from src.utils.audio_io import DEFAULT_SR, load_audio, save_audio  # noqa: E402

log = logging.getLogger("sona.engine")

CHECKPOINT_PATH = (
    REPO
    / "results"
    / "kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07"
    / "extracted"
    / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25"
    / "last.pt"
)
CHECKPOINT_NAME = "gtcrn_ch96_h96_b4_mw1p25_mr0p25"
PREFERENCES_PATH = REPO / "results" / "web_demo_preferences.json"

EPS = 1e-12


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), EPS)))


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav)) + EPS)) if wav.size else 0.0


def waveform_summary(wav: np.ndarray, bins: int = 700) -> dict:
    """Downsampled min/max envelope for canvas waveform rendering."""
    n = wav.size
    if n == 0:
        return {"mins": [], "maxs": [], "bins": 0}
    bins = min(bins, n)
    edges = np.linspace(0, n, bins + 1, dtype=np.int64)
    mins, maxs = [], []
    for i in range(bins):
        chunk = wav[edges[i] : max(edges[i + 1], edges[i] + 1)]
        mins.append(round(float(chunk.min()), 4))
        maxs.append(round(float(chunk.max()), 4))
    return {"mins": mins, "maxs": maxs, "bins": bins}


def spectrum_summary(
    wav: np.ndarray, sr: int = DEFAULT_SR, n_fft: int = 2048, bands: int = 96
) -> dict:
    """Per-band magnitude in log-spaced bands, 50 Hz .. Nyquist.

    `levels_db` is the frame-average (dominated by speech); `floor_db` is the
    10th-percentile frame level per band — the steady background-noise floor.
    """
    if wav.size < n_fft:
        wav = np.pad(wav, (0, n_fft - wav.size))
    hop = n_fft // 2
    n_frames = 1 + (wav.size - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wav[idx] * window[None, :]
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (n_frames, F)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    edges = np.geomspace(50.0, sr / 2.0, bands + 1)
    band_of = np.clip(np.searchsorted(edges, freqs, side="right") - 1, -1, bands - 1)
    indicator = np.zeros((freqs.size, bands), dtype=np.float64)
    valid = (freqs >= edges[0]) & (band_of >= 0)
    indicator[valid, band_of[valid]] = 1.0
    counts = np.maximum(indicator.sum(axis=0), 1.0)
    band_power = power @ indicator / counts  # (n_frames, bands)

    levels = 10.0 * np.log10(band_power.mean(axis=0) + EPS)
    floors = 10.0 * np.log10(np.percentile(band_power, 10, axis=0) + EPS)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return {
        "freqs_hz": [round(float(c), 1) for c in centers],
        "levels_db": [round(float(v), 2) for v in levels],
        "floor_db": [round(float(v), 2) for v in floors],
    }


def estimate_snr_db(wav: np.ndarray, sr: int = DEFAULT_SR) -> float:
    """Crude activity-vs-floor SNR proxy: 90th vs 10th percentile of frame RMS (dB)."""
    frame = int(0.032 * sr)
    if wav.size < frame * 4:
        return 0.0
    n = wav.size // frame
    frames = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1) + EPS)
    floor = float(np.percentile(rms, 10))
    active = float(np.percentile(rms, 90))
    return round(20.0 * np.log10(max(active, EPS) / max(floor, EPS)), 2)


def describe_audio(wav: np.ndarray, sr: int = DEFAULT_SR) -> dict:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    return {
        "duration_s": round(wav.size / sr, 3),
        "sample_rate": sr,
        "peak": round(peak, 4),
        "peak_db": round(_db(peak), 2),
        "rms_db": round(_db(_rms(wav)), 2),
        "est_snr_db": estimate_snr_db(wav, sr),
        "waveform": waveform_summary(wav),
        "spectrum": spectrum_summary(wav, sr=sr),
    }


class SonaEngine:
    """Cached GTCRN enhancer + scene classifier + preference memory."""

    def __init__(
        self,
        checkpoint_path: Path = CHECKPOINT_PATH,
        preferences_path: Path = PREFERENCES_PATH,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.memory = PreferenceMemory(preferences_path)
        self.classifier = SceneClassifier(sr=DEFAULT_SR)
        self._enhancer = None
        self._status = "cold"  # cold | loading | ready | error
        self._error: str | None = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    @property
    def status(self) -> dict:
        return {
            "engine": self._status,
            "error": self._error,
            "checkpoint": CHECKPOINT_NAME,
            "checkpoint_exists": self.checkpoint_path.exists(),
            "device": getattr(self._enhancer, "device", None) or (self.device or "auto"),
            "sample_rate": DEFAULT_SR,
        }

    def load_async(self) -> None:
        threading.Thread(target=self.ensure_loaded, daemon=True).start()

    def ensure_loaded(self):
        with self._lock:
            if self._enhancer is not None:
                return self._enhancer
            if not self.checkpoint_path.exists():
                self._status, self._error = "error", (
                    f"checkpoint not found: {self.checkpoint_path}"
                )
                raise FileNotFoundError(self._error)
            self._status = "loading"
            try:
                t0 = time.perf_counter()
                self._enhancer = load_neural_enhancer(
                    self.checkpoint_path, device=self.device, name=CHECKPOINT_NAME
                )
                self._status, self._error = "ready", None
                log.info(
                    "loaded %s on %s in %.2fs",
                    CHECKPOINT_NAME,
                    self._enhancer.device,
                    time.perf_counter() - t0,
                )
            except Exception as exc:  # pragma: no cover - depends on env
                self._status, self._error = "error", f"failed to load checkpoint: {exc}"
                log.exception("checkpoint load failed")
                raise
            return self._enhancer

    # -- scene ---------------------------------------------------------------

    def classify(self, wav: np.ndarray) -> dict:
        pred = self.classifier.predict(wav, sr=DEFAULT_SR)
        return {
            "scene": pred.label,
            "confidence": round(float(pred.confidence), 4),
            "features": {k: round(float(v), 5) for k, v in pred.features.items()},
            "suggested_mode": SCENE_TO_MODE.get(pred.label, "speech"),
        }

    # -- settings ------------------------------------------------------------

    def resolve_settings(
        self,
        mode: str | None,
        scene: str | None,
        denoise_strength: float | None,
        safety_floor: float | None,
        eq_gains_db: list[float] | None,
        compressor_threshold_db: float | None,
        compressor_ratio: float | None,
        use_memory: bool,
        bypass_eq: bool | None = None,
        bypass_compressor: bool | None = None,
    ) -> tuple[ControllerSettings, str, bool]:
        """Returns (settings, resolved_mode, memory_used)."""
        controller = SonaController(memory=self.memory)
        record = self.memory.get(scene) if use_memory and scene and not mode else None
        resolved_mode = (mode or (record.mode if record else None) or controller.mode_for_scene(scene)).lower()
        settings = controller.settings(
            mode=mode,
            scene=scene,
            denoise_strength=denoise_strength,
            safety_floor=safety_floor,
            bypass_eq=bypass_eq,
            bypass_compressor=bypass_compressor,
            use_memory=use_memory,
        )
        overrides: dict = {}
        if eq_gains_db is not None:
            gains = [float(np.clip(g, -12.0, 12.0)) for g in eq_gains_db][:8]
            gains += [0.0] * (8 - len(gains))
            overrides["eq_gains_db"] = tuple(gains)
        if compressor_threshold_db is not None:
            overrides["compressor_threshold_db"] = float(np.clip(compressor_threshold_db, -60.0, 0.0))
        if compressor_ratio is not None:
            overrides["compressor_ratio"] = float(np.clip(compressor_ratio, 1.0, 12.0))
        if overrides:
            settings = replace(settings, **overrides)
        return settings, resolved_mode, record is not None

    # -- enhancement ---------------------------------------------------------

    def enhance(
        self,
        input_path: Path,
        output_path: Path,
        residual_path: Path | None = None,
        mode: str | None = None,
        scene: str | None = None,
        auto_scene: bool = False,
        denoise_strength: float | None = None,
        safety_floor: float | None = None,
        eq_gains_db: list[float] | None = None,
        compressor_threshold_db: float | None = None,
        compressor_ratio: float | None = None,
        bypass_eq: bool | None = None,
        bypass_compressor: bool | None = None,
        use_memory: bool = True,
        remember: bool = False,
        normalize_output: bool = False,
    ) -> dict:
        enhancer = self.ensure_loaded()
        noisy, sr = load_audio(input_path, sr=DEFAULT_SR, mono=True)

        stages: list[dict] = []
        t_total = time.perf_counter()

        # 1. scene analysis
        t0 = time.perf_counter()
        scene_result = None
        if auto_scene and not scene:
            scene_result = self.classify(noisy)
            scene = scene_result["scene"]
        stages.append({"stage": "scene_analysis", "seconds": round(time.perf_counter() - t0, 4)})

        # 2. controller resolution
        t0 = time.perf_counter()
        settings, resolved_mode, memory_used = self.resolve_settings(
            mode=mode,
            scene=scene,
            denoise_strength=denoise_strength,
            safety_floor=safety_floor,
            eq_gains_db=eq_gains_db,
            compressor_threshold_db=compressor_threshold_db,
            compressor_ratio=compressor_ratio,
            bypass_eq=bypass_eq,
            bypass_compressor=bypass_compressor,
            use_memory=use_memory,
        )
        stages.append({"stage": "mode_controller", "seconds": round(time.perf_counter() - t0, 4)})

        # 3. GTCRN neural enhancement (+ strength blend)
        t0 = time.perf_counter()
        with self._infer_lock:
            enhanced = enhancer.enhance(noisy, sr=sr)
        strength = float(settings.mask_strength)
        blended = noisy + strength * (enhanced - noisy)
        stages.append({"stage": "gtcrn_enhancement", "seconds": round(time.perf_counter() - t0, 4)})

        # 4. EQ / compressor / limiter
        t0 = time.perf_counter()
        with self._infer_lock:
            processed = apply_post_processing(
                blended,
                settings=settings,
                sr=sr,
                n_fft=enhancer.tf.n_fft,
                hop=enhancer.tf.hop,
                device=enhancer.device,
            )
        stages.append({"stage": "eq_compressor", "seconds": round(time.perf_counter() - t0, 4)})

        # 5. export (optional peak normalization to input level for easier A/B listening)
        t0 = time.perf_counter()
        normalized = False
        if normalize_output:
            in_peak_pre = float(np.max(np.abs(noisy))) if noisy.size else 0.0
            out_peak_pre = float(np.max(np.abs(processed))) if processed.size else 0.0
            target = min(max(in_peak_pre, 0.1), float(settings.limiter_ceiling))
            if out_peak_pre > 1e-6:
                processed = (processed * (target / out_peak_pre)).astype(np.float32)
                normalized = True
        # removed noise = input minus the output actually delivered to the user
        residual = (noisy - processed).astype(np.float32)
        if residual_path is not None:
            save_audio(residual_path, residual, sr)
        save_audio(output_path, processed, sr)
        stages.append({"stage": "export_wav", "seconds": round(time.perf_counter() - t0, 4)})

        processing_s = time.perf_counter() - t_total
        if remember and scene:
            self.memory.set(scene, resolved_mode, settings)

        duration_s = noisy.size / sr
        in_peak = float(np.max(np.abs(noisy))) if noisy.size else 0.0
        out_peak = float(np.max(np.abs(processed))) if processed.size else 0.0
        in_energy = float(np.mean(np.square(noisy))) if noisy.size else 0.0
        res_energy = float(np.mean(np.square(residual))) if residual.size else 0.0
        noise_removed_pct = float(np.clip(100.0 * res_energy / max(in_energy, EPS), 0.0, 100.0))
        analytics = {
            "input_duration_s": round(duration_s, 3),
            "output_duration_s": round(processed.size / sr, 3),
            "scene": scene or "",
            "scene_confidence": scene_result["confidence"] if scene_result else None,
            "auto_scene": bool(scene_result),
            "mode": resolved_mode,
            "denoise_strength": round(strength, 3),
            "safety_floor": round(float(settings.safety_floor), 3),
            "processing_time_s": round(processing_s, 3),
            "realtime_factor": round(processing_s / max(duration_s, EPS), 3),
            "input_peak": round(in_peak, 4),
            "output_peak": round(out_peak, 4),
            "input_rms_db": round(_db(_rms(noisy)), 2),
            "output_rms_db": round(_db(_rms(processed)), 2),
            "residual_rms_db": round(_db(_rms(residual)), 2),
            "residual_peak": round(float(np.max(np.abs(residual))) if residual.size else 0.0, 4),
            "noise_removed_pct": round(noise_removed_pct, 1),
            "loudness_change_db": round(_db(_rms(processed)) - _db(_rms(noisy)), 2),
            "normalized_output": normalized,
            "limiter_ceiling": float(settings.limiter_ceiling),
            "limiter_engaged": out_peak >= float(settings.limiter_ceiling) - 1e-3,
            "memory_used": memory_used,
            "remembered": bool(remember and scene),
            "checkpoint": CHECKPOINT_NAME,
            "device": enhancer.device,
            "output_path": str(output_path),
            "stages": stages,
            "settings": asdict(settings),
        }
        if scene_result:
            analytics["scene_features"] = scene_result["features"]
        return analytics

    # -- presets -------------------------------------------------------------

    @staticmethod
    def presets() -> dict:
        return {
            "modes": {
                name: {"description": p.description, "settings": asdict(p.settings)}
                for name, p in MODE_PRESETS.items()
            },
            "scene_to_mode": dict(SCENE_TO_MODE),
            "scenes": list(SceneClassifier.labels),
        }
