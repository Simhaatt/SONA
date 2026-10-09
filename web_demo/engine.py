"""SONA web-demo engine: model registry, Layer B controller, metrics and streaming.

Thin wrapper over the research code in `src/` — every model loads once and is reused
across requests, unlike `scripts/enhance_controlled.py` which reloads per file.

What the demo exposes from the FYP code base:

* every trained enhancer (GTCRN, complex-mask GTCRN, causal Transformer, Mask-GRU,
  Mask-TCN) and the classical DSP baselines (Wiener, spectral subtraction,
  soft/hard thresholding), all through the same `enhance(wav)` interface;
* the Layer B controller (modes, EQ, compressor, limiter, scene memory) plus the
  mains-hum notch from `src.models.eq`;
* reference-free online adaptation (`OnlineAdapter`, strict-guard settings);
* the reference-based metric suite (`src.eval.metrics`) when a clean reference exists;
* block-streaming (COLA overlap-add) simulation from `sona-final-month/realtime`.
"""

from __future__ import annotations

import csv
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

WEB_DEMO = Path(__file__).resolve().parent
REPO = WEB_DEMO.parent

import sys  # noqa: E402

for _p in (REPO, REPO / "sona-final-month" / "realtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from src.baselines import BASELINES  # noqa: E402
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
from src.eval.metrics import available_metrics, compute_metrics  # noqa: E402
from src.frontend import SpectralFrontend  # noqa: E402
from src.models.adapt import make_adapter  # noqa: E402
from src.models.eq import hum_notch_curve  # noqa: E402
from src.models.rl_attention import OnlineAdapter  # noqa: E402
from src.utils.audio_io import DEFAULT_SR, load_audio, save_audio  # noqa: E402

log = logging.getLogger("sona.engine")

# The models are tiny (<1M params): a few threads give steadier per-block latency than
# letting torch fan out across every core and contend with the rest of the machine.
torch.set_num_threads(min(4, torch.get_num_threads()))

RESULTS = REPO / "results"
FINAL_TABLES = REPO / "sona-final-month" / "FINAL_TABLES"
PREFERENCES_PATH = RESULTS / "web_demo_preferences.json"
EPS = 1e-12

# --------------------------------------------------------------------------- registry

_GTCRN_DIR = RESULTS / "kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07" / "extracted"
_RUN2_DIR = RESULTS / "kaggle_voicebank_online_2026-07-03_run2" / "extracted"


@dataclass(frozen=True)
class EngineSpec:
    """One selectable enhancement method. Benchmarks are the frozen published numbers."""

    id: str
    label: str
    family: str  # "neural" | "classical"
    summary: str
    path: Path | None = None
    benchmark: dict = field(default_factory=dict)
    online_capable: bool = False  # real-valued band-gain model -> OnlineAdapter can wrap it


ENGINES: dict[str, EngineSpec] = {
    spec.id: spec
    for spec in [
        EngineSpec(
            "gtcrn", "GTCRN", "neural",
            "Locked FYP model. Gated temporal-convolution + band GRU, magnitude mask, "
            "multi-resolution spectral loss. Best SI-SDRi of all causal models tested.",
            _GTCRN_DIR / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25" / "last.pt",
            {"set": "VoiceBank 824", "si_sdr_i": 9.927, "pesq": 2.521, "stoi": 0.930, "lsd": 6.47,
             "ms_snsd_si_sdr_i": 3.153},
            online_capable=True,
        ),
        EngineSpec(
            "complex_gtcrn", "Complex GTCRN", "neural",
            "Same backbone with a bounded complex ratio mask (fixes phase too). "
            "+0.31 dB SI-SDRi over the magnitude twin in the 5k-clip A/B, −0.016 PESQ.",
            REPO / "sona-final-month" / "COMPLEX_MASK_FULL5K" / "ckpt_complex_gtcrn" / "last.pt",
            {"set": "VoiceBank 824 (5k-clip A/B)", "si_sdr_i": 10.039, "stoi": 0.931, "lsd": 6.20},
        ),
        EngineSpec(
            "attention", "Causal Transformer", "neural",
            "Causal self-attention mask estimator. Slightly better STOI/LSD than GTCRN, "
            "lower SI-SDRi. Base model for the online-adaptation study.",
            RESULTS / "kaggle_voicebank_strict_guard_full824_2026-07-05" / "extracted" / "ckpt_attention" / "last.pt",
            {"set": "VoiceBank 824", "si_sdr_i": 9.082, "pesq": 2.501, "stoi": 0.932, "lsd": 5.97,
             "ms_snsd_si_sdr_i": 2.586},
            online_capable=True,
        ),
        EngineSpec(
            "mask_gru", "Mask GRU", "neural",
            "Recurrent band-mask baseline (190k params). Fastest learned model.",
            _RUN2_DIR / "ckpt_mask_gru" / "last.pt",
            {"set": "VoiceBank 300", "si_sdr_i": 9.382, "pesq": 2.435, "stoi": 0.928, "lsd": 6.20},
            online_capable=True,
        ),
        EngineSpec(
            "mask_tcn", "Mask TCN", "neural",
            "Causal dilated temporal-convolution band-mask baseline.",
            _RUN2_DIR / "ckpt_mask_tcn" / "last.pt",
            {"set": "VoiceBank 300", "si_sdr_i": 9.268, "pesq": 2.408, "stoi": 0.931, "lsd": 6.28},
            online_capable=True,
        ),
        EngineSpec(
            "wiener", "Wiener filter", "classical",
            "Decision-directed a-priori SNR Wiener gain (Ephraim & Malah). Training-free.",
            benchmark={"set": "VoiceBank 300", "si_sdr_i": -3.024, "pesq": 1.401, "stoi": 0.820, "lsd": 11.35},
        ),
        EngineSpec(
            "spectral_subtraction", "Spectral subtraction", "classical",
            "Boll-style power subtraction with a minimum-statistics noise estimate.",
            benchmark={"set": "VoiceBank 300", "si_sdr_i": -1.061, "pesq": 1.878, "stoi": 0.879, "lsd": 9.33},
        ),
        EngineSpec(
            "soft_threshold", "Soft threshold", "classical",
            "Soft spectral gating against the estimated noise floor.",
            benchmark={"set": "VoiceBank 300", "si_sdr_i": -7.003, "pesq": 1.132, "stoi": 0.723, "lsd": 16.15},
        ),
        EngineSpec(
            "hard_threshold", "Hard threshold", "classical",
            "Binary spectral gate: keep bins above the noise floor, zero the rest.",
            benchmark={"set": "VoiceBank 300", "si_sdr_i": -4.918, "pesq": 1.128, "stoi": 0.761, "lsd": 14.52},
        ),
    ]
}
DEFAULT_ENGINE = "gtcrn"
CHECKPOINT_PATH = ENGINES[DEFAULT_ENGINE].path
CHECKPOINT_NAME = "gtcrn_ch96_h96_b4_mw1p25_mr0p25"

# Strict-guard online adaptation: the configuration that never degraded quality in the
# VoiceBank/MS-SNSD study (attention_online_lowrank_s1_a2_d0p01_go0p0001).
ONLINE_KW = dict(
    steps=12, lr=0.1, anchor_weight=2.0, max_gain_delta=0.01,
    smooth_time_weight=1.0, guard_max_output_delta=1e-4,
)

STREAM_BLOCKS = {  # block_ms -> (block, hop) samples at 16 kHz, 50 % overlap
    32: (512, 256),
    64: (1024, 512),
    128: (2048, 1024),
    256: (4096, 2048),
}

REFERENCE_METRICS = ["si_sdr", "si_sdr_i", "snr_i", "stoi", "estoi", "lsd", "pesq"]

# --------------------------------------------------------------------------- analysis


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), EPS)))


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav)) + EPS)) if wav.size else 0.0


def waveform_summary(wav: np.ndarray, bins: int = 800) -> dict:
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


def _frames_power(wav: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if wav.size < n_fft:
        wav = np.pad(wav, (0, n_fft - wav.size))
    n_frames = 1 + (wav.size - n_fft) // hop
    window = np.hanning(n_fft).astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wav[idx] * window[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2 / (window.sum() ** 2)  # (T, F)


def spectrum_summary(wav: np.ndarray, sr: int = DEFAULT_SR, n_fft: int = 2048, bands: int = 96) -> dict:
    """Per-band level in log-spaced bands, 50 Hz .. Nyquist.

    `levels_db` is the frame-average (dominated by speech); `floor_db` is the
    10th-percentile frame level per band — the steady background-noise floor.
    """
    power = _frames_power(wav, n_fft, n_fft // 2)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    edges = np.geomspace(50.0, sr / 2.0, bands + 1)
    band_of = np.clip(np.searchsorted(edges, freqs, side="right") - 1, -1, bands - 1)
    indicator = np.zeros((freqs.size, bands), dtype=np.float64)
    valid = (freqs >= edges[0]) & (band_of >= 0)
    indicator[valid, band_of[valid]] = 1.0
    centers = np.sqrt(edges[:-1] * edges[1:])
    # Low log-spaced bands are narrower than one FFT bin (7.8 Hz at 2048/16k), so they
    # catch no bin and would read as silence -- a false sawtooth. Use the nearest bin.
    for b in np.flatnonzero(indicator.sum(axis=0) == 0):
        indicator[int(np.argmin(np.abs(freqs - centers[b]))), b] = 1.0
    band_power = power @ indicator / np.maximum(indicator.sum(axis=0), 1.0)
    levels = 10.0 * np.log10(band_power.mean(axis=0) + EPS)
    floors = 10.0 * np.log10(np.percentile(band_power, 10, axis=0) + EPS)
    return {
        "freqs_hz": [round(float(c), 1) for c in centers],
        "levels_db": [round(float(v), 2) for v in levels],
        "floor_db": [round(float(v), 2) for v in floors],
    }


SPEC_FLOOR_DB, SPEC_CEIL_DB = -110.0, -20.0  # shared absolute scale -> panels are comparable


def spectrogram_summary(wav: np.ndarray, max_frames: int = 480, n_freq: int = 128) -> dict:
    """Log-power spectrogram, quantised to 0..255 rows of `n_freq` bins (low -> high freq).

    Uses one fixed dB scale for every signal so input / enhanced / removed panels share colour
    meaning: the same colour is the same absolute level everywhere.
    """
    n_fft, hop = 512, 128
    power = _frames_power(wav, n_fft, hop)  # (T, 257)
    t = power.shape[0]
    if t > max_frames:  # average adjacent frames
        edges = np.linspace(0, t, max_frames + 1, dtype=np.int64)
        power = np.stack([power[a:max(b, a + 1)].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    f = power.shape[1] - 1  # drop Nyquist -> 256 bins
    power = power[:, :f].reshape(power.shape[0], n_freq, f // n_freq).mean(axis=2)
    db = 10.0 * np.log10(power + 1e-20)
    q = np.clip((db - SPEC_FLOOR_DB) / (SPEC_CEIL_DB - SPEC_FLOOR_DB), 0.0, 1.0)
    return {
        "frames": int(q.shape[0]),
        "bins": n_freq,
        "data": (q * 255.0).round().astype(np.uint8).ravel().tolist(),
        "floor_db": SPEC_FLOOR_DB,
        "ceil_db": SPEC_CEIL_DB,
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


def describe_audio(wav: np.ndarray, sr: int = DEFAULT_SR, spectrogram: bool = True) -> dict:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    out = {
        "duration_s": round(wav.size / sr, 3),
        "sample_rate": sr,
        "peak": round(peak, 4),
        "peak_db": round(_db(peak), 2),
        "rms_db": round(_db(_rms(wav)), 2),
        "est_snr_db": estimate_snr_db(wav, sr),
        "waveform": waveform_summary(wav),
        "spectrum": spectrum_summary(wav, sr=sr),
    }
    if spectrogram:
        out["spectrogram"] = spectrogram_summary(wav)
    return out


def reference_metrics(clean: np.ndarray | None, estimate: np.ndarray, noisy: np.ndarray) -> dict | None:
    """Full reference metric suite; unavailable backends (e.g. PESQ on py3.14) map to None."""
    if clean is None:
        return None
    n = min(clean.size, estimate.size, noisy.size)
    m = compute_metrics(clean[:n], estimate[:n], noisy[:n], sr=DEFAULT_SR, include=REFERENCE_METRICS)
    return {k: (None if v is None or not np.isfinite(v) else round(float(v), 4)) for k, v in m.items()}


def noise_floor_reduction_db(noisy: np.ndarray, enhanced: np.ndarray) -> float:
    """Reference-free proxy: drop of the 10th-percentile band floor, averaged 300 Hz..8 kHz."""
    a, b = spectrum_summary(noisy), spectrum_summary(enhanced)
    diffs = [fi - fo for f, fi, fo in zip(a["freqs_hz"], a["floor_db"], b["floor_db"]) if f >= 300]
    return round(float(np.mean(diffs)), 2) if diffs else 0.0


def apply_hum_notch(wav: np.ndarray, base_hz: float, sr: int = DEFAULT_SR, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Notch mains hum at `base_hz` and harmonics (src.models.eq.hum_notch_curve)."""
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).view(1, -1)
    window = torch.hann_window(n_fft, periodic=True)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, window=window, center=True, return_complex=True)
    curve = hum_notch_curve(sr=sr, n_fft=n_fft, base_hz=base_hz, n_harmonics=6, width_hz=6.0)
    y = torch.istft(spec * curve.view(1, -1, 1), n_fft=n_fft, hop_length=hop, window=window,
                    center=True, length=x.shape[-1])
    return y.squeeze(0).numpy().astype(np.float32)


# --------------------------------------------------------------------------- engine


class SonaEngine:
    """Cached enhancers + scene classifier + preference memory."""

    def __init__(self, preferences_path: Path = PREFERENCES_PATH, device: str | None = None) -> None:
        self.device = device or "cpu"
        self.memory = PreferenceMemory(preferences_path)
        self.classifier = SceneClassifier(sr=DEFAULT_SR)
        self._frontend = SpectralFrontend(sr=DEFAULT_SR, n_fft=512, hop=128, n_bands=32)
        self._enhancers: dict[str, object] = {}
        self._status = "cold"  # cold | loading | ready | error
        self._error: str | None = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self.checkpoint_path = CHECKPOINT_PATH

    # -- lifecycle -----------------------------------------------------------

    @property
    def status(self) -> dict:
        default = self._enhancers.get(DEFAULT_ENGINE)
        return {
            "engine": self._status,
            "error": self._error,
            "checkpoint": CHECKPOINT_NAME,
            "checkpoint_exists": self.checkpoint_path.exists(),
            "device": getattr(default, "device", None) or self.device,
            "sample_rate": DEFAULT_SR,
            "loaded_engines": sorted(self._enhancers),
            "metrics_available": {**available_metrics(), "si_sdr/stoi/lsd": True},
        }

    def load_async(self) -> None:
        threading.Thread(target=self.ensure_loaded, daemon=True).start()

    def ensure_loaded(self, engine_id: str = DEFAULT_ENGINE):
        """Return the enhancer for `engine_id`, loading (and caching) it on first use."""
        spec = ENGINES.get(engine_id)
        if spec is None:
            raise KeyError(f"unknown engine '{engine_id}'")
        with self._lock:
            if engine_id in self._enhancers:
                return self._enhancers[engine_id]
            if spec.family == "classical":
                enhancer = BASELINES[engine_id](self._frontend)
                self._enhancers[engine_id] = enhancer
                return enhancer
            if not spec.path or not spec.path.exists():
                msg = f"checkpoint not found: {spec.path}"
                if engine_id == DEFAULT_ENGINE:
                    self._status, self._error = "error", msg
                raise FileNotFoundError(msg)
            if engine_id == DEFAULT_ENGINE:
                self._status = "loading"
            try:
                t0 = time.perf_counter()
                enhancer = load_neural_enhancer(spec.path, device=self.device, name=engine_id)
                self._enhancers[engine_id] = enhancer
                if engine_id == DEFAULT_ENGINE:
                    self._status, self._error = "ready", None
                log.info("loaded %s on %s in %.2fs", engine_id, enhancer.device, time.perf_counter() - t0)
                return enhancer
            except Exception as exc:  # pragma: no cover - depends on env
                if engine_id == DEFAULT_ENGINE:
                    self._status, self._error = "error", f"failed to load checkpoint: {exc}"
                log.exception("checkpoint load failed: %s", engine_id)
                raise

    def engines(self) -> list[dict]:
        out = []
        for spec in ENGINES.values():
            params = None
            enh = self._enhancers.get(spec.id)
            if enh is not None and hasattr(enh, "model"):
                params = int(sum(p.numel() for p in enh.model.parameters()))
            out.append({
                "id": spec.id,
                "label": spec.label,
                "family": spec.family,
                "summary": spec.summary,
                "benchmark": spec.benchmark,
                "online_capable": spec.online_capable,
                "available": spec.family == "classical" or bool(spec.path and spec.path.exists()),
                "params": params,
                "default": spec.id == DEFAULT_ENGINE,
            })
        return out

    def _run(self, engine_id: str, wav: np.ndarray, online: bool = False) -> tuple[np.ndarray, dict]:
        """Raw enhancement (no Layer B post-processing). Returns (output, info)."""
        enhancer = self.ensure_loaded(engine_id)
        info: dict = {"engine": engine_id}
        with self._infer_lock:
            t0 = time.perf_counter()
            if online:
                spec = ENGINES[engine_id]
                if not spec.online_capable:
                    raise ValueError(f"online adaptation is not available for '{spec.label}'")
                frozen = enhancer.enhance(wav, sr=DEFAULT_SR)
                adapter = OnlineAdapter(
                    enhancer.model, enhancer.tf, device=enhancer.device, name=f"{engine_id}_online",
                    adapter=make_adapter("lowrank", enhancer.tf.n_bands), **ONLINE_KW,
                )
                out = adapter.enhance(wav, sr=DEFAULT_SR)
                diff = float(np.max(np.abs(out - frozen))) if out.size else 0.0
                info["online"] = {
                    "accepted": diff > 1e-7,
                    "max_sample_change": round(diff, 6),
                    "correction_norm": round(float(adapter.adapter.correction_norm()), 5),
                    "frozen_output": frozen,
                }
            else:
                out = enhancer.enhance(wav, sr=DEFAULT_SR)
            info["seconds"] = time.perf_counter() - t0
        return np.asarray(out, dtype=np.float32)[: wav.size], info

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

    # -- studio enhancement --------------------------------------------------

    def enhance(
        self,
        input_path: Path,
        output_path: Path,
        residual_path: Path | None = None,
        reference: np.ndarray | None = None,
        engine_id: str = DEFAULT_ENGINE,
        online_adapt: bool = False,
        hum_notch_hz: float | None = None,
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
            mode=mode, scene=scene, denoise_strength=denoise_strength, safety_floor=safety_floor,
            eq_gains_db=eq_gains_db, compressor_threshold_db=compressor_threshold_db,
            compressor_ratio=compressor_ratio, bypass_eq=bypass_eq,
            bypass_compressor=bypass_compressor, use_memory=use_memory,
        )
        stages.append({"stage": "mode_controller", "seconds": round(time.perf_counter() - t0, 4)})

        # 3. hum notch (optional, before the denoiser so the model sees less tonal noise)
        source = noisy
        if hum_notch_hz:
            t0 = time.perf_counter()
            source = apply_hum_notch(noisy, hum_notch_hz, sr=sr)
            stages.append({"stage": "hum_notch", "seconds": round(time.perf_counter() - t0, 4)})

        # 4. enhancement (+ optional online adaptation) and strength blend
        enhanced, run_info = self._run(engine_id, source, online=online_adapt)
        strength = float(settings.mask_strength)
        blended = source + strength * (enhanced - source)
        stages.append({"stage": "online_adaptation" if online_adapt else "enhancement",
                       "seconds": round(run_info["seconds"], 4)})

        # 5. EQ / compressor / limiter
        t0 = time.perf_counter()
        with self._infer_lock:
            processed = apply_post_processing(blended, settings=settings, sr=sr, n_fft=512, hop=128, device="cpu")
        stages.append({"stage": "eq_compressor", "seconds": round(time.perf_counter() - t0, 4)})

        # 6. export (optional peak normalization to input level for easier A/B listening)
        t0 = time.perf_counter()
        normalized = False
        if normalize_output:
            in_peak_pre = float(np.max(np.abs(noisy))) if noisy.size else 0.0
            out_peak_pre = float(np.max(np.abs(processed))) if processed.size else 0.0
            target = min(max(in_peak_pre, 0.1), float(settings.limiter_ceiling))
            if out_peak_pre > 1e-6:
                processed = (processed * (target / out_peak_pre)).astype(np.float32)
                normalized = True
        residual = (noisy - processed).astype(np.float32)  # removed = input minus delivered output
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
        spec = ENGINES[engine_id]
        analytics = {
            "engine": engine_id,
            "engine_label": spec.label,
            "engine_family": spec.family,
            "input_duration_s": round(duration_s, 3),
            "output_duration_s": round(processed.size / sr, 3),
            "scene": scene or "",
            "scene_confidence": scene_result["confidence"] if scene_result else None,
            "auto_scene": bool(scene_result),
            "mode": resolved_mode,
            "denoise_strength": round(strength, 3),
            "safety_floor": round(float(settings.safety_floor), 3),
            "hum_notch_hz": hum_notch_hz or None,
            "processing_time_s": round(processing_s, 3),
            "realtime_factor": round(processing_s / max(duration_s, EPS), 3),
            "input_peak": round(in_peak, 4),
            "output_peak": round(out_peak, 4),
            "input_rms_db": round(_db(_rms(noisy)), 2),
            "output_rms_db": round(_db(_rms(processed)), 2),
            "residual_rms_db": round(_db(_rms(residual)), 2),
            "residual_peak": round(float(np.max(np.abs(residual))) if residual.size else 0.0, 4),
            "noise_removed_pct": round(float(np.clip(100.0 * res_energy / max(in_energy, EPS), 0.0, 100.0)), 1),
            "noise_floor_reduction_db": noise_floor_reduction_db(noisy, processed),
            "loudness_change_db": round(_db(_rms(processed)) - _db(_rms(noisy)), 2),
            "est_snr_in_db": estimate_snr_db(noisy, sr),
            "est_snr_out_db": estimate_snr_db(processed, sr),
            "normalized_output": normalized,
            "limiter_ceiling": float(settings.limiter_ceiling),
            "limiter_engaged": out_peak >= float(settings.limiter_ceiling) - 1e-3,
            "memory_used": memory_used,
            "remembered": bool(remember and scene),
            "checkpoint": CHECKPOINT_NAME if engine_id == DEFAULT_ENGINE else engine_id,
            "device": self.device,
            "output_path": str(output_path),
            "stages": stages,
            "settings": asdict(settings),
        }
        if scene_result:
            analytics["scene_features"] = scene_result["features"]
        if online_adapt:
            online = dict(run_info["online"])
            frozen = online.pop("frozen_output")
            if reference is not None:
                frozen_m = reference_metrics(reference, frozen, noisy)
                adapted_m = reference_metrics(reference, enhanced, noisy)
                online["si_sdr_i_frozen"] = frozen_m.get("si_sdr_i")
                online["si_sdr_i_adapted"] = adapted_m.get("si_sdr_i")
            analytics["online"] = online
        if reference is not None:
            # score both the raw model output (fair, matches the paper) and the delivered output
            analytics["metrics"] = {
                "input": reference_metrics(reference, noisy, noisy),
                "model": reference_metrics(reference, enhanced, noisy),
                "delivered": reference_metrics(reference, processed, noisy),
            }
        return analytics

    # -- comparison ----------------------------------------------------------

    def compare(self, noisy: np.ndarray, engine_ids: list[str], reference: np.ndarray | None,
                save_dir: Path) -> list[dict]:
        """Raw output of each engine on the same clip — same harness as the paper tables."""
        rows = []
        duration = noisy.size / DEFAULT_SR
        for engine_id in engine_ids:
            spec = ENGINES[engine_id]
            row: dict = {"engine": engine_id, "label": spec.label, "family": spec.family}
            try:
                out, info = self._run(engine_id, noisy)
            except Exception as exc:  # noqa: BLE001 - one broken engine shouldn't sink the table
                log.exception("compare failed for %s", engine_id)
                row["error"] = str(exc)
                rows.append(row)
                continue
            audio_id = uuid.uuid4().hex
            save_audio(save_dir / f"{audio_id}.wav", out, DEFAULT_SR)
            enh = self._enhancers.get(engine_id)
            row.update({
                "audio_id": audio_id,
                "seconds": round(info["seconds"], 4),
                "rtf": round(info["seconds"] / max(duration, EPS), 4),
                "params": int(sum(p.numel() for p in enh.model.parameters())) if hasattr(enh, "model") else 0,
                "noise_floor_reduction_db": noise_floor_reduction_db(noisy, out),
                "est_snr_db": estimate_snr_db(out),
                "metrics": reference_metrics(reference, out, noisy),
                "waveform": waveform_summary(out, bins=240),
            })
            rows.append(row)
        return rows

    # -- streaming -----------------------------------------------------------

    def stream(self, noisy: np.ndarray, block_ms: int, engine_id: str, reference: np.ndarray | None) -> tuple[np.ndarray, dict]:
        """Simulate realtime block streaming (COLA overlap-add, `sona-final-month/realtime`)."""
        from stream_processor import StreamProcessor

        block, hop = STREAM_BLOCKS[block_ms]
        enhancer = self.ensure_loaded(engine_id)
        timings: list[float] = []

        def process(chunk: np.ndarray) -> np.ndarray:
            t0 = time.perf_counter()
            out = enhancer.enhance(chunk.astype(np.float32), sr=DEFAULT_SR)
            timings.append(time.perf_counter() - t0)
            return out

        sp = StreamProcessor(block_size=block, hop_size=hop, process=process)
        outs: list[np.ndarray] = []
        with self._infer_lock:
            for i in range(0, noisy.size, hop):
                outs.extend(sp.push(noisy[i : i + hop]))
            outs.append(sp.flush())
        streamed = np.concatenate(outs)[: noisy.size].astype(np.float32)
        if streamed.size < noisy.size:
            streamed = np.pad(streamed, (0, noisy.size - streamed.size))

        offline, _ = self._run(engine_id, noisy)
        deadline_ms = 1000.0 * hop / DEFAULT_SR
        times_ms = [round(t * 1000.0, 3) for t in timings]
        report = {
            "engine": engine_id,
            "engine_label": ENGINES[engine_id].label,
            "block_ms": block_ms,
            "block_samples": block,
            "hop_samples": hop,
            "hop_ms": round(deadline_ms, 2),
            "algorithmic_latency_ms": round(1000.0 * block / DEFAULT_SR, 1),
            "blocks": len(times_ms),
            "block_times_ms": times_ms,
            "mean_block_ms": round(float(np.mean(times_ms)), 3) if times_ms else 0.0,
            "p95_block_ms": round(float(np.percentile(times_ms, 95)), 3) if times_ms else 0.0,
            "max_block_ms": round(float(np.max(times_ms)), 3) if times_ms else 0.0,
            "deadline_misses": int(sum(t > deadline_ms for t in times_ms)),
            "rtf": round(float(np.sum(timings)) / max(noisy.size / DEFAULT_SR, EPS), 4),
            "metrics_input": reference_metrics(reference, noisy, noisy),
            "metrics_stream": reference_metrics(reference, streamed, noisy),
            "metrics_offline": reference_metrics(reference, offline, noisy),
            "floor_reduction_stream_db": noise_floor_reduction_db(noisy, streamed),
            "floor_reduction_offline_db": noise_floor_reduction_db(noisy, offline),
        }
        return streamed, report

    # -- presets / research --------------------------------------------------

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


RESEARCH_TABLES = [
    ("voicebank", "VoiceBank-DEMAND test — 824 clips", "voicebank_full824.csv",
     "Main benchmark. Noise types in the test set are unseen during training."),
    ("ms_snsd", "MS-SNSD external validation — 1,100 clips", "ms_snsd_external.csv",
     "A different corpus the models never saw. Speech quality still improves (GTCRN +3.15 dB SI-SDR), but intelligibility stays at the original level (STOI 0.818 original vs 0.816): the honest generalisation gap. Original LSD is omitted because the LSD formula changed after these runs."),
    ("online", "Test-time online adaptation — frozen vs online", "online_adaptation_story.csv",
     "SONA's original hypothesis. Unguarded adaptation hurts; the strict guard is safe but the gain is ~0."),
    ("classical", "Classical DSP baselines — synthetic dev set", "classical_synthetic.csv",
     "Training-free methods on synthetic noise (illustrative only)."),
]


BASELINES_PATH = WEB_DEMO / "data" / "baselines.json"
ORIGINAL = "Original (no processing)"
VOICEBANK_ORIGINAL_PESQ = 1.968  # noisy-input PESQ measured in the project harness (Kaggle, 824 clips)


FYP_BENCHMARK_CSV = REPO / "sona-final-month" / "FYP_BENCHMARK" / "summary.csv"


def _baselines() -> dict:
    """Original-input scores per test set.

    VoiceBank uses the frozen passthrough row from the FYP benchmark (the numbers in the paper).
    MS-SNSD has no frozen passthrough run, so it comes from `tools/compute_baselines.py`; its LSD
    is dropped because the LSD formula changed after the frozen model runs, so it is not comparable.
    """
    import json

    try:
        base = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        base = {}
    if "ms_snsd" in base:
        base["ms_snsd"].pop("lsd", None)
    if FYP_BENCHMARK_CSV.is_file():
        with FYP_BENCHMARK_CSV.open(encoding="utf-8") as fh:
            row = next((r for r in csv.DictReader(fh) if r.get("method") == "passthrough"), None)
        if row:
            base["voicebank"] = {k: round(float(row[k]), 4) for k in ("si_sdr", "snr", "stoi", "estoi", "lsd") if row.get(k)}
            base["voicebank"]["n"] = int(float(row.get("n", 824)))
    return base


def _original_row(columns: list[str], base: dict, label: str = ORIGINAL, pesq: float | None = None) -> dict:
    """A table row for the unprocessed noisy input: improvements are 0 by definition."""
    values = {
        "si_sdr_i": 0.0, "snr_i": 0.0, "si_sdr": base.get("si_sdr"), "snr": base.get("snr"),
        "stoi": base.get("stoi"), "estoi": base.get("estoi"), "lsd": base.get("lsd"),
        "pesq": pesq if pesq is not None else base.get("pesq"), "n": base.get("n"),
    }
    row = {}
    for c in columns:
        if c in ("label", "method"):
            row[c] = label
        else:
            v = values.get(c)
            row[c] = "" if v is None else str(v)
    row["_baseline"] = "1"
    return row


def research_tables() -> list[dict]:
    """Frozen result tables from `sona-final-month/FINAL_TABLES` as JSON, each with the original input as a baseline row."""
    out = []
    for key, title, fname, note in RESEARCH_TABLES:
        path = FINAL_TABLES / fname
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        out.append({"key": key, "title": title, "note": note, "columns": list(rows[0].keys()) if rows else [], "rows": rows})
        if key == "classical":  # this table already measured the untouched input as "passthrough"
            for r in rows:
                if r.get("method") == "passthrough":
                    r["method"], r["_baseline"] = ORIGINAL, "1"
    complex_csv = REPO / "sona-final-month" / "COMPLEX_MASK_FULL5K" / "summary.csv"
    if complex_csv.is_file():
        with complex_csv.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        out.insert(1, {
            "key": "complex",
            "title": "Complex vs magnitude mask — 5k-clip A/B, 824 test clips",
            "note": "Complex mask wins SI-SDRi by +0.31 dB (p < 1e-6) but loses 0.016 PESQ (p = 0.005): a trade-off.",
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
        })
    base = _baselines()
    vb, ms = base.get("voicebank", {}), base.get("ms_snsd", {})
    for table in out:
        cols = table["columns"]
        if table["key"] in ("voicebank", "complex") and vb:
            table["rows"].insert(0, _original_row(cols, vb, pesq=VOICEBANK_ORIGINAL_PESQ))
        elif table["key"] == "ms_snsd" and ms:
            table["rows"].insert(0, _original_row(cols, ms))
        elif table["key"] == "online":
            if ms:
                table["rows"].insert(2, _original_row(cols, ms, label="MS-SNSD original (no processing)"))
            if vb:
                table["rows"].insert(0, _original_row(cols, vb, label="VoiceBank original (no processing)", pesq=VOICEBANK_ORIGINAL_PESQ))
    return out
