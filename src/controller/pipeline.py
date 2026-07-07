"""Offline controlled enhancement pipeline for SONA Layer B."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..models import Compressor, NeuralEnhancer, ParametricEQ
from ..train.train_supervised import build_frontend, build_model
from ..utils.audio_io import DEFAULT_SR, load_audio, save_audio
from ..utils.config import Config
from .memory import PreferenceMemory
from .modes import ControllerSettings, SonaController


def load_neural_enhancer(
    checkpoint_path: str | Path,
    device: str | None = None,
    name: str = "sona_layer_a",
) -> NeuralEnhancer:
    """Load a supervised SONA checkpoint as a `NeuralEnhancer`."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    cfg = Config(checkpoint["cfg"])
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return NeuralEnhancer(model, tf, device=device, name=name)


def apply_post_processing(
    wav: np.ndarray,
    settings: ControllerSettings,
    sr: int = DEFAULT_SR,
    n_fft: int = 512,
    hop: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Apply EQ, compressor, and limiter to a mono waveform."""

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).view(1, -1).to(device)
    window = torch.hann_window(n_fft, periodic=True, device=device)
    spec = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    phase = torch.exp(1j * torch.angle(spec))
    mag = spec.abs()

    eq = ParametricEQ(sr=sr, n_fft=n_fft).to(device)
    eq_gains = _match_eq_bands(settings.eq_gains_db, eq.n_bands).to(device)
    mag = eq(mag, eq_gains)

    compressor = Compressor(
        threshold_db=settings.compressor_threshold_db,
        ratio=settings.compressor_ratio,
        makeup_db=settings.compressor_makeup_db,
        ceiling_db=None,
    ).to(device)
    mag = compressor(mag)

    y = torch.istft(
        mag * phase,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=x.shape[-1],
    )
    out = y.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return _limit_peak(out, settings.limiter_ceiling)


def enhance_file(
    checkpoint_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    mode: str | None = None,
    scene: str | None = None,
    denoise_strength: float | None = None,
    safety_floor: float | None = None,
    memory_path: str | Path | None = None,
    remember: bool = False,
    use_memory: bool = True,
    device: str | None = None,
) -> dict[str, str | float | bool]:
    """Enhance one WAV file with Layer A GTCRN plus Layer B mode controls."""

    memory = PreferenceMemory(memory_path) if memory_path else None
    controller = SonaController(memory=memory)
    chosen_mode = mode or controller.mode_for_scene(scene)
    memory_record = memory.get(scene) if memory and scene and use_memory and not mode else None
    actual_mode = mode or (memory_record.mode if memory_record else chosen_mode)
    settings = controller.settings(
        mode=mode,
        scene=scene,
        denoise_strength=denoise_strength,
        safety_floor=safety_floor,
        use_memory=use_memory,
    )
    noisy, sr = load_audio(input_path, sr=DEFAULT_SR, mono=True)
    enhancer = load_neural_enhancer(checkpoint_path, device=device)
    enhanced = enhancer.enhance(noisy, sr=sr)

    strength = float(settings.mask_strength)
    blended = noisy + strength * (enhanced - noisy)
    processed = apply_post_processing(
        blended,
        settings=settings,
        sr=sr,
        n_fft=enhancer.tf.n_fft,
        hop=enhancer.tf.hop,
        device=enhancer.device,
    )
    save_audio(output_path, processed, sr)
    if remember and memory and scene:
        memory.set(scene, actual_mode, settings)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "mode": actual_mode,
        "scene": scene or "",
        "memory_used": bool(memory_record),
        "remembered": bool(remember and memory and scene),
        "mask_strength": settings.mask_strength,
        "safety_floor": settings.safety_floor,
        "compressor_threshold_db": settings.compressor_threshold_db,
        "compressor_ratio": settings.compressor_ratio,
    }


def _match_eq_bands(gains_db: tuple[float, ...], n_bands: int) -> torch.Tensor:
    gains = list(gains_db)
    if len(gains) < n_bands:
        gains.extend([0.0] * (n_bands - len(gains)))
    return torch.tensor(gains[:n_bands], dtype=torch.float32)


def _limit_peak(wav: np.ndarray, ceiling: float) -> np.ndarray:
    ceiling = float(max(0.0, min(1.0, ceiling)))
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > ceiling > 0.0:
        wav = wav * (ceiling / peak)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)
