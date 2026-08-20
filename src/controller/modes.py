"""Mode presets and scene routing for the SONA Layer B controller."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory import PreferenceMemory


@dataclass(frozen=True)
class ControllerSettings:
    """Concrete audio-control values consumed by the offline/realtime pipeline."""

    mask_strength: float
    safety_floor: float
    eq_gains_db: tuple[float, ...]
    compressor_threshold_db: float
    compressor_ratio: float
    compressor_makeup_db: float = 0.0
    limiter_ceiling: float = 0.98
    bypass_eq: bool = False
    bypass_compressor: bool = False


@dataclass(frozen=True)
class ModePreset:
    """Named product mode plus its controller settings."""

    name: str
    description: str
    settings: ControllerSettings


NEUTRAL_EQ_8 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

MODE_PRESETS: dict[str, ModePreset] = {
    "speech": ModePreset(
        name="speech",
        description="Speech clarity with moderate denoising and presence lift.",
        settings=ControllerSettings(
            # mask_strength=1.0 means "trust the model's own gain estimate": the
            # pipeline blends as noisy + strength*(enhanced-noisy), so strength>1.0
            # EXTRAPOLATES beyond what GTCRN predicted. Measured against real
            # clean references (sona-final-month/COMPLEX_MASK_FINDINGS.md-adjacent
            # sweep), SI-SDR peaks at ~1.0 and falls monotonically above it — the
            # previous 1.05 here was a small, consistent quality loss.
            mask_strength=1.0,
            safety_floor=0.06,
            eq_gains_db=(-2.0, -1.5, -1.0, 0.0, 0.8, 1.5, 1.0, 0.3),
            compressor_threshold_db=-26.0,
            compressor_ratio=2.2,
            compressor_makeup_db=0.2,
        ),
    ),
    "aggressive_speech": ModePreset(
        name="aggressive_speech",
        description="Stronger speech denoising for obvious fan, traffic, or room noise.",
        settings=ControllerSettings(
            # was 1.25: extrapolating 25% past the model's gain estimate. Measured
            # on AirConditioner/Traffic/Babble clips at -5..10 dB SNR against real
            # clean references, 1.25 lost ~1.4 dB SI-SDR on average vs 1.0, and
            # actively made hard clips WORSE than doing nothing (e.g. AirConditioner
            # @ -5 dB: +0.21 dB at strength=1.0 vs -2.04 dB at strength=1.25). The
            # "stronger denoising" this mode should offer comes from the EQ/
            # compressor settings below, not from overshooting the model's mask.
            mask_strength=1.0,
            safety_floor=0.03,
            eq_gains_db=(-3.0, -2.2, -1.2, 0.0, 0.7, 1.2, 0.8, 0.0),
            compressor_threshold_db=-24.0,
            compressor_ratio=1.6,
            compressor_makeup_db=0.0,
            limiter_ceiling=0.96,
        ),
    ),
    "music": ModePreset(
        name="music",
        description="Gentler denoising with fuller lows and less speech-focused compression.",
        settings=ControllerSettings(
            mask_strength=0.72,
            safety_floor=0.16,
            eq_gains_db=(1.5, 1.2, 0.8, 0.0, 0.0, 0.4, 0.8, 0.6),
            compressor_threshold_db=-20.0,
            compressor_ratio=1.4,
            compressor_makeup_db=0.0,
        ),
    ),
    "podcast": ModePreset(
        name="podcast",
        description="Voice-forward leveling for spoken content.",
        settings=ControllerSettings(
            mask_strength=0.92,
            safety_floor=0.10,
            eq_gains_db=(-2.5, -2.0, -0.5, 0.8, 1.2, 1.8, 1.0, 0.0),
            compressor_threshold_db=-28.0,
            compressor_ratio=2.8,
            compressor_makeup_db=0.4,
        ),
    ),
    "gaming": ModePreset(
        name="gaming",
        description="Low-fatigue clarity with restrained dynamics.",
        settings=ControllerSettings(
            mask_strength=0.82,
            safety_floor=0.12,
            eq_gains_db=(-1.0, -0.5, 0.0, 0.4, 0.6, 1.2, 1.6, 1.2),
            compressor_threshold_db=-22.0,
            compressor_ratio=1.8,
            compressor_makeup_db=0.0,
        ),
    ),
    "hearing": ModePreset(
        name="hearing",
        description="Speech-preserving high-clarity mode with stronger leveling.",
        settings=ControllerSettings(
            mask_strength=0.96,
            safety_floor=0.18,
            eq_gains_db=(-3.0, -2.0, -1.0, 0.6, 1.8, 3.0, 3.0, 2.0),
            compressor_threshold_db=-32.0,
            compressor_ratio=3.2,
            compressor_makeup_db=0.6,
            limiter_ceiling=0.94,
        ),
    ),
    "studio": ModePreset(
        name="studio",
        description="Cleaner output with near-neutral tone and conservative compression.",
        settings=ControllerSettings(
            mask_strength=0.88,
            safety_floor=0.06,
            eq_gains_db=NEUTRAL_EQ_8,
            compressor_threshold_db=-18.0,
            compressor_ratio=1.25,
            compressor_makeup_db=0.0,
        ),
    ),
}

SCENE_TO_MODE: dict[str, str] = {
    "speech": "speech",
    "music": "music",
    "traffic": "aggressive_speech",
    "fan": "aggressive_speech",
    "crowd": "hearing",
    "mixed": "podcast",
    "unknown": "speech",
}


class SonaController:
    """Resolve user mode/scene choices into concrete audio-control settings."""

    def __init__(
        self,
        presets: dict[str, ModePreset] | None = None,
        scene_to_mode: dict[str, str] | None = None,
        memory: PreferenceMemory | None = None,
    ) -> None:
        self.presets = presets or MODE_PRESETS
        self.scene_to_mode = scene_to_mode or SCENE_TO_MODE
        self.memory = memory

    def preset_names(self) -> tuple[str, ...]:
        return tuple(self.presets)

    def mode_for_scene(self, scene: str | None) -> str:
        if not scene:
            return "speech"
        return self.scene_to_mode.get(scene.lower(), "speech")

    def settings(
        self,
        mode: str | None = None,
        scene: str | None = None,
        denoise_strength: float | None = None,
        safety_floor: float | None = None,
        bypass_eq: bool | None = None,
        bypass_compressor: bool | None = None,
        use_memory: bool = True,
    ) -> ControllerSettings:
        memory_record = self.memory.get(scene) if use_memory and self.memory and not mode else None
        mode_name = (mode or (memory_record.mode if memory_record else None) or self.mode_for_scene(scene)).lower()
        if mode_name not in self.presets:
            raise ValueError(f"unknown mode '{mode_name}'. options: {list(self.presets)}")
        settings = memory_record.settings if memory_record else self.presets[mode_name].settings
        if denoise_strength is not None:
            settings = replace(settings, mask_strength=_clamp(denoise_strength, 0.0, 1.25))
        if safety_floor is not None:
            settings = replace(settings, safety_floor=_clamp(safety_floor, 0.0, 0.5))
        if bypass_eq is not None:
            settings = replace(settings, bypass_eq=bool(bypass_eq))
        if bypass_compressor is not None:
            settings = replace(settings, bypass_compressor=bool(bypass_compressor))
        return settings


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
