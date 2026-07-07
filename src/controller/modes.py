"""Mode presets and scene routing for the SONA Layer B controller."""

from __future__ import annotations

from dataclasses import dataclass, replace


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
            mask_strength=1.0,
            safety_floor=0.08,
            eq_gains_db=(-2.0, -1.5, -1.0, 0.0, 1.0, 2.0, 1.5, 0.5),
            compressor_threshold_db=-26.0,
            compressor_ratio=2.2,
            compressor_makeup_db=1.0,
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
            compressor_makeup_db=0.5,
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
            compressor_makeup_db=1.5,
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
            compressor_makeup_db=0.5,
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
            compressor_makeup_db=2.0,
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
    "traffic": "hearing",
    "fan": "speech",
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
    ) -> None:
        self.presets = presets or MODE_PRESETS
        self.scene_to_mode = scene_to_mode or SCENE_TO_MODE

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
    ) -> ControllerSettings:
        mode_name = (mode or self.mode_for_scene(scene)).lower()
        if mode_name not in self.presets:
            raise ValueError(f"unknown mode '{mode_name}'. options: {list(self.presets)}")
        settings = self.presets[mode_name].settings
        if denoise_strength is not None:
            settings = replace(settings, mask_strength=_clamp(denoise_strength, 0.0, 1.25))
        if safety_floor is not None:
            settings = replace(settings, safety_floor=_clamp(safety_floor, 0.0, 0.5))
        return settings


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
