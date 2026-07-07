"""Layer B controller and offline post-processing tests."""

import numpy as np

from src.controller import ControllerSettings, SonaController, apply_post_processing


def test_controller_modes_and_scene_routing():
    controller = SonaController()
    assert "speech" in controller.preset_names()
    assert controller.mode_for_scene("traffic") == "hearing"

    speech = controller.settings(mode="speech")
    music = controller.settings(mode="music")
    assert speech.mask_strength > music.mask_strength
    assert len(speech.eq_gains_db) == 8


def test_controller_overrides_are_clamped():
    controller = SonaController()
    settings = controller.settings(mode="speech", denoise_strength=2.0, safety_floor=-1.0)
    assert settings.mask_strength == 1.25
    assert settings.safety_floor == 0.0


def test_post_processing_neutral_keeps_shape_and_bounds():
    sr = 16000
    t = np.arange(sr // 2, dtype=np.float32) / sr
    wav = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    settings = ControllerSettings(
        mask_strength=1.0,
        safety_floor=0.0,
        eq_gains_db=(0.0,) * 8,
        compressor_threshold_db=0.0,
        compressor_ratio=1.0,
        compressor_makeup_db=0.0,
    )
    out = apply_post_processing(wav, settings, sr=sr, n_fft=256, hop=64, device="cpu")
    assert out.shape == wav.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
    assert np.max(np.abs(out - wav)) < 1e-4
