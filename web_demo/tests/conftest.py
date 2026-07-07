import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

WEB_DEMO = Path(__file__).resolve().parents[1]
if str(WEB_DEMO) not in sys.path:
    sys.path.insert(0, str(WEB_DEMO))

import app as app_module  # noqa: E402
from src.controller import PreferenceMemory  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    """Flask test client with preference memory isolated to a temp file."""
    original_memory = app_module.engine.memory
    app_module.engine.memory = PreferenceMemory(tmp_path / "prefs.json")
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c
    app_module.engine.memory = original_memory


@pytest.fixture()
def tone_wav(tmp_path):
    """A valid 1-second 16 kHz WAV file."""
    sr = 16000
    t = np.arange(sr) / sr
    wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, wav, sr)
    return path
