"""Backend tests for the SONA web demo. Run: cd web_demo && python -m pytest tests -q"""

import io
from pathlib import Path

import app as app_module

SCENES = {"speech", "music", "traffic", "fan", "crowd", "mixed", "unknown"}


def _load_sample(client) -> dict:
    res = client.post("/api/sample")
    assert res.status_code == 200, res.get_json()
    return res.get_json()


# ── health / static ──────────────────────────────────────────────────────────


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert data["checkpoint_exists"] is True
    assert data["sample_available"] is True


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"SONA" in res.data


def test_modes_listed(client):
    data = client.get("/api/modes").get_json()
    assert set(data["modes"]) == app_module.MODES
    assert {"speech", "music", "podcast", "gaming", "hearing", "studio"} <= set(data["modes"])
    assert set(data["scenes"]) == SCENES


# ── sample & upload ──────────────────────────────────────────────────────────


def test_sample_file_exists():
    assert app_module.SAMPLE_PATH.is_file(), "bundled sample missing — run tools/make_sample.py"


def test_sample_endpoint(client):
    info = _load_sample(client)
    assert info["duration_s"] > 0
    assert info["sample_rate"] == 16000
    # the bundled sample must be genuinely noisy so the demo shows clear enhancement
    assert info["est_snr_db"] < 20, f"sample looks too clean (est SNR {info['est_snr_db']} dB)"
    assert info["waveform"]["bins"] > 0
    assert len(info["spectrum"]["levels_db"]) > 0
    assert (app_module.UPLOADS_DIR / f"{info['file_id']}.wav").is_file()


def test_upload_valid_wav(client, tone_wav):
    with tone_wav.open("rb") as fh:
        res = client.post("/api/upload", data={"file": (fh, "tone.wav")})
    assert res.status_code == 200
    info = res.get_json()
    assert abs(info["duration_s"] - 1.0) < 0.05


def test_upload_rejects_bad_extension(client):
    res = client.post("/api/upload", data={"file": (io.BytesIO(b"hello"), "notes.txt")})
    assert res.status_code == 415
    assert "unsupported" in res.get_json()["error"]


def test_upload_rejects_undecodable_audio(client):
    res = client.post("/api/upload", data={"file": (io.BytesIO(b"\x00" * 2048), "fake.wav")})
    assert res.status_code == 415
    assert "decode" in res.get_json()["error"]


def test_upload_requires_file(client):
    res = client.post("/api/upload", data={})
    assert res.status_code == 400


# ── scene classification ─────────────────────────────────────────────────────


def test_classify_sample(client):
    info = _load_sample(client)
    res = client.post("/api/classify", json={"file_id": info["file_id"]})
    assert res.status_code == 200
    data = res.get_json()
    assert data["scene"] in SCENES
    assert 0.0 <= data["confidence"] <= 1.0
    assert "features" in data


def test_classify_rejects_bad_id(client):
    assert client.post("/api/classify", json={"file_id": "../../etc"}).status_code == 400
    assert client.post("/api/classify", json={"file_id": "0" * 32}).status_code == 404


# ── enhancement ──────────────────────────────────────────────────────────────


def test_enhance_sample_produces_wav(client):
    info = _load_sample(client)
    res = client.post(
        "/api/enhance",
        json={"file_id": info["file_id"], "auto_scene": True, "normalize_output": True},
    )
    assert res.status_code == 200, res.get_json()
    data = res.get_json()
    out_path = Path(data["analytics"]["output_path"])
    assert out_path.is_file()
    assert out_path.parent == app_module.OUTPUTS_DIR
    a = data["analytics"]
    assert a["scene"] in SCENES
    assert a["mode"] in app_module.MODES
    assert a["processing_time_s"] > 0
    assert a["realtime_factor"] > 0
    assert abs(a["output_duration_s"] - a["input_duration_s"]) < 0.05
    assert [s["stage"] for s in a["stages"]] == [
        "scene_analysis", "mode_controller", "gtcrn_enhancement", "eq_compressor", "export_wav",
    ]
    # residual (removed noise) is computed and exposed
    assert "residual" in data and data["residual"]["waveform"]["bins"] > 0
    assert "residual_rms_db" in a
    assert 0.0 <= a["noise_removed_pct"] <= 100.0
    assert (app_module.OUTPUTS_DIR / f"{data['output_id']}_residual.wav").is_file()
    # output + residual are streamable, output downloadable
    assert client.get(f"/api/audio/output/{data['output_id']}").status_code == 200
    assert client.get(f"/api/audio/residual/{data['output_id']}").status_code == 200
    dl = client.get(f"/api/download/{data['output_id']}")
    assert dl.status_code == 200
    assert "attachment" in dl.headers["Content-Disposition"]


def test_enhance_honors_bypass_flags(client):
    info = _load_sample(client)
    res = client.post(
        "/api/enhance",
        json={"file_id": info["file_id"], "mode": "aggressive_speech",
              "bypass_eq": True, "bypass_compressor": True},
    )
    assert res.status_code == 200, res.get_json()
    settings = res.get_json()["analytics"]["settings"]
    assert settings["bypass_eq"] is True
    assert settings["bypass_compressor"] is True


def test_enhance_validates_mode_and_scene(client):
    info = _load_sample(client)
    assert client.post("/api/enhance", json={"file_id": info["file_id"], "mode": "bass_boost"}).status_code == 400
    assert client.post("/api/enhance", json={"file_id": info["file_id"], "scene": "space"}).status_code == 400


def test_enhance_missing_input(client):
    assert client.post("/api/enhance", json={"file_id": "f" * 32}).status_code == 404


# ── download safety ──────────────────────────────────────────────────────────


def test_download_rejects_traversal(client):
    # non-hex ids are rejected outright; path separators never reach the handler
    assert client.get("/api/download/notanid").status_code == 400
    assert client.get("/api/download/..%2f..%2fapp.py").status_code in (400, 404)
    assert client.get("/api/audio/output/" + "a" * 32).status_code == 404


# ── preference memory ────────────────────────────────────────────────────────


def test_preferences_roundtrip(client):
    res = client.post(
        "/api/preferences",
        json={"scene": "traffic", "mode": "hearing", "denoise_strength": 0.9},
    )
    assert res.status_code == 200
    assert res.get_json()["saved"] is True

    data = client.get("/api/preferences").get_json()
    assert "traffic" in data["preferences"]
    rec = data["preferences"]["traffic"]
    assert rec["mode"] == "hearing"
    assert rec["settings"]["mask_strength"] == 0.9

    assert client.delete("/api/preferences/traffic").get_json()["deleted"] is True
    assert "traffic" not in client.get("/api/preferences").get_json()["preferences"]


def test_preferences_validation(client):
    assert client.post("/api/preferences", json={"scene": "moon", "mode": "speech"}).status_code == 400
    assert client.post("/api/preferences", json={"scene": "speech", "mode": "nope"}).status_code == 400
