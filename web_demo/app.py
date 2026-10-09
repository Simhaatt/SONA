"""SONA web demo — local Flask backend + static frontend.

Run (from the repo root or web_demo/):
    python web_demo/app.py
then open http://localhost:7860
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from engine import (
    DEFAULT_ENGINE,
    DEFAULT_SR,
    ENGINES,
    PREFERENCES_PATH,
    REPO,
    STREAM_BLOCKS,
    WEB_DEMO,
    SonaEngine,
    describe_audio,
    estimate_snr_db,
    reference_metrics,
    research_tables,
    waveform_summary,
)
from src.controller import MODE_PRESETS
from src.utils.audio_io import load_audio, save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("sona.web")

HOST, PORT = "127.0.0.1", 7860
UPLOADS_DIR = REPO / "results" / "web_demo_uploads"
OUTPUTS_DIR = REPO / "results" / "web_demo_outputs"
SAMPLES_DIR = WEB_DEMO / "samples"
SAMPLE_PATH = SAMPLES_DIR / "sample_noisy.wav"
LIBRARY_PATH = SAMPLES_DIR / "library.json"
STATIC_DIR = WEB_DEMO / "static"

ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
MAX_UPLOAD_MB = 60
MAX_DURATION_S = 300.0
MAX_ANALYSIS_S = 60.0  # compare / streaming run several passes — keep them snappy
ID_RE = re.compile(r"^[0-9a-f]{32}$")
SAMPLE_ID_RE = re.compile(r"^[a-z0-9_]{1,64}$")
SCENES = {"speech", "music", "traffic", "fan", "crowd", "mixed", "unknown"}
MODES = set(MODE_PRESETS)  # follows the controller package, incl. new presets
HUM_NOTCH = {0, 50, 60}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

engine = SonaEngine()
_names: dict[str, str] = {}  # file_id -> original display name (in-memory only)


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message, self.status = message, status


@app.errorhandler(ApiError)
def _api_error(err: ApiError):
    return jsonify({"error": err.message}), err.status


@app.errorhandler(HTTPException)
def _http_error(err: HTTPException):
    return jsonify({"error": err.description or err.name}), err.code or 500


@app.errorhandler(Exception)
def _unhandled(err: Exception):
    log.exception("unhandled error")
    return jsonify({"error": "internal server error"}), 500


# -- helpers -------------------------------------------------------------------


def _check_id(file_id: str) -> str:
    if not isinstance(file_id, str) or not ID_RE.match(file_id):
        raise ApiError("invalid file id", 400)
    return file_id


def _input_path(file_id: str) -> Path:
    path = UPLOADS_DIR / f"{_check_id(file_id)}.wav"
    if not path.is_file():
        raise ApiError("input audio not found; upload or select a sample first", 404)
    return path


def _reference_path(file_id: str) -> Path:
    return UPLOADS_DIR / f"{_check_id(file_id)}_ref.wav"


def _load_reference(file_id: str) -> np.ndarray | None:
    path = _reference_path(file_id)
    if not path.is_file():
        return None
    wav, _ = load_audio(path, sr=DEFAULT_SR, mono=True)
    return wav


def _output_path(output_id: str) -> Path:
    path = OUTPUTS_DIR / f"{_check_id(output_id)}.wav"
    if not path.is_file():
        raise ApiError("output audio not found", 404)
    return path


def _reference_info(ref: np.ndarray) -> dict:
    return {"duration_s": round(ref.size / DEFAULT_SR, 3), "waveform": waveform_summary(ref)}


def _register_input(wav: np.ndarray, name: str, reference: np.ndarray | None = None, sample: dict | None = None) -> dict:
    file_id = uuid.uuid4().hex
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_audio(UPLOADS_DIR / f"{file_id}.wav", wav, DEFAULT_SR)
    if reference is not None:
        save_audio(_reference_path(file_id), reference, DEFAULT_SR)
    _names[file_id] = name
    info = describe_audio(wav, DEFAULT_SR)
    info.update({
        "file_id": file_id,
        "name": name,
        "sample": sample,
        "reference": _reference_info(reference) if reference is not None else None,
    })
    return info


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ApiError("expected a JSON object body", 400)
    return data


def _opt_float(data: dict, key: str, lo: float, hi: float) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return float(np.clip(float(value), lo, hi))
    except (TypeError, ValueError):
        raise ApiError(f"'{key}' must be a number", 400)


def _opt_bool(data: dict, key: str) -> bool | None:
    value = data.get(key)
    return None if value is None else bool(value)


def _engine_id(value, default: str = DEFAULT_ENGINE) -> str:
    engine_id = str(value or default)
    if engine_id not in ENGINES:
        raise ApiError(f"unknown engine '{engine_id}'; options: {sorted(ENGINES)}", 400)
    return engine_id


def _decode_upload(field: str = "file") -> tuple[np.ndarray, str]:
    file = request.files.get(field)
    if file is None or not file.filename:
        raise ApiError(f"no file provided (multipart field '{field}')", 400)
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ApiError(
            f"unsupported file type '{ext or 'none'}'; allowed: " + ", ".join(sorted(ALLOWED_EXTENSIONS)),
            415,
        )
    tmp_path = UPLOADS_DIR / f"tmp_{uuid.uuid4().hex}{ext}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        file.save(tmp_path)
        try:
            wav, _ = load_audio(tmp_path, sr=DEFAULT_SR, mono=True)
        except Exception:
            raise ApiError("could not decode audio — is this a valid audio file?", 415)
    finally:
        tmp_path.unlink(missing_ok=True)
    if wav.size == 0:
        raise ApiError("audio file is empty", 415)
    if wav.size / DEFAULT_SR > MAX_DURATION_S:
        raise ApiError(f"audio longer than {MAX_DURATION_S:.0f}s is not supported in the demo", 413)
    # display name only — storage always uses a fresh UUID
    safe_name = re.sub(r"[^\w .()\-]", "_", Path(file.filename).name)[:80]
    return wav, safe_name


def _library() -> list[dict]:
    if not LIBRARY_PATH.is_file():
        return []
    try:
        entries = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in entries if (SAMPLES_DIR / f"{e['id']}_noisy.wav").is_file()]


def _analysis_input(data: dict) -> tuple[str, np.ndarray, np.ndarray | None]:
    file_id = str(data.get("file_id", ""))
    wav, _ = load_audio(_input_path(file_id), sr=DEFAULT_SR, mono=True)
    if wav.size / DEFAULT_SR > MAX_ANALYSIS_S:
        raise ApiError(f"compare / realtime analysis is limited to {MAX_ANALYSIS_S:.0f}s clips", 413)
    return file_id, wav, _load_reference(file_id)


# -- frontend ----------------------------------------------------------------


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(STATIC_DIR, filename)


# -- status ------------------------------------------------------------------


@app.get("/api/health")
def health():
    status = engine.status
    status.update({
        "ok": True,
        "sample_available": SAMPLE_PATH.is_file() or bool(_library()),
        "outputs_dir": str(OUTPUTS_DIR),
        "preferences_path": str(PREFERENCES_PATH),
    })
    return jsonify(status)


@app.get("/api/modes")
def modes():
    return jsonify(engine.presets())


@app.get("/api/engines")
def engines():
    return jsonify({"engines": engine.engines(), "default": DEFAULT_ENGINE,
                    "stream_blocks_ms": sorted(STREAM_BLOCKS)})


@app.get("/api/research")
def research():
    return jsonify({"tables": research_tables()})


# -- audio input ---------------------------------------------------------------


@app.get("/api/samples")
def samples():
    items = _library()
    if SAMPLE_PATH.is_file():
        items.append({"id": "legacy", "title": "Demo mix (no clean reference)", "dataset": "MS-SNSD mix",
                      "noise": "AirConditioner / Traffic", "snr_db": 3.0, "duration_s": None,
                      "has_reference": False})
    for item in items:
        item.setdefault("has_reference", (SAMPLES_DIR / f"{item['id']}_clean.wav").is_file())
    return jsonify({"samples": items})


@app.post("/api/sample")
def use_sample():
    data = request.get_json(silent=True) or {}
    sample_id = str(data.get("sample_id") or "legacy")
    if sample_id == "legacy":
        if not SAMPLE_PATH.is_file():
            raise ApiError("bundled sample missing; run tools/make_sample.py", 500)
        wav, _ = load_audio(SAMPLE_PATH, sr=DEFAULT_SR, mono=True)
        return jsonify(_register_input(wav, "sample_noisy.wav"))
    if not SAMPLE_ID_RE.match(sample_id):
        raise ApiError("invalid sample id", 400)
    meta = next((e for e in _library() if e["id"] == sample_id), None)
    if meta is None:
        raise ApiError(f"unknown sample '{sample_id}'", 404)
    wav, _ = load_audio(SAMPLES_DIR / f"{sample_id}_noisy.wav", sr=DEFAULT_SR, mono=True)
    clean_path = SAMPLES_DIR / f"{sample_id}_clean.wav"
    ref = load_audio(clean_path, sr=DEFAULT_SR, mono=True)[0] if clean_path.is_file() else None
    return jsonify(_register_input(wav, f"{meta['title']}.wav", reference=ref, sample=meta))


@app.post("/api/upload")
def upload():
    wav, name = _decode_upload("file")
    return jsonify(_register_input(wav, name))


@app.post("/api/reference")
def upload_reference():
    """Attach a clean reference to an input so the full metric suite can be computed."""
    file_id = _check_id(str(request.form.get("file_id", "")))
    noisy, _ = load_audio(_input_path(file_id), sr=DEFAULT_SR, mono=True)
    ref, _ = _decode_upload("file")
    if abs(ref.size - noisy.size) > 0.25 * DEFAULT_SR:
        raise ApiError(
            f"reference length {ref.size / DEFAULT_SR:.2f}s does not match input "
            f"{noisy.size / DEFAULT_SR:.2f}s — it must be the clean version of the same clip",
            422,
        )
    ref = ref[: noisy.size] if ref.size >= noisy.size else np.pad(ref, (0, noisy.size - ref.size))
    save_audio(_reference_path(file_id), ref, DEFAULT_SR)
    return jsonify({"file_id": file_id, "reference": _reference_info(ref)})


@app.delete("/api/reference/<file_id>")
def delete_reference(file_id: str):
    path = _reference_path(file_id)
    existed = path.is_file()
    path.unlink(missing_ok=True)
    return jsonify({"deleted": existed})


# -- scene -------------------------------------------------------------------


@app.post("/api/classify")
def classify():
    data = _json_body()
    path = _input_path(str(data.get("file_id", "")))
    wav, _ = load_audio(path, sr=DEFAULT_SR, mono=True)
    return jsonify(engine.classify(wav))


# -- enhancement ---------------------------------------------------------------


@app.post("/api/enhance")
def enhance():
    data = _json_body()
    file_id = str(data.get("file_id", ""))
    input_path = _input_path(file_id)
    engine_id = _engine_id(data.get("engine"))

    mode = data.get("mode")
    if mode is not None:
        mode = str(mode).lower()
        if mode not in MODES:
            raise ApiError(f"unknown mode '{mode}'; options: {sorted(MODES)}", 400)
    scene = data.get("scene")
    if scene is not None:
        scene = str(scene).lower()
        if scene not in SCENES:
            raise ApiError(f"unknown scene '{scene}'; options: {sorted(SCENES)}", 400)

    eq = data.get("eq_gains_db")
    if eq is not None:
        if not isinstance(eq, list) or len(eq) > 8:
            raise ApiError("'eq_gains_db' must be a list of up to 8 numbers", 400)
        try:
            eq = [float(g) for g in eq]
        except (TypeError, ValueError):
            raise ApiError("'eq_gains_db' must contain numbers", 400)

    try:
        hum = int(data.get("hum_notch_hz") or 0)
    except (TypeError, ValueError):
        raise ApiError("'hum_notch_hz' must be 0, 50 or 60", 400)
    if hum not in HUM_NOTCH:
        raise ApiError("'hum_notch_hz' must be 0, 50 or 60", 400)

    online = bool(data.get("online_adapt", False))
    if online and not ENGINES[engine_id].online_capable:
        raise ApiError(f"online adaptation is not available for '{ENGINES[engine_id].label}'", 400)

    output_id = uuid.uuid4().hex
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{output_id}.wav"
    residual_path = OUTPUTS_DIR / f"{output_id}_residual.wav"

    try:
        analytics = engine.enhance(
            input_path=input_path,
            output_path=output_path,
            residual_path=residual_path,
            reference=_load_reference(file_id),
            engine_id=engine_id,
            online_adapt=online,
            hum_notch_hz=float(hum) if hum else None,
            mode=mode,
            scene=scene,
            auto_scene=bool(data.get("auto_scene", False)),
            denoise_strength=_opt_float(data, "denoise_strength", 0.0, 1.25),
            safety_floor=_opt_float(data, "safety_floor", 0.0, 0.5),
            eq_gains_db=eq,
            compressor_threshold_db=_opt_float(data, "compressor_threshold_db", -60.0, 0.0),
            compressor_ratio=_opt_float(data, "compressor_ratio", 1.0, 12.0),
            bypass_eq=_opt_bool(data, "bypass_eq"),
            bypass_compressor=_opt_bool(data, "bypass_compressor"),
            use_memory=bool(data.get("use_memory", True)),
            remember=bool(data.get("remember", False)),
            normalize_output=bool(data.get("normalize_output", False)),
        )
    except ApiError:
        raise
    except FileNotFoundError as exc:
        raise ApiError(str(exc), 503)
    except Exception:
        log.exception("enhancement failed")
        output_path.unlink(missing_ok=True)
        residual_path.unlink(missing_ok=True)
        raise ApiError("enhancement failed — see server log for details", 500)

    out_wav, _ = load_audio(output_path, sr=DEFAULT_SR, mono=True)
    res_wav, _ = load_audio(residual_path, sr=DEFAULT_SR, mono=True)
    return jsonify({
        "output_id": output_id,
        "analytics": analytics,
        "output": describe_audio(out_wav, DEFAULT_SR),
        "residual": describe_audio(res_wav, DEFAULT_SR),
    })


@app.post("/api/compare")
def compare():
    data = _json_body()
    _, wav, ref = _analysis_input(data)
    requested = data.get("engines") or [e for e in ENGINES]
    if not isinstance(requested, list) or not requested:
        raise ApiError("'engines' must be a non-empty list", 400)
    engine_ids = [_engine_id(e) for e in requested]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = engine.compare(wav, engine_ids, ref, OUTPUTS_DIR)
    return jsonify({
        "has_reference": ref is not None,
        "duration_s": round(wav.size / DEFAULT_SR, 3),
        "input_est_snr_db": describe_audio(wav, spectrogram=False)["est_snr_db"],
        # the unprocessed clip: the baseline every method is compared against
        "original": {"metrics": reference_metrics(ref, wav, wav), "est_snr_db": estimate_snr_db(wav)},
        "rows": rows,
    })


@app.post("/api/stream")
def stream():
    data = _json_body()
    _, wav, ref = _analysis_input(data)
    try:
        block_ms = int(data.get("block_ms", 64))
    except (TypeError, ValueError):
        raise ApiError("'block_ms' must be an integer", 400)
    if block_ms not in STREAM_BLOCKS:
        raise ApiError(f"'block_ms' must be one of {sorted(STREAM_BLOCKS)}", 400)
    engine_id = _engine_id(data.get("engine"))
    if ENGINES[engine_id].family != "neural":
        raise ApiError("streaming simulation runs the neural engines", 400)
    streamed, report = engine.stream(wav, block_ms, engine_id, ref)
    output_id = uuid.uuid4().hex
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_audio(OUTPUTS_DIR / f"{output_id}.wav", streamed, DEFAULT_SR)
    report["output_id"] = output_id
    report["waveform"] = waveform_summary(streamed)
    return jsonify(report)


# -- audio streaming / download ------------------------------------------------


@app.get("/api/audio/input/<file_id>")
def audio_input(file_id: str):
    return send_file(_input_path(file_id), mimetype="audio/wav", conditional=True)


@app.get("/api/audio/reference/<file_id>")
def audio_reference(file_id: str):
    path = _reference_path(file_id)
    if not path.is_file():
        raise ApiError("no clean reference for this input", 404)
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.get("/api/audio/output/<output_id>")
def audio_output(output_id: str):
    return send_file(_output_path(output_id), mimetype="audio/wav", conditional=True)


@app.get("/api/audio/residual/<output_id>")
def audio_residual(output_id: str):
    path = OUTPUTS_DIR / f"{_check_id(output_id)}_residual.wav"
    if not path.is_file():
        raise ApiError("residual audio not found", 404)
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.get("/api/download/<output_id>")
def download(output_id: str):
    return send_file(
        _output_path(output_id),
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"sona_enhanced_{output_id[:8]}.wav",
        conditional=True,
    )


# -- preference memory ---------------------------------------------------------


@app.get("/api/preferences")
def preferences_all():
    return jsonify({"path": str(PREFERENCES_PATH), "preferences": engine.memory.to_dict()})


@app.post("/api/preferences")
def preferences_set():
    data = _json_body()
    scene = str(data.get("scene", "")).lower()
    if scene not in SCENES:
        raise ApiError(f"'scene' must be one of {sorted(SCENES)}", 400)
    mode = str(data.get("mode", "")).lower()
    if mode not in MODES:
        raise ApiError(f"'mode' must be one of {sorted(MODES)}", 400)
    eq = data.get("eq_gains_db")
    if eq is not None and (not isinstance(eq, list) or len(eq) > 8):
        raise ApiError("'eq_gains_db' must be a list of up to 8 numbers", 400)
    settings, _, _ = engine.resolve_settings(
        mode=mode,
        scene=scene,
        denoise_strength=_opt_float(data, "denoise_strength", 0.0, 1.25),
        safety_floor=_opt_float(data, "safety_floor", 0.0, 0.5),
        eq_gains_db=[float(g) for g in eq] if eq else None,
        compressor_threshold_db=_opt_float(data, "compressor_threshold_db", -60.0, 0.0),
        compressor_ratio=_opt_float(data, "compressor_ratio", 1.0, 12.0),
        bypass_eq=_opt_bool(data, "bypass_eq"),
        bypass_compressor=_opt_bool(data, "bypass_compressor"),
        use_memory=False,
    )
    record = engine.memory.set(scene, mode, settings)
    return jsonify({"saved": True, "scene": record.scene, "mode": record.mode, "updated_at": record.updated_at})


@app.delete("/api/preferences/<scene>")
def preferences_delete(scene: str):
    scene = scene.lower()
    if scene not in SCENES:
        raise ApiError(f"unknown scene '{scene}'", 400)
    return jsonify({"deleted": engine.memory.delete(scene), "scene": scene})


@app.delete("/api/preferences")
def preferences_clear():
    engine.memory.clear()
    return jsonify({"cleared": True})


if __name__ == "__main__":
    import sys
    import threading
    import webbrowser

    engine.load_async()  # warm the GTCRN checkpoint while the loading screen plays
    url = f"http://localhost:{PORT}"
    print("\n  SONA web demo")
    print(f"  ->  {url}")
    print("  (keep this window open; press Ctrl+C to stop)\n")
    if "--open" in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
