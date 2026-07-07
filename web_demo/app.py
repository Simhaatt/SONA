"""SONA web demo — local Flask backend + static frontend.

Run:
    cd "D:\\noise cancellation\\web_demo"
    python app.py
then open http://localhost:7860
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from engine import (
    DEFAULT_SR,
    PREFERENCES_PATH,
    REPO,
    WEB_DEMO,
    SonaEngine,
    describe_audio,
)
from src.controller import MODE_PRESETS
from src.utils.audio_io import load_audio, save_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("sona.web")

HOST, PORT = "127.0.0.1", 7860
UPLOADS_DIR = REPO / "results" / "web_demo_uploads"
OUTPUTS_DIR = REPO / "results" / "web_demo_outputs"
SAMPLE_PATH = WEB_DEMO / "samples" / "sample_noisy.wav"
STATIC_DIR = WEB_DEMO / "static"

ALLOWED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}
MAX_UPLOAD_MB = 60
MAX_DURATION_S = 300.0
ID_RE = re.compile(r"^[0-9a-f]{32}$")
SCENES = {"speech", "music", "traffic", "fan", "crowd", "mixed", "unknown"}
MODES = set(MODE_PRESETS)  # follows the controller package, incl. new presets

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


def _check_id(file_id: str) -> str:
    if not isinstance(file_id, str) or not ID_RE.match(file_id):
        raise ApiError("invalid file id", 400)
    return file_id


def _input_path(file_id: str) -> Path:
    path = UPLOADS_DIR / f"{_check_id(file_id)}.wav"
    if not path.is_file():
        raise ApiError("input audio not found; upload or select the sample first", 404)
    return path


def _output_path(output_id: str) -> Path:
    path = OUTPUTS_DIR / f"{_check_id(output_id)}.wav"
    if not path.is_file():
        raise ApiError("output audio not found", 404)
    return path


def _register_input(wav: np.ndarray, name: str) -> dict:
    file_id = uuid.uuid4().hex
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_audio(UPLOADS_DIR / f"{file_id}.wav", wav, DEFAULT_SR)
    _names[file_id] = name
    info = describe_audio(wav, DEFAULT_SR)
    info.update({"file_id": file_id, "name": name})
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
    status.update(
        {
            "ok": True,
            "sample_available": SAMPLE_PATH.is_file(),
            "outputs_dir": str(OUTPUTS_DIR),
            "preferences_path": str(PREFERENCES_PATH),
        }
    )
    return jsonify(status)


@app.get("/api/modes")
def modes():
    return jsonify(engine.presets())


# -- audio input ---------------------------------------------------------------


@app.post("/api/sample")
def use_sample():
    if not SAMPLE_PATH.is_file():
        raise ApiError("bundled sample missing; run tools/make_sample.py", 500)
    wav, _ = load_audio(SAMPLE_PATH, sr=DEFAULT_SR, mono=True)
    return jsonify(_register_input(wav, "sample_noisy.wav"))


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if file is None or not file.filename:
        raise ApiError("no file provided (multipart field 'file')", 400)
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ApiError(
            f"unsupported file type '{ext or 'none'}'; allowed: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS)),
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
    return jsonify(_register_input(wav, safe_name))


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
    input_path = _input_path(str(data.get("file_id", "")))

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

    output_id = uuid.uuid4().hex
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / f"{output_id}.wav"
    residual_path = OUTPUTS_DIR / f"{output_id}_residual.wav"

    if not engine.checkpoint_path.exists():
        raise ApiError(f"model checkpoint missing: {engine.checkpoint_path}", 503)
    try:
        analytics = engine.enhance(
            input_path=input_path,
            output_path=output_path,
            residual_path=residual_path,
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
    return jsonify(
        {
            "output_id": output_id,
            "analytics": analytics,
            "output": describe_audio(out_wav, DEFAULT_SR),
            "residual": describe_audio(res_wav, DEFAULT_SR),
        }
    )


# -- audio streaming / download ------------------------------------------------


@app.get("/api/audio/input/<file_id>")
def audio_input(file_id: str):
    return send_file(_input_path(file_id), mimetype="audio/wav", conditional=True)


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
    engine.load_async()  # warm the GTCRN checkpoint while the loading screen plays
    url = f"http://{HOST}:{PORT}"
    print("\n  SONA web demo")
    print(f"  ->  {url}\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
