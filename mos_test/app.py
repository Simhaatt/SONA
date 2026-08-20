"""Standalone local Flask app for the in-house MOS listening test.

Serves the clips selected by ``scripts/select_mos_clips.py`` in a randomized order per rater,
collects ACR 1-5 ratings, and appends them to ``responses.csv``. Condition labels are never sent
to the browser -- only opaque clip_ids -- so raters are blind to which method produced each clip.

Run:
    cd "E:\\Rohit\\FYP\\noise cancellation\\mos_test"
    python app.py
then share http://<your-lan-ip>:7861 with raters on the same network, or run locally per-rater.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from flask import Flask, jsonify, request, send_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("mos_test")

HOST, PORT = "0.0.0.0", 7861
HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio"
MANIFEST_PATH = HERE / "clip_manifest.csv"
RESPONSES_PATH = HERE / "responses.csv"
STATIC_DIR = HERE / "static"

app = Flask(__name__, static_folder=None)


def _load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@app.get("/")
def index():
    return send_file(STATIC_DIR / "index.html")


@app.get("/api/clips")
def api_clips():
    rows = _load_manifest()
    # Only expose clip_id to the client -- condition/uid stay server-side until analysis.
    return jsonify([{"clip_id": r["clip_id"]} for r in rows])


@app.get("/api/audio/<clip_id>")
def api_audio(clip_id: str):
    safe = "".join(c for c in clip_id if c.isalnum())
    path = AUDIO_DIR / f"{safe}.wav"
    if not path.is_file():
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="audio/wav")


@app.post("/api/submit")
def api_submit():
    payload = request.get_json(force=True)
    rater_id = str(payload.get("rater_id", "")).strip()[:80]
    ratings = payload.get("ratings", [])
    if not rater_id or not isinstance(ratings, list) or not ratings:
        return jsonify({"error": "missing rater_id or ratings"}), 400

    manifest_by_id = {r["clip_id"]: r for r in _load_manifest()}
    is_new = not RESPONSES_PATH.exists()
    with open(RESPONSES_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh, fieldnames=["rater_id", "clip_id", "uid", "condition", "is_attention_check", "rating"]
        )
        if is_new:
            w.writeheader()
        for entry in ratings:
            clip_id = str(entry.get("clip_id", ""))
            rating = entry.get("rating")
            row = manifest_by_id.get(clip_id)
            if row is None or not isinstance(rating, int) or not (1 <= rating <= 5):
                continue
            w.writerow(
                {
                    "rater_id": rater_id,
                    "clip_id": clip_id,
                    "uid": row["uid"],
                    "condition": row["condition"],
                    "is_attention_check": row["is_attention_check"],
                    "rating": rating,
                }
            )
    log.info("recorded %d ratings from rater=%s", len(ratings), rater_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"{MANIFEST_PATH} not found -- run scripts/select_mos_clips.py first")
    app.run(host=HOST, port=PORT, debug=False)
