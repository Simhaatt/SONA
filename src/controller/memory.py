"""Persistent preference memory for SONA Layer B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .modes import ControllerSettings


DEFAULT_MEMORY_PATH = Path.home() / ".sona" / "preferences.json"


@dataclass(frozen=True)
class PreferenceRecord:
    """One remembered user preference for a detected scene."""

    scene: str
    mode: str
    settings: ControllerSettings
    updated_at: str


class PreferenceMemory:
    """Small JSON-backed store for scene-specific mode/control choices."""

    def __init__(self, path: str | Path = DEFAULT_MEMORY_PATH) -> None:
        self.path = Path(path)

    def get(self, scene: str | None) -> PreferenceRecord | None:
        if not scene:
            return None
        data = self._read()
        raw = data.get(_key(scene))
        return _record_from_dict(raw) if raw else None

    def set(self, scene: str, mode: str, settings: ControllerSettings) -> PreferenceRecord:
        scene_key = _key(scene)
        record = PreferenceRecord(
            scene=scene_key,
            mode=mode.lower(),
            settings=settings,
            updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        data = self._read()
        data[scene_key] = _record_to_dict(record)
        self._write(data)
        return record

    def delete(self, scene: str) -> bool:
        data = self._read()
        existed = data.pop(_key(scene), None) is not None
        if existed:
            self._write(data)
        return existed

    def clear(self) -> None:
        self._write({})

    def all(self) -> dict[str, PreferenceRecord]:
        return {scene: _record_from_dict(raw) for scene, raw in self._read().items()}

    def to_dict(self) -> dict[str, Any]:
        return self._read()

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"preference memory must be a JSON object: {self.path}")
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(self.path)


def _key(scene: str) -> str:
    return scene.strip().lower() or "unknown"


def _record_to_dict(record: PreferenceRecord) -> dict[str, Any]:
    return {
        "scene": record.scene,
        "mode": record.mode,
        "settings": asdict(record.settings),
        "updated_at": record.updated_at,
    }


def _record_from_dict(raw: dict[str, Any]) -> PreferenceRecord:
    settings_raw = raw["settings"]
    return PreferenceRecord(
        scene=str(raw["scene"]),
        mode=str(raw["mode"]),
        settings=ControllerSettings(
            mask_strength=float(settings_raw["mask_strength"]),
            safety_floor=float(settings_raw["safety_floor"]),
            eq_gains_db=tuple(float(x) for x in settings_raw["eq_gains_db"]),
            compressor_threshold_db=float(settings_raw["compressor_threshold_db"]),
            compressor_ratio=float(settings_raw["compressor_ratio"]),
            compressor_makeup_db=float(settings_raw.get("compressor_makeup_db", 0.0)),
            limiter_ceiling=float(settings_raw.get("limiter_ceiling", 0.98)),
        ),
        updated_at=str(raw["updated_at"]),
    )
