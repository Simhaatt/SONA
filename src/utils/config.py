"""Lightweight YAML config with dotted access and deep-merge overrides.

Configs are the single source of truth for every experiment (STFT params, band count,
method, training hyperparameters). Keeping them declarative makes runs reproducible and
makes ablations a matter of swapping/overriding a few keys.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml


class Config(dict):
    """A dict that also supports attribute access and nested `get('a.b.c')` lookups."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
            self[key] = val
        return val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Fetch a nested value via 'a.b.c'. Returns `default` if any level is missing."""
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> dict:
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, Mapping):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path, overrides: Mapping | None = None) -> Config:
    """Load a YAML config.

    Supports a top-level ``defaults: <relative_path>`` key so method configs can inherit
    from ``configs/default.yaml`` and only specify their differences.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    base: dict = {}
    parent = raw.pop("defaults", None)
    if parent is not None:
        parent_path = (path.parent / parent).resolve()
        base = load_config(parent_path).to_dict()

    merged = _deep_merge(base, raw)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config(merged)
