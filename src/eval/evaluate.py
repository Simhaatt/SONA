"""Run an `Enhancer` over a dataset and collect per-sample metric rows.

Method-agnostic: anything implementing `Enhancer.enhance` is scored identically, which is what
lets the benchmark table put classical baselines and learned models side by side.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np

from ..data.dataset import Sample
from ..frontend import Enhancer
from ..utils.audio_io import save_audio
from .metrics import compute_metrics


def evaluate_method(
    enhancer: Enhancer,
    dataset: Iterable[Sample],
    include_metrics: list[str] | None = None,
    save_audio_dir: str | Path | None = None,
    method_name: str | None = None,
) -> list[dict]:
    """Return one row per sample: identity + metrics + real-time factor (RTF)."""
    name = method_name or getattr(enhancer, "name", enhancer.__class__.__name__)
    rows: list[dict] = []
    save_dir = Path(save_audio_dir) if save_audio_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for s in dataset:
        t0 = time.perf_counter()
        enhanced = enhancer.enhance(s.noisy, s.sr)
        elapsed = time.perf_counter() - t0
        audio_dur = len(s.noisy) / s.sr
        rtf = elapsed / audio_dur if audio_dur > 0 else float("nan")

        m = compute_metrics(
            clean=s.clean,
            enhanced=enhanced,
            noisy=s.noisy,
            sr=s.sr,
            reference_text=s.text,
            include=include_metrics,
        )
        row = {
            "method": name,
            "uid": s.uid,
            "noise_type": s.noise_type,
            "snr_db": s.snr_db,
            "rtf": rtf,
            **m,
        }
        rows.append(row)
        if save_dir:
            save_audio(save_dir / f"{s.uid}__{name}.wav", enhanced, s.sr)
    return rows


def _numeric_mean(values: list) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float)) and v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def aggregate(rows: list[dict], group_keys: tuple[str, ...] = ("method",)) -> list[dict]:
    """Group rows and average numeric metric columns within each group."""
    skip = set(group_keys) | {"uid", "noise_type", "snr_db"}
    metric_cols = [
        k for k in (rows[0].keys() if rows else []) if k not in skip and k not in ("method",)
    ]

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r.get(k) for k in group_keys)
        groups.setdefault(key, []).append(r)

    out = []
    for key, grp in groups.items():
        summary = dict(zip(group_keys, key))
        summary["n"] = len(grp)
        for col in metric_cols:
            summary[col] = _numeric_mean([g.get(col) for g in grp])
        out.append(summary)
    return out
