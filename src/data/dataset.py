"""Datasets that yield clean/noisy pairs.

`PairDataset`      -- real corpora (VoiceBank-DEMAND layout, or a CSV manifest of clean,noisy paths).
`SyntheticNoisyDataset` -- self-contained generator (tones + colored noise + optional bursts) so
                     the full pipeline and tests run with zero downloads.

A `Sample` is the unit the eval harness and trainers consume.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils.audio_io import DEFAULT_SR, load_audio
from .noise_bank import add_burst_noise, mix_at_snr


@dataclass
class Sample:
    clean: np.ndarray | None
    noisy: np.ndarray
    sr: int = DEFAULT_SR
    uid: str = ""
    noise_type: str = "unknown"
    snr_db: float | None = None
    text: str | None = None
    meta: dict = field(default_factory=dict)


class PairDataset:
    """Clean/noisy pairs from disk.

    Two layouts supported:
      * VoiceBank-DEMAND: ``clean_dir`` and ``noisy_dir`` with matching filenames.
      * Manifest CSV with columns ``clean,noisy[,text,noise_type,snr_db]``.
    """

    def __init__(
        self,
        clean_dir: str | Path | None = None,
        noisy_dir: str | Path | None = None,
        manifest: str | Path | None = None,
        sr: int = DEFAULT_SR,
        limit: int | None = None,
    ) -> None:
        self.sr = sr
        self.limit = limit
        self._rows: list[dict] = []

        if manifest is not None:
            with open(manifest, newline="", encoding="utf-8") as fh:
                self._rows = list(csv.DictReader(fh))
        elif clean_dir is not None and noisy_dir is not None:
            clean_dir, noisy_dir = Path(clean_dir), Path(noisy_dir)
            for noisy_path in sorted(noisy_dir.rglob("*.wav")):
                clean_path = clean_dir / noisy_path.name
                if clean_path.exists():
                    self._rows.append({"clean": str(clean_path), "noisy": str(noisy_path)})
        else:
            raise ValueError("provide either `manifest` or both `clean_dir` and `noisy_dir`")

        if limit:
            self._rows = self._rows[:limit]

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Sample]:
        for row in self._rows:
            clean, _ = load_audio(row["clean"], sr=self.sr) if row.get("clean") else (None, self.sr)
            noisy, _ = load_audio(row["noisy"], sr=self.sr)
            yield Sample(
                clean=clean,
                noisy=noisy,
                sr=self.sr,
                uid=Path(row["noisy"]).stem,
                noise_type=row.get("noise_type", "unknown"),
                snr_db=float(row["snr_db"]) if row.get("snr_db") else None,
                text=row.get("text"),
            )


class SyntheticNoisyDataset:
    """Generates reproducible clean/noisy speech-like pairs with no external data.

    Clean = sum of a few harmonically-related sinusoids with an amplitude envelope (a crude but
    band-limited speech surrogate). Noise = colored Gaussian, optionally with impulsive bursts to
    exercise the non-stationary / burst-noise path.
    """

    NOISE_TYPES = ("white", "pink", "burst")

    def __init__(
        self,
        n: int = 8,
        sr: int = DEFAULT_SR,
        duration: float = 2.0,
        snr_db_range: tuple[float, float] = (0.0, 10.0),
        seed: int = 0,
        noise_types: tuple[str, ...] | None = None,
    ) -> None:
        self.n = n
        self.sr = sr
        self.duration = duration
        self.snr_db_range = snr_db_range
        self.seed = seed
        # Restrict which noise types appear -> lets us build train/test splits with *disjoint*
        # noise types for the distribution-shift experiment (e.g. train {white,pink}, test {burst}).
        self.noise_types = tuple(noise_types) if noise_types else self.NOISE_TYPES

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self.n):
            rng = np.random.default_rng(self.seed * 1000 + i)
            length = int(self.sr * self.duration)
            clean = self._make_clean(length, rng)
            noise_type = self.noise_types[i % len(self.noise_types)]
            noise = self._make_noise(length, noise_type, rng)
            snr = float(rng.uniform(*self.snr_db_range))
            noisy, _ = mix_at_snr(clean, noise, snr, rng)
            if noise_type == "burst":
                noisy = add_burst_noise(noisy, sr=self.sr, rng=rng)
            yield Sample(
                clean=clean,
                noisy=noisy.astype(np.float32),
                sr=self.sr,
                uid=f"synth_{i:03d}",
                noise_type=noise_type,
                snr_db=snr,
            )

    def _make_clean(self, length: int, rng: np.random.Generator) -> np.ndarray:
        t = np.arange(length) / self.sr
        f0 = rng.uniform(120, 240)
        sig = np.zeros(length)
        for h in range(1, 6):
            sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 2 * np.pi))

        # Syllable envelope with *real silence gaps* (leading silence + voiced/unvoiced segments),
        # so noise-floor estimators have noise-only regions to latch onto -- like real speech.
        env = self._syllable_envelope(length, rng)
        sig *= env
        sig /= np.max(np.abs(sig)) + 1e-8
        return (0.7 * sig).astype(np.float32)

    def _syllable_envelope(self, length: int, rng: np.random.Generator) -> np.ndarray:
        env = np.zeros(length, dtype=np.float64)
        lead = int(0.25 * self.sr)  # leading silence
        pos = lead
        while pos < length:
            voiced = rng.random() < 0.6
            seg = int(self.sr * rng.uniform(0.15, 0.45))
            end = min(pos + seg, length)
            if voiced:
                w = np.hanning(2 * (end - pos)) if end > pos else np.array([])
                env[pos:end] = w[: end - pos] if len(w) else 0.0
            pos = end + int(self.sr * rng.uniform(0.05, 0.2))  # silence gap after segment
        return env

    def _make_noise(self, length: int, kind: str, rng: np.random.Generator) -> np.ndarray:
        white = rng.standard_normal(length).astype(np.float32)
        if kind == "white" or kind == "burst":
            return white
        if kind == "pink":  # 1/f-ish via cumulative filtering
            b = np.array([0.049922, -0.095993, 0.050612, -0.004408])
            a = np.array([1.0, -2.494956, 2.017265, -0.522189])
            from scipy.signal import lfilter

            return lfilter(b, a, white).astype(np.float32)
        return white
