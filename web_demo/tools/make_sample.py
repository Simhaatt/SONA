"""Create the bundled demo sample as genuinely noisy speech.

Priority:
1. A real pre-mixed MS-SNSD noisy test clip (fan/traffic-style noise, 0 dB SNR
   preferred) from `external_data/ms_snsd_sona/test.csv`.
2. Mix MS-SNSD clean speech with fan/traffic noise at 3 dB SNR.
3. Synthetic speech-like signal + pink noise (keeps the demo self-contained).

Usage:  python tools/make_sample.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

WEB_DEMO = Path(__file__).resolve().parents[1]
REPO = WEB_DEMO.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.audio_io import load_audio, save_audio  # noqa: E402

SR = 16000
TARGET_SECONDS = 8.0
MIX_SNR_DB = 3.0
OUT_PATH = WEB_DEMO / "samples" / "sample_noisy.wav"

MANIFEST = REPO / "external_data" / "ms_snsd_sona" / "test.csv"
CLEAN_DIR = REPO / "external_data" / "MS-SNSD" / "clean_test"
NOISE_DIR = REPO / "external_data" / "MS-SNSD" / "noise_test"
# stationary/background noises make the enhancement obvious to see and hear
PREFERRED_NOISE_TYPES = ("AirConditioner", "Traffic", "VacuumCleaner", "Babble")
PREFERRED_NOISE_FILES = ("AirConditioner_1.wav", "Traffic_1.wav", "Babble_1.wav")


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    if x.size >= n:
        return x[:n]
    reps = int(np.ceil(n / x.size))
    return np.tile(x, reps)[:n]


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = np.sqrt(np.mean(clean**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    noisy = clean + noise * (target_noise_rms / noise_rms)
    peak = float(np.max(np.abs(noisy)))
    if peak > 0.95:
        noisy = noisy * (0.95 / peak)
    return noisy.astype(np.float32)


def _from_manifest(n: int) -> np.ndarray | None:
    """Pick a real pre-mixed noisy MS-SNSD clip: preferred noise type, lowest SNR."""
    if not MANIFEST.is_file():
        return None
    with MANIFEST.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("noisy")]
    rows = [r for r in rows if r.get("noise_type") in PREFERRED_NOISE_TYPES]
    rows.sort(
        key=lambda r: (
            abs(float(r.get("snr_db", 99))),  # 0 dB first, then 5 / -5 ...
            PREFERRED_NOISE_TYPES.index(r["noise_type"]),
        )
    )
    for row in rows[:60]:
        path = Path(row["noisy"])
        if not path.is_file():
            continue
        wav, _ = load_audio(path, sr=SR, mono=True)
        if wav.size >= n * 0.75 and float(np.sqrt(np.mean(wav**2))) > 0.005:
            print(f"real noisy clip: {path.name} ({row['noise_type']} @ {row['snr_db']} dB SNR)")
            return _fit_length(wav, min(n, wav.size))
    return None


def _pick_clean(n: int) -> np.ndarray | None:
    if not CLEAN_DIR.is_dir():
        return None
    for path in sorted(CLEAN_DIR.glob("clnsp*.wav"))[:200]:
        wav, _ = load_audio(path, sr=SR, mono=True)
        if wav.size >= n and float(np.sqrt(np.mean(wav**2))) > 0.01:
            print(f"clean: {path.name}")
            return wav[:n]
    return None


def _pick_noise(n: int) -> np.ndarray | None:
    if not NOISE_DIR.is_dir():
        return None
    candidates = [NOISE_DIR / f for f in PREFERRED_NOISE_FILES if (NOISE_DIR / f).exists()]
    candidates += sorted(NOISE_DIR.glob("*.wav"))
    for path in candidates:
        wav, _ = load_audio(path, sr=SR, mono=True)
        if wav.size > SR and float(np.sqrt(np.mean(wav**2))) > 1e-4:
            print(f"noise: {path.name}")
            return _fit_length(wav, n)
    return None


def _synthetic_speechlike(n: int, rng: np.random.Generator) -> np.ndarray:
    """Vowel-ish harmonic bursts with syllable-rate amplitude modulation."""
    t = np.arange(n) / SR
    f0 = 120.0 + 30.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 9))
    syllables = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 3.0 * t + rng.uniform(0, np.pi))))
    envelope = np.convolve(syllables, np.hanning(1200) / 600.0, mode="same")
    return (voiced * envelope * 0.25).astype(np.float32)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    spectrum /= np.maximum(np.sqrt(freqs), 1.0)
    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return pink / (np.max(np.abs(pink)) + 1e-9)


def main() -> None:
    n = int(TARGET_SECONDS * SR)
    noisy = _from_manifest(n)
    if noisy is None:
        clean, noise = _pick_clean(n), _pick_noise(n)
        if clean is not None and noise is not None:
            print(f"mixing clean + noise at {MIX_SNR_DB} dB SNR")
            noisy = _mix_at_snr(clean, noise, MIX_SNR_DB)
    if noisy is None:
        print("MS-SNSD not found; generating synthetic sample")
        rng = np.random.default_rng(2026)
        noisy = _mix_at_snr(_synthetic_speechlike(n, rng), _pink_noise(n, rng), MIX_SNR_DB)
    save_audio(OUT_PATH, noisy, SR)
    print(f"wrote {OUT_PATH} ({noisy.size / SR:.1f}s @ {SR} Hz)")


if __name__ == "__main__":
    main()
