"""Full speech-enhancement metric suite with graceful optional-dependency handling.

Every method is scored by the *same* function so the comparison table is apples-to-apples.

Metrics
-------
Reference-based (need clean signal):
  * si_sdr, si_sdr_i   -- scale-invariant SDR and its improvement over the noisy input
  * snr, snr_i         -- classic SNR and improvement
  * lsd                -- log-spectral distance (lower is better)
  * stoi, estoi        -- intelligibility (pystoi)
  * pesq               -- perceptual quality (needs the `pesq` package; skipped if absent)
Reference-free:
  * dnsmos             -- DNSMOS perceptual MOS (needs an ONNX model; skipped if absent)
  * wer                -- word error rate via ASR (needs an ASR backend + transcript; skipped)

Missing optional backends are reported as ``None`` (not an error) so the harness still runs
locally on Python 3.14 where e.g. `pesq` has no wheel. The full suite populates on Kaggle.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

EPS = 1e-8


# ----------------------------------------------------------------------- helpers
def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    return a[:n], b[:n]


# ------------------------------------------------------------- reference metrics
def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB (Le Roux et al., 2019)."""
    ref, est = _align(reference, estimate)
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + EPS)
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + EPS) / (np.dot(noise, noise) + EPS)))


def snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref, est = _align(reference, estimate)
    noise = est - ref
    return float(10.0 * np.log10((np.dot(ref, ref) + EPS) / (np.dot(noise, noise) + EPS)))


def lsd(reference: np.ndarray, estimate: np.ndarray, n_fft: int = 512, hop: int = 128) -> float:
    """Log-spectral distance (dB), averaged over frames. Lower is better.

    Power spectra are floored relative to their own peak (-80 dB) rather than at an absolute
    epsilon, so digital-silence regions don't dominate the score.
    """
    from src.dsp import STFT

    st = STFT(n_fft=n_fft, hop=hop)
    ref, est = _align(reference, estimate)
    rp = np.abs(st.stft(ref)) ** 2
    ep = np.abs(st.stft(est)) ** 2
    rp = np.maximum(rp, rp.max() * 1e-8)
    ep = np.maximum(ep, ep.max() * 1e-8)
    r = 10.0 * np.log10(rp)
    e = 10.0 * np.log10(ep)
    n = min(r.shape[1], e.shape[1])
    diff = r[:, :n] - e[:, :n]
    return float(np.mean(np.sqrt(np.mean(diff**2, axis=0))))


def stoi_metric(reference: np.ndarray, estimate: np.ndarray, sr: int, extended: bool = False) -> float:
    from pystoi import stoi as _stoi

    ref, est = _align(reference, estimate)
    return float(_stoi(ref, est, sr, extended=extended))


def pesq_metric(reference: np.ndarray, estimate: np.ndarray, sr: int) -> float:
    from pesq import pesq as _pesq

    ref, est = _align(reference, estimate)
    mode = "wb" if sr >= 16000 else "nb"
    return float(_pesq(sr, ref, est, mode))


def dnsmos_metric(estimate: np.ndarray, sr: int) -> float:
    """DNSMOS overall MOS. Requires a DNSMOS ONNX model on disk; see eval/dnsmos.py (optional)."""
    from .dnsmos import score as _dnsmos  # local optional module

    return float(_dnsmos(estimate, sr))


def wer_metric(reference_text: str, estimate: np.ndarray, sr: int) -> float:
    from .asr import transcribe, word_error_rate  # local optional module

    hyp = transcribe(estimate, sr)
    return float(word_error_rate(reference_text, hyp))


# --------------------------------------------------------------------- registry
# Each entry: name -> (callable(sample_dict) -> float, needs_reference)
def _try(fn: Callable[[], float]) -> float | None:
    try:
        return fn()
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 - a bad clip shouldn't kill the whole run
        warnings.warn(f"metric failed: {exc}")
        return None


METRIC_REGISTRY: dict[str, str] = {
    "si_sdr": "reference",
    "si_sdr_i": "reference",
    "snr": "reference",
    "snr_i": "reference",
    "lsd": "reference",
    "stoi": "reference",
    "estoi": "reference",
    "pesq": "reference",
    "dnsmos": "reference_free",
    "wer": "text",
}


def available_metrics() -> dict[str, bool]:
    """Report which optional backends are importable in this environment."""
    checks = {
        "pystoi": "pystoi",
        "pesq": "pesq",
        "dnsmos(onnxruntime)": "onnxruntime",
    }
    out = {}
    for label, mod in checks.items():
        try:
            __import__(mod)
            out[label] = True
        except ImportError:
            out[label] = False
    return out


def compute_metrics(
    clean: np.ndarray | None,
    enhanced: np.ndarray,
    noisy: np.ndarray | None = None,
    sr: int = 16000,
    reference_text: str | None = None,
    include: list[str] | None = None,
) -> dict[str, float | None]:
    """Score one enhanced clip. Returns a dict; unavailable metrics map to ``None``.

    `noisy` enables the *_i (improvement) metrics: improvement = score(enhanced) - score(noisy).
    """
    want = set(include) if include else set(METRIC_REGISTRY)
    out: dict[str, float | None] = {}

    if clean is not None:
        # Improvement metrics implicitly need their base score, even if only the *_i was requested.
        need_si = want & {"si_sdr", "si_sdr_i"}
        need_snr = want & {"snr", "snr_i"}

        si_val = _try(lambda: si_sdr(clean, enhanced)) if need_si else None
        snr_val = _try(lambda: snr(clean, enhanced)) if need_snr else None
        if "si_sdr" in want:
            out["si_sdr"] = si_val
        if "snr" in want:
            out["snr"] = snr_val
        if "lsd" in want:
            out["lsd"] = _try(lambda: lsd(clean, enhanced))
        if "stoi" in want:
            out["stoi"] = _try(lambda: stoi_metric(clean, enhanced, sr, extended=False))
        if "estoi" in want:
            out["estoi"] = _try(lambda: stoi_metric(clean, enhanced, sr, extended=True))
        if "pesq" in want:
            out["pesq"] = _try(lambda: pesq_metric(clean, enhanced, sr))

        if noisy is not None:
            if "si_sdr_i" in want:
                base = _try(lambda: si_sdr(clean, noisy))
                out["si_sdr_i"] = None if (si_val is None or base is None) else si_val - base
            if "snr_i" in want:
                base = _try(lambda: snr(clean, noisy))
                out["snr_i"] = None if (snr_val is None or base is None) else snr_val - base

    if "dnsmos" in want:
        out["dnsmos"] = _try(lambda: dnsmos_metric(enhanced, sr))
    if "wer" in want and reference_text is not None:
        out["wer"] = _try(lambda: wer_metric(reference_text, enhanced, sr))

    return out
