"""ASR backend for WER: transcribe + word-error-rate against a pseudo-reference.

VoiceBank-DEMAND (built on VCTK) does not ship ground-truth transcripts alongside the
clean/noisy wav pairs used by this benchmark. We therefore use a *differential WER*: transcribe
the **clean** reference once with an ASR model to get a pseudo-ground-truth transcript, then
transcribe each method's output and measure WER against that same pseudo-reference. This isolates
what enhancement does to downstream ASR accuracy without needing external transcript files, and
is the standard workaround when a corpus doesn't distribute text.

Backend is `faster-whisper` (CTranslate2, GPU-capable). Import is lazy and optional, matching the
graceful-degradation pattern used throughout `src/eval`: if the package isn't installed, callers
get an `ImportError` that `metrics._try` already catches and turns into `None`.
"""

from __future__ import annotations

import os

import numpy as np

_MODEL = None
_MODEL_SIZE = None
_DLL_DIRS_ADDED = False


def _add_pip_cuda_dll_dirs() -> None:
    """On Windows, CTranslate2's GPU backend needs cublas/cudnn DLLs on the search path.

    ``pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`` puts them under
    ``site-packages/nvidia/*/bin`` rather than on PATH, so CTranslate2 can't find them unless we
    register the directory with ``os.add_dll_directory`` first. No-op elsewhere / if absent.
    """
    global _DLL_DIRS_ADDED
    if _DLL_DIRS_ADDED or not hasattr(os, "add_dll_directory"):
        return
    _DLL_DIRS_ADDED = True
    try:
        import nvidia.cublas as cublas
        import nvidia.cudnn as cudnn

        for pkg in (cublas, cudnn):
            pkg_dir = next(iter(pkg.__path__), None)  # namespace package -> no __file__
            if not pkg_dir:
                continue
            bin_dir = os.path.join(pkg_dir, "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
    except ImportError:
        pass


def _get_model(model_size: str | None = None):
    global _MODEL, _MODEL_SIZE
    model_size = model_size or os.environ.get("SONA_WHISPER_MODEL", "small.en")
    if _MODEL is not None and _MODEL_SIZE == model_size:
        return _MODEL

    _add_pip_cuda_dll_dirs()
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        _MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    except RuntimeError:
        # GPU CUDA/cuDNN runtime not loadable (missing system libs) -> fall back to CPU.
        _MODEL = WhisperModel(model_size, device="cpu", compute_type="int8")
    _MODEL_SIZE = model_size
    return _MODEL


def _to_whisper_audio(wav: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz mono float32, the sample rate faster-whisper expects for ndarray input."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    target_sr = 16000
    if sr != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return wav


def transcribe(wav: np.ndarray, sr: int, model_size: str | None = None) -> str:
    """Transcribe one clip. Returns the concatenated, stripped segment text."""
    model = _get_model(model_size)
    audio = _to_whisper_audio(wav, sr)
    segments, _info = model.transcribe(audio, language="en", vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return " ".join(text.split())  # collapse whitespace


def word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """WER between two transcripts, case/punctuation-insensitive (jiwer's standard transform)."""
    import jiwer

    if not reference_text.strip():
        return float("nan")
    return float(
        jiwer.wer(
            reference_text,
            hypothesis_text,
            reference_transform=jiwer.wer_standardize,
            hypothesis_transform=jiwer.wer_standardize,
        )
    )
