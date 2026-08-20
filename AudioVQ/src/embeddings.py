"""Audio embedding extraction.

Embeddings come from a pretrained self-supervised speech model (WavLM-Base+ by default) with
mean-pooling over time, which is the standard way to turn a frame-level representation into a
single utterance vector for retrieval.

Design notes that matter for the experiments downstream:
  * The embedding model is FROZEN and never fine-tuned -- the study is about quantizing a given
    embedding space, not about learning a better one.
  * Extraction is cached to .npy so quantizer experiments (which are re-run many times) never pay
    the extraction cost twice, and so every method is compared on byte-identical vectors.
  * Metadata (speaker, noise type, SNR) travels with the vectors in a parallel CSV, enabling the
    downstream semantic-retrieval evaluation without re-reading audio.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

SR = 16000

BUNDLES = {
    "wavlm_base_plus": "WAVLM_BASE_PLUS",
    "wav2vec2_base": "WAV2VEC2_BASE",
    "hubert_base": "HUBERT_BASE",
}


def load_model(name: str = "wavlm_base_plus", device: str | None = None):
    import torchaudio

    if name not in BUNDLES:
        raise ValueError(f"unknown model '{name}'; options: {list(BUNDLES)}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bundle = getattr(torchaudio.pipelines, BUNDLES[name])
    model = bundle.get_model().to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, device, bundle.sample_rate


def _load_wav(path: Path, target_sr: int = SR) -> np.ndarray:
    import torchaudio

    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy()


@torch.no_grad()
def embed_files(paths: list[Path], model, device: str, model_sr: int,
                batch_log: int = 500, max_seconds: float = 20.0) -> np.ndarray:
    """Mean-pooled embeddings, one row per file. Clips longer than `max_seconds` are truncated
    so a handful of long recordings cannot dominate runtime or memory."""
    out = []
    t0 = time.perf_counter()
    max_len = int(max_seconds * model_sr)
    for i, p in enumerate(paths):
        wav = _load_wav(p, model_sr)
        if wav.size > max_len:
            wav = wav[:max_len]
        if wav.size < model_sr // 10:  # pad very short clips to 100 ms
            wav = np.pad(wav, (0, model_sr // 10 - wav.size))
        x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
        feats, _ = model.extract_features(x)
        # last transformer layer, mean over time
        emb = feats[-1].mean(dim=1).squeeze(0).cpu().numpy()
        out.append(emb)
        if (i + 1) % batch_log == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"    {i+1}/{len(paths)} ({rate:.1f} clips/s)", flush=True)
    return np.asarray(out, dtype=np.float32)


def save_bank(out_dir: Path, name: str, vectors: np.ndarray, meta: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", vectors)
    if meta:
        keys = list(meta[0].keys())
        with open(out_dir / f"{name}_meta.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(meta)
    print(f"  saved {name}: {vectors.shape} -> {out_dir / (name + '.npy')}")


def load_bank(out_dir: Path, name: str) -> tuple[np.ndarray, list[dict]]:
    vectors = np.load(out_dir / f"{name}.npy")
    meta_path = out_dir / f"{name}_meta.csv"
    meta = []
    if meta_path.exists():
        with open(meta_path, newline="", encoding="utf-8") as fh:
            meta = list(csv.DictReader(fh))
    return vectors, meta
