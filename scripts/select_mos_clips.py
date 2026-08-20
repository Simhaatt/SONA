"""Select and export the audio clips for the in-house MOS listening test.

Picks a fixed, reproducible sample of VoiceBank-DEMAND test utterances, runs each through every
condition (noisy passthrough, best classical baseline, GTCRN, complex-mask GTCRN, optionally
DeepFilterNet2), loudness-matches every condition to the *original noisy clip's* RMS level (so
raters compare quality, not level), and writes one wav per (uid, condition) plus a manifest CSV
mapping a blinded clip_id -> (uid, condition) for later analysis. One obviously-clean attention
check clip is included and clearly flagged in the manifest (excluded from blinding).

Usage:
    python scripts/select_mos_clips.py --gtcrn-checkpoint "...\\last.pt" \\
        --complex-checkpoint "...\\last.pt" --out mos_test
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fixed, reproducible pick of 7 VoiceBank-DEMAND test speakers/utterances spanning the manifest.
SEED = 2026
N_UTTERANCES = 7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",
                    default=str(ROOT / "vbd_local" / "test_manifest.csv"))
    ap.add_argument("--gtcrn-checkpoint", default=None)
    ap.add_argument("--complex-checkpoint", default=None)
    ap.add_argument("--deepfilternet2-dir", default=None,
                    help="dir of pre-generated DeepFilterNet2 wavs, named <uid>.wav (optional)")
    ap.add_argument("--out", default=str(ROOT / "mos_test"))
    ap.add_argument("--n-utterances", type=int, default=N_UTTERANCES)
    args = ap.parse_args()

    import numpy as np
    import torch

    from src.baselines import SpectralSubtraction
    from src.data import PairDataset
    from src.frontend import SpectralFrontend
    from src.models import NeuralEnhancer
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.audio_io import load_audio, save_audio
    from src.utils.config import Config

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    samples = list(PairDataset(manifest=args.manifest))
    rng = random.Random(SEED)
    picked = rng.sample(samples, k=min(args.n_utterances, len(samples)))

    frontend = SpectralFrontend()
    spectral_sub = SpectralSubtraction(frontend)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_neural(ckpt_path, label):
        checkpoint = torch.load(ckpt_path, map_location=device)
        cfg = Config(checkpoint["cfg"])
        tf = build_frontend(cfg).to(device)
        model = build_model(cfg, tf).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return NeuralEnhancer(model, tf, device=device, name=label)

    gtcrn = _load_neural(args.gtcrn_checkpoint, "gtcrn") if args.gtcrn_checkpoint else None
    complex_gtcrn = (
        _load_neural(args.complex_checkpoint, "complex_gtcrn") if args.complex_checkpoint else None
    )
    dfn2_dir = Path(args.deepfilternet2_dir) if args.deepfilternet2_dir else None

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))

    def _match_level(x: np.ndarray, target_rms: float) -> np.ndarray:
        cur = _rms(x)
        if cur < 1e-8:
            return x
        y = x * (target_rms / cur)
        peak = float(np.max(np.abs(y)))
        if peak > 0.98:
            y = y * (0.98 / peak)
        return y.astype(np.float32)

    manifest_rows: list[dict] = []
    clip_counter = 0

    for s in picked:
        target_rms = _rms(s.noisy)
        conditions: dict[str, np.ndarray] = {"noisy": s.noisy}
        conditions["spectral_subtraction"] = spectral_sub.enhance(s.noisy, s.sr)
        if gtcrn is not None:
            conditions["gtcrn"] = gtcrn.enhance(s.noisy, s.sr)
        if complex_gtcrn is not None:
            conditions["complex_gtcrn"] = complex_gtcrn.enhance(s.noisy, s.sr)
        if dfn2_dir is not None:
            cand = dfn2_dir / f"{s.uid}.wav"
            if cand.is_file():
                wav, _ = load_audio(cand, sr=s.sr, mono=True)
                conditions["deepfilternet2"] = wav

        for cond_name, wav in conditions.items():
            clip_counter += 1
            clip_id = f"c{clip_counter:03d}"
            normed = _match_level(np.asarray(wav, dtype=np.float32), target_rms)
            save_audio(audio_dir / f"{clip_id}.wav", normed, s.sr)
            manifest_rows.append(
                {"clip_id": clip_id, "uid": s.uid, "condition": cond_name, "is_attention_check": 0}
            )

    # One attention-check clip: the cleanest available signal (unprocessed clean reference),
    # obviously good, so an inattentive/careless rater is easy to catch.
    clean_pick = next((s for s in picked if s.clean is not None), None)
    if clean_pick is not None:
        clip_counter += 1
        clip_id = f"c{clip_counter:03d}"
        target_rms = _rms(clean_pick.noisy)
        normed = _match_level(np.asarray(clean_pick.clean, dtype=np.float32), target_rms)
        save_audio(audio_dir / f"{clip_id}.wav", normed, clean_pick.sr)
        manifest_rows.append(
            {"clip_id": clip_id, "uid": clean_pick.uid, "condition": "clean_attention_check",
             "is_attention_check": 1}
        )

    manifest_path = out / "clip_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "uid", "condition", "is_attention_check"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Wrote {len(manifest_rows)} clips to {audio_dir}")
    print(f"Manifest: {manifest_path}")
    print("Conditions present:", sorted({r["condition"] for r in manifest_rows}))


if __name__ == "__main__":
    main()
