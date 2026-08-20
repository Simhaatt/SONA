#!/usr/bin/env python3
"""Controlled A/B: magnitude gain mask vs complex ratio mask, same backbone.

Trains two GTCRN models that are identical except for the output representation —
`gtcrn` (real per-band gain on magnitude, noisy phase preserved) vs `complex_gtcrn`
(per-band complex ratio mask, modifies magnitude AND phase) — under IDENTICAL data,
epochs, loss and optimiser, then evaluates both on the VoiceBank test set. This
isolates "complex mask vs magnitude gain" as the single variable, which is the fair
way to test whether the complex mask actually helps (rather than comparing against
the Kaggle checkpoint, which had far more training).

    python scripts/complex_mask_ab.py --epochs 30 --n-eval 300

Local metrics: SI-SDR(i), SNR(i), STOI, ESTOI, LSD, DNSMOS. PESQ needs a C build
unavailable for this Python (the repo runs PESQ on Kaggle); DNSMOS is the local
perceptual proxy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# The SONA repo root: scripts/ -> sona-final-month/ -> <repo>.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.dataset import PairDataset  # noqa: E402
from src.eval.metrics import compute_metrics  # noqa: E402
from src.models import NeuralEnhancer, TorchFrontend  # noqa: E402
from src.train.dataset_torch import TorchPairDataset  # noqa: E402
from src.train.train_supervised import build_frontend, build_model, train  # noqa: E402
from src.utils.config import Config  # noqa: E402

# PESQ (needs a C build unavailable for this Python) and DNSMOS (Kaggle-only module)
# are not available locally; SI-SDR/SNR/STOI/ESTOI/LSD are the local metric set.
EVAL_METRICS = ["si_sdr", "si_sdr_i", "snr", "snr_i", "stoi", "estoi", "lsd"]


def make_cfg(kind: str, epochs: int, seed: int) -> Config:
    return Config({
        "seed": seed,
        "sr": 16000,
        "stft": {"n_fft": 512, "hop": 128, "win_length": 512, "center": True},
        "bands": {"n_bands": 32, "scale": "bark"},
        "model": {"kind": kind, "ch": 96, "hidden": 96, "n_blocks": 4},
        "train": {"epochs": epochs, "batch_size": 8, "lr": 3e-4, "weight_decay": 1e-5,
                  "mag_weight": 1.25, "mr_mag_weight": 0.25, "snr_weight": 0.3,
                  "crop_seconds": 2.0},
    })


def load_samples(clean_dir, noisy_dir, limit=None):
    ds = PairDataset(clean_dir=clean_dir, noisy_dir=noisy_dir, limit=limit)
    return list(ds)


def evaluate(model, tf, test_samples, device, n_eval):
    enh = NeuralEnhancer(model, tf, device=device, name="eval")
    samples = test_samples[:n_eval]
    acc = {m: [] for m in EVAL_METRICS}
    for s in samples:
        y = enh.enhance(s.noisy, s.sr)
        met = compute_metrics(s.clean, y, noisy=s.noisy, sr=s.sr, include=EVAL_METRICS)
        for m in EVAL_METRICS:
            v = met.get(m)
            if v is not None and np.isfinite(v):
                acc[m].append(v)
    return {m: (float(np.mean(v)) if v else None) for m, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=REPO / "vbd_local")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "COMPLEX_MASK_AB")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=None, help="cap train pairs")
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kinds", nargs="+", default=["gtcrn", "complex_gtcrn"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"device={device}  epochs={args.epochs}  n_eval={args.n_eval}")
    print("loading data ...")
    train_samples = load_samples(args.data / "train" / "clean",
                                 args.data / "train" / "noisy", limit=args.n_train)
    test_samples = load_samples(args.data / "test" / "clean",
                                args.data / "test" / "noisy")
    print(f"  train pairs: {len(train_samples)}   test pairs: {len(test_samples)}")

    crop = int(2.0 * 16000)
    train_ds = TorchPairDataset(train_samples, crop_len=crop, train=True, seed=args.seed)

    results = {}
    for kind in args.kinds:
        print(f"\n===== training {kind} =====")
        cfg = make_cfg(kind, args.epochs, args.seed)
        torch.manual_seed(args.seed)  # identical init RNG state per model
        np.random.seed(args.seed)
        t0 = time.perf_counter()
        model = train(cfg, train_ds, val_dataset=None, device=device,
                      ckpt_dir=args.out_dir / f"ckpt_{kind}")
        train_s = time.perf_counter() - t0
        tf = build_frontend(cfg).to(device)
        model.eval()
        print(f"  trained in {train_s:.0f}s, params={model.num_params()}; evaluating ...")
        metrics = evaluate(model, tf, test_samples, device, args.n_eval)
        metrics["_params"] = model.num_params()
        metrics["_train_seconds"] = round(train_s, 1)
        results[kind] = metrics
        print(f"  {kind}: " + "  ".join(
            f"{m}={metrics[m]:.4f}" for m in EVAL_METRICS if metrics[m] is not None))

    (args.out_dir / "ab_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- comparison ----
    print("\n================ A/B COMPARISON ================")
    print(f"{'metric':10s} " + "".join(f"{k:>16s}" for k in args.kinds) + f"{'delta':>12s}")
    higher_better = {"si_sdr", "si_sdr_i", "snr", "snr_i", "stoi", "estoi", "dnsmos"}
    wins = 0
    total = 0
    for m in EVAL_METRICS:
        vals = [results[k].get(m) for k in args.kinds]
        if any(v is None for v in vals) or len(args.kinds) != 2:
            continue
        delta = vals[1] - vals[0]  # complex - magnitude
        better = (delta > 0) if m in higher_better else (delta < 0)
        total += 1
        wins += int(better)
        flag = "  <-- complex better" if better else ""
        print(f"{m:10s} " + "".join(f"{v:16.4f}" for v in vals) + f"{delta:+12.4f}{flag}")
    print(f"\ncomplex wins {wins}/{total} metrics")
    verdict = "POSITIVE" if wins > total / 2 else ("MIXED" if wins == total / 2 else "NEGATIVE")
    print(f"VERDICT: complex mask is {verdict} vs magnitude baseline (same backbone, {args.epochs} epochs)")
    (args.out_dir / "verdict.txt").write_text(
        f"complex wins {wins}/{total} metrics -> {verdict}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
