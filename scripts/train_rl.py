"""Train the RL gain policy, then run the headline ablation: frozen vs online-adapted.

Usage:
    python scripts/train_rl.py --config configs/model_rl.yaml
    python scripts/train_rl.py --config configs/model_rl.yaml --epochs 12

Evaluates on the val set:
  * Wiener (classical reference)
  * rl_frozen   -- policy mean gain, no inference-time adaptation
  * rl_online   -- same policy + reference-free online adaptation (the novelty)
and breaks results down by noise type, so the benefit on burst / non-stationary noise is visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.baselines import WienerFilter
from src.data import SyntheticNoisyDataset
from src.eval.evaluate import aggregate, evaluate_method
from src.eval.report import markdown_table, save_rows_csv
from src.frontend import SpectralFrontend
from src.models import NeuralEnhancer, OnlineAdapter
from src.train.dataset_torch import TorchPairDataset
from src.train.train_rl import build_policy, train_rl
from src.train.train_supervised import build_frontend
from src.utils import get_logger, load_config, set_seed

log = get_logger("train_rl_script")


def _synth(cfg, key, default_n, seed_off):
    s = cfg.get_path(f"data.{key}", {}) or {}
    return list(SyntheticNoisyDataset(
        n=int(s.get("n", default_n)), sr=int(cfg.get_path("sr", 16000)),
        duration=float(s.get("duration", 2.0)),
        snr_db_range=tuple(s.get("snr_db_range", (-5.0, 10.0))),
        seed=int(cfg.get_path("seed", 1234)) + seed_off,
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_rl.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    set_seed(int(cfg.get_path("seed", 1234)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = _synth(cfg, "synthetic", 60, 0)
    val_samples = _synth(cfg, "val_synthetic", 24, 777)
    sr = int(cfg.get_path("sr", 16000))
    crop = int(sr * float(cfg.get_path("train.crop_seconds", 2.0)))
    train_ds = TorchPairDataset(train_samples, crop_len=crop, train=True)

    policy = train_rl(cfg, train_ds, device=device, ckpt_dir=ROOT / "checkpoints")

    tf = build_frontend(cfg).to(device)
    # NOTE: reuse the *trained* policy weights (train_rl returns the trained instance).
    fe = SpectralFrontend.from_config(cfg)
    metrics = list(cfg.get_path("metrics", []))
    o = cfg.get_path("online", {})

    methods = [
        WienerFilter(fe),
        NeuralEnhancer(policy, tf, device=device, name="rl_frozen"),
        OnlineAdapter(policy, tf, device=device, name="rl_online",
                      steps=int(o.get("steps", 8)), lr=float(o.get("lr", 0.1))),
    ]

    rows = []
    for m in methods:
        name = getattr(m, "name", m.__class__.__name__)
        rows += evaluate_method(m, val_samples, include_metrics=metrics, method_name=name)

    results_dir = ROOT / cfg.get_path("paths.results_dir", "results")
    print("\n=== Overall (val) ===\n")
    print(markdown_table(aggregate(rows, group_keys=("method",))))
    print("\n=== By noise type (frozen vs online adaptation) ===\n")
    by_noise = aggregate(rows, group_keys=("method", "noise_type"))
    print(markdown_table(by_noise))
    save_rows_csv(rows, results_dir / "rl_per_sample.csv")
    save_rows_csv(by_noise, results_dir / "rl_by_noise.csv")


if __name__ == "__main__":
    main()
