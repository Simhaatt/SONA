"""Distribution-shift experiment: train on seen noise, test on UNSEEN noise.

Compares, on the unseen-noise test set:
  * wiener      -- classical reference
  * rl_frozen   -- trained policy, no inference-time adaptation
  * rl_online   -- same policy + reference-free online adaptation (SQUIM on Kaggle, heuristic local)

The headline number is the rl_online - rl_frozen delta on unseen noise: the paper's core claim.

Usage:
    python scripts/eval_shift.py --config configs/shift.yaml
    python scripts/eval_shift.py --config configs/shift.yaml --epochs 12

Locally (Python 3.14, no torchaudio) the reward falls back to the spectral heuristic and data is
synthetic -> this validates the *machinery*. The real gate runs on Kaggle (SQUIM + real noise via
`data.manifest` / real noise banks); see notebooks/kaggle_benchmark.ipynb.
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
from src.eval.reward_squim import make_reward, squim_available
from src.frontend import SpectralFrontend
from src.models import NeuralEnhancer, OnlineAdapter, make_adapter
from src.train.dataset_torch import TorchPairDataset
from src.train.train_rl import build_policy, train_rl
from src.train.train_supervised import build_frontend
from src.utils import get_logger, load_config, set_seed

log = get_logger("eval_shift")


def _synth(cfg, key, default_n, seed_off, noise_types):
    s = cfg.get_path(f"data.{key}", {}) or {}
    return list(SyntheticNoisyDataset(
        n=int(s.get("n", default_n)), sr=int(cfg.get_path("sr", 16000)),
        duration=float(s.get("duration", 2.0)),
        snr_db_range=tuple(s.get("snr_db_range", (-5.0, 10.0))),
        seed=int(cfg.get_path("seed", 1234)) + seed_off,
        noise_types=tuple(noise_types),
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shift.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ablate-adapters", action="store_true",
                    help="compare adapter dimensionalities (scalar/lowrank/band_bias/film)")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    set_seed(int(cfg.get_path("seed", 1234)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_noise = cfg.get_path("shift.train_noise", ["white", "pink"])
    test_noise = cfg.get_path("shift.test_noise", ["burst"])
    log.info("Shift: train noise=%s  test noise(unseen)=%s", train_noise, test_noise)
    log.info("SQUIM reward available: %s (else spectral heuristic)", squim_available())

    train_samples = _synth(cfg, "synthetic", 60, 0, train_noise)
    test_samples = _synth(cfg, "test_synthetic", 24, 777, test_noise)
    sr = int(cfg.get_path("sr", 16000))
    crop = int(sr * float(cfg.get_path("train.crop_seconds", 2.0)))
    train_ds = TorchPairDataset(train_samples, crop_len=crop, train=True)

    policy = train_rl(cfg, train_ds, device=device, ckpt_dir=ROOT / "checkpoints")

    tf = build_frontend(cfg).to(device)
    fe = SpectralFrontend.from_config(cfg)
    metrics = list(cfg.get_path("metrics", []))
    o = cfg.get_path("online", {})
    reward_fn = make_reward(o.get("reward", "auto"), device=device, sr=sr)
    steps, olr = int(o.get("steps", 12)), float(o.get("lr", 0.1))

    def _online(kind_name, adapter_kind):
        return OnlineAdapter(
            policy, tf, device=device, name=kind_name, steps=steps, lr=olr,
            reward_fn=reward_fn, adapter=make_adapter(adapter_kind, tf.n_bands))

    methods = [
        WienerFilter(fe),
        NeuralEnhancer(policy, tf, device=device, name="rl_frozen"),
    ]
    if args.ablate_adapters:
        # Dimensionality ablation: scalar(1) < lowrank(r) < band_bias(n) < film(2n).
        for ak in ("scalar", "lowrank", "band_bias", "film"):
            methods.append(_online(f"rl_online_{ak}", ak))
    else:
        methods.append(_online("rl_online", o.get("adapter", "band_bias")))

    rows = []
    for m in methods:
        rows += evaluate_method(m, test_samples, include_metrics=metrics,
                                method_name=getattr(m, "name", m.__class__.__name__))

    summary = aggregate(rows, group_keys=("method",))
    print("\n=== Distribution-shift results on UNSEEN noise %s ===\n" % test_noise)
    print(markdown_table(summary))

    by = {r["method"]: r for r in summary}
    frozen = by.get("rl_frozen")
    if frozen:
        print("\ndelta(online - frozen) on unseen noise:")
        for name, r in by.items():
            if not name.startswith("rl_online"):
                continue
            parts = []
            for k in ("si_sdr_i", "pesq", "stoi", "dnsmos"):
                fo, on = frozen.get(k), r.get(k)
                if isinstance(fo, (int, float)) and isinstance(on, (int, float)):
                    parts.append(f"{k} {on - fo:+.3f}")
            print(f"  {name}: " + " | ".join(parts))

    results_dir = ROOT / cfg.get_path("paths.results_dir", "results")
    save_rows_csv(rows, results_dir / "shift_per_sample.csv")
    save_rows_csv(summary, results_dir / "shift_summary.csv")


if __name__ == "__main__":
    main()
