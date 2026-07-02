"""Train a learned mask model, then benchmark it against the classical baselines.

Usage:
    python scripts/train.py --config configs/model_supervised.yaml
    python scripts/train.py --config configs/model_supervised.yaml --epochs 10   # override

Uses the synthetic generator when no dataset paths are configured (runs anywhere, CPU or GPU).
Point `data.manifest` at VoiceBank-DEMAND for real training. Produces a combined comparison
table (baselines + the trained model) so you immediately see where the model lands.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.baselines import BASELINES
from src.data import PairDataset, SyntheticNoisyDataset
from src.eval.evaluate import aggregate, evaluate_method
from src.eval.report import markdown_table, save_rows_csv
from src.frontend import SpectralFrontend
from src.train.dataset_torch import TorchPairDataset
from src.train.train_supervised import build_frontend, make_enhancer, train
from src.utils import get_logger, load_config, set_seed

log = get_logger("train_script")


def _synth(cfg, key, default_n):
    s = cfg.get_path(f"data.{key}", {}) or {}
    return list(SyntheticNoisyDataset(
        n=int(s.get("n", default_n)),
        sr=int(cfg.get_path("sr", 16000)),
        duration=float(s.get("duration", 2.0)),
        snr_db_range=tuple(s.get("snr_db_range", (-5.0, 10.0))),
        seed=int(cfg.get_path("seed", 1234)) + (0 if key == "synthetic" else 777),
    ))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/model_supervised.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    set_seed(int(cfg.get_path("seed", 1234)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data
    d = cfg.get_path("data", {})
    if d.get("manifest"):
        train_samples = list(PairDataset(manifest=d["manifest"], sr=cfg.get_path("sr", 16000)))
        val_samples = list(PairDataset(manifest=d.get("val_manifest", d["manifest"]),
                                       sr=cfg.get_path("sr", 16000), limit=d.get("val_limit", 50)))
    else:
        log.info("No manifest -> synthetic train/val")
        train_samples = _synth(cfg, "synthetic", 60)
        val_samples = _synth(cfg, "val_synthetic", 20)

    sr = int(cfg.get_path("sr", 16000))
    crop = int(sr * float(cfg.get_path("train.crop_seconds", 2.0)))
    train_ds = TorchPairDataset(train_samples, crop_len=crop, train=True)
    val_ds = TorchPairDataset(val_samples, crop_len=crop, train=False)

    # ---- train
    model = train(cfg, train_ds, val_ds, device=device, ckpt_dir=ROOT / "checkpoints")

    # ---- benchmark trained model vs classical baselines on the val set
    tf = build_frontend(cfg).to(device)
    tf.load_state_dict({k: v for k, v in tf.state_dict().items()})  # buffers already set
    model_name = f"neural_{cfg.get_path('model.kind','model')}"
    enhancer = make_enhancer(model, tf, device=device, name=model_name)

    metrics = list(cfg.get_path("metrics", []))
    fe = SpectralFrontend.from_config(cfg)
    rows = []
    for name, cls in BASELINES.items():
        rows += evaluate_method(cls(fe), val_samples, include_metrics=metrics, method_name=name)
    rows += evaluate_method(enhancer, val_samples, include_metrics=metrics, method_name=model_name)

    summary = aggregate(rows, group_keys=("method",))
    results_dir = ROOT / cfg.get_path("paths.results_dir", "results")
    save_rows_csv(summary, results_dir / f"summary_{model_name}.csv")
    md = markdown_table(summary)
    (results_dir / f"summary_{model_name}.md").write_text(md, encoding="utf-8")
    print("\n=== Trained model vs baselines (val) ===\n")
    print(md)


if __name__ == "__main__":
    main()
