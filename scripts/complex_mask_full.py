#!/usr/bin/env python3
"""Full-corpus A/B: magnitude gain mask vs complex ratio mask on VoiceBank-DEMAND.

Promotes the local pilot (sona-final-month/scripts/complex_mask_ab.py, which showed
complex +0.32 dB SI-SDR on a 1200-clip subset) to a full-corpus, PESQ-backed run so
the result is publishable.

Trains two GTCRN models that are IDENTICAL except for the output representation:
  * `gtcrn`          real per-band gain on magnitude, noisy phase preserved
  * `complex_gtcrn`  per-band complex ratio mask, modifies magnitude AND phase
under the same data, epochs, loss, optimiser and backbone width (ch/hidden/blocks).
The only variable is the mask type — the fair test of the hypothesis.

Both are evaluated on the full VoiceBank test set with the standard metric suite
(SI-SDR(i), SNR(i), PESQ, STOI, ESTOI, LSD). Emits per_sample.csv + summary.csv/md
in the same format as kaggle_run.py, so build_final_tables.py / significance.py
(paired Wilcoxon + bootstrap CIs) consume it directly.

    python scripts/complex_mask_full.py --vbd-root /path/to/VoiceBank-DEMAND \
        --epochs 50 --out results_complex_full
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _add_repo_to_path(repo: str | None) -> Path:
    root = Path(repo).resolve() if repo else ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def make_cfg(kind: str, args) -> "object":
    from src.utils.config import Config
    return Config({
        "seed": args.seed,
        "sr": 16000,
        "stft": {"n_fft": 512, "hop": 128, "win_length": 512, "center": True},
        "bands": {"n_bands": 32, "scale": "bark"},
        "model": {"kind": kind, "ch": args.ch, "hidden": args.hidden,
                  "n_blocks": args.n_blocks},
        "train": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                  "weight_decay": 1e-5, "mag_weight": args.mag_weight,
                  "mr_mag_weight": args.mr_mag_weight, "snr_weight": args.snr_weight,
                  "crop_seconds": 2.0},
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="repo root (default: this file's parent)")
    ap.add_argument("--vbd-root", default=None,
                    help="VoiceBank-DEMAND folder (standard clean_trainset_wav/... layout)")
    ap.add_argument("--local-root", default=None,
                    help="alternative: a folder with train/{clean,noisy} and "
                         "test/{clean,noisy} subdirs (the sona-final-month vbd_local layout)")
    ap.add_argument("--out", default="results_complex_full")
    ap.add_argument("--limit-train", type=int, default=None, help="None = full trainset")
    ap.add_argument("--limit-test", type=int, default=824)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--mag-weight", type=float, default=1.25)
    ap.add_argument("--mr-mag-weight", type=float, default=0.25)
    ap.add_argument("--snr-weight", type=float, default=0.3,
                    help="scale-anchor term; needed so the complex mask does not "
                         "drift output level under the scale-invariant SI-SDR loss")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kinds", default="gtcrn,complex_gtcrn")
    args = ap.parse_args()

    repo = _add_repo_to_path(args.repo)
    import torch

    from scripts.prepare_data import build_manifest
    from src.data import PairDataset
    from src.eval.evaluate import aggregate, evaluate_method
    from src.eval.metrics import available_metrics
    from src.eval.report import latex_table, markdown_table, save_rows_csv
    from src.models import NeuralEnhancer
    from src.train.dataset_torch import TorchPairDataset
    from src.train.train_supervised import build_frontend, build_model, train
    from src.utils import get_logger, set_seed

    log = get_logger("complex_full")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log.info("device=%s | metric backends=%s", device, available_metrics())

    # ---- data ----
    if args.local_root:
        # sona-final-month vbd_local layout: train/{clean,noisy}, test/{clean,noisy}
        root = Path(args.local_root)
        train_samples = list(PairDataset(clean_dir=root / "train" / "clean",
                                         noisy_dir=root / "train" / "noisy",
                                         limit=args.limit_train))
        test_samples = list(PairDataset(clean_dir=root / "test" / "clean",
                                        noisy_dir=root / "test" / "noisy",
                                        limit=args.limit_test))
    else:
        if not args.vbd_root:
            raise SystemExit("provide --vbd-root (standard layout) or --local-root (vbd_local layout)")
        vbd = Path(args.vbd_root)
        build_manifest(vbd, "clean_trainset_wav", "noisy_trainset_wav", out / "train.csv")
        build_manifest(vbd, "clean_testset_wav", "noisy_testset_wav", out / "test.csv")
        train_samples = list(PairDataset(manifest=out / "train.csv", limit=args.limit_train))
        test_samples = list(PairDataset(manifest=out / "test.csv", limit=args.limit_test))
    log.info("train=%d test=%d", len(train_samples), len(test_samples))
    if not train_samples:
        raise SystemExit("No TRAIN pairs found — need clean_trainset_wav/ + noisy_trainset_wav/.")
    if not test_samples:
        raise SystemExit("No TEST pairs found — need clean_testset_wav/ + noisy_testset_wav/.")

    metrics = ["si_sdr", "si_sdr_i", "snr", "snr_i", "pesq", "stoi", "estoi", "lsd"]
    crop = int(16000 * 2.0)
    train_ds = TorchPairDataset(train_samples, crop_len=crop, train=True, seed=args.seed)

    rows: list[dict] = []
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    for kind in kinds:
        cfg = make_cfg(kind, args)
        set_seed(args.seed)  # identical init RNG per model — the only diff is mask type
        model = train(cfg, train_ds, device=device, ckpt_dir=out / f"ckpt_{kind}")
        tf = build_frontend(cfg).to(device)
        name = "magnitude_gtcrn" if kind == "gtcrn" else kind
        log.info("evaluating %s (%d params, is_complex=%s)",
                 name, model.num_params(), getattr(model, "is_complex", False))
        rows += evaluate_method(
            NeuralEnhancer(model, tf, device=device, name=name),
            test_samples, include_metrics=metrics)

    # ---- report (same format as kaggle_run.py) ----
    summary = aggregate(rows, group_keys=("method",))
    save_rows_csv(rows, out / "per_sample.csv")
    save_rows_csv(summary, out / "summary.csv")
    (out / "summary.md").write_text(markdown_table(summary), encoding="utf-8")
    (out / "summary.tex").write_text(latex_table(summary), encoding="utf-8")
    print("\n=== VoiceBank-DEMAND test: magnitude vs complex mask ===\n")
    print(markdown_table(summary))

    by = {r["method"]: r for r in summary}
    mag, cx = by.get("magnitude_gtcrn"), by.get("complex_gtcrn")
    if mag and cx:
        higher_better = {"si_sdr", "si_sdr_i", "snr", "snr_i", "pesq", "stoi", "estoi"}
        print("\ndelta (complex - magnitude):")
        wins = total = 0
        for k in metrics:
            mv, cv = mag.get(k), cx.get(k)
            if mv is None or cv is None:
                continue
            d = cv - mv
            better = (d > 0) if k in higher_better else (d < 0)
            total += 1
            wins += int(better)
            print(f"  {k:9s} {mv:8.4f} -> {cv:8.4f}   {d:+.4f}"
                  + ("  <-- complex better" if better else ""))
        verdict = "POSITIVE" if wins > total / 2 else ("MIXED" if wins * 2 == total else "NEGATIVE")
        print(f"\ncomplex wins {wins}/{total} metrics -> {verdict}")


if __name__ == "__main__":
    main()
