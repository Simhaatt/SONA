"""End-to-end real-data benchmark for Kaggle (or any GPU box with torchaudio).

Runs the full comparison on VoiceBank-DEMAND (whose test set uses noise types UNSEEN in training —
so it is itself a distribution-shift benchmark):

  classical baselines  +  supervised MaskGRU  +  RL frozen  +  RL online (SQUIM reward)

and prints/saves the master table + the online-vs-frozen delta.

Usage (inside the Kaggle notebook or CLI):
    python scripts/kaggle_run.py \
        --vbd-root /kaggle/input/voicebank-demand \
        --out /kaggle/working/results \
        --limit-train 3000 --limit-test 824 --epochs 30 --rl-epochs 30

VoiceBank-DEMAND layout expected under --vbd-root: clean_trainset_wav / noisy_trainset_wav /
clean_testset_wav / noisy_testset_wav (attach a Kaggle mirror and adjust the path).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_repo_to_path(repo: str | None) -> Path:
    root = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="repo root (default: this file's parent)")
    ap.add_argument("--vbd-root", required=True, help="VoiceBank-DEMAND folder")
    ap.add_argument("--out", default="results_kaggle")
    ap.add_argument("--limit-train", type=int, default=3000)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30, help="supervised epochs")
    ap.add_argument("--rl-epochs", type=int, default=30)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ROOT = _add_repo_to_path(args.repo)

    import torch

    from scripts.prepare_data import build_manifest  # reuse manifest builder
    from src.baselines import BASELINES
    from src.data import PairDataset
    from src.eval.evaluate import aggregate, evaluate_method
    from src.eval.metrics import available_metrics
    from src.eval.report import latex_table, markdown_table, save_rows_csv
    from src.eval.reward_squim import make_reward, squim_available
    from src.frontend import SpectralFrontend
    from src.models import NeuralEnhancer, OnlineAdapter
    from src.train.dataset_torch import TorchPairDataset
    from src.train.train_rl import train_rl
    from src.train.train_supervised import build_frontend, train
    from src.utils import get_logger, load_config, set_seed

    log = get_logger("kaggle")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1234)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log.info("device=%s | metric backends=%s | SQUIM=%s",
             device, available_metrics(), squim_available())

    # ---- manifests + data
    vbd = Path(args.vbd_root)
    build_manifest(vbd, "clean_trainset_wav", "noisy_trainset_wav", out / "train.csv")
    build_manifest(vbd, "clean_testset_wav", "noisy_testset_wav", out / "test.csv")
    train_samples = list(PairDataset(manifest=out / "train.csv", limit=args.limit_train))
    test_samples = list(PairDataset(manifest=out / "test.csv", limit=args.limit_test))
    log.info("train=%d test=%d", len(train_samples), len(test_samples))

    metrics = ["si_sdr", "si_sdr_i", "snr", "snr_i", "pesq", "stoi", "estoi", "lsd"]
    fe = SpectralFrontend(sr=16000, n_fft=512, n_bands=32)
    rows = []

    # ---- classical baselines
    for name, cls in BASELINES.items():
        log.info("baseline %s", name)
        rows += evaluate_method(cls(fe), test_samples, include_metrics=metrics, method_name=name)

    crop = int(16000 * 2.0)
    train_ds = TorchPairDataset(train_samples, crop_len=crop, train=True)

    # ---- supervised MaskGRU
    cfg_sup = load_config(ROOT / "configs/model_supervised.yaml")
    cfg_sup["train"]["epochs"] = args.epochs
    model = train(cfg_sup, train_ds, device=device, ckpt_dir=out / "ckpt_sup")
    tf_sup = build_frontend(cfg_sup).to(device)
    rows += evaluate_method(NeuralEnhancer(model, tf_sup, device=device, name="neural_mask_gru"),
                            test_samples, include_metrics=metrics)

    # ---- RL policy: frozen vs online (SQUIM reward) on the unseen-noise test set
    cfg_rl = load_config(ROOT / "configs/model_rl.yaml")
    cfg_rl["train"]["epochs"] = args.rl_epochs
    policy = train_rl(cfg_rl, train_ds, device=device, ckpt_dir=out / "ckpt_rl")
    tf_rl = build_frontend(cfg_rl).to(device)
    reward = make_reward("auto", device=device, sr=16000)
    rows += evaluate_method(NeuralEnhancer(policy, tf_rl, device=device, name="rl_frozen"),
                            test_samples, include_metrics=metrics)
    rows += evaluate_method(
        OnlineAdapter(policy, tf_rl, device=device, name="rl_online", steps=12, lr=0.1,
                      reward_fn=reward),
        test_samples, include_metrics=metrics)

    # ---- report
    summary = aggregate(rows, group_keys=("method",))
    save_rows_csv(rows, out / "per_sample.csv")
    save_rows_csv(summary, out / "summary.csv")
    (out / "summary.md").write_text(markdown_table(summary), encoding="utf-8")
    (out / "summary.tex").write_text(latex_table(summary), encoding="utf-8")
    print("\n=== VoiceBank-DEMAND test (unseen noises) ===\n")
    print(markdown_table(summary))

    by = {r["method"]: r for r in summary}
    if "rl_online" in by and "rl_frozen" in by:
        for k in ("si_sdr_i", "pesq", "stoi", "estoi"):
            fo, on = by["rl_frozen"].get(k), by["rl_online"].get(k)
            if isinstance(fo, (int, float)) and isinstance(on, (int, float)):
                print(f"delta(online - frozen) {k}: {on - fo:+.3f}")


if __name__ == "__main__":
    main()
