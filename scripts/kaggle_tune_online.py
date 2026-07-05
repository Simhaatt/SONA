"""Focused Kaggle tuner for supervised online adaptation.

This is deliberately narrower than ``kaggle_run.py``: it trains selected supervised models once,
scores their frozen outputs, then evaluates a grid of online-adaptation settings. Use it after the
full Scenario A run shows the supervised-online path needs tuning.
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


def _csv_items(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _csv_ints(text: str) -> list[int]:
    return [int(x) for x in _csv_items(text)]


def _csv_floats(text: str) -> list[float]:
    return [float(x) for x in _csv_items(text)]


def _csv_optional_floats(text: str) -> list[float | None]:
    out: list[float | None] = []
    for x in _csv_items(text):
        out.append(None if x.lower() in ("none", "off", "unclipped") else float(x))
    return out


def _fmt_float(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _fmt_optional_float(x: float | None) -> str:
    return "none" if x is None else _fmt_float(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="repo root")
    ap.add_argument("--vbd-root", required=True, help="VoiceBank-DEMAND folder")
    ap.add_argument("--out", default="results_online_tune")
    ap.add_argument("--limit-train", type=int, default=2000)
    ap.add_argument("--limit-test", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--sup-models", default="attention",
                    help="comma list, usually 'attention' for quick tuning")
    ap.add_argument("--adapters", default="scalar,lowrank")
    ap.add_argument("--anchors", default="0.2,0.5,1.0")
    ap.add_argument("--steps", default="3,6")
    ap.add_argument("--max-gain-deltas", default="none",
                    help="comma list of trust-region gain deltas; use 'none' to disable")
    ap.add_argument("--smooth-time-weight", type=float, default=0.0)
    ap.add_argument("--smooth-band-weight", type=float, default=0.0)
    ap.add_argument("--guard-min-reward-deltas", default="none",
                    help="comma list; require adapted reward - frozen reward >= value, or none")
    ap.add_argument("--guard-max-output-deltas", default="none",
                    help="comma list; require normalized output drift <= value, or none")
    ap.add_argument("--guard-min-mask-polarities", default="none",
                    help="comma list; require adapted mask polarity >= value, or none")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--mask-polarity-weight", type=float, default=0.0,
                    help="training-time confidence loss weight for supervised model")
    ap.add_argument("--mask-polarity-target", type=float, default=0.0,
                    help="desired mean mask polarity in [0,1] for supervised training")
    ap.add_argument("--reward-kind", default="auto",
                    help="'auto' uses SQUIM on Kaggle, spectral fallback locally")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ROOT = _add_repo_to_path(args.repo)

    import torch

    from scripts.prepare_data import build_manifest
    from src.data import PairDataset
    from src.eval.evaluate import aggregate, evaluate_method
    from src.eval.metrics import available_metrics
    from src.eval.report import latex_table, markdown_table, save_rows_csv
    from src.eval.reward_squim import make_reward, squim_available
    from src.models import NeuralEnhancer, OnlineAdapter, make_adapter
    from src.train.dataset_torch import TorchPairDataset
    from src.train.train_supervised import build_frontend, train
    from src.utils import get_logger, load_config, set_seed

    log = get_logger("online_tune")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1234)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log.info("device=%s | metric backends=%s | SQUIM=%s",
             device, available_metrics(), squim_available())
    log.info("grid adapters=%s anchors=%s steps=%s",
             _csv_items(args.adapters), _csv_floats(args.anchors), _csv_ints(args.steps))
    log.info("trust region max_gain_deltas=%s smooth_time=%.4f smooth_band=%.4f",
             _csv_optional_floats(args.max_gain_deltas),
             args.smooth_time_weight, args.smooth_band_weight)
    log.info("guards min_reward_deltas=%s max_output_deltas=%s",
             _csv_optional_floats(args.guard_min_reward_deltas),
             _csv_optional_floats(args.guard_max_output_deltas))
    log.info("mask polarity guards min_mask_polarities=%s",
             _csv_optional_floats(args.guard_min_mask_polarities))

    vbd = Path(args.vbd_root)
    build_manifest(vbd, "clean_trainset_wav", "noisy_trainset_wav", out / "train.csv")
    build_manifest(vbd, "clean_testset_wav", "noisy_testset_wav", out / "test.csv")
    train_samples = list(PairDataset(manifest=out / "train.csv", limit=args.limit_train))
    test_samples = list(PairDataset(manifest=out / "test.csv", limit=args.limit_test))
    log.info("train=%d test=%d", len(train_samples), len(test_samples))
    if not train_samples or not test_samples:
        raise SystemExit("VoiceBank-DEMAND train/test pairs were not found.")

    metrics = ["si_sdr", "si_sdr_i", "snr", "snr_i", "pesq", "stoi", "estoi", "lsd"]
    crop = int(16000 * 2.0)
    train_ds = TorchPairDataset(train_samples, crop_len=crop, train=True)

    cfg_sup = load_config(ROOT / "configs/model_supervised.yaml")
    cfg_sup["train"]["epochs"] = args.epochs
    cfg_sup["train"]["mask_polarity_weight"] = args.mask_polarity_weight
    cfg_sup["train"]["mask_polarity_target"] = args.mask_polarity_target
    tf = build_frontend(cfg_sup).to(device)
    reward = make_reward(args.reward_kind, device=device, sr=16000)

    rows: list[dict] = []
    for kind in _csv_items(args.sup_models):
        log.info("train supervised %s", kind)
        cfg_sup["model"] = {"kind": kind}
        model = train(cfg_sup, train_ds, device=device, ckpt_dir=out / f"ckpt_{kind}")

        frozen_name = f"neural_{kind}"
        rows += evaluate_method(
            NeuralEnhancer(model, tf, device=device, name=frozen_name),
            test_samples, include_metrics=metrics, method_name=frozen_name)

        for adapter_kind in _csv_items(args.adapters):
            for steps in _csv_ints(args.steps):
                for anchor in _csv_floats(args.anchors):
                    for max_delta in _csv_optional_floats(args.max_gain_deltas):
                        for guard_reward in _csv_optional_floats(args.guard_min_reward_deltas):
                            for guard_output in _csv_optional_floats(args.guard_max_output_deltas):
                                for guard_polarity in _csv_optional_floats(args.guard_min_mask_polarities):
                                    guard_suffix = ""
                                    if guard_reward is not None:
                                        guard_suffix += f"_gr{_fmt_float(guard_reward)}"
                                    if guard_output is not None:
                                        guard_suffix += f"_go{_fmt_float(guard_output)}"
                                    if guard_polarity is not None:
                                        guard_suffix += f"_gp{_fmt_float(guard_polarity)}"
                                    name = (
                                        f"{kind}_online_{adapter_kind}"
                                        f"_s{steps}_a{_fmt_float(anchor)}"
                                        f"_d{_fmt_optional_float(max_delta)}"
                                        f"{guard_suffix}"
                                    )
                                    log.info("eval %s", name)
                                    rows += evaluate_method(
                                        OnlineAdapter(
                                            model, tf, device=device, name=name, steps=steps, lr=args.lr,
                                            anchor_weight=anchor, max_gain_delta=max_delta,
                                            smooth_time_weight=args.smooth_time_weight,
                                            smooth_band_weight=args.smooth_band_weight,
                                            guard_min_reward_delta=guard_reward,
                                            guard_max_output_delta=guard_output,
                                            guard_min_mask_polarity=guard_polarity,
                                            reward_fn=reward,
                                            adapter=make_adapter(adapter_kind, tf.n_bands),
                                        ),
                                        test_samples, include_metrics=metrics, method_name=name)

        summary = aggregate(rows, group_keys=("method",))
        save_rows_csv(rows, out / "per_sample.csv")
        save_rows_csv(summary, out / "summary.csv")
        (out / "summary.md").write_text(markdown_table(summary), encoding="utf-8")
        (out / "summary.tex").write_text(latex_table(summary), encoding="utf-8")

    summary = aggregate(rows, group_keys=("method",))
    print("\n=== Online adaptation tuning summary ===\n")
    print(markdown_table(summary))

    by = {r["method"]: r for r in summary}
    print("\nDelta vs frozen base:")
    for kind in _csv_items(args.sup_models):
        base = by.get(f"neural_{kind}")
        if not base:
            continue
        for name, row in by.items():
            if not name.startswith(f"{kind}_online_"):
                continue
            parts = []
            for key in ("si_sdr_i", "pesq", "stoi", "estoi", "lsd"):
                b, v = base.get(key), row.get(key)
                if isinstance(b, (int, float)) and isinstance(v, (int, float)):
                    parts.append(f"{key} {v - b:+.3f}")
            print(f"  {name}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
