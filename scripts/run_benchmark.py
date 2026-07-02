"""Run a full benchmark: every configured method over a dataset -> tables + figures.

Usage:
    python scripts/run_benchmark.py --config configs/baselines.yaml
    python scripts/run_benchmark.py --config configs/baselines.yaml --limit 5 --save-audio

With no dataset paths in the config it uses the built-in synthetic generator (no downloads),
so this runs anywhere. Point `data.clean_dir`/`data.noisy_dir` (or `data.manifest`) at
VoiceBank-DEMAND for the real benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import BASELINES
from src.data import PairDataset, SyntheticNoisyDataset
from src.eval.evaluate import aggregate, evaluate_method
from src.eval.metrics import available_metrics
from src.eval.report import latex_table, markdown_table, plot_comparison, save_rows_csv
from src.frontend import SpectralFrontend
from src.utils import get_logger, load_config, set_seed

log = get_logger("benchmark")


def build_dataset(cfg, limit=None):
    d = cfg.get_path("data", {})
    if d.get("manifest") or (d.get("clean_dir") and d.get("noisy_dir")):
        log.info("Using real PairDataset")
        return PairDataset(
            clean_dir=d.get("clean_dir"),
            noisy_dir=d.get("noisy_dir"),
            manifest=d.get("manifest"),
            sr=int(cfg.get_path("sr", 16000)),
            limit=limit or d.get("limit"),
        )
    s = d.get("synthetic", {})
    log.info("No dataset paths configured -> using SyntheticNoisyDataset")
    return list(
        SyntheticNoisyDataset(
            n=limit or int(s.get("n", 20)),
            sr=int(cfg.get_path("sr", 16000)),
            duration=float(s.get("duration", 2.0)),
            snr_db_range=tuple(s.get("snr_db_range", (0.0, 10.0))),
            seed=int(cfg.get_path("seed", 1234)),
        )
    )


def build_methods(cfg, frontend):
    methods = cfg.get_path("methods", None) or {name: {} for name in BASELINES}
    built = []
    for name, kwargs in methods.items():
        if name not in BASELINES:
            log.warning("unknown method '%s' (skipping)", name)
            continue
        built.append(BASELINES[name](frontend, **(kwargs or {})))
    return built


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--limit", type=int, default=None, help="cap number of samples")
    ap.add_argument("--save-audio", action="store_true")
    ap.add_argument("--figures", type=int, default=2, help="how many comparison figures to save")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(int(cfg.get_path("seed", 1234)))

    log.info("Optional metric backends: %s", available_metrics())
    frontend = SpectralFrontend.from_config(cfg)
    dataset = build_dataset(cfg, limit=args.limit)
    dataset = list(dataset)
    methods = build_methods(cfg, frontend)
    metrics = list(cfg.get_path("metrics", []))
    results_dir = ROOT / cfg.get_path("paths.results_dir", "results")

    all_rows = []
    for enh in methods:
        log.info("Evaluating %s over %d samples", enh.name, len(dataset))
        rows = evaluate_method(
            enh, dataset, include_metrics=metrics,
            save_audio_dir=(results_dir / "audio") if args.save_audio else None,
        )
        all_rows += rows

    save_rows_csv(all_rows, results_dir / "per_sample.csv")

    summary = aggregate(all_rows, group_keys=("method",))
    save_rows_csv(summary, results_dir / "summary.csv")
    md = markdown_table(summary)
    (results_dir / "summary.md").write_text(md, encoding="utf-8")
    (results_dir / "summary.tex").write_text(latex_table(summary), encoding="utf-8")

    # Per-noise-type breakdown (supports the non-stationary/burst analysis in the paper).
    by_noise = aggregate(all_rows, group_keys=("method", "noise_type"))
    save_rows_csv(by_noise, results_dir / "summary_by_noise.csv")

    # A few before/after figures for the report/demo.
    fig_dir = ROOT / cfg.get_path("paths.figures_dir", "results/figures")
    best = methods[-1]
    for s in dataset[: args.figures]:
        enh = best.enhance(s.noisy, s.sr)
        plot_comparison(s.clean, s.noisy, enh, s.sr, fig_dir / f"{s.uid}_{best.name}.png",
                        title=f"{best.name} | {s.noise_type} | {s.snr_db:.1f} dB")

    print("\n=== Benchmark summary ===\n")
    print(md)
    print(f"\nResults written to {results_dir}")


if __name__ == "__main__":
    main()
