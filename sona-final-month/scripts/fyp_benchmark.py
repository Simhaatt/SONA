#!/usr/bin/env python3
"""FYP core benchmark: learned adaptive model vs classical DSP baselines.

Runs hard/soft thresholding, spectral subtraction, Wiener filtering, and the
learned per-band GTCRN model on the SAME VoiceBank-DEMAND test clips, so the
comparison the project is built around (Objective 5) is on identical data.

    python scripts/fyp_benchmark.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# The SONA repo root: scripts/ -> sona-final-month/ -> <repo>.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from src.baselines import BASELINES  # noqa: E402
from src.data.dataset import PairDataset  # noqa: E402
from src.eval.evaluate import aggregate, evaluate_method  # noqa: E402
from src.eval.report import markdown_table, save_rows_csv  # noqa: E402
from src.frontend import SpectralFrontend  # noqa: E402
from src.models import NeuralEnhancer  # noqa: E402
from src.train.train_supervised import build_frontend, build_model  # noqa: E402
from src.utils.config import Config  # noqa: E402

DATA = REPO / "vbd_local"
OUT = Path(__file__).resolve().parents[1] / "FYP_BENCHMARK"
CKPT = REPO / ("results/kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/"
               "extracted/ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt")
METRICS = ["si_sdr", "si_sdr_i", "snr", "snr_i", "stoi", "estoi", "lsd"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = list(PairDataset(clean_dir=DATA / "test" / "clean",
                            noisy_dir=DATA / "test" / "noisy", limit=824))
    print(f"device={device}  test clips={len(test)}")

    fe = SpectralFrontend(sr=16000, n_fft=512, n_bands=32)
    rows = []
    # classical DSP baselines
    for name, cls in BASELINES.items():
        print(f"  classical: {name}")
        rows += evaluate_method(cls(fe), test, include_metrics=METRICS, method_name=name)

    # learned adaptive model (GTCRN)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = Config(ck["cfg"])
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"  learned: GTCRN ({model.num_params():,} params)")
    rows += evaluate_method(NeuralEnhancer(model, tf, device=device, name="learned_gtcrn"),
                            test, include_metrics=METRICS)

    summary = aggregate(rows, group_keys=("method",))
    save_rows_csv(rows, OUT / "per_sample.csv")
    save_rows_csv(summary, OUT / "summary.csv")
    (OUT / "summary.md").write_text(markdown_table(summary), encoding="utf-8")
    print("\n=== FYP benchmark: classical DSP vs learned adaptive (same 824 clips) ===\n")
    print(markdown_table(summary))


if __name__ == "__main__":
    main()
