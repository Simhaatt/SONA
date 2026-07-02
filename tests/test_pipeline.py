"""End-to-end integration: baselines run over synthetic data and produce a coherent table."""

import numpy as np

from src.frontend import SpectralFrontend
from src.baselines import BASELINES
from src.data import SyntheticNoisyDataset
from src.eval.evaluate import evaluate_method, aggregate
from src.eval.report import markdown_table


def _run_all(include=None):
    fe = SpectralFrontend(sr=16000, n_fft=512, n_bands=32)
    ds = list(SyntheticNoisyDataset(n=6, duration=1.5, seed=1))
    all_rows = []
    for name, cls in BASELINES.items():
        enh = cls(fe)
        all_rows += evaluate_method(enh, ds, include_metrics=include, method_name=name)
    return all_rows


def test_pipeline_runs_and_reports():
    rows = _run_all(include=["si_sdr", "si_sdr_i", "snr", "snr_i", "lsd", "stoi", "estoi"])
    assert rows
    summary = aggregate(rows, group_keys=("method",))
    table = markdown_table(summary)
    assert "passthrough" in table and "wiener" in table
    # snr_i must populate even though only the base 'snr' was implicitly needed.
    assert all("snr_i" in r for r in summary)
    # Passthrough improvement is ~0 (only STFT reconstruction error separates it from noisy).
    pt = next(r for r in summary if r["method"] == "passthrough")
    assert abs(pt["si_sdr_i"]) < 0.5


def test_snr_baselines_improve_si_sdr():
    # On the (now realistic, with silence gaps) synthetic set, the SNR-based methods should
    # improve SI-SDR over the noisy input on average. This validates the whole eval chain.
    rows = _run_all(include=["si_sdr", "si_sdr_i"])
    summary = {r["method"]: r for r in aggregate(rows)}
    assert summary["wiener"]["si_sdr_i"] > 1.0, summary["wiener"]
    assert summary["spectral_subtraction"]["si_sdr_i"] > 0.0, summary["spectral_subtraction"]


def test_lsd_is_reasonable_scale():
    rows = _run_all(include=["lsd"])
    summary = {r["method"]: r for r in aggregate(rows)}
    # LSD should be a sane dB scale (single/low-double digits), not inflated by silence.
    assert 0.0 < summary["wiener"]["lsd"] < 40.0, summary["wiener"]


def test_output_length_matches_input():
    fe = SpectralFrontend(n_fft=512, n_bands=32)
    ds = list(SyntheticNoisyDataset(n=2, duration=1.0, seed=3))
    enh = BASELINES["wiener"](fe)
    for s in ds:
        y = enh.enhance(s.noisy, s.sr)
        assert len(y) == len(s.noisy)
