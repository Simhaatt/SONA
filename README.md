# Adaptive Audio Noise Cancellation (Learning-Based Frequency-Domain)

Real-time speech denoiser: **STFT → per-band attenuation (gain mask) → ISTFT/overlap-add**, with
the suppression gains chosen by methods ranging from classical filters to a learned, *online-adaptive*
policy. Built as a fair, equal-footing **benchmark study** to support a paper / patent / FYP.

> Central design principle: every method — classical or learned — shares the **same STFT
> front-end, the same per-band gain output space, and the same evaluation harness**. Only the
> gain-estimation step differs. That is what makes the comparison publishable.

## Status

| Milestone | State |
|---|---|
| M0 scaffold + utils | ✅ |
| M1 DSP core (STFT/ISTFT, bands, features) + eval harness | ✅ (perfect-reconstruction tested) |
| M2 classical baselines (hard/soft threshold, spectral subtraction, Wiener) + data pipeline | ✅ |
| M3 learned baselines (MaskGRU / MaskTCN / Attention) + supervised trainer | ✅ (beats baselines on synthetic; overfit-probe tested) |
| M4 RL agent + RL-attention hybrid + reference-free online adaptation (the novelty) | ✅ mechanism built & tested; frozen policy beats Wiener incl. burst. **Open experiment:** show online > frozen under noise-distribution shift (real data). |
| M5 full benchmark tables + M6 real-time latency/RTF profiling | ⏳ |
| M7 edge deployment (hardware chosen from M6 data) | ⏳ deferred |

## Install

```bash
pip install -r requirements.txt
```

Local dev here runs on Python 3.14 (CPU torch). `pesq`/`dnsmos`/`wer` have no wheels on 3.14 and
degrade to `null` in the metrics table. The **full metric suite runs on Kaggle (Python 3.11)**.

## Quick start (no downloads — synthetic data)

```bash
python scripts/run_benchmark.py --config configs/baselines.yaml --limit 12
```

Writes `results/summary.md`, `summary.tex`, `summary.csv`, `per_sample.csv`,
`summary_by_noise.csv`, and before/after figures under `results/figures/`.

## Real data (VoiceBank-DEMAND)

```bash
# attach/download the corpus, then:
python scripts/prepare_data.py --root /path/to/VoiceBank-DEMAND
# point the config at data/manifests/test.csv (data.manifest), then run_benchmark.py
```

## Layout

```
src/dsp/         STFT/ISTFT, band mapping, features        (shared front-end)
src/frontend.py  SpectralFrontend + Enhancer interface
src/baselines/   classical methods (all subclass Enhancer)
src/data/        datasets, noise mixing, burst augmentation
src/eval/        metrics, evaluate, report (tables + figures)
src/models/      learned models (mask net, attention, RL)  [M3/M4]
src/train/       training loops + losses                   [M3/M4]
scripts/         prepare_data, run_benchmark
configs/         YAML experiment configs (inherit default.yaml)
tests/           DSP correctness + end-to-end pipeline
```

## Tests

```bash
python -m pytest -q
```

Key guarantees: STFT→ISTFT perfect reconstruction; band gain partition-of-unity;
end-to-end benchmark runs and the SNR-based baselines improve SI-SDR.

## Metrics

SI-SDR & SI-SDRi, SNR & SNRi, PESQ (wb), STOI/ESTOI, DNSMOS, LSD, WER, plus latency/RTF (and
params/MACs during profiling). Missing optional backends report `null` rather than failing.
```
