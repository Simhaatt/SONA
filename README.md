# SONA — Self-supervised Online Neural Audio engine

Real-time speech enhancement that **adapts after deployment, with no clean reference and no
cloud**. Audio flows through **STFT → per-band attenuation (+ EQ / dynamic-range) → ISTFT /
overlap-add**, where the gains are set by a lightweight learned controller. At inference the
controller is refined online by maximizing a **pretrained non-intrusive quality estimator
(TorchAudio-SQUIM)** — the reference-free reward.

Built as a fair, equal-footing **benchmark study + a defensible novelty** to support a journal
paper, a provisional patent, and the FYP demo.

> **Central design principle:** every method — classical or learned — shares the **same STFT
> front-end, the same per-band gain output space, and the same evaluation harness**. Only the
> gain-estimation step differs. That is what makes the comparison publishable.

## The idea (why this is novel)

Plain ML frequency masking (RNNoise, DeepFilterNet, …) is well-trodden and not patentable. SONA's
core is different and narrow enough to defend:

> A learned controller adjusts the parameters of an **interpretable DSP chain**, and a
> **low-dimensional** adaptation module is updated **online at inference** by maximizing a
> **reference-free** quality signal — stable and cheap enough to run on edge hardware.

- **Layer A (research / patent core):** reference-free online adaptation of a small controller.
  Headline result = it beats a frozen model of equal size under **noise-distribution shift**.
- **Layer B (product / demo / dependent claims):** the *same* controller drives EQ + dynamic
  range; plus audio-scene context, a self-learning preference loop, GUI, and Raspberry Pi.

See the full plan in the design doc referenced at the bottom.

## Status

| Phase | State |
|---|---|
| Foundation: shared front-end, classical baselines, eval harness, learned models (M0–M3) | ✅ |
| RL policy + attention hybrid + reference-free `OnlineAdapter` (M4) | ✅ |
| **P2** non-intrusive SQUIM reward + pluggable reward in adapter + distribution-shift eval + Kaggle path | In progress: real-data path works; online adaptation improves PESQ in RL but still needs tuning to preserve SI-SDR/STOI |
| **P3** adaptation-dimensionality ablation (scalar / low-rank / band-bias / FiLM) | ✅ (low-dim adaptation wins locally) |
| **P4** differentiable parametric EQ + hum notch + compressor/limiter | ✅ built & tested |
| P4 unified controller (EQ/DRC in the action space) + scene classifier + preference loop | ⏳ |
| P5 real-time streaming, latency/RTF/power, ONNX/TFLite export, GUI, Raspberry Pi | ⏳ |
| P6 paper write-up + provisional patent draft | ⏳ |

38 tests passing.

## Install

```bash
pip install -r requirements.txt
```

Local dev here runs on **Python 3.14 (CPU torch, no torchaudio)**: `pesq`/`dnsmos`/`wer` and
`SQUIM` have no wheels, so metrics degrade to `null` and the reward falls back to a spectral
heuristic. The **full metric suite + SQUIM reward run on Kaggle (Python 3.11, T4)**.

## Quick start (no downloads — synthetic data)

```bash
# classical + learned benchmark on synthetic speech-like data
python scripts/run_benchmark.py --config configs/baselines.yaml --limit 12

# the headline experiment: train on seen noise, test on UNSEEN noise (frozen vs online-adaptive)
python scripts/eval_shift.py --config configs/shift.yaml --epochs 12

# adaptation-dimensionality ablation (scalar / low-rank / band-bias / FiLM)
python scripts/eval_shift.py --config configs/shift.yaml --epochs 12 --ablate-adapters
```

## Real data (VoiceBank-DEMAND, on Kaggle)

Open `notebooks/kaggle_benchmark.ipynb`, attach the SONA repo + a VoiceBank-DEMAND dataset, set
the two paths, and run. It calls `scripts/kaggle_run.py`, which builds manifests and runs the full
comparison (classical + supervised MaskGRU/TCN/Attention, RL frozen vs **RL online with SQUIM
reward**, and anchored supervised-online variants such as `attention_online`) on the VoiceBank-
DEMAND test set (whose noises are unseen in training → a built-in distribution-shift benchmark),
with the full metric suite.

Latest checked artifacts are recorded under `results/`. Layer A is now locked on
`gtcrn_ch96_h96_b4_mw1p25_mr0p25`: a compact GTCRN with multi-resolution spectral loss. Mamba-GTCRN,
SONA-G governed GTCRN, and SONA-R residual GTCRN were tested and not promoted. See
`LAYER_A_RESULTS.md` and `SONA_STATUS.md` for the final table and external-validation story.

## Layer B controlled enhancement

The first product/controller layer lives in `src/controller/`. It adds mode presets, scene-to-mode
routing, EQ, compressor settings, denoise-strength blending, and peak limiting around a trained
Layer A checkpoint.

```bash
python scripts/enhance_controlled.py --checkpoint "results/kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/extracted/ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt" --input "path/to/noisy.wav" --output "results/audio/noisy__sona_speech.wav" --mode speech
```

Available modes: `speech`, `music`, `podcast`, `gaming`, `hearing`, `studio`.
Available scenes for auto-routing: `speech`, `music`, `traffic`, `fan`, `crowd`, `mixed`, `unknown`.

Preference memory stores scene-specific choices in JSON:

```bash
python scripts/manage_preferences.py --memory "results/preferences.json" set --scene traffic --mode hearing --denoise-strength 0.9
python scripts/manage_preferences.py --memory "results/preferences.json" list
python scripts/enhance_controlled.py --memory "results/preferences.json" --scene traffic --checkpoint "results/kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/extracted/ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt" --input "path/to/noisy.wav" --output "results/audio/noisy__sona_traffic.wav"
```

Scene classification is available as a lightweight local heuristic:

```bash
python scripts/classify_scene.py --input "path/to/noisy.wav"
python scripts/enhance_controlled.py --auto-scene --memory "results/preferences.json" --checkpoint "results/kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/extracted/ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt" --input "path/to/noisy.wav" --output "results/audio/noisy__sona_auto.wav"
```

## Layout

```
src/dsp/          STFT/ISTFT, band mapping, features        (shared front-end, numpy)
src/frontend.py   SpectralFrontend + Enhancer interface
src/baselines/    classical methods (hard/soft threshold, spectral sub, Wiener)
src/data/         datasets, noise mixing, burst aug, noise-type splits
src/eval/         metrics, evaluate, report, reward_squim   (SQUIM reward + fallback)
src/models/       torch front-end, mask/attention/RL models, OnlineAdapter,
                  adapt.py (adaptation modules), eq.py (differentiable EQ/DRC)
src/controller/   Layer B mode presets, scene routing, controlled enhancement pipeline
src/train/        supervised + RL training loops, losses
scripts/          prepare_data, run_benchmark, eval_shift, kaggle_run
notebooks/        kaggle_benchmark.ipynb (real-data run on the T4)
configs/          YAML experiment configs (inherit default.yaml)
tests/            DSP correctness, pipeline, reward, adapters, EQ/DRC
```

## Tests

```bash
python -m pytest -q
```

Key guarantees: STFT→ISTFT perfect reconstruction; band-gain partition-of-unity; SNR-based
baselines improve SI-SDR; SQUIM/heuristic reward is bounded; EQ unity = identity; each adapter is
identity at reset; online adaptation runs with every adapter.

## Metrics

SI-SDR & SI-SDRi, SNR & SNRi, PESQ (wb), STOI/ESTOI, DNSMOS, LSD, WER, plus latency/RTF (and
params/MACs during profiling). Missing optional backends report `null` rather than failing.

---

Design doc / roadmap: `C:\Users\hp\.claude\plans\c-users-hp-appdata-local-packages-53192-vectorized-wirth.md`
(SONA plan — Layers A/B, datasets, phased milestones, patent-claim sketch).
