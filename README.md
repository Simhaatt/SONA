# SONA: Self-supervised Online Neural Audio engine

SONA removes background noise from recorded speech: fans, traffic, babble, typing and machines. A compact learned model (a 722,209-parameter GTCRN) predicts how much of each frequency band to keep. It runs about 10× faster than real time on one laptop CPU core, and the whole system works offline.

Final-year project, Department of Electrical and Electronics Engineering, Sri Sivasubramaniya Nadar College of Engineering, Chennai.
Team: A. Rohit Daniel, Simhaa T. T., Vijayaraghavan A. · Supervisor: Dr. G. R. Venkatakrishnan.
Phase I (this repository) is the software system and its evaluation. Phase II moves the same model onto an embedded live microphone-to-headphone device.

Every method, classical or learned, shares the same STFT front-end (512-point Hann window, 128 hop, 32 Bark bands), the same per-band gain output and the same evaluation harness. Only the gain estimator changes, so a score difference comes from the estimator alone.

## Results

**VoiceBank-DEMAND test set, 824 clips** (speakers and noise types unseen in training):

| Method | SI-SDRi (dB) | PESQ | STOI | ESTOI | LSD (dB) |
|---|---:|---:|---:|---:|---:|
| Original (no processing) | 0.00 | 1.97 | 0.921 | 0.787 | 10.50 |
| Spectral subtraction | −0.25 | — | 0.884 | 0.734 | 8.86 |
| Wiener filter | −1.93 | — | 0.824 | 0.677 | 10.87 |
| Hard threshold | −3.69 | — | 0.767 | 0.573 | 13.78 |
| Soft threshold | −5.82 | — | 0.728 | 0.532 | 15.28 |
| Causal Transformer | 9.08 | 2.50 | 0.932 | 0.834 | 5.97 |
| **GTCRN (locked model)** | **9.93** | **2.52** | 0.930 | 0.832 | 6.47 |

- All four classical filters score below the unprocessed input, in every noise type and SNR level.
- **MS-SNSD, 1,100 clips never used in training:** GTCRN gains +3.15 dB SI-SDR, but intelligibility does not improve (STOI 0.818 before, 0.816 after).
- **Complex vs magnitude mask:** +0.31 dB SI-SDR, −0.016 PESQ (p = 0.005). It is a trade-off, not a win.
- **Guarded test-time (online) adaptation**, the project's original idea, is safe but gives about zero gain. It is reported as a negative result.
- **Quantisation:** one 960-parameter layer (0.133 % of weights) carries 98.7 % of the INT8 loss. Keeping it in FP32 gives a 916.9 KB model with no detectable loss.
- **Speech recognition:** enhancement raises Whisper's corpus word error rate by 23–47 %.

Frozen tables: `sona-final-month/FINAL_TABLES/`. Significance tests: `sona-final-month/SIGNIFICANCE/significance.md`. Full write-up: `paper_bundle/`.

**Paper:** *Where the Loss Lives: Concentrated Damage in Audio Compression and Evaluation*, submitted to the 7th International Conference on Cognitive Computing and Cyber Physical Systems (IC4S 2026).

## Web application

```bash
pip install -r requirements.txt
python web_demo/app.py --open        # http://localhost:7860
```

The app ships with ten test clips that have clean references, plus the five trained models it needs. It has five pages:
- **Enhance:** pick a test clip or upload your own file.
- **Results:** instant before/after playback, spectrograms and quality scores against the original.
- **Compare:** runs nine methods on the same clip.
- **Real-time:** a block-streaming simulation.
- **About:** every benchmark table.

Microphone input is Phase II. See [`web_demo/README.md`](web_demo/README.md). `python web_demo/tools/package_demo.py` builds a portable zip with one-click launchers for Windows, macOS and Linux.

`SONA_Project_Explainer.html` explains the whole project from scratch, with interactive demos.

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

Available modes: `speech`, `aggressive_speech`, `music`, `podcast`, `gaming`, `hearing`, `studio`.
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

For audibly noisy clips, try `--mode aggressive_speech`, or use `--bypass-eq` /
`--bypass-compressor` to hear the neural denoiser without post-processing.

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
web_demo/         Flask offline demo (port 7860)

PERCEPTUAL_MP/    mixed-precision quantization study (stem-exemption result)
sona-final-month/ frozen result tables, significance battery, figures, complex-mask A/B,
                  scene-classifier evaluation, realtime streaming demo, FYP report
AudioVQ/          data-oblivious vector quantization for audio retrieval (separate paper)
mos_test/         blinded MOS listening-test app (awaiting raters)

results/          per-experiment summaries and per-clip artefacts
vbd_local/        local 16 kHz VoiceBank-DEMAND copy (3.0 GB, gitignored, regenerable)
external_data/    MS-SNSD corpus + SONA-mixed test set
```

Corpus locations resolve from environment variables, falling back to paths derived from each
file's own location — so the tree can be moved or copied to another drive without edits.

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

