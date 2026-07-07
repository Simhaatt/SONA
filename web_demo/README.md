# SONA Web Demo

Local web app for the SONA neural audio enhancement engine. It wraps the locked
Layer A GTCRN checkpoint and the Layer B controller (`src/controller`) behind a
Flask backend with a single-page studio-style frontend.

![stack](https://img.shields.io/badge/stack-Flask%20%2B%20vanilla%20JS-informational)

## What it does

- Load the bundled noisy sample (a real pre-mixed MS-SNSD test clip, AirConditioner noise @ 0 dB SNR) or upload audio.
- Classify the acoustic scene (speech / music / traffic / fan / crowd / mixed / unknown).
- Enhance with the locked `gtcrn_ch96_h96_b4_mw1p25_mr0p25` checkpoint plus
  mode presets, denoise strength, safety floor, 8-band EQ, compressor, limiter.
- Play original vs enhanced vs removed-noise residual (A/B/N), download the enhanced WAV.
- Remember/restore per-scene preferences (JSON store).
- Analytics: durations, peaks, input/output/residual RMS, estimated noise removed %,
  loudness change, processing time, realtime factor, per-stage timings.
- Visualizations: input/output waveforms on a **shared amplitude scale**
  (per-signal scaling only via the visible "normalized view" toggle),
  removed-noise residual view (`input − enhanced`), spectrum overlay with
  10th-percentile noise-floor curves and a "background noise above 2 kHz"
  reduction readout, and a draggable before/after comparison.
- A "Low-noise input detected; enhancement may be subtle." notice appears for
  inputs with a high activity-to-floor ratio (and again after enhancement if
  less than ~3 % of energy was removed).

## Install

Everything runs on the packages already used by the repo (torch, numpy,
soundfile) plus Flask:

```
pip install flask
```

(No other new dependencies. No retraining, no GPU required — CPU realtime
factor for the 8 s sample is about 0.06×.)

## Run

```
cd "D:\noise cancellation\web_demo"
python app.py
```

Then open **http://localhost:7860**.

## Demo flow

1. Wait for the loading screen (the GTCRN checkpoint warms in the background).
2. Click **Use sample noisy audio** (or drop your own WAV/FLAC/OGG/MP3).
3. Optionally click **Classify scene**, or leave the **auto** toggle on.
4. Pick a mode preset and/or adjust the sliders.
5. Click **Enhance Audio** and watch the pipeline stages.
6. A/B the original vs enhanced audio, inspect analytics, download the WAV.
7. Use **Remember for scene** so the same scene restores your settings next time.

## Paths

| Thing | Path |
| --- | --- |
| Layer A checkpoint (read-only) | `results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07\extracted\ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25\last.pt` |
| Bundled sample | `web_demo\samples\sample_noisy.wav` (regenerate: `python tools\make_sample.py`) |
| Uploaded/registered inputs | `results\web_demo_uploads\` |
| Enhanced outputs | `results\web_demo_outputs\` |
| Preference memory | `results\web_demo_preferences.json` |

## API (all local)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | engine status, checkpoint presence, device |
| `/api/modes` | GET | mode presets + scene→mode routing |
| `/api/sample` | POST | register bundled sample as input |
| `/api/upload` | POST | multipart upload (validated, ≤60 MB, ≤300 s) |
| `/api/classify` | POST | scene label + confidence + features |
| `/api/enhance` | POST | full pipeline; returns analytics + output id |
| `/api/audio/input/<id>`, `/api/audio/output/<id>`, `/api/audio/residual/<id>` | GET | stream WAV |
| `/api/download/<id>` | GET | download enhanced WAV |
| `/api/preferences` | GET/POST/DELETE | scene preference store |

Safety: uploads are extension-checked and decode-verified, stored under fresh
UUID names; audio/download routes only accept 32-char hex ids inside the fixed
demo folders, so user input can never address arbitrary paths. Errors return
JSON without tracebacks; details go to the server log.

## Tests

```
cd "D:\noise cancellation\web_demo"
python -m pytest tests -q
```

17 tests cover health, sample presence, upload validation, classification,
end-to-end enhancement (real checkpoint, produces a WAV), download-path safety,
and the preference-memory round trip. The preference store is isolated to a
temp file during tests.

## Known limitations

- Offline file processing only — the realtime mic pipeline is a later roadmap stage.
- Everything is resampled to 16 kHz mono (the pipeline's native format).
- MP3 upload support depends on the installed libsndfile version.
- The Layer B compressor noticeably reduces RMS level; the
  "Normalize output loudness" toggle (default on) rescales the output peak to
  the input peak for fair A/B listening.
- The scene classifier is the deterministic feature-based starter, not a trained model.
- Uploaded-name display list is in-memory; restarting the server forgets display names (files remain).
