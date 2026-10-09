# SONA Web Demo

Local web app for the SONA speech-enhancement final-year project. A Flask backend
wraps the research code in `src/` (every trained model, the classical baselines,
the Layer B controller, the metric suite and the streaming processor) behind a
multi-page frontend. Nothing leaves the machine.

![stack](https://img.shields.io/badge/stack-Flask%20%2B%20vanilla%20JS-informational)

## Run

```
pip install flask          # everything else is already in requirements.txt
python web_demo/app.py
```

Then open **http://localhost:7860**. CPU only, no GPU needed.

## Pages

The app has one task per page; advanced settings live in a slide-in **Settings** panel.

**Home** — what SONA does, with buttons to try a sample clip or use your own file.

**Enhance** — choose audio from the **sample library** (10 real test-set mixtures
from MS-SNSD and VoiceBank-DEMAND, each with its clean reference) or **upload** a
file (WAV/FLAC/OGG/MP3, up to 5 min; an optional clean reference can be attached).
Press Enhance; the defaults work well.

**Settings panel** (from Enhance or "Adjust and re-run" on Results)
- *Model:* any of **9 engines** — GTCRN (locked FYP model), complex-mask GTCRN,
  causal Transformer, Mask-GRU, Mask-TCN, Wiener, spectral subtraction, soft and
  hard thresholding — plus experimental **online adaptation** (strict-guard
  `OnlineAdapter`, with a report on whether the guard accepted the update).
- *Sound:* 7 mode presets or automatic, noise-removal strength, **50/60 Hz hum
  filter**, loudness matching; advanced EQ, compressor and bypass switches.
- *Scene:* automatic scene detection, manual scene, per-scene remembered settings.

**Results** — player with Original / Enhanced / Removed / Clean (switch while
playing with `1`–`4`, `Space` to play/pause), a live audio visualiser,
before/after **spectrograms** on a shared dB scale and four headline numbers.
"More detail" holds the full metric table (SI-SDR(i), SNRi, STOI, ESTOI, LSD, PESQ
when installed — for both the raw model output and the delivered output),
waveforms, spectrum, a before/after slider, stage timings and this session's runs.

**Compare** — run any set of engines on the current clip through the same harness
(raw output, no post-processing): ranking chart plus a sortable table with a play
button per method.

**Real-time** — streams the clip through the FYP's COLA block processor
(`sona-final-month/realtime/stream_processor.py`) at 32/64/128/256 ms blocks:
delay, per-block compute time vs the deadline, late blocks, speed, and streamed vs
whole-file quality. This is a simulation on a file; live microphone processing is
Phase 2 (hardware).

**About** — model cards, the frozen benchmark tables from
`sona-final-month/FINAL_TABLES`, and the key findings.

Dark futuristic theme by default with a light option; animations switch off when
the system asks for reduced motion; layout works down to phone width.

## Paths

| Thing | Path |
| --- | --- |
| Model checkpoints (read-only) | see `ENGINES` in `engine.py` |
| Sample library | `web_demo/samples/*_noisy.wav`, `*_clean.wav`, `library.json` (rebuild: `python web_demo/tools/build_sample_library.py`) |
| Legacy demo mix | `web_demo/samples/sample_noisy.wav` (`python web_demo/tools/make_sample.py`) |
| Uploaded inputs (+ `_ref.wav` references) | `results/web_demo_uploads/` |
| Enhanced / compare / stream outputs | `results/web_demo_outputs/` |
| Preference memory | `results/web_demo_preferences.json` |

## API (all local)

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | GET | engine status, loaded engines, available metric backends |
| `/api/engines` | GET | the 9 engines with summaries and benchmark numbers |
| `/api/modes` | GET | mode presets + scene→mode routing |
| `/api/samples` | GET | sample library |
| `/api/sample` | POST | register a library sample (`sample_id`) or the legacy mix |
| `/api/upload` | POST | multipart upload (validated, ≤60 MB, ≤300 s) |
| `/api/reference` | POST / DELETE | attach / remove a clean reference for an input |
| `/api/classify` | POST | scene label + confidence + features |
| `/api/enhance` | POST | full pipeline (`engine`, `online_adapt`, `hum_notch_hz`, controller settings) |
| `/api/compare` | POST | run several engines on one clip (≤60 s) |
| `/api/stream` | POST | streaming simulation (`block_ms`, `engine`; ≤60 s) |
| `/api/research` | GET | frozen benchmark tables |
| `/api/audio/{input,reference,output,residual}/<id>` | GET | stream WAV |
| `/api/download/<id>` | GET | download enhanced WAV |
| `/api/preferences` | GET/POST/DELETE | scene preference store |

Safety: uploads are extension-checked and decode-verified, stored under fresh
UUID names; audio routes only accept 32-char hex ids (sample ids are
`[a-z0-9_]`) inside the fixed demo folders, so user input can never address
arbitrary paths. Errors return JSON without tracebacks.

## Tests

```
cd web_demo
python -m pytest tests -q
```

29 tests: health, upload validation, classification, end-to-end enhancement
with the real checkpoints, engine/hum/online-adaptation validation, reference
metrics (the locked model must improve a real test mixture by >3 dB SI-SDR),
compare, streaming, research tables, download-path safety and the preference
store (isolated to a temp file).

## Known limitations

- Works on audio files only. Live microphone input is planned for Phase 2
  (hardware); the command-line `sona-final-month/realtime/realtime_demo.py --mode live`
  is the existing prototype.
- Everything is resampled to 16 kHz mono (the models' native format).
- PESQ shows "n/a" on Python 3.14 (no `pesq` wheel); all other metrics work.
- Streaming enhances each block independently, so quality is lower than
  whole-file processing — the Real-time view measures exactly that trade-off.
- Real-time numbers are for this CPU under its current load, not an edge board.
