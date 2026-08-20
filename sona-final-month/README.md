# SONA — Final-Month Software (Week 1: results consolidation)

Self-contained analysis package that turns the scattered `results/**/summary.csv`
and `per_sample.csv` files from the main SONA repo (`..`) into
the frozen tables, significance tests, and figures the thesis / paper / viva cite.

This folder contains **software only** — no training, no model code, no new
experiments. It reads the existing results and produces publication-ready
artefacts. See `../noise cancellation/NEXT_STEPS_PLAN.md` for the full month plan.

## Install

```
pip install -r requirements.txt
```

(Only pandas / numpy / scipy / matplotlib / pytest — no torch, no GPU. The
live-mic realtime mode additionally needs `sounddevice`, listed as optional.)

## One command, one green check

```
python scripts/check_all.py
```

Chains everything that can be verified without you being in the room: rebuilds
the results tables, runs the significance battery, regenerates the figures,
runs the realtime unit tests, runs the realtime selftest (synthetic audio, no
mic needed), and runs the scene-classifier self-test. Prints a PASS/FAIL
summary per stage and exits non-zero if anything broke — use it as a
pre-flight check before a supervisor meeting or before trusting a thesis
number. `--skip-significance` skips the slower per-clip stage;
`--results-dir` overrides the results location (defaults to
`../results`).

## Run just the results pipeline

```
python scripts/run_all.py --results-dir "../results"
```

`--results-dir` defaults to `../results` relative to this
folder, so on this machine you can usually just run `python scripts/run_all.py`.

Outputs:

| Folder | Contents |
| --- | --- |
| `FINAL_TABLES/` | master + curated tables (csv + markdown) |
| `SIGNIFICANCE/` | paired Wilcoxon + bootstrap-CI tables (csv + markdown) |
| `figures/` | png charts for the thesis/slides |
| `SCENE_EVAL_SELFTEST/` | written by `check_all.py`'s scene-classifier stage |

## Individual stages

**1. Aggregate all runs into frozen tables** (already run once — the real
outputs are committed in `FINAL_TABLES/`):

```
python scripts/build_final_tables.py --results-dir "../results"
```

Produces `master_all_runs`, `voicebank_full824`, `ms_snsd_external`,
`online_adaptation_story`, `classical_synthetic`.

**2. Statistical significance** (needs the per_sample.csv files in the results
tree — run on your machine where `E:` is a normal path):

```
python scripts/significance.py --preset all --results-dir "../results"
```

Compares, on shared clips (paired, aligned on `uid`):
GTCRN vs Attention, GTCRN vs Mamba-GTCRN, GTCRN vs SONA-G, GTCRN vs SONA-R,
and strict-guard online vs frozen — reporting mean difference, 95% bootstrap CI,
and two-sided Wilcoxon p per metric. For a one-off pair use the `--a-file /
--a-method / --b-file / --b-method` form (see the script header).

**3. Figures** (reads `FINAL_TABLES/`):

```
python scripts/make_figures.py
# with a before/after spectrogram from the web-demo sample + an enhanced wav:
python scripts/make_figures.py \
    --sample-wav "../noise cancellation/web_demo/samples/sample_noisy.wav" \
    --enhanced-wav "path/to/enhanced.wav"
```

## Realtime demo (Week 2)

`realtime/` contains a hardware-independent block-streaming core plus a demo CLI
that runs the locked GTCRN checkpoint in a realtime loop. It reuses the main
repo's `src.controller.load_neural_enhancer` (array-in / array-out); if torch or
the checkpoint is unavailable it falls back to a spectral noise-gate so the path
still runs.

```
# self-test (synthetic tone, prints latency + RTF; no mic, no torch needed)
python realtime/realtime_demo.py --mode selftest

# offline: stream a wav through the realtime path and write the enhanced wav
python realtime/realtime_demo.py --mode wav --in noisy.wav --out enhanced.wav

# live mic -> speaker (install sounddevice first)
pip install sounddevice
python realtime/realtime_demo.py --mode live --seconds 20
```

`--block` / `--hop` trade latency against smoothness (default 1024/256 =
~64 ms algorithmic latency at 16 kHz, 75 % overlap-add). Unit tests prove the
overlap-add reconstructs the input exactly (COLA):

```
cd realtime && python -m pytest -q
```

## Scene-classifier evaluation (Week 2)

`scripts/eval_scene_classifier.py` scores the deterministic scene classifier on a
folder of labelled recordings and writes a confusion matrix + per-class
precision/recall/F1, for the Layer B section of the thesis.

```
# validate the harness with no audio / no repo import:
python scripts/eval_scene_classifier.py --self-test

# real run: record a few clips per scene, list them in a manifest, then:
python scripts/eval_scene_classifier.py --manifest recordings/manifest.csv
```

See `recordings/manifest.example.csv` for the manifest format (columns:
`path,label`; labels: speech / music / traffic / fan / crowd / mixed / unknown).

## Notes

- `FINAL_TABLES/` is shipped pre-populated with the **real** aggregated numbers
  so the tables are usable immediately. Re-running stage 1 regenerates them.
- `SIGNIFICANCE/` and `figures/` are generated locally by `run_all.py` because
  the significance stage needs the large `per_sample.csv` files and the
  spectrogram needs audio — neither is duplicated into this folder.
- `lsd` is treated as lower-is-better everywhere (highlighting, "better_is").
- The classical baseline table is on the synthetic dev set (illustrative only);
  the headline neural comparisons are on VoiceBank-DEMAND and MS-SNSD.
