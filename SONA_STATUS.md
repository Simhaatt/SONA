# SONA Status

Last updated: 2026-07-03

## Current Goal

Tune trust-region online adaptation before launching the full Scenario B paper run.

## Current Run State

Kaggle Scenario A retry completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_voicebank_online_2026-07-03_run2\sona_artifacts.zip`
- `D:\noise cancellation\results\kaggle_voicebank_online_2026-07-03_run2\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_online_2026-07-03_run2\extracted\summary.csv`

Result: the anchored supervised-online method did not pass the thesis gate yet.

New tuning tooling has been added:

- `D:\noise cancellation\scripts\kaggle_tune_online.py`
- `D:\noise cancellation\sona_kaggle_tune_online.ipynb`

This trains the supervised base once, then evaluates a grid of online adapter settings.

Tuning run 1 completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run1\sona_online_tune_artifacts.zip`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run1\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run1\extracted\summary.csv`

Tuning validation run 2 completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run2_300clip_scalar\sona_online_tune_artifacts.zip`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run2_300clip_scalar\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run2_300clip_scalar\extracted\summary.csv`

New trust-region methodology has been added after reviewing external model advice:

- `OnlineAdapter` supports `max_gain_delta` to clip online gain changes around the frozen gain.
- `OnlineAdapter` supports correction smoothness penalties:
  - `smooth_time_weight`
  - `smooth_band_weight`
- `scripts\kaggle_tune_online.py` exposes:
  - `--max-gain-deltas`
  - `--smooth-time-weight`
  - `--smooth-band-weight`
- `sona_kaggle_tune_online.ipynb` exposes the same knobs.
- Local tests cover trust-region gain bounds.

## What We Are Doing Right Now

We are testing the improved thesis version, not building new code.

The current Kaggle run is checking whether online adaptation works better when applied to strong supervised models instead of only the weaker RL policy:

- frozen baseline: `neural_attention`
- online version: `attention_online`
- frozen baseline: `neural_mask_gru`
- online version: `mask_gru_online`

This is the important retry after the first run showed `rl_online` improved PESQ but hurt SI-SDR/STOI.

## Latest Results

Scenario A retry, VoiceBank-DEMAND, 2000 train clips, 300 test clips, 40 epochs:

- `neural_attention`: SI-SDRi 8.942, PESQ 2.497, STOI 0.932
- `attention_online`: SI-SDRi 5.349, PESQ 2.479, STOI 0.923
- `neural_mask_gru`: SI-SDRi 9.382, PESQ 2.435, STOI 0.928
- `mask_gru_online`: SI-SDRi 5.142, PESQ 2.425, STOI 0.916
- `rl_frozen`: SI-SDRi 5.969, PESQ 2.436, STOI 0.927
- `rl_online`: SI-SDRi 3.885, PESQ 2.570, STOI 0.922

Interpretation:

- `rl_online` still gives the best PESQ, but loses SI-SDRi/STOI.
- Anchored supervised-online did not improve the supervised frozen baselines.
- Full Scenario B should not be launched yet.

Tuning run 1, `attention` only, 120 test clips:

- baseline `neural_attention`: SI-SDRi 8.889, PESQ 2.514, STOI 0.931
- best PESQ config: `attention_online_lowrank_s3_a1`
  - PESQ +0.015
  - SI-SDRi -2.270
  - not acceptable because SI-SDRi still drops too much
- best balanced config: `attention_online_scalar_s6_a1`
  - PESQ +0.012
  - SI-SDRi -0.299
  - STOI -0.001
  - this is the first promising candidate

Tuning validation run 2, `attention`, 300 test clips, scalar adapter, 6 steps, anchor 1.0:

- `neural_attention`: SI-SDRi 9.116, PESQ 2.537, STOI 0.933
- `attention_online_scalar_s6_a1`: SI-SDRi 8.759, PESQ 2.546, STOI 0.932
- delta: PESQ +0.008, SI-SDRi -0.357, STOI -0.001

Interpretation:

- The PESQ gain survived on 300 clips.
- SI-SDRi stayed within the allowed 0.5 dB tolerance.
- This is now a plausible tuned online setting.

External advice made the acceptance criteria stricter:

- Do not claim success from +0.008 PESQ alone.
- Target a guarded/trust-region result:
  - PESQ gain ideally >= +0.03
  - SI-SDRi drop <= 0.10-0.20 dB, or improvement
  - STOI/ESTOI essentially unchanged
- Next runs should test trust-region clipping before the full Scenario B paper run.

## What Happens Next

The next step is a trust-region tuning pass around the current best scalar adapter.

Suggested grid:

- base model: `attention`
- adapter: `scalar`
- anchors: `1.0,2.0`
- steps: `4,6`
- max gain deltas: `0.02,0.05,0.1`
- test clips: `300`

If this improves the PESQ/SI-SDR/STOI tradeoff, validate the best setting on the full 824-clip test set.
If it does not, add guarded multi-objective SQUIM reward and/or mask-polarization next.

## Done

- Built the SONA repo in `D:\noise cancellation`.
- Confirmed the core FFT/STFT -> per-band processing -> ISTFT/IFFT pipeline remains intact.
- Ran the first real VoiceBank-DEMAND Scenario A.
- Found the first thesis result:
  - Neural supervised models beat classical baselines clearly.
  - `rl_online` improved PESQ/LSD but lost SI-SDR/SNR/STOI versus `rl_frozen`.
  - This means the thesis is promising but not proven yet.
- Fixed the GPU cuDNN crash in SQUIM-backed online adaptation.
- Generalized `OnlineAdapter` so it can wrap supervised models, not only the RL policy.
- Added anchored online adaptation with `online_anchor_weight` to discourage drifting away from the frozen model output.
- Added Kaggle runner support for:
  - `attention_online`
  - `mask_gru_online`
  - `--online-supervised`
  - `--online-supervised-adapter`
  - `--online-anchor-weight`
- Added tests for the supervised-online anchored path.
- Verified tests:
  - Full suite: 39 passing.
  - Focused online/reward/adapter tests: 12 passing after trust-region changes.
- Committed the update:
  - `61c1dcb feat: add anchored supervised online adaptation`
- Refreshed repo upload zip:
  - `D:\noise cancellation\sona.zip`
- Created the upload-ready Kaggle notebook:
  - `D:\noise cancellation\sona_kaggle_run_all_online.ipynb`

## Do Next

1. Do not run full Scenario B yet.
2. Upload refreshed `D:\noise cancellation\sona.zip` to Kaggle.
3. Run `D:\noise cancellation\sona_kaggle_tune_online.ipynb`.
4. In Block 4, use:
   - base model: `attention`
   - adapter: `scalar`
   - anchors: `1.0,2.0`
   - steps: `4,6`
   - max gain deltas: `0.02,0.05,0.1`
   - test clips: `300`
5. Pick the setting with the best PESQ gain while SI-SDRi and STOI stay nearly unchanged.
6. Validate the winner on the full 824-clip test set.

## Decision Gate

Compare:

- `attention_online` vs `neural_attention`
- `mask_gru_online` vs `neural_mask_gru`

Good outcome:

- online variant improves PESQ/STOI/DNSMOS, or at least PESQ, while keeping SI-SDRi close.

If good:

- Run full Scenario B with larger train/test and ablations.

If not good:

- Tune anchor strength, adapter type, reward weights, and adaptation steps before launching the full run.

## Current Files To Use

- Main repo folder: `D:\noise cancellation`
- Kaggle repo zip: `D:\noise cancellation\sona.zip`
- Kaggle notebook: `D:\noise cancellation\sona_kaggle_run_all_online.ipynb`
- Tuning notebook: `D:\noise cancellation\sona_kaggle_tune_online.ipynb`
- Living status file: `D:\noise cancellation\SONA_STATUS.md`
