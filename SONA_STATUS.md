# SONA Status

Last updated: 2026-07-03

## Current Goal

Tune the online adaptation settings on a smaller Scenario A grid before launching the full Scenario B paper run.

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

## What Happens Next

When the Kaggle run finishes, inspect the final table and especially:

- `attention_online` vs `neural_attention`
- `mask_gru_online` vs `neural_mask_gru`

The next step is a tuning pass:

- increase/decrease `ONLINE_ANCHOR`
- try `band_bias` or `scalar` instead of `lowrank`
- change SQUIM reward weights
- reduce/increase adaptation steps

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
  - Full suite: 38 passing.
  - Focused online/reward/adapter tests: 15 passing.
- Committed the update:
  - `61c1dcb feat: add anchored supervised online adaptation`
- Refreshed repo upload zip:
  - `D:\noise cancellation\sona.zip`
- Created the upload-ready Kaggle notebook:
  - `D:\noise cancellation\sona_kaggle_run_all_online.ipynb`

## Do Next

1. Do not run full Scenario B yet.
2. Upload refreshed `D:\noise cancellation\sona.zip` to `sona-repo` as a new Kaggle dataset version.
3. Run `D:\noise cancellation\sona_kaggle_tune_online.ipynb`.
4. Start with the conservative grid already in the notebook:
   - base model: `attention`
   - adapters: `scalar,lowrank`
   - anchors: `0.2,0.5,1.0`
   - steps: `3,6`
   - test clips: `120`
5. Choose the setting with the best PESQ/STOI improvement while SI-SDRi stays close to `neural_attention`.

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
