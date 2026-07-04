# SONA Status

Last updated: 2026-07-04

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

Full-test validation run 3 completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run3_trust_region\sona_online_tune_artifacts.zip`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run3_trust_region\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run3_trust_region\extracted\summary.csv`

Note: despite the folder name, this artifact contains the previous scalar setting on the full 824-clip test set, not the trust-region grid.

Tuning run 4 completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run4_trust_region_grid\sona_online_tune_artifacts.zip`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run4_trust_region_grid\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-03_run4_trust_region_grid\extracted\summary.csv`

Note: this run did not apply gain clipping because all method names ended in `_dnone`. It tested scalar adapter steps/anchors only, not the actual `max_gain_delta` trust-region grid.

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

Full-test validation run 3, `attention`, 824 test clips, scalar adapter, 6 steps, anchor 1.0:

- `neural_attention`: SI-SDRi 9.124, PESQ 2.505, STOI 0.932
- `attention_online_scalar_s6_a1`: SI-SDRi 8.876, PESQ 2.490, STOI 0.930
- delta: PESQ -0.015, SI-SDRi -0.248, STOI -0.002

Interpretation:

- The 300-clip PESQ gain did not hold on the full 824-clip test set.
- The online setting is still safe-ish on SI-SDRi, but it is not beneficial on average.
- Do not use `attention_online_scalar_s6_a1` as the headline result.

Tuning run 4, `attention`, 300 test clips, scalar adapter, no gain-delta clipping:

- `neural_attention`: SI-SDRi 9.133, PESQ 2.538, STOI 0.933
- best PESQ config: `attention_online_scalar_s6_a2_dnone`
  - SI-SDRi 8.773
  - PESQ 2.550
  - STOI 0.932
  - delta: PESQ +0.012, SI-SDRi -0.360, STOI -0.001

Interpretation:

- Anchor 2.0 with 6 steps is slightly better on PESQ than anchor 1.0, but SI-SDRi still drops about 0.36 dB.
- Because `dnone` was used, the actual trust-region clipping hypothesis is still untested.

Overnight trust-region sweep notebook added:

- `D:\noise cancellation\sona_kaggle_overnight_trust_region.ipynb`
- Purpose: run a large unattended grid for 4-6 hours on Kaggle T4.
- Grid:
  - base model: `attention`
  - adapters: `scalar,lowrank`
  - anchors: `0.5,1.0,2.0,4.0`
  - steps: `1,2,3,4,6`
  - max gain deltas: `0.01,0.02,0.05,0.1,none`
  - test clips: `300`
- The notebook ranks configs by a safety-aware score and bundles `overnight_ranked_configs.csv`.

Overnight trust-region sweep completed and ranked output was stored in:

- `D:\noise cancellation\results\kaggle_overnight_trust_region_2026-07-04\overnight_ranked_configs.csv`

Result summary, `attention`, 300 test clips, 200 online configs:

- Best safety-aware config: `attention_online_lowrank_s1_a2_d0p02`
  - PESQ 2.548, delta +0.015
  - SI-SDRi 9.009, delta -0.102
  - STOI 0.932710, delta -0.000193
  - ESTOI 0.851445, delta -0.000517
  - LSD 5.990, delta -0.017
- Best strict-safe config: `attention_online_lowrank_s2_a0p5_d0p01`
  - PESQ 2.544, delta +0.011
  - SI-SDRi 9.075, delta -0.036
  - STOI 0.932821, delta -0.000082
  - ESTOI 0.851757, delta -0.000205
  - LSD 5.978, delta -0.029
- Best raw PESQ config: `attention_online_lowrank_s1_a1_d0p05`
  - PESQ delta +0.023
  - SI-SDRi delta -0.360
  - not safe enough for the next headline candidate

Interpretation:

- Actual trust-region clipping works better than the previous `_dnone` run.
- `max_gain_delta=0.02` gives the best balanced gain but sits just outside the strict 0.10 dB SI-SDRi drop target.
- `max_gain_delta=0.01` is much safer, with smaller PESQ gain and almost no STOI/ESTOI movement.
- Next validation should use the full 824-clip VoiceBank test set on the `lowrank` trust-region neighborhood.

Full 824-clip trust-region validation completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_online_tune_2026-07-04_full824_trust_region\sona_online_tune_artifacts.zip`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-04_full824_trust_region\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-04_full824_trust_region\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_online_tune_2026-07-04_full824_trust_region\extracted\ckpt_attention\last.pt`

Full 824 result, `attention`, `lowrank`, trust-region grid:

- frozen `neural_attention`:
  - SI-SDRi 9.098
  - PESQ 2.503
  - STOI 0.932225
  - ESTOI 0.834488
  - LSD 5.973
- best online config: `attention_online_lowrank_s1_a2_d0p02`
  - SI-SDRi 9.018, delta -0.079
  - PESQ 2.513, delta +0.009
  - STOI 0.931543, delta -0.000682
  - ESTOI 0.833655, delta -0.000833
  - LSD 5.960, delta -0.012
- safest useful config: `attention_online_lowrank_s1_a2_d0p01`
  - SI-SDRi 9.073, delta -0.025
  - PESQ 2.511, delta +0.008
  - STOI 0.931799, delta -0.000426
  - ESTOI 0.834136, delta -0.000351
  - LSD 5.954, delta -0.019

Interpretation:

- The trust-region online adapter now passes the full 824-clip strict safety gate.
- The gain is real but small: PESQ improves by about +0.009 while SI-SDRi drops less than 0.10 dB and STOI/ESTOI are nearly unchanged.
- Treat this as a valid Layer A v1 online candidate, not the final headline architecture.
- Next work should improve effect size and novelty using guarded reward / mask-polarization and then the Bark-GTCRN backbone.

External advice made the acceptance criteria stricter:

- Do not claim success from +0.008 PESQ alone.
- Target a guarded/trust-region result:
  - PESQ gain ideally >= +0.03
  - SI-SDRi drop <= 0.10-0.20 dB, or improvement
  - STOI/ESTOI essentially unchanged
- Next runs should test trust-region clipping before the full Scenario B paper run.

## What Happens Next

Layer A v1 is now a valid but small online-adaptation result.

Immediate next steps:

1. Add an external-validation runner for the existing MS-SNSD manifest.
2. Evaluate frozen `neural_attention` and the best trust-region online candidate on MS-SNSD.
3. Start the next methodology step to increase effect size:
   - guarded multi-objective reward,
   - mask-polarization confidence,
   - then Bark/ERB GTCRN as the stronger backbone.

Do not present the current +0.009 PESQ as the final thesis result. Present it as evidence that trust-region online adaptation can improve perceptual quality without breaking SI-SDR/STOI on the full VoiceBank test set.

Architecture direction from external model advice:

- Do not add every idea at once.
- If trust-region tuning on `neural_attention` is still weak, the next backbone should be a Bark/ERB gain-only GTCRN-style model.
- The next architecture path is:
  1. plain Bark-GTCRN frozen gain estimator,
  2. Bark-GTCRN + trust-region online adapter,
  3. add guarded multi-objective reward and mask-polarization,
  4. only then consider Mamba as a causal noise-memory branch,
  5. RL should be a later governor for update/no-update and trust radius, not the direct gain generator.
- Working future names:
  - `SONA-HGCRN`: Hybrid Gain-Control Recurrent Network
  - `SONA-MP-GTCRN`: Mask-Polarized Trust-Region GTCRN
  - `SONA-MAD-MambaGTCRN`: memory-augmented future extension

Staged build plan toward the large architecture:

1. Finish trust-region tuning on the current `neural_attention` backbone.
2. Build a frozen Bark-GTCRN gain model as the next stronger edge-friendly backbone.
3. Add the trust-region online adapter to Bark-GTCRN.
4. Add guarded multi-objective reward so PESQ gains are only accepted when SI-SDR/STOI remain safe.
5. Add mask polarization as a reference-free confidence-restoration mechanism.
6. Add retrieval memory for safe adapter states / noise prototypes.
7. Add a causal Mamba noise-memory branch only after GTCRN + safe TTA are working.
8. Add RL last as a governor for update/no-update, trust radius, adapter strength, or memory choice, not as the direct gain generator.

Reason for staging: the full SONA-MAD/MambaGTCRN architecture is the destination, but building it all at once would be hard to debug and weak for ablation. Each module must prove its contribution before becoming part of the headline system.

External validation dataset state:

- MS-SNSD was downloaded to `D:\noise cancellation\external_data\MS-SNSD-master.zip`.
- ZIP size: 2.278 GB.
- Extracted dataset: `D:\noise cancellation\external_data\MS-SNSD`.
- Extracted size: 2.86 GB.
- Folder counts:
  - `clean_train`: 23075 wavs, 2.043 GB
  - `clean_test`: 1100 wavs, 0.391 GB
  - `noise_train`: 128 wavs, 0.353 GB
  - `noise_test`: 51 wavs, 0.071 GB
- Added `scripts\preprocess_ms_snsd.py`.
- Generated SONA external test manifest:
  - `D:\noise cancellation\external_data\ms_snsd_sona\test.csv`
  - 1100 clean/noisy pairs
  - SNRs cycled evenly across `-5,0,5,10` dB
  - generated noisy mixture folder size: 0.391 GB

Next external-validation work: add an evaluation runner that loads a trained VoiceBank model/checkpoint and scores it on `external_data\ms_snsd_sona\test.csv`, rather than retraining on MS-SNSD.

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
2. Add a script/notebook to evaluate the saved VoiceBank model on `external_data\ms_snsd_sona\test.csv`.
3. Evaluate:
   - frozen `neural_attention`
   - `attention_online_lowrank_s1_a2_d0p02`
   - `attention_online_lowrank_s1_a2_d0p01`
4. If external validation is safe, keep the trust-region adapter as Layer A v1.
5. Then implement guarded reward / mask-polarization to improve effect size before Scenario B.

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
