# SONA Status

Last updated: 2026-07-05

## Current Goal

Build the dual-mask reference-free governor on top of the externally validated GTCRN +
multi-resolution spectral-loss backbone.

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

We are moving from Layer A attention tuning to Layer B architecture improvement.

Added methodology:

- Bark-band `gtcrn` supervised backbone.
- Gated causal time-frequency convolution blocks over Bark-band features.
- Per-band causal GRU decoder.
- Same output contract as existing models: per-band gain logits.
- Same strict guarded online adapter path can now be tested on GTCRN.
- Kaggle tuner now supports GTCRN architecture/loss sweep controls:
  - `--gtcrn-chs`
  - `--gtcrn-hiddens`
  - `--gtcrn-blocks`
  - `--mag-weights`

Created upload-ready notebook:

- `D:\noise cancellation\sona_voicebank_gtcrn_pilot.ipynb`
- `D:\noise cancellation\sona_voicebank_gtcrn_sweep.ipynb`

Run purpose:

- Train multiple `neural_gtcrn_*` capacity/loss variants on VoiceBank.
- Evaluate frozen GTCRN variants on 300 VoiceBank test clips.
- Evaluate strict guarded online GTCRN variants with the locked Layer A guard.
- Save checkpoints and result CSV/MD files into the output zip.

## Latest Results

### VoiceBank Mamba-GTCRN Full-824 Validation

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_full824_2026-07-07\sona_voicebank_mamba_gtcrn_full824.zip`
- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_full824_2026-07-07\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_full824_2026-07-07\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_full824_2026-07-07\MAMBA_GTCRN_FULL824_RESULT.md`

Full 824-clip VoiceBank result:

- `neural_mamba_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 9.844146
- PESQ 2.504768
- STOI 0.927906
- ESTOI 0.830768
- LSD 6.617599
- RTF 0.012139

Delta against locked attention full-824:

- SI-SDRi +0.762146
- PESQ +0.003768
- STOI -0.004094
- ESTOI -0.003232
- LSD +0.647599

Delta against best GTCRN full-824 rerun:

- SI-SDRi -0.082897
- PESQ -0.015946
- STOI -0.002010
- ESTOI -0.001486
- LSD +0.147906

Interpretation:

- First Mamba-lite GTCRN did not pass the full-824 gate.
- It still beats locked attention on SI-SDRi and slightly on PESQ.
- It is worse than the validated GTCRN branch on all main metrics.
- Do not run MS-SNSD external validation for this Mamba-lite checkpoint.
- Keep `gtcrn_ch96_h96_b4_mw1p25_mr0p25` as the current main backbone.
- Architecture decision after reviewing external recommendations:
  - Do not switch to RWKV now.
  - Do not jump directly to the RL governor yet.
  - Pause the failed pre-GRU Mamba-lite path.
  - Next SONA architecture should keep GTCRN stable and add a dual-mask reference-free governor.

Next proposed architecture:

- `SONA-G`: Governed GTCRN.
- GTCRN encoder and GRU remain the validated backbone.
- Add two mask heads:
  - conservative speech-preserving mask
  - aggressive noise-suppressing mask
- Add a tiny reference-free governor that predicts blend weight `alpha(t,b)`.
- Final mask:
  - `M_final = alpha * M_aggressive + (1 - alpha) * M_conservative`
  - optional adaptive safety floor to prevent speech over-suppression.
- This directly targets the current weakness:
  - keep SI-SDRi/PESQ gains
  - recover STOI/ESTOI/LSD.
- Implementation added:
  - model kind `governed_gtcrn`
  - notebook `D:\noise cancellation\sona_voicebank_governed_gtcrn_full824.ipynb`

### VoiceBank SONA-G Governed-GTCRN Full-824 Validation

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_governed_gtcrn_full824_2026-07-07\sona_voicebank_governed_gtcrn_full824.zip`
- `D:\noise cancellation\results\kaggle_voicebank_governed_gtcrn_full824_2026-07-07\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_governed_gtcrn_full824_2026-07-07\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_governed_gtcrn_full824_2026-07-07\per_sample.csv`
- `D:\noise cancellation\results\kaggle_voicebank_governed_gtcrn_full824_2026-07-07\ckpt_governed_gtcrn_ch96_h96_b4_mw1p25_mr0p25\last.pt`

Full-824 VoiceBank result:

- `neural_governed_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 9.887606
- PESQ 2.454897
- STOI 0.930346
- ESTOI 0.830752
- LSD 6.641260
- RTF 0.006678

Strict online result:

- `governed_gtcrn_ch96_h96_b4_mw1p25_mr0p25_online_lowrank_s1_a2_d0p01_go0p0001`
- SI-SDRi 9.887574
- PESQ 2.454831
- STOI 0.930347
- ESTOI 0.830757
- LSD 6.641396
- RTF 0.056753

Delta vs current best GTCRN VoiceBank full-824:

- SI-SDRi -0.039437
- PESQ -0.065817
- STOI +0.000430
- ESTOI -0.001502
- LSD +0.171567

Interpretation:

- SONA-G ran successfully and produced a checkpoint.
- It fails the full-824 success gate.
- It only improves STOI by a negligible amount.
- It hurts PESQ and LSD compared with the current best GTCRN.
- Strict online adaptation again changes almost nothing while increasing RTF.
- Do not run MS-SNSD external validation for this governed checkpoint.
- Keep `gtcrn_ch96_h96_b4_mw1p25_mr0p25` as the current main architecture.
- The dual-mask governor idea is not dead, but this naive independent-head version is not the winning implementation.

Next controlled architecture:

- `SONA-R`: Residual GTCRN.
- Keep the externally validated GTCRN encoder + GRU backbone unchanged.
- Add a small bounded post-GRU logit refiner:
  - `logits = base_logits + 0.25 * tanh(refine(hidden))`
- This is intentionally smaller than Mamba or independent dual heads.
- Goal:
  - preserve GTCRN SI-SDRi/PESQ
  - reduce LSD
  - recover STOI/ESTOI without destabilizing the mask.
- Implementation added:
  - model kind `residual_gtcrn`
  - notebook `D:\noise cancellation\sona_voicebank_residual_gtcrn_full824.ipynb`

### VoiceBank SONA-R Residual-GTCRN Full-824 Validation

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_residual_gtcrn_full824_2026-07-07\sona_voicebank_residual_gtcrn_full824.zip`
- `D:\noise cancellation\results\kaggle_voicebank_residual_gtcrn_full824_2026-07-07\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_residual_gtcrn_full824_2026-07-07\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_residual_gtcrn_full824_2026-07-07\per_sample.csv`
- `D:\noise cancellation\results\kaggle_voicebank_residual_gtcrn_full824_2026-07-07\ckpt_residual_gtcrn_ch96_h96_b4_mw1p25_mr0p25\last.pt`

Full-824 VoiceBank result:

- `neural_residual_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 9.868019
- PESQ 2.519997
- STOI 0.929206
- ESTOI 0.830742
- LSD 6.507774
- RTF 0.006820

Strict online result:

- `residual_gtcrn_ch96_h96_b4_mw1p25_mr0p25_online_lowrank_s1_a2_d0p01_go0p0001`
- SI-SDRi 9.867788
- PESQ 2.519939
- STOI 0.929210
- ESTOI 0.830747
- LSD 6.507849
- RTF 0.059431

Delta vs current best GTCRN VoiceBank full-824:

- SI-SDRi -0.059024
- PESQ -0.000717
- STOI -0.000710
- ESTOI -0.001512
- LSD +0.038081

Interpretation:

- SONA-R ran successfully and produced a checkpoint.
- It is much closer than SONA-G and essentially ties GTCRN on PESQ.
- It does not beat the current best GTCRN.
- SI-SDRi, STOI, ESTOI, and LSD are all slightly worse than GTCRN.
- Strict online adaptation again changes almost nothing while increasing RTF.
- Do not promote this SONA-R checkpoint.
- Stop architecture hopping for now; lock GTCRN as the Layer A winner.

### Layer A Packaging and Layer B Controller Starter

Implemented after locking GTCRN:

- Layer A result package:
  - `D:\noise cancellation\LAYER_A_RESULTS.md`
- Layer B controller package:
  - `D:\noise cancellation\src\controller\__init__.py`
  - `D:\noise cancellation\src\controller\modes.py`
  - `D:\noise cancellation\src\controller\pipeline.py`
- Offline controlled enhancement CLI:
  - `D:\noise cancellation\scripts\enhance_controlled.py`
- Preference memory CLI:
  - `D:\noise cancellation\scripts\manage_preferences.py`
- Controller tests:
  - `D:\noise cancellation\tests\test_controller.py`

Layer B starter capabilities:

- mode presets:
  - Speech
  - Music
  - Podcast
  - Gaming
  - Hearing
  - Studio
- scene-to-mode routing:
  - speech
  - music
  - traffic
  - fan
  - crowd
  - mixed
  - unknown
- controller outputs:
  - mask/denoise strength
  - speech-preservation safety floor value
  - 8-band EQ gains
  - compressor threshold
  - compressor ratio
  - compressor makeup
  - limiter ceiling
- offline pipeline:
  - load Layer A checkpoint
  - enhance WAV
  - blend denoise strength
  - apply EQ
  - apply compressor
  - apply peak limiter
  - save output WAV
- preference memory:
  - JSON-backed scene preference store
  - remembers selected mode and controller settings per scene
  - optional memory lookup in `enhance_controlled.py`
  - optional `--remember` path to save the chosen scene settings

Note:

- The current safety floor is a controller setting recorded for the action space.
- True per-frame mask-floor enforcement should be wired into model inference later if needed.

### VoiceBank Mamba-GTCRN Pilot

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_pilot_2026-07-07\sona_voicebank_mamba_gtcrn_pilot.zip`
- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_pilot_2026-07-07\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_pilot_2026-07-07\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_pilot_2026-07-07\MAMBA_GTCRN_PILOT_RESULT.md`

300-clip VoiceBank result:

- `neural_mamba_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 10.468651
- PESQ 2.567920
- STOI 0.927073
- ESTOI 0.845595
- LSD 6.807549
- RTF 0.011878

Interpretation:

- Mamba-GTCRN improves SI-SDRi and PESQ strongly.
- Mamba-GTCRN improves ESTOI versus the full-824 GTCRN loss-focused reference.
- Mamba-GTCRN still hurts STOI and LSD.
- This is a promising Mamba signal but not clean enough to promote before full-824 validation.
- Strict online still changes no aggregate metrics.

### MS-SNSD GTCRN External Validation

Artifact stored in:

- `D:\noise cancellation\results\kaggle_ms_snsd_gtcrn_external_eval_2026-07-07\sona_ms_snsd_gtcrn_external_eval.zip`
- `D:\noise cancellation\results\kaggle_ms_snsd_gtcrn_external_eval_2026-07-07\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_ms_snsd_gtcrn_external_eval_2026-07-07\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_ms_snsd_gtcrn_external_eval_2026-07-07\GTCRN_EXTERNAL_MS_SNSD_RESULT.md`

Full 1100-clip MS-SNSD external result for:

- `gtcrn_ch96_h96_b4_mw1p25_mr0p25`

Frozen GTCRN:

- SI-SDRi 3.153363
- PESQ 1.402709
- STOI 0.816160
- ESTOI 0.690363
- LSD 11.643679
- RTF 0.005767

Delta against full MS-SNSD locked attention baseline:

- SI-SDRi +0.567298
- PESQ +0.079496
- STOI +0.012615
- ESTOI +0.026042
- LSD -0.263802
- RTF +0.004652

Strict online GTCRN:

- identical aggregate quality metrics to frozen GTCRN
- RTF 0.043299

Interpretation:

- This is the strongest GTCRN result so far.
- GTCRN + multi-resolution spectral loss beats attention on all main MS-SNSD external metrics.
- GTCRN is now externally validated as the main backbone candidate.
- Locked attention remains a useful safe fallback and ablation baseline.
- Stop spending time on strict online adaptation for now; the next meaningful step is Mamba/state-space memory.

Mamba/state-space methodology added:

- `mamba_gtcrn` model registered.
- Uses GTCRN gated TF encoder plus a causal Mamba-lite selective state-space memory block.
- Avoids custom CUDA dependency so Kaggle can run reliably.
- First pilot notebook:
  - `D:\noise cancellation\sona_voicebank_mamba_gtcrn_pilot.ipynb`

### VoiceBank GTCRN Loss-Focused Full-824 Sweep

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_2026-07-06\sona_voicebank_gtcrn_loss_focus_full824.zip`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_2026-07-06\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_2026-07-06\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_2026-07-06\ranking_frozen_loss_focus.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_2026-07-06\GTCRN_LOSS_FOCUS_FULL824_RESULT.md`

Rerun artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07\sona_voicebank_gtcrn_loss_focus_full824_1.zip`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07\ranking_frozen_loss_focus_1.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07\GTCRN_LOSS_FOCUS_FULL824_RERUN_RESULT.md`

Result:

- All 6 loss-focused GTCRN configs completed on full 824-clip VoiceBank.
- Multi-resolution spectral loss helped PESQ/LSD versus the previous GTCRN full-824 winner.

Best balanced config:

- `neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 9.916320
- PESQ 2.520302
- STOI 0.929926
- ESTOI 0.832056
- LSD 6.480913
- RTF 0.006147

Delta against locked full-824 attention Layer A:

- SI-SDRi +0.834320
- PESQ +0.019302
- STOI -0.002074
- ESTOI -0.001944
- LSD +0.510913

Delta against previous full-824 GTCRN winner:

- SI-SDRi -0.043510
- PESQ +0.046537
- STOI +0.000737
- ESTOI +0.002097
- LSD -0.211493

Interpretation:

- GTCRN now beats locked attention on full VoiceBank SI-SDRi and PESQ.
- GTCRN still does not fully beat attention on STOI, ESTOI, or LSD.
- This is strong enough to external-validate as a branch result, but not enough to replace the safe attention baseline yet.
- Strict guarded online adaptation still changes metrics only at tiny numerical scale.

Rerun confirmation:

- Same best family won again:
  - `neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- Rerun metrics:
  - SI-SDRi 9.927043
  - PESQ 2.520714
  - STOI 0.929916
  - ESTOI 0.832254
  - LSD 6.469693
- Rerun delta against locked full-824 attention:
  - SI-SDRi +0.845043
  - PESQ +0.019714
  - STOI -0.002084
  - ESTOI -0.001746
  - LSD +0.499693
- This strengthens the conclusion that GTCRN + multi-resolution spectral loss consistently improves SI-SDRi/PESQ but still trails attention on intelligibility/spectral distortion.

External validation tooling added:

- `scripts\eval_external_ms_snsd.py` now names methods from checkpoint config or `--method-name`.
- GTCRN external notebook:
  - `D:\noise cancellation\sona_ms_snsd_gtcrn_external_eval.ipynb`
- Kaggle-safe GTCRN checkpoint zip:
  - `D:\noise cancellation\sona_gtcrn_loss_focus_checkpoint.zip`

### VoiceBank GTCRN Winner Full-824 Validation

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_winner_full824_2026-07-06\sona_voicebank_gtcrn_winner_full824.zip`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_winner_full824_2026-07-06\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_winner_full824_2026-07-06\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_winner_full824_2026-07-06\GTCRN_WINNER_FULL824_RESULT.md`

Config:

- `gtcrn_ch96_h96_b4_mw0p75`
- `ch`: 96
- `hidden`: 96
- `n_blocks`: 4
- `mag_weight`: 0.75

Frozen full-824 result:

- SI-SDRi 9.959830
- PESQ 2.473765
- STOI 0.929189
- ESTOI 0.829959
- LSD 6.692406
- RTF 0.006482

Strict online full-824 result:

- SI-SDRi 9.959632
- PESQ 2.473744
- STOI 0.929190
- ESTOI 0.829948
- LSD 6.692643
- RTF 0.057075

Delta against locked full-824 attention Layer A:

- SI-SDRi +0.877830
- PESQ -0.027235
- STOI -0.002811
- ESTOI -0.004041
- LSD +0.722406

Interpretation:

- GTCRN clearly improves SI-SDRi.
- GTCRN does not beat locked attention on PESQ, STOI, ESTOI, or LSD on the full 824 test set.
- Strict guarded online still changes almost nothing.
- Do not move to MS-SNSD yet.
- Next methodology is loss-focused GTCRN tuning with stronger spectral pressure.

New methodology added:

- Multi-resolution log-magnitude loss for supervised training.
- Kaggle tuner exposes `--mr-mag-weights`.
- New notebook:
  - `D:\noise cancellation\sona_voicebank_gtcrn_loss_focus_full824.ipynb`

### VoiceBank GTCRN Capacity/Loss Sweep

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_sweep_2026-07-06\sona_voicebank_gtcrn_sweep.zip`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_sweep_2026-07-06\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_sweep_2026-07-06\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_sweep_2026-07-06\ranking_frozen_gtcrn.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_sweep_2026-07-06\GTCRN_SWEEP_RESULT.md`

Result:

- All 16 GTCRN configs completed.
- All 16 checkpoint folders were present.
- Strict guarded online rows were identical to frozen rows, so this run should be interpreted as a frozen GTCRN backbone sweep.

Best PESQ:

- `neural_gtcrn_ch64_h128_b4_mw0p5`
- SI-SDRi 10.323924
- PESQ 2.616119
- STOI 0.926852
- ESTOI 0.844060
- LSD 7.181668

Best SI-SDRi:

- `neural_gtcrn_ch64_h96_b5_mw0p5`
- SI-SDRi 10.708559
- PESQ 2.553011
- STOI 0.927991
- ESTOI 0.847378
- LSD 6.942425

Selected balanced winner:

- `neural_gtcrn_ch96_h96_b4_mw0p75`
- SI-SDRi 10.485253
- PESQ 2.561575
- STOI 0.930060
- ESTOI 0.847581
- LSD 6.748218

Interpretation:

- GTCRN strongly beats attention on SI-SDRi and beats attention on PESQ.
- Attention is still better on STOI/ESTOI/LSD.
- The selected winner is the best compromise because it has the best GTCRN STOI/ESTOI/LSD balance while keeping strong SI-SDRi/PESQ.
- Next gate is full 824-clip VoiceBank validation for `ch96_h96_b4_mw0p75`.

### VoiceBank GTCRN Layer B Pilot

Artifact stored in:

- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_pilot_2026-07-05\sona_voicebank_gtcrn_pilot.zip`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_pilot_2026-07-05\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_pilot_2026-07-05\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_pilot_2026-07-05\extracted\ckpt_gtcrn\last.pt`

Result on 300 VoiceBank test clips:

- `neural_gtcrn`:
  - SI-SDRi 10.250210
  - PESQ 2.549814
  - STOI 0.929735
  - ESTOI 0.845171
  - LSD 6.789342
  - RTF 0.005500
- `gtcrn_online_lowrank_s1_a2_d0p01_go0p0001`:
  - identical aggregate metrics to frozen GTCRN
  - RTF 0.062614

Comparison against the 300-clip attention mask-polarity training pilot:

- GTCRN improves SI-SDRi strongly:
  - 10.250210 vs 9.145090
  - delta +1.105120
- GTCRN improves PESQ slightly:
  - 2.549814 vs 2.538868
  - delta +0.010946
- GTCRN loses STOI:
  - 0.929735 vs 0.932966
  - delta -0.003231
- GTCRN loses ESTOI:
  - 0.845171 vs 0.852221
  - delta -0.007050
- GTCRN has worse LSD:
  - 6.789342 vs 6.009981
  - delta +0.779361

Interpretation:

- Bark-band GTCRN is promising because it gives a large SI-SDRi gain and a small PESQ gain.
- It is not final-best because intelligibility and spectral distortion are worse than attention.
- Strict online GTCRN did not change aggregate metrics under the current guard.
- Next gain should come from GTCRN capacity/loss tuning, not online adaptation.

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

External-validation tooling added:

- `D:\noise cancellation\scripts\eval_external_ms_snsd.py`
- `D:\noise cancellation\sona_ms_snsd_external_eval.ipynb`

Purpose:

- Load the saved VoiceBank `neural_attention` checkpoint from:
  - `D:\noise cancellation\results\kaggle_online_tune_2026-07-04_full824_trust_region\extracted\ckpt_attention\last.pt`
- Score it on:
  - `D:\noise cancellation\external_data\ms_snsd_sona\test.csv`
- Compare:
  - frozen `neural_attention`
  - `attention_online_lowrank_s1_a2_d0p01`
  - `attention_online_lowrank_s1_a2_d0p02`

Smoke test:

- Ran 2 MS-SNSD samples locally on CPU.
- Checkpoint load, manifest load, frozen eval, and online eval all work.
- Local Python lacks PESQ, so full metric external validation should run in Kaggle/Python 3.11 or another env where `pesq` installs.

MS-SNSD external validation completed on Kaggle, but the session output zip was lost. Main summary was recovered from the downloaded notebook outputs and stored in:

- `D:\noise cancellation\results\kaggle_ms_snsd_external_2026-07-04_recovered\summary.md`
- `D:\noise cancellation\results\kaggle_ms_snsd_external_2026-07-04_recovered\summary.csv`
- `D:\noise cancellation\results\kaggle_ms_snsd_external_2026-07-04_recovered\recovered_notebook_output.txt`
- `D:\noise cancellation\results\kaggle_ms_snsd_external_2026-07-04_recovered\kaggle-benchmark-nc-ipynb.ipynb`

External validation result, MS-SNSD, 1100 clips:

- frozen `neural_attention`:
  - SI-SDRi 2.586
  - PESQ 1.323
  - STOI 0.803545
  - ESTOI 0.664321
  - LSD 11.907
- `attention_online_lowrank_s1_a2_d0p01`:
  - SI-SDRi 2.600, delta +0.014
  - PESQ 1.314, delta -0.009
  - STOI 0.803181, delta -0.000364
  - ESTOI 0.663548, delta -0.000773
  - LSD 11.968, delta +0.061
- `attention_online_lowrank_s1_a2_d0p02`:
  - SI-SDRi 2.609, delta +0.023
  - PESQ 1.307, delta -0.016
  - STOI 0.802818, delta -0.000726
  - ESTOI 0.662818, delta -0.001503
  - LSD 12.017, delta +0.110

Interpretation:

- External validation did not reproduce the VoiceBank PESQ gain.
- Online adaptation is structurally safe on SI-SDR/STOI but hurts PESQ/LSD on MS-SNSD.
- Do not lock the current trust-region adapter as a final Layer A result.
- Use this as evidence that the current adapter is dataset-sensitive.
- Next methodology should add an external-safety gate / guarded reward before claiming online adaptation generalization.

Guarded online-update methodology added:

- `OnlineAdapter` now supports post-adaptation accept/reject guards:
  - `guard_min_reward_delta`
  - `guard_max_output_delta`
- If the adapted gain fails the guard, the adapter returns the frozen model gain for that clip.
- `scripts\kaggle_tune_online.py` exposes:
  - `--guard-min-reward-deltas`
  - `--guard-max-output-deltas`
- `scripts\eval_external_ms_snsd.py` exposes the same guard sweep knobs.
- Added upload-ready guarded external pilot notebook:
  - `D:\noise cancellation\sona_ms_snsd_guarded_eval.ipynb`
- Verified tests:
  - full local suite: 40 passed.

Next guarded pilot:

- Dataset: MS-SNSD external validation.
- Limit: 300 clips first, not full 1100.
- Candidates:
  - `max_gain_delta`: `0.01,0.02`
  - `guard_max_output_delta`: `none,0.0005,0.001,0.002`
  - `guard_min_reward_delta`: `none`
- If a guarded setting prevents PESQ/LSD loss on MS-SNSD, rerun the same setting on full MS-SNSD and then VoiceBank.

Guarded MS-SNSD pilot completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_ms_snsd_guarded_2026-07-04_pilot300\sona_ms_snsd_external_eval.zip`
- `D:\noise cancellation\results\kaggle_ms_snsd_guarded_2026-07-04_pilot300\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_ms_snsd_guarded_2026-07-04_pilot300\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_ms_snsd_guarded_2026-07-04_pilot300\extracted\per_sample.csv`

Guarded pilot result, MS-SNSD, 300 clips:

- frozen `neural_attention`:
  - SI-SDRi 2.645
  - PESQ 1.317
  - STOI 0.803395
  - ESTOI 0.665840
  - LSD 11.947
- least bad guarded config: `attention_online_lowrank_s1_a2_d0p02_go0p0005`
  - SI-SDRi 2.656, delta +0.010
  - PESQ 1.313, delta -0.004
  - STOI 0.803348, delta -0.000048
  - ESTOI 0.665662, delta -0.000178
  - LSD 11.957, delta +0.009
- no guarded config in this pilot achieved PESQ >= frozen and LSD <= frozen.

Interpretation:

- The output-drift guard reduces external damage but does not solve the external PESQ/LSD failure.
- Do not run full 1100 guarded MS-SNSD with this exact grid.
- The next step should be a stricter/no-update baseline and/or a better guard signal, then mask-polarization. The current simple output-drift guard alone is not enough.

Strict guarded MS-SNSD pilot notebook added:

- `D:\noise cancellation\sona_ms_snsd_strict_guard_eval.ipynb`

Purpose:

- Run a second 300-clip MS-SNSD pilot before any full 1100 run.
- Include explicit no-update rows with `STEPS = "0,1"`.
- Test stricter output-drift guards:
  - `guard_max_output_delta = "0.0,0.0001,0.00025,0.0005"`
- Keep:
  - `max_gain_delta = "0.01,0.02"`
  - `guard_min_reward_delta = "none"`

Decision rule:

- Only continue to full 1100 if at least one strict guarded setting has PESQ >= frozen and LSD <= frozen on the 300-clip pilot.

Strict guarded MS-SNSD pilot completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_ms_snsd_strict_guard_2026-07-04_pilot300\sona_ms_snsd_external_eval.zip`
- `D:\noise cancellation\results\kaggle_ms_snsd_strict_guard_2026-07-04_pilot300\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_ms_snsd_strict_guard_2026-07-04_pilot300\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_ms_snsd_strict_guard_2026-07-04_pilot300\extracted\per_sample.csv`

Strict guarded pilot result, MS-SNSD, 300 clips:

- frozen `neural_attention`:
  - SI-SDRi 2.645365
  - PESQ 1.316965
  - STOI 0.803395
  - ESTOI 0.665840
  - LSD 11.947144
- best strict guarded config: `attention_online_lowrank_s1_a2_d0p01_go0p0001`
  - SI-SDRi 2.646190, delta +0.000825
  - PESQ 1.316994, delta +0.000029
  - STOI 0.803395, delta -0.00000006
  - ESTOI 0.665841, delta +0.00000086
  - LSD 11.946880, delta -0.000264
- second strict guarded pass: `attention_online_lowrank_s1_a2_d0p02_go0p00025`
  - SI-SDRi 2.645878, delta +0.000513
  - PESQ 1.316985, delta +0.000020
  - STOI 0.803396, delta +0.00000031
  - ESTOI 0.665841, delta +0.00000116
  - LSD 11.947130, delta -0.000014

Interpretation:

- Strict output guard can prevent MS-SNSD degradation, but the effect size is essentially zero.
- This is useful as a safety/fallback mechanism, not a strong enhancement result.
- Do not claim it as a major online-adaptation win.
- Next full MS-SNSD run should test only the two strict pass configs above, not the whole grid.

Overnight strict-guard sweep notebook added:

- `D:\noise cancellation\sona_ms_snsd_overnight_strict_guard_sweep.ipynb`

Purpose:

- Run a long full-1100 MS-SNSD robustness sweep overnight.
- No retraining; loads the saved VoiceBank `ckpt_attention\last.pt`.
- Grid:
  - `LIMIT = 0`
  - `ADAPTER = "lowrank"`
  - `ANCHORS = "2.0"`
  - `STEPS = "1"`
  - `MAX_GAIN_DELTAS = "0.005,0.01,0.02,0.03,0.05"`
  - `GUARD_MAX_OUTPUT_DELTAS = "0.0,0.00005,0.0001,0.00025,0.0005"`
  - `GUARD_MIN_REWARD_DELTAS = "none"`
- Total: 1 frozen baseline + 25 online variants.
- Expected runtime on Kaggle T4: roughly 8-11 hours.

Overnight strict-guard sweep completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_ms_snsd_overnight_strict_guard_sweep_2026-07-05\sona_ms_snsd_overnight_strict_guard_sweep.zip`
- `D:\noise cancellation\results\kaggle_ms_snsd_overnight_strict_guard_sweep_2026-07-05\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_ms_snsd_overnight_strict_guard_sweep_2026-07-05\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_ms_snsd_overnight_strict_guard_sweep_2026-07-05\extracted\per_sample.csv`

Full MS-SNSD overnight result, 1100 clips:

- frozen `neural_attention`:
  - SI-SDRi 2.586065
  - PESQ 1.323213
  - STOI 0.803545
  - ESTOI 0.664321
  - LSD 11.907481
- best non-trivial passing config: `attention_online_lowrank_s1_a2_d0p01_go0p0001`
  - SI-SDRi 2.586486, delta +0.000421
  - PESQ 1.323278, delta +0.000065
  - STOI 0.803532, delta -0.000013
  - ESTOI 0.664307, delta -0.000014
  - LSD 11.907373, delta -0.000108

Interpretation:

- The strict guard is full-MS-SNSD non-degrading.
- Effect size is essentially zero, so this is a safety/fallback result rather than a strong enhancement result.
- This guard can be considered externally safe enough to test on full VoiceBank.

VoiceBank strict-guard full-test notebook added:

- `D:\noise cancellation\sona_voicebank_strict_guard_full824.ipynb`
- Purpose: train `attention` once and evaluate full 824 VoiceBank clips with:
  - adapter: `lowrank`
  - steps: `1`
  - anchor: `2.0`
  - max gain delta: `0.01`
  - guard max output delta: `0.0001`

VoiceBank strict-guard full-test completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_voicebank_strict_guard_full824_2026-07-05\sona_voicebank_strict_guard_full824.zip`
- `D:\noise cancellation\results\kaggle_voicebank_strict_guard_full824_2026-07-05\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_strict_guard_full824_2026-07-05\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_strict_guard_full824_2026-07-05\extracted\per_sample.csv`
- `D:\noise cancellation\results\kaggle_voicebank_strict_guard_full824_2026-07-05\extracted\ckpt_attention\last.pt`

Full VoiceBank strict-guard result, 824 clips:

- frozen `neural_attention`:
  - SI-SDRi 9.081744
  - PESQ 2.501244
  - STOI 0.932215
  - ESTOI 0.834390
  - LSD 5.970064
- strict guarded online: `attention_online_lowrank_s1_a2_d0p01_go0p0001`
  - SI-SDRi 9.079047, delta -0.002697
  - PESQ 2.503420, delta +0.002175
  - STOI 0.932198, delta -0.000017
  - ESTOI 0.834362, delta -0.000028
  - LSD 5.962981, delta -0.007083

Layer A v1 decision:

- Strict guarded Layer A v1 is now locked as the safe fallback foundation.
- It is non-degrading on full MS-SNSD and positive on full VoiceBank PESQ/LSD.
- The effect size is still tiny, so this is not the final thesis-strength architecture.
- Next methodology should move to Layer B: mask-polarization confidence gating, then Bark/ERB GTCRN.

Layer B mask-polarization confidence gate added:

- `OnlineAdapter` now supports:
  - `guard_min_mask_polarity`
- Mask polarity is a reference-free confidence proxy:
  - `0.0` means an uncertain gain mask near `0.5`
  - `1.0` means a near-binary/confident mask near `0` or `1`
- If the adapted mask polarity is below the threshold, the adapter falls back to the frozen gain.
- `scripts\kaggle_tune_online.py` exposes:
  - `--guard-min-mask-polarities`
- `scripts\eval_external_ms_snsd.py` exposes the same knob.
- Added pilot notebook:
  - `D:\noise cancellation\sona_ms_snsd_mask_polarity_pilot.ipynb`
- Verified tests:
  - full local suite: 42 passed.

Mask-polarity confidence pilot completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_ms_snsd_mask_polarity_pilot_2026-07-05\sona_ms_snsd_mask_polarity_pilot.zip`
- `D:\noise cancellation\results\kaggle_ms_snsd_mask_polarity_pilot_2026-07-05\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_ms_snsd_mask_polarity_pilot_2026-07-05\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_ms_snsd_mask_polarity_pilot_2026-07-05\extracted\per_sample.csv`

Pilot result, MS-SNSD, 300 clips:

- Best passing polarity rows were numerically identical to the strict safe fallback:
  - `attention_online_lowrank_s1_a2_d0p01_go0p0001_gp0p2`
  - PESQ delta +0.000029
  - SI-SDRi delta +0.000825
  - STOI delta approximately 0
  - ESTOI delta approximately 0
  - LSD delta -0.000264
- Polarity thresholds did not create a stronger candidate in this inference-only form.

Interpretation:

- Inference-only mask-polarity gating mostly behaves like the existing strict guard.
- It is safe, but it does not improve effect size.
- Next Layer B step should be a training-time mask-polarization loss or confidence objective, not just an inference threshold.

Training-time mask-polarization objective added:

- Added `mask_polarity_loss` in `src\train\losses.py`.
- Supervised training now accepts:
  - `mask_polarity_weight`
  - `mask_polarity_target`
- `scripts\kaggle_tune_online.py` exposes:
  - `--mask-polarity-weight`
  - `--mask-polarity-target`
- Added pilot notebook:
  - `D:\noise cancellation\sona_voicebank_mask_polarity_train_pilot.ipynb`
- Pilot settings:
  - train on VoiceBank only
  - `LIMIT_TRAIN = 2000`
  - `LIMIT_TEST = 300`
  - `MASK_POLARITY_WEIGHT = 0.03`
  - `MASK_POLARITY_TARGET = 0.30`
  - strict guard still enabled with `max_gain_delta=0.01`, `guard_max_output_delta=0.0001`
- Verified tests:
  - full local suite: 43 passed.

VoiceBank mask-polarity training pilot completed and artifacts were stored in:

- `D:\noise cancellation\results\kaggle_voicebank_mask_polarity_train_pilot_2026-07-05\sona_voicebank_mask_polarity_train_pilot.zip`
- `D:\noise cancellation\results\kaggle_voicebank_mask_polarity_train_pilot_2026-07-05\extracted\summary.md`
- `D:\noise cancellation\results\kaggle_voicebank_mask_polarity_train_pilot_2026-07-05\extracted\summary.csv`
- `D:\noise cancellation\results\kaggle_voicebank_mask_polarity_train_pilot_2026-07-05\extracted\per_sample.csv`
- `D:\noise cancellation\results\kaggle_voicebank_mask_polarity_train_pilot_2026-07-05\extracted\ckpt_attention\last.pt`

Pilot result, VoiceBank, 300 clips:

- polarized frozen `neural_attention`:
  - SI-SDRi 9.145090
  - PESQ 2.538868
  - STOI 0.932966
  - ESTOI 0.852221
  - LSD 6.009981
- best guarded online: `attention_online_lowrank_s1_a2_d0p01_go0p0001_gp0p2`
  - SI-SDRi 9.141361, delta -0.003729
  - PESQ 2.540431, delta +0.001563
  - STOI 0.932942, delta -0.000023
  - ESTOI 0.852209, delta -0.000012
  - LSD 6.006973, delta -0.003008

Interpretation:

- Training-time mask-polarization is safe on the VoiceBank pilot.
- It does not materially improve effect size over the locked strict guarded Layer A fallback.
- Do not over-tune the attention backbone further; move to Bark/ERB GTCRN for the stronger Layer B backbone.

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

1. Keep `gtcrn_ch96_h96_b4_mw1p25_mr0p25` as the current main backbone.
2. Do not promote the current `governed_gtcrn` checkpoint.
3. Do not promote the current `residual_gtcrn` checkpoint.
4. Do not run MS-SNSD external validation for SONA-G or SONA-R.
5. Treat GTCRN + multi-resolution spectral loss as the locked Layer A winner.
6. Use GTCRN VoiceBank + MS-SNSD external results as the main publishable Layer A evidence.
7. Test preference memory with several scene/mode choices and real noisy audio.
8. Add a simple scene classifier and route its labels into the controller.
9. Build the first GUI/web demo around the offline pipeline.
10. Then move to realtime mic/speaker streaming.
11. RL governor remains postponed until the action space is stable and clearly improves over strict online adaptation.

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
- External validation notebook: `D:\noise cancellation\sona_ms_snsd_external_eval.ipynb`
- GTCRN external validation notebook: `D:\noise cancellation\sona_ms_snsd_gtcrn_external_eval.ipynb`
- GTCRN loss-focused checkpoint zip: `D:\noise cancellation\sona_gtcrn_loss_focus_checkpoint.zip`
- Guarded external pilot notebook: `D:\noise cancellation\sona_ms_snsd_guarded_eval.ipynb`
- Strict guarded external pilot notebook: `D:\noise cancellation\sona_ms_snsd_strict_guard_eval.ipynb`
- Overnight strict-guard sweep notebook: `D:\noise cancellation\sona_ms_snsd_overnight_strict_guard_sweep.ipynb`
- VoiceBank strict-guard full-test notebook: `D:\noise cancellation\sona_voicebank_strict_guard_full824.ipynb`
- Mask-polarity pilot notebook: `D:\noise cancellation\sona_ms_snsd_mask_polarity_pilot.ipynb`
- VoiceBank mask-polarity training pilot notebook: `D:\noise cancellation\sona_voicebank_mask_polarity_train_pilot.ipynb`
- VoiceBank GTCRN pilot notebook: `D:\noise cancellation\sona_voicebank_gtcrn_pilot.ipynb`
- VoiceBank GTCRN sweep notebook: `D:\noise cancellation\sona_voicebank_gtcrn_sweep.ipynb`
- VoiceBank GTCRN winner full-test notebook: `D:\noise cancellation\sona_voicebank_gtcrn_winner_full824.ipynb`
- VoiceBank GTCRN loss-focused full-test notebook: `D:\noise cancellation\sona_voicebank_gtcrn_loss_focus_full824.ipynb`
- VoiceBank Mamba-GTCRN pilot notebook: `D:\noise cancellation\sona_voicebank_mamba_gtcrn_pilot.ipynb`
- VoiceBank Mamba-GTCRN full-824 notebook: `D:\noise cancellation\sona_voicebank_mamba_gtcrn_full824.ipynb`
- VoiceBank SONA-G governed-GTCRN full-824 notebook: `D:\noise cancellation\sona_voicebank_governed_gtcrn_full824.ipynb`
- VoiceBank SONA-R residual-GTCRN full-824 notebook: `D:\noise cancellation\sona_voicebank_residual_gtcrn_full824.ipynb`
- Layer A results package: `D:\noise cancellation\LAYER_A_RESULTS.md`
- Layer B controller package: `D:\noise cancellation\src\controller`
- Offline controlled enhancement CLI: `D:\noise cancellation\scripts\enhance_controlled.py`
- Preference memory CLI: `D:\noise cancellation\scripts\manage_preferences.py`
- Living status file: `D:\noise cancellation\SONA_STATUS.md`
