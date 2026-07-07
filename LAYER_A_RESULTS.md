# SONA Layer A Results

Layer A is the denoising backbone search. The locked winner is:

`gtcrn_ch96_h96_b4_mw1p25_mr0p25`

This is a compact Bark/ERB-band GTCRN trained with magnitude loss weight `1.25` and
multi-resolution spectral loss weight `0.25`.

## Final Decision

- Promote GTCRN + multi-resolution spectral loss as the Layer A backbone.
- Do not promote Mamba-GTCRN, SONA-G governed GTCRN, or SONA-R residual GTCRN.
- Use the GTCRN VoiceBank and MS-SNSD external validation results as the publishable Layer A
  evidence.
- Move next to Layer B controller, product pipeline, and realtime packaging around the locked GTCRN.

## VoiceBank Full-824 Results

| Model | SI-SDRi | PESQ | STOI | ESTOI | LSD | Decision |
|---|---:|---:|---:|---:|---:|---|
| GTCRN + MR loss | 9.927043 | 2.520714 | 0.929916 | 0.832254 | 6.469693 | Winner |
| Mamba-GTCRN | 9.844146 | 2.504768 | 0.927906 | 0.830768 | 6.617599 | Not promoted |
| SONA-G governed GTCRN | 9.887606 | 2.454897 | 0.930346 | 0.830752 | 6.641260 | Not promoted |
| SONA-R residual GTCRN | 9.868019 | 2.519997 | 0.929206 | 0.830742 | 6.507774 | Not promoted |

## MS-SNSD External Validation

GTCRN external full-1100 result:

| Model | SI-SDRi | PESQ | STOI | ESTOI | LSD |
|---|---:|---:|---:|---:|---:|
| GTCRN + MR loss | 3.153363 | 1.402709 | 0.816160 | 0.690363 | 11.643679 |

Delta versus locked attention external:

| Metric | Delta |
|---|---:|
| SI-SDRi | +0.567298 |
| PESQ | +0.079496 |
| STOI | +0.012615 |
| ESTOI | +0.026042 |
| LSD | -0.263802 |

## Interpretation

GTCRN is the best Layer A choice because it is the only architecture that combines strong
VoiceBank performance with positive MS-SNSD external validation. The extra-complexity attempts did
not improve the full-test result:

- Mamba-GTCRN added long-memory structure but regressed on all main full-824 metrics versus GTCRN.
- SONA-G tested independent conservative/aggressive heads, but lost too much PESQ and LSD.
- SONA-R tested a bounded post-GRU residual head and came close on PESQ, but still lost SI-SDRi,
  STOI, ESTOI, and LSD.

The story is now coherent: a compact GTCRN with the right spectral objective is stronger than
heavier or more adaptive variants for this Layer A setting.

## Key Artifacts

- GTCRN VoiceBank full-test:
  `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07`
- GTCRN MS-SNSD external:
  `D:\noise cancellation\results\kaggle_ms_snsd_gtcrn_external_eval_2026-07-07`
- Mamba-GTCRN full-test:
  `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_full824_2026-07-07`
- SONA-G governed-GTCRN full-test:
  `D:\noise cancellation\results\kaggle_voicebank_governed_gtcrn_full824_2026-07-07`
- SONA-R residual-GTCRN full-test:
  `D:\noise cancellation\results\kaggle_voicebank_residual_gtcrn_full824_2026-07-07`

## Next Work

Layer B starts from the locked GTCRN checkpoint and adds:

- unified controller
- mode presets
- EQ and compressor controls
- scene routing
- preference memory
- realtime pipeline
- GUI or web app
- edge packaging

RL remains planned as a later controller/governor, not a raw-mask generator.
