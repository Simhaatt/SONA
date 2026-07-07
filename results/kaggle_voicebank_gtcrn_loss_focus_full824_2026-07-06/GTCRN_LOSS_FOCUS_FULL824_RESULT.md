# SONA GTCRN Loss-Focused Full-824 Result

Source artifact:

- `C:\Users\hp\Downloads\sona_voicebank_gtcrn_loss_focus_full824.zip`

Stored artifact:

- `D:\noise cancellation\results\kaggle_voicebank_gtcrn_loss_focus_full824_2026-07-06\sona_voicebank_gtcrn_loss_focus_full824.zip`

## Result

The run completed all 6 loss-focused GTCRN configs on full 824-clip VoiceBank.

Best balanced config:

- `neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 9.916320
- PESQ 2.520302
- STOI 0.929926
- ESTOI 0.832056
- LSD 6.480913
- RTF 0.006147

## Comparison Against Locked Full-824 Attention Layer A

Locked attention reference:

- SI-SDRi 9.082
- PESQ 2.501
- STOI 0.932
- ESTOI 0.834
- LSD 5.970

Best GTCRN delta:

- SI-SDRi +0.834320
- PESQ +0.019302
- STOI -0.002074
- ESTOI -0.001944
- LSD +0.510913

## Comparison Against Previous GTCRN Full-824 Winner

Previous GTCRN full-824 winner:

- `neural_gtcrn_ch96_h96_b4_mw0p75`
- SI-SDRi 9.959830
- PESQ 2.473765
- STOI 0.929189
- ESTOI 0.829959
- LSD 6.692406

Loss-focused best delta:

- SI-SDRi -0.043510
- PESQ +0.046537
- STOI +0.000737
- ESTOI +0.002097
- LSD -0.211493

## Interpretation

The multi-resolution spectral loss helped. GTCRN now beats locked attention on SI-SDRi and PESQ
on the full 824-clip VoiceBank test set.

It still does not fully beat attention on STOI, ESTOI, or LSD. This means GTCRN is now a stronger
candidate but not a clean final replacement for the locked strict guarded attention baseline.

Strict online adaptation still changes metrics only at tiny numerical scale.

## Decision

Do not claim GTCRN as final winner yet.

Best current GTCRN candidate:

- `gtcrn_ch96_h96_b4_mw1p25_mr0p25`

Recommended next step:

- Run MS-SNSD external validation for this best GTCRN checkpoint/config as a branch result.
- Keep locked strict guarded attention as the safe main baseline unless external GTCRN is clearly better.
