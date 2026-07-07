# SONA Mamba-GTCRN Full-824 Result

Source artifact:

- `C:\Users\hp\Downloads\sona_voicebank_mamba_gtcrn_full824.zip`

Stored artifact:

- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_full824_2026-07-07\sona_voicebank_mamba_gtcrn_full824.zip`

## Result

Full 824-clip VoiceBank validation for:

- `mamba_gtcrn_ch96_h96_b4_mw1p25_mr0p25`

Frozen Mamba-GTCRN:

- SI-SDRi 9.844146
- PESQ 2.504768
- STOI 0.927906
- ESTOI 0.830768
- LSD 6.617599
- RTF 0.012139

Strict online Mamba-GTCRN:

- SI-SDRi 9.844080
- PESQ 2.504692
- STOI 0.927911
- ESTOI 0.830766
- LSD 6.617839
- RTF 0.074449

## Comparison Against Locked Attention Full-824

- SI-SDRi +0.762146
- PESQ +0.003768
- STOI -0.004094
- ESTOI -0.003232
- LSD +0.647599

## Comparison Against Best GTCRN Full-824 Rerun

Best GTCRN reference:

- `gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 9.927043
- PESQ 2.520714
- STOI 0.929916
- ESTOI 0.832254
- LSD 6.469693

Mamba-GTCRN delta:

- SI-SDRi -0.082897
- PESQ -0.015946
- STOI -0.002010
- ESTOI -0.001486
- LSD +0.147906

## Interpretation

The first Mamba-lite GTCRN candidate did not pass the full-824 gate.

It still beats locked attention on SI-SDRi and slightly on PESQ, but it is worse than the validated
GTCRN branch on all main metrics. The 300-clip Mamba pilot was over-optimistic.

Strict online adaptation still changes almost nothing.

## Decision

Do not run MS-SNSD external validation for this Mamba-lite checkpoint.

Keep `gtcrn_ch96_h96_b4_mw1p25_mr0p25` as the current main backbone. Next Mamba step should be a
tuning/redesign round, not external validation.
