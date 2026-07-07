# SONA GTCRN Loss-Focused Full-824 Rerun Result

Source artifact:

- `C:\Users\hp\Downloads\sona_voicebank_gtcrn_loss_focus_full824 (1).zip`

Extracted analysis folder:

- `C:\Users\hp\Documents\Codex\2026-07-03\re\sona_voicebank_gtcrn_loss_focus_full824_1_analysis_2026-07-07`

## Result

This is a rerun of the same 6 loss-focused GTCRN configs. It is not byte-identical to the previous
run; metrics changed slightly, but the same best configuration family won again.

Best balanced config:

- `neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25`
- SI-SDRi 9.927043
- PESQ 2.520714
- STOI 0.929916
- ESTOI 0.832254
- LSD 6.469693

## Comparison Against Locked Full-824 Attention Layer A

Locked attention reference:

- SI-SDRi 9.082
- PESQ 2.501
- STOI 0.932
- ESTOI 0.834
- LSD 5.970

Best GTCRN delta:

- SI-SDRi +0.845043
- PESQ +0.019714
- STOI -0.002084
- ESTOI -0.001746
- LSD +0.499693

## Interpretation

The rerun confirms the previous decision:

- GTCRN with multi-resolution spectral loss beats attention on SI-SDRi and PESQ.
- It still trails attention on STOI, ESTOI, and LSD.
- `gtcrn_ch96_h96_b4_mw1p25_mr0p25` remains the best current GTCRN branch for external validation.

## Next Step

Run MS-SNSD external validation for:

- `gtcrn_ch96_h96_b4_mw1p25_mr0p25`
