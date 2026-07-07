# SONA GTCRN MS-SNSD External Result

Source artifact:

- `C:\Users\hp\Downloads\sona_ms_snsd_gtcrn_external_eval.zip`

Stored artifact:

- `D:\noise cancellation\results\kaggle_ms_snsd_gtcrn_external_eval_2026-07-07\sona_ms_snsd_gtcrn_external_eval.zip`

## Result

Full 1100-clip MS-SNSD external validation for:

- `gtcrn_ch96_h96_b4_mw1p25_mr0p25`

Frozen GTCRN:

- SI-SDRi 3.153363
- PESQ 1.402709
- STOI 0.816160
- ESTOI 0.690363
- LSD 11.643679
- RTF 0.005767

Strict online GTCRN:

- identical aggregate metrics to frozen GTCRN
- RTF 0.043299

## Comparison Against Full MS-SNSD Attention Baseline

Locked full MS-SNSD attention baseline:

- SI-SDRi 2.586065
- PESQ 1.323213
- STOI 0.803545
- ESTOI 0.664321
- LSD 11.907481
- RTF 0.001115

GTCRN delta:

- SI-SDRi +0.567298
- PESQ +0.079496
- STOI +0.012615
- ESTOI +0.026042
- LSD -0.263802
- RTF +0.004652

## Interpretation

This is the strongest GTCRN result so far.

On MS-SNSD external validation, GTCRN beats locked attention on all main quality metrics:

- denoising strength
- perceptual quality
- intelligibility
- spectral distortion

The branch is now externally validated. Attention remains useful as the safe fallback, but GTCRN
should be treated as the stronger Layer B backbone for the next architecture step.

## Decision

Promote GTCRN + multi-resolution spectral loss to the main backbone candidate for Mamba integration.

Do not spend more time on strict online adaptation right now because the strict online row remains
identical to frozen GTCRN at aggregate level.
