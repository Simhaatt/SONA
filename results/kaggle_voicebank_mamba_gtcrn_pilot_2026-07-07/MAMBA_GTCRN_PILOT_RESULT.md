# SONA Mamba-GTCRN VoiceBank Pilot Result

Source artifact:

- `C:\Users\hp\Downloads\sona_voicebank_mamba_gtcrn_pilot.zip`

Stored artifact:

- `D:\noise cancellation\results\kaggle_voicebank_mamba_gtcrn_pilot_2026-07-07\sona_voicebank_mamba_gtcrn_pilot.zip`

## Result

300-clip VoiceBank pilot for:

- `mamba_gtcrn_ch96_h96_b4_mw1p25_mr0p25`

Frozen Mamba-GTCRN:

- SI-SDRi 10.468651
- PESQ 2.567920
- STOI 0.927073
- ESTOI 0.845595
- LSD 6.807549
- RTF 0.011878

Strict online Mamba-GTCRN:

- identical aggregate quality metrics to frozen
- RTF 0.066317

## Comparison

Against 300-clip attention mask-polarity pilot:

- SI-SDRi +1.323561
- PESQ +0.029052
- STOI -0.005893
- ESTOI -0.006626
- LSD +0.797568

Against full-824 GTCRN loss-focused rerun reference:

- SI-SDRi +0.541608
- PESQ +0.047206
- STOI -0.002843
- ESTOI +0.013341
- LSD +0.337856

## Interpretation

Mamba-GTCRN is promising but not clean yet.

It improves SI-SDRi and PESQ strongly, and it improves ESTOI compared with the full-824 GTCRN
loss-focused reference. However, STOI and LSD are worse, so the Mamba block needs either full-test
confirmation or tuning before it can be promoted.

Strict online adaptation still changes no aggregate metrics.

## Decision

Run a full-824 VoiceBank validation for the same Mamba-GTCRN config before MS-SNSD external
validation.
