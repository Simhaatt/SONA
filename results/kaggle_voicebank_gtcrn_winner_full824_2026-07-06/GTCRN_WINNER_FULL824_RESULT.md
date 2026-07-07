# SONA GTCRN Winner Full-824 Result

Source artifact:

- `C:\Users\hp\Downloads\sona_voicebank_gtcrn_winner_full824.zip`

Extracted analysis folder:

- `C:\Users\hp\Documents\Codex\2026-07-03\re\sona_voicebank_gtcrn_winner_full824_analysis_2026-07-06`

## Result

Full 824-clip VoiceBank validation for selected sweep winner:

- Config: `gtcrn_ch96_h96_b4_mw0p75`
- `ch`: 96
- `hidden`: 96
- `n_blocks`: 4
- `mag_weight`: 0.75

Frozen result:

- SI-SDRi 9.959830
- PESQ 2.473765
- STOI 0.929189
- ESTOI 0.829959
- LSD 6.692406
- RTF 0.006482

Strict online result:

- SI-SDRi 9.959632
- PESQ 2.473744
- STOI 0.929190
- ESTOI 0.829948
- LSD 6.692643
- RTF 0.057075

## Comparison Against Locked Full-824 Attention Layer A

Locked full-824 attention reference:

- SI-SDRi 9.082
- PESQ 2.501
- STOI 0.932
- ESTOI 0.834
- LSD 5.970

GTCRN delta:

- SI-SDRi +0.877830
- PESQ -0.027235
- STOI -0.002811
- ESTOI -0.004041
- LSD +0.722406

## Interpretation

The selected GTCRN winner did not fully hold on the 824-clip test set.

It is clearly stronger for SI-SDRi, but loses PESQ, STOI, ESTOI, and LSD against the locked
attention Layer A fallback. Strict guarded online adaptation still changes almost nothing.

## Decision

Do not promote GTCRN as the final Layer A replacement yet.

Keep locked strict guarded attention as the safe paper baseline. Treat GTCRN as a promising Layer B
branch for denoising strength, but it needs loss/architecture tuning to recover perceptual quality.

## Next Step

Run a loss-focused GTCRN round rather than MS-SNSD external validation:

- keep candidate sizes near `ch96_h96_b4`
- increase spectral/perceptual pressure
- try `mag_weight`: 1.0, 1.25, 1.5
- optionally add multi-resolution magnitude loss or band-smoothness loss
- judge against full 824 VoiceBank before external validation
