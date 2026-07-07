# SONA GTCRN Sweep Result

Source artifact:

- `C:\Users\hp\Downloads\sona_voicebank_gtcrn_sweep.zip`

Extracted analysis folder:

- `C:\Users\hp\Documents\Codex\sona_voicebank_gtcrn_sweep_analysis_2026-07-06`

## Result

The sweep completed all 16 GTCRN configs and produced checkpoints for each config.

Online adaptation rows were identical to frozen rows under the strict guard, so this run should be
interpreted as a frozen GTCRN backbone sweep.

## Best Configs

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

Best intelligibility / lowest LSD balance:

- `neural_gtcrn_ch96_h96_b4_mw0p75`
- SI-SDRi 10.485253
- PESQ 2.561575
- STOI 0.930060
- ESTOI 0.847581
- LSD 6.748218

## Interpretation

GTCRN improved over the first GTCRN pilot and strongly beats attention on SI-SDRi. It also beats
attention on PESQ for many configs.

However, attention is still better on STOI/ESTOI/LSD. GTCRN is not final yet.

## Recommendation

Use `neural_gtcrn_ch96_h96_b4_mw0p75` as the next candidate because it is the best compromise:

- better SI-SDRi than attention
- better PESQ than attention
- best GTCRN STOI/ESTOI/LSD balance
- not the heaviest config

Next run should validate this candidate on full 824 VoiceBank test clips, then MS-SNSD external
validation if the full-test result holds.
