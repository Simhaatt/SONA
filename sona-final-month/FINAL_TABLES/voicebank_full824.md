# VoiceBank-DEMAND full 824-clip test set

| label | method | si_sdr_i | pesq | stoi | estoi | lsd | rtf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Attention (baseline) | neural_attention | 9.0817 | 2.5012 | 0.9322 | 0.8344 | 5.9701 | 0.0019 |
| GTCRN + MR loss (WINNER) | neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25 | 9.9270 | 2.5207 | 0.9299 | 0.8323 | 6.4697 | 0.0061 |
| Mamba-GTCRN | neural_mamba_gtcrn_ch96_h96_b4_mw1p25_mr0p25 | 9.8441 | 2.5048 | 0.9279 | 0.8308 | 6.6176 | 0.0121 |
| SONA-G governed | neural_governed_gtcrn_ch96_h96_b4_mw1p25_mr0p25 | 9.8876 | 2.4549 | 0.9303 | 0.8308 | 6.6413 | 0.0067 |
| SONA-R residual | neural_residual_gtcrn_ch96_h96_b4_mw1p25_mr0p25 | 9.8680 | 2.5200 | 0.9292 | 0.8307 | 6.5078 | 0.0068 |
