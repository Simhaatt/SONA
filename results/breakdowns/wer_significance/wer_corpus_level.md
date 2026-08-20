# Corpus-level WER (total errors / total reference words)

Standard ASR aggregation, weighting each clip by its reference length instead of averaging per-clip rates. Avoids the zero-inflation that collapses the effective sample in the per-clip analysis. Uncertainty from a **paired** clip bootstrap (10000 replicates): each replicate resamples clip indices once and scores both methods on the same clips.

Difference is method minus baseline, so **negative is better**.

| method | split | condition | clips | words | corpus WER | baseline | diff | rel % | 95% CI | boot p |
|---|---|---|---|---|---|---|---|---|---|---|
| neural_gtcrn | noise | bus | 164 | 1214 | 0.0151 | 0.0210 | -0.0059 | -28.0 | [-0.0145, +0.0017] | 0.1424 |
| neural_gtcrn | noise | cafe | 164 | 1238 | 0.0903 | 0.0572 | +0.0331 | +57.9 | [+0.0098, +0.0591] | 0.0034 |
| neural_gtcrn | noise | living | 166 | 1207 | 0.0466 | 0.0388 | +0.0078 | +20.0 | [-0.0093, +0.0245] | 0.3690 |
| neural_gtcrn | noise | office | 164 | 1249 | 0.0202 | 0.0172 | +0.0030 | +17.3 | [-0.0042, +0.0101] | 0.4084 |
| neural_gtcrn | noise | psquare | 166 | 1275 | 0.0416 | 0.0382 | +0.0035 | +9.1 | [-0.0104, +0.0179] | 0.6212 |
| neural_gtcrn | overall | all | 824 | 6183 | 0.0428 | 0.0345 | +0.0083 | +24.1 | [+0.0014, +0.0155] | 0.0200 |
| neural_gtcrn | snr | 12.5 | 207 | 1537 | 0.0317 | 0.0351 | -0.0034 | -9.7 | [-0.0148, +0.0082] | 0.5538 |
| neural_gtcrn | snr | 17.5 | 204 | 1525 | 0.0162 | 0.0136 | +0.0026 | +19.1 | [-0.0062, +0.0106] | 0.5208 |
| neural_gtcrn | snr | 2.5 | 207 | 1555 | 0.0917 | 0.0641 | +0.0276 | +43.1 | [+0.0059, +0.0513] | 0.0104 |
| neural_gtcrn | snr | 7.5 | 206 | 1566 | 0.0310 | 0.0248 | +0.0062 | +24.9 | [-0.0003, +0.0131] | 0.0646 |
| neural_complex_gtcrn | noise | bus | 164 | 1214 | 0.0189 | 0.0210 | -0.0022 | -10.4 | [-0.0120, +0.0068] | 0.6564 |
| neural_complex_gtcrn | noise | cafe | 164 | 1238 | 0.0759 | 0.0572 | +0.0187 | +32.7 | [-0.0003, +0.0409] | 0.0550 |
| neural_complex_gtcrn | noise | living | 166 | 1207 | 0.0517 | 0.0388 | +0.0129 | +33.2 | [-0.0030, +0.0293] | 0.1040 |
| neural_complex_gtcrn | noise | office | 164 | 1249 | 0.0171 | 0.0172 | -0.0000 | -0.2 | [-0.0068, +0.0065] | 0.9866 |
| neural_complex_gtcrn | noise | psquare | 166 | 1275 | 0.0410 | 0.0382 | +0.0028 | +7.4 | [-0.0079, +0.0136] | 0.5950 |
| neural_complex_gtcrn | overall | all | 824 | 6183 | 0.0409 | 0.0345 | +0.0064 | +18.6 | [+0.0006, +0.0128] | 0.0314 |
| neural_complex_gtcrn | snr | 12.5 | 207 | 1537 | 0.0338 | 0.0351 | -0.0013 | -3.7 | [-0.0138, +0.0110] | 0.8488 |
| neural_complex_gtcrn | snr | 17.5 | 204 | 1525 | 0.0202 | 0.0136 | +0.0065 | +47.8 | [-0.0003, +0.0136] | 0.0608 |
| neural_complex_gtcrn | snr | 2.5 | 207 | 1555 | 0.0796 | 0.0641 | +0.0155 | +24.2 | [-0.0016, +0.0345] | 0.0734 |
| neural_complex_gtcrn | snr | 7.5 | 206 | 1566 | 0.0296 | 0.0248 | +0.0048 | +19.4 | [-0.0021, +0.0124] | 0.1792 |
| spectral_subtraction | noise | bus | 164 | 1214 | 0.0439 | 0.0210 | +0.0229 | +108.7 | [+0.0119, +0.0360] | 0.0001 |
| spectral_subtraction | noise | cafe | 164 | 1238 | 0.1021 | 0.0572 | +0.0450 | +78.6 | [+0.0242, +0.0679] | 0.0001 |
| spectral_subtraction | noise | living | 166 | 1207 | 0.0496 | 0.0388 | +0.0107 | +27.7 | [-0.0052, +0.0261] | 0.1870 |
| spectral_subtraction | noise | office | 164 | 1249 | 0.0520 | 0.0172 | +0.0349 | +203.0 | [+0.0225, +0.0483] | 0.0001 |
| spectral_subtraction | noise | psquare | 166 | 1275 | 0.0554 | 0.0382 | +0.0172 | +45.1 | [+0.0041, +0.0312] | 0.0096 |
| spectral_subtraction | overall | all | 824 | 6183 | 0.0607 | 0.0345 | +0.0262 | +75.9 | [+0.0193, +0.0334] | 0.0001 |
| spectral_subtraction | snr | 12.5 | 207 | 1537 | 0.0548 | 0.0351 | +0.0197 | +56.0 | [+0.0078, +0.0314] | 0.0014 |
| spectral_subtraction | snr | 17.5 | 204 | 1525 | 0.0355 | 0.0136 | +0.0219 | +160.4 | [+0.0132, +0.0316] | 0.0001 |
| spectral_subtraction | snr | 2.5 | 207 | 1555 | 0.1004 | 0.0641 | +0.0363 | +56.6 | [+0.0186, +0.0557] | 0.0001 |
| spectral_subtraction | snr | 7.5 | 206 | 1566 | 0.0515 | 0.0248 | +0.0267 | +107.8 | [+0.0125, +0.0418] | 0.0002 |
| wiener | noise | bus | 164 | 1214 | 0.0652 | 0.0210 | +0.0442 | +209.9 | [+0.0228, +0.0687] | 0.0001 |
| wiener | noise | cafe | 164 | 1238 | 0.1341 | 0.0572 | +0.0769 | +134.6 | [+0.0475, +0.1110] | 0.0001 |
| wiener | noise | living | 166 | 1207 | 0.0852 | 0.0388 | +0.0464 | +119.5 | [+0.0225, +0.0716] | 0.0002 |
| wiener | noise | office | 164 | 1249 | 0.0597 | 0.0172 | +0.0425 | +247.5 | [+0.0279, +0.0590] | 0.0001 |
| wiener | noise | psquare | 166 | 1275 | 0.0730 | 0.0382 | +0.0348 | +91.1 | [+0.0161, +0.0556] | 0.0001 |
| wiener | overall | all | 824 | 6183 | 0.0834 | 0.0345 | +0.0489 | +141.7 | [+0.0386, +0.0598] | 0.0001 |
| wiener | snr | 12.5 | 207 | 1537 | 0.0681 | 0.0351 | +0.0329 | +93.6 | [+0.0169, +0.0505] | 0.0002 |
| wiener | snr | 17.5 | 204 | 1525 | 0.0430 | 0.0136 | +0.0294 | +215.3 | [+0.0165, +0.0428] | 0.0001 |
| wiener | snr | 2.5 | 207 | 1555 | 0.1381 | 0.0641 | +0.0740 | +115.5 | [+0.0466, +0.1042] | 0.0001 |
| wiener | snr | 7.5 | 206 | 1566 | 0.0834 | 0.0248 | +0.0586 | +236.5 | [+0.0396, +0.0800] | 0.0001 |
