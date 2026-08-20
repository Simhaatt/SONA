# WER condition split -- paired significance

Baseline: `passthrough`. Mean difference is method minus baseline, so **negative is better** (lower WER). Bootstrap 95 % CI over 10000 resamples; two-sided Wilcoxon signed-rank.

`n_eff` is the number of clips whose WER actually differs between the two methods -- per-clip WER is zero-inflated, so most pairs are exact ties and the effective sample is far smaller than 824.

Multiplicity: Benjamini-Hochberg FDR is applied **within each (method, split) family** -- the tests a reader interprets together (the 5 noise types for one method, or its 4 SNR bins). `q` is the FDR-adjusted value; judge significance on `q`, not `p`.

| method | split | condition | n_eff/n | mean WER | baseline | diff | rel % | 95% CI | p | q | sig(FDR) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| neural_gtcrn | noise | bus | 20/164 | 0.0200 | 0.0344 | -0.0143 | -41.7 | [-0.0284, -0.0013] | 0.021 | 0.095 | no |
| neural_gtcrn | noise | cafe | 37/164 | 0.0907 | 0.0631 | +0.0276 | +43.8 | [+0.0040, +0.0527] | 0.038 | 0.095 | no |
| neural_gtcrn | noise | living | 28/166 | 0.0512 | 0.0440 | +0.0072 | +16.3 | [-0.0111, +0.0256] | 0.508 | 0.635 | no |
| neural_gtcrn | noise | office | 16/164 | 0.0220 | 0.0203 | +0.0018 | +8.7 | [-0.0070, +0.0101] | 0.659 | 0.659 | no |
| neural_gtcrn | noise | psquare | 28/166 | 0.0527 | 0.0513 | +0.0014 | +2.8 | [-0.0235, +0.0230] | 0.432 | 0.635 | no |
| neural_gtcrn | overall | all | 129/824 | 0.0473 | 0.0426 | +0.0047 | +11.1 | [-0.0037, +0.0130] | 0.270 | 0.270 | no |
| neural_gtcrn | snr | 12.5 | 27/207 | 0.0353 | 0.0487 | -0.0134 | -27.5 | [-0.0341, +0.0037] | 0.123 | 0.247 | no |
| neural_gtcrn | snr | 17.5 | 16/204 | 0.0202 | 0.0178 | +0.0024 | +13.7 | [-0.0095, +0.0136] | 0.568 | 0.568 | no |
| neural_gtcrn | snr | 2.5 | 66/207 | 0.0951 | 0.0711 | +0.0240 | +33.7 | [+0.0017, +0.0474] | 0.095 | 0.247 | no |
| neural_gtcrn | snr | 7.5 | 20/206 | 0.0384 | 0.0325 | +0.0059 | +18.1 | [-0.0034, +0.0156] | 0.253 | 0.338 | no |
| neural_complex_gtcrn | noise | bus | 21/164 | 0.0223 | 0.0344 | -0.0120 | -35.0 | [-0.0288, +0.0026] | 0.097 | 0.243 | no |
| neural_complex_gtcrn | noise | cafe | 45/164 | 0.0804 | 0.0631 | +0.0173 | +27.4 | [-0.0040, +0.0396] | 0.166 | 0.276 | no |
| neural_complex_gtcrn | noise | living | 27/166 | 0.0609 | 0.0440 | +0.0169 | +38.5 | [-0.0009, +0.0359] | 0.073 | 0.243 | no |
| neural_complex_gtcrn | noise | office | 12/164 | 0.0193 | 0.0203 | -0.0010 | -5.0 | [-0.0089, +0.0066] | 0.783 | 0.783 | no |
| neural_complex_gtcrn | noise | psquare | 22/166 | 0.0522 | 0.0513 | +0.0009 | +1.8 | [-0.0232, +0.0202] | 0.314 | 0.392 | no |
| neural_complex_gtcrn | overall | all | 127/824 | 0.0471 | 0.0426 | +0.0045 | +10.4 | [-0.0036, +0.0125] | 0.166 | 0.166 | no |
| neural_complex_gtcrn | snr | 12.5 | 33/207 | 0.0351 | 0.0487 | -0.0137 | -28.0 | [-0.0360, +0.0046] | 0.238 | 0.258 | no |
| neural_complex_gtcrn | snr | 17.5 | 20/204 | 0.0253 | 0.0178 | +0.0075 | +42.5 | [-0.0032, +0.0184] | 0.130 | 0.258 | no |
| neural_complex_gtcrn | snr | 2.5 | 53/207 | 0.0886 | 0.0711 | +0.0176 | +24.7 | [-0.0017, +0.0376] | 0.107 | 0.258 | no |
| neural_complex_gtcrn | snr | 7.5 | 21/206 | 0.0389 | 0.0325 | +0.0064 | +19.7 | [-0.0042, +0.0176] | 0.258 | 0.258 | no |
| spectral_subtraction | noise | bus | 28/164 | 0.0574 | 0.0344 | +0.0230 | +66.9 | [+0.0087, +0.0396] | 0.004 | 0.007 | **yes** |
| spectral_subtraction | noise | cafe | 45/164 | 0.1149 | 0.0631 | +0.0519 | +82.3 | [+0.0269, +0.0798] | 2.88e-04 | 7.20e-04 | **yes** |
| spectral_subtraction | noise | living | 33/166 | 0.0608 | 0.0440 | +0.0168 | +38.1 | [-0.0017, +0.0363] | 0.130 | 0.130 | no |
| spectral_subtraction | noise | office | 30/164 | 0.0564 | 0.0203 | +0.0361 | +178.3 | [+0.0230, +0.0505] | 1.66e-06 | 8.30e-06 | **yes** |
| spectral_subtraction | noise | psquare | 32/166 | 0.0676 | 0.0513 | +0.0163 | +31.7 | [-0.0047, +0.0358] | 0.037 | 0.046 | **yes** |
| spectral_subtraction | overall | all | 168/824 | 0.0714 | 0.0426 | +0.0288 | +67.5 | [+0.0202, +0.0377] | 6.39e-11 | 6.39e-11 | **yes** |
| spectral_subtraction | snr | 12.5 | 41/207 | 0.0673 | 0.0487 | +0.0186 | +38.3 | [+0.0018, +0.0339] | 0.004 | 0.004 | **yes** |
| spectral_subtraction | snr | 17.5 | 31/204 | 0.0459 | 0.0178 | +0.0281 | +158.2 | [+0.0164, +0.0415] | 2.66e-05 | 1.06e-04 | **yes** |
| spectral_subtraction | snr | 2.5 | 55/207 | 0.1083 | 0.0711 | +0.0372 | +52.4 | [+0.0173, +0.0594] | 0.002 | 0.002 | **yes** |
| spectral_subtraction | snr | 7.5 | 41/206 | 0.0635 | 0.0325 | +0.0310 | +95.5 | [+0.0126, +0.0507] | 0.002 | 0.002 | **yes** |
| wiener | noise | bus | 38/164 | 0.0805 | 0.0344 | +0.0462 | +134.3 | [+0.0204, +0.0765] | 0.001 | 0.001 | **yes** |
| wiener | noise | cafe | 45/164 | 0.1395 | 0.0631 | +0.0765 | +121.3 | [+0.0454, +0.1115] | 3.94e-06 | 9.84e-06 | **yes** |
| wiener | noise | living | 52/166 | 0.1029 | 0.0440 | +0.0589 | +133.9 | [+0.0294, +0.0915] | 2.14e-04 | 3.57e-04 | **yes** |
| wiener | noise | office | 46/164 | 0.0697 | 0.0203 | +0.0494 | +243.8 | [+0.0324, +0.0677] | 4.44e-07 | 2.22e-06 | **yes** |
| wiener | noise | psquare | 50/166 | 0.0923 | 0.0513 | +0.0410 | +80.0 | [+0.0159, +0.0680] | 0.005 | 0.005 | **yes** |
| wiener | overall | all | 231/824 | 0.0970 | 0.0426 | +0.0544 | +127.6 | [+0.0422, +0.0672] | 4.64e-18 | 4.64e-18 | **yes** |
| wiener | snr | 12.5 | 56/207 | 0.0873 | 0.0487 | +0.0386 | +79.3 | [+0.0189, +0.0600] | 3.01e-04 | 3.01e-04 | **yes** |
| wiener | snr | 17.5 | 42/204 | 0.0511 | 0.0178 | +0.0333 | +187.4 | [+0.0189, +0.0478] | 1.08e-05 | 1.44e-05 | **yes** |
| wiener | snr | 2.5 | 75/207 | 0.1514 | 0.0711 | +0.0803 | +113.0 | [+0.0486, +0.1146] | 4.94e-06 | 9.88e-06 | **yes** |
| wiener | snr | 7.5 | 58/206 | 0.0975 | 0.0325 | +0.0650 | +200.0 | [+0.0400, +0.0936] | 6.56e-07 | 2.63e-06 | **yes** |
