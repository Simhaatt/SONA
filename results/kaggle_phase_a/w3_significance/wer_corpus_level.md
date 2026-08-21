# Corpus-level WER (total errors / total reference words)

Standard ASR aggregation, weighting each clip by its reference length instead of averaging per-clip rates. Avoids the zero-inflation that collapses the effective sample in the per-clip analysis. Uncertainty from a **paired** clip bootstrap (10000 replicates): each replicate resamples clip indices once and scores both methods on the same clips.

Difference is method minus baseline, so **negative is better**.

| method | split | condition | clips | words | corpus WER | baseline | diff | rel % | 95% CI | boot p |
|---|---|---|---|---|---|---|---|---|---|---|
| neural_gtcrn | noise | bus | 164 | 1219 | 0.0515 | 0.0376 | +0.0140 | +37.2 | [-0.0026, +0.0312] | 0.1008 |
| neural_gtcrn | noise | cafe | 164 | 1246 | 0.1745 | 0.1350 | +0.0395 | +29.3 | [+0.0105, +0.0707] | 0.0056 |
| neural_gtcrn | noise | living | 166 | 1209 | 0.1031 | 0.0982 | +0.0049 | +5.0 | [-0.0199, +0.0308] | 0.7154 |
| neural_gtcrn | noise | office | 164 | 1259 | 0.0430 | 0.0288 | +0.0142 | +49.1 | [+0.0026, +0.0269] | 0.0128 |
| neural_gtcrn | noise | psquare | 166 | 1274 | 0.0665 | 0.0570 | +0.0095 | +16.7 | [-0.0075, +0.0274] | 0.2850 |
| neural_gtcrn | overall | all | 824 | 6207 | 0.0876 | 0.0711 | +0.0165 | +23.1 | [+0.0069, +0.0260] | 0.0004 |
| neural_gtcrn | snr | 12.5 | 207 | 1546 | 0.0388 | 0.0416 | -0.0028 | -6.7 | [-0.0158, +0.0105] | 0.6548 |
| neural_gtcrn | snr | 17.5 | 204 | 1530 | 0.0374 | 0.0370 | +0.0004 | +1.1 | [-0.0095, +0.0110] | 0.9680 |
| neural_gtcrn | snr | 2.5 | 207 | 1556 | 0.1931 | 0.1375 | +0.0556 | +40.5 | [+0.0287, +0.0834] | 0.0001 |
| neural_gtcrn | snr | 7.5 | 206 | 1575 | 0.0800 | 0.0678 | +0.0123 | +18.1 | [-0.0070, +0.0313] | 0.2160 |
| spectral_subtraction | noise | bus | 164 | 1219 | 0.0856 | 0.0376 | +0.0481 | +128.0 | [+0.0268, +0.0701] | 0.0001 |
| spectral_subtraction | noise | cafe | 164 | 1246 | 0.2363 | 0.1350 | +0.1013 | +75.1 | [+0.0662, +0.1397] | 0.0001 |
| spectral_subtraction | noise | living | 166 | 1209 | 0.1212 | 0.0982 | +0.0230 | +23.4 | [-0.0037, +0.0501] | 0.0950 |
| spectral_subtraction | noise | office | 164 | 1259 | 0.1016 | 0.0288 | +0.0728 | +252.5 | [+0.0488, +0.0980] | 0.0001 |
| spectral_subtraction | noise | psquare | 166 | 1274 | 0.1149 | 0.0570 | +0.0579 | +101.7 | [+0.0322, +0.0850] | 0.0001 |
| spectral_subtraction | overall | all | 824 | 6207 | 0.1320 | 0.0711 | +0.0609 | +85.6 | [+0.0484, +0.0736] | 0.0001 |
| spectral_subtraction | snr | 12.5 | 207 | 1546 | 0.0986 | 0.0416 | +0.0570 | +137.1 | [+0.0366, +0.0788] | 0.0001 |
| spectral_subtraction | snr | 17.5 | 204 | 1530 | 0.0808 | 0.0370 | +0.0438 | +118.4 | [+0.0247, +0.0635] | 0.0001 |
| spectral_subtraction | snr | 2.5 | 207 | 1556 | 0.2151 | 0.1375 | +0.0776 | +56.4 | [+0.0441, +0.1134] | 0.0001 |
| spectral_subtraction | snr | 7.5 | 206 | 1575 | 0.1326 | 0.0678 | +0.0649 | +95.7 | [+0.0441, +0.0875] | 0.0001 |
| wiener | noise | bus | 164 | 1219 | 0.1310 | 0.0376 | +0.0934 | +248.7 | [+0.0661, +0.1231] | 0.0001 |
| wiener | noise | cafe | 164 | 1246 | 0.3183 | 0.1350 | +0.1833 | +135.8 | [+0.1329, +0.2358] | 0.0001 |
| wiener | noise | living | 166 | 1209 | 0.2162 | 0.0982 | +0.1180 | +120.2 | [+0.0844, +0.1543] | 0.0001 |
| wiener | noise | office | 164 | 1259 | 0.1673 | 0.0288 | +0.1385 | +480.6 | [+0.1066, +0.1719] | 0.0001 |
| wiener | noise | psquare | 166 | 1274 | 0.1698 | 0.0570 | +0.1129 | +198.2 | [+0.0820, +0.1463] | 0.0001 |
| wiener | overall | all | 824 | 6207 | 0.2005 | 0.0711 | +0.1294 | +181.9 | [+0.1133, +0.1465] | 0.0001 |
| wiener | snr | 12.5 | 207 | 1546 | 0.1408 | 0.0416 | +0.0992 | +238.5 | [+0.0716, +0.1290] | 0.0001 |
| wiener | snr | 17.5 | 204 | 1530 | 0.1370 | 0.0370 | +0.1000 | +270.4 | [+0.0730, +0.1289] | 0.0001 |
| wiener | snr | 2.5 | 207 | 1556 | 0.3128 | 0.1375 | +0.1754 | +127.5 | [+0.1337, +0.2196] | 0.0001 |
| wiener | snr | 7.5 | 206 | 1575 | 0.2100 | 0.0678 | +0.1422 | +209.8 | [+0.1135, +0.1718] | 0.0001 |
