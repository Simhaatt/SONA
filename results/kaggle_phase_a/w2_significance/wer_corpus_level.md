# Corpus-level WER (total errors / total reference words)

Standard ASR aggregation, weighting each clip by its reference length instead of averaging per-clip rates. Avoids the zero-inflation that collapses the effective sample in the per-clip analysis. Uncertainty from a **paired** clip bootstrap (10000 replicates): each replicate resamples clip indices once and scores both methods on the same clips.

Difference is method minus baseline, so **negative is better**.

| method | split | condition | clips | words | corpus WER | baseline | diff | rel % | 95% CI | boot p |
|---|---|---|---|---|---|---|---|---|---|---|
| neural_gtcrn | noise | AirConditioner | 218 | 6926 | 0.1353 | 0.0807 | +0.0546 | +67.6 | [+0.0419, +0.0676] | 0.0001 |
| neural_gtcrn | noise | AirportAnnouncements | 91 | 2908 | 0.2191 | 0.1228 | +0.0962 | +78.4 | [+0.0696, +0.1260] | 0.0001 |
| neural_gtcrn | noise | Babble | 184 | 5818 | 0.3449 | 0.2393 | +0.1057 | +44.2 | [+0.0833, +0.1300] | 0.0001 |
| neural_gtcrn | noise | CopyMachine | 35 | 1123 | 0.4358 | 0.2475 | +0.1883 | +76.1 | [+0.1282, +0.2558] | 0.0001 |
| neural_gtcrn | noise | Munching | 180 | 5721 | 0.1221 | 0.0677 | +0.0544 | +80.4 | [+0.0364, +0.0739] | 0.0001 |
| neural_gtcrn | noise | Neighbor | 166 | 5313 | 0.3095 | 0.2752 | +0.0342 | +12.4 | [+0.0057, +0.0622] | 0.0184 |
| neural_gtcrn | noise | ShuttingDoor | 165 | 5319 | 0.0860 | 0.0442 | +0.0417 | +94.3 | [+0.0290, +0.0565] | 0.0001 |
| neural_gtcrn | noise | Typing | 44 | 1379 | 0.0661 | 0.0536 | +0.0125 | +23.2 | [-0.0082, +0.0308] | 0.2090 |
| neural_gtcrn | noise | VacuumCleaner | 17 | 530 | 0.1928 | 0.1360 | +0.0568 | +41.8 | [+0.0095, +0.1075] | 0.0160 |
| neural_gtcrn | overall | all | 1100 | 35037 | 0.2016 | 0.1375 | +0.0641 | +46.6 | [+0.0559, +0.0726] | 0.0001 |
| neural_gtcrn | snr | -5.0 | 275 | 8695 | 0.4808 | 0.3380 | +0.1428 | +42.3 | [+0.1201, +0.1658] | 0.0001 |
| neural_gtcrn | snr | 0.0 | 275 | 8840 | 0.2320 | 0.1470 | +0.0850 | +57.8 | [+0.0669, +0.1024] | 0.0001 |
| neural_gtcrn | snr | 10.0 | 275 | 8725 | 0.0306 | 0.0222 | +0.0084 | +37.9 | [+0.0042, +0.0128] | 0.0001 |
| neural_gtcrn | snr | 5.0 | 275 | 8777 | 0.0644 | 0.0439 | +0.0205 | +46.8 | [+0.0134, +0.0278] | 0.0001 |
| spectral_subtraction | noise | AirConditioner | 218 | 6926 | 0.0982 | 0.0807 | +0.0175 | +21.6 | [+0.0067, +0.0279] | 0.0014 |
| spectral_subtraction | noise | AirportAnnouncements | 91 | 2908 | 0.1887 | 0.1228 | +0.0659 | +53.6 | [+0.0465, +0.0869] | 0.0001 |
| spectral_subtraction | noise | Babble | 184 | 5818 | 0.3129 | 0.2393 | +0.0736 | +30.8 | [+0.0541, +0.0935] | 0.0001 |
| spectral_subtraction | noise | CopyMachine | 35 | 1123 | 0.2972 | 0.2475 | +0.0497 | +20.1 | [+0.0128, +0.0884] | 0.0086 |
| spectral_subtraction | noise | Munching | 180 | 5721 | 0.1214 | 0.0677 | +0.0537 | +79.4 | [+0.0386, +0.0684] | 0.0001 |
| spectral_subtraction | noise | Neighbor | 166 | 5313 | 0.3224 | 0.2752 | +0.0472 | +17.1 | [+0.0227, +0.0697] | 0.0001 |
| spectral_subtraction | noise | ShuttingDoor | 165 | 5319 | 0.0793 | 0.0442 | +0.0351 | +79.3 | [+0.0256, +0.0448] | 0.0001 |
| spectral_subtraction | noise | Typing | 44 | 1379 | 0.0997 | 0.0536 | +0.0461 | +86.0 | [+0.0201, +0.0723] | 0.0004 |
| spectral_subtraction | noise | VacuumCleaner | 17 | 530 | 0.1714 | 0.1360 | +0.0354 | +26.0 | [-0.0066, +0.0875] | 0.1156 |
| spectral_subtraction | overall | all | 1100 | 35037 | 0.1838 | 0.1375 | +0.0463 | +33.7 | [+0.0398, +0.0530] | 0.0001 |
| spectral_subtraction | snr | -5.0 | 275 | 8695 | 0.4179 | 0.3380 | +0.0800 | +23.7 | [+0.0617, +0.0983] | 0.0001 |
| spectral_subtraction | snr | 0.0 | 275 | 8840 | 0.2027 | 0.1470 | +0.0557 | +37.9 | [+0.0395, +0.0716] | 0.0001 |
| spectral_subtraction | snr | 10.0 | 275 | 8725 | 0.0392 | 0.0222 | +0.0170 | +76.7 | [+0.0121, +0.0221] | 0.0001 |
| spectral_subtraction | snr | 5.0 | 275 | 8777 | 0.0767 | 0.0439 | +0.0328 | +74.6 | [+0.0263, +0.0394] | 0.0001 |
| wiener | noise | AirConditioner | 218 | 6926 | 0.1093 | 0.0807 | +0.0285 | +35.3 | [+0.0175, +0.0399] | 0.0001 |
| wiener | noise | AirportAnnouncements | 91 | 2908 | 0.2409 | 0.1228 | +0.1181 | +96.1 | [+0.0877, +0.1511] | 0.0001 |
| wiener | noise | Babble | 184 | 5818 | 0.3884 | 0.2393 | +0.1491 | +62.3 | [+0.1253, +0.1745] | 0.0001 |
| wiener | noise | CopyMachine | 35 | 1123 | 0.3956 | 0.2475 | +0.1480 | +59.8 | [+0.0899, +0.2092] | 0.0001 |
| wiener | noise | Munching | 180 | 5721 | 0.1569 | 0.0677 | +0.0892 | +131.9 | [+0.0706, +0.1092] | 0.0001 |
| wiener | noise | Neighbor | 166 | 5313 | 0.3771 | 0.2752 | +0.1019 | +37.0 | [+0.0685, +0.1342] | 0.0001 |
| wiener | noise | ShuttingDoor | 165 | 5319 | 0.1073 | 0.0442 | +0.0630 | +142.5 | [+0.0509, +0.0758] | 0.0001 |
| wiener | noise | Typing | 44 | 1379 | 0.1584 | 0.0536 | +0.1048 | +195.4 | [+0.0725, +0.1394] | 0.0001 |
| wiener | noise | VacuumCleaner | 17 | 530 | 0.1883 | 0.1360 | +0.0523 | +38.5 | [+0.0068, +0.1006] | 0.0234 |
| wiener | overall | all | 1100 | 35037 | 0.2269 | 0.1375 | +0.0894 | +65.1 | [+0.0805, +0.0985] | 0.0001 |
| wiener | snr | -5.0 | 275 | 8695 | 0.4882 | 0.3380 | +0.1502 | +44.4 | [+0.1274, +0.1732] | 0.0001 |
| wiener | snr | 0.0 | 275 | 8840 | 0.2610 | 0.1470 | +0.1141 | +77.6 | [+0.0932, +0.1352] | 0.0001 |
| wiener | snr | 10.0 | 275 | 8725 | 0.0590 | 0.0222 | +0.0368 | +165.6 | [+0.0296, +0.0443] | 0.0001 |
| wiener | snr | 5.0 | 275 | 8777 | 0.1008 | 0.0439 | +0.0569 | +129.5 | [+0.0470, +0.0668] | 0.0001 |
