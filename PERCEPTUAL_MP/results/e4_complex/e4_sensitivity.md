# E4 -- stem exemption on a second architecture (complex-mask GTCRN)

Same backbone, same runtime, same splits; only the output representation differs (2-channel complex ratio mask, +97 parameters). Dev split p254/p256, 200 clips. Deltas are variant minus FP32.

| variant | size KB | dSI-SDR | 95% CI | mean abs dSI-SDR |
|---|---|---|---|---|
| only_stem | 2852.6 | +0.0055 | [+0.0015, +0.0096] | 0.0176 |
| only_block0_d1 | 2369.4 | +0.0004 | [-0.0023, +0.0030] | 0.0124 |
| only_block1_d2 | 2369.4 | -0.0009 | [-0.0030, +0.0010] | 0.0086 |
| only_block2_d4 | 2369.4 | +0.0001 | [-0.0018, +0.0018] | 0.0083 |
| only_block3_d8 | 2369.4 | +0.0016 | [-0.0004, +0.0037] | 0.0086 |
| only_head | 2854.4 | +0.0042 | [+0.0030, +0.0053] | 0.0067 |
| uniform_int8 | 952.2 | -0.0510 | [-0.0709, -0.0318] | 0.1056 |
| mixed_stemfp32 | 917.4 | +0.0041 | [+0.0000, +0.0080] | 0.0193 |

## Verdict

Most sensitive single layer: **only_stem**

Sensitivity order (by mean |dSI-SDR|): stem > block0_d1 > block1_d2 > block3_d8 > block2_d4 > head

Stem's share of total INT8 degradation: **92.1 %**

**stem dominant -- replicates the magnitude-model finding**
