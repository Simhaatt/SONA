# Condition-gated enhancement (R1)

Remedy for the ASR degradation. The decision is *which signal the recogniser consumes*; the human listener gets the enhanced signal either way.

| routing | corpus WER | vs always-enhance |
|---|---|---|
| always enhance (current) | 0.0428 | — |
| never enhance | 0.0345 | -0.0083 |
| **oracle, per noise type** (upper bound, not deployable) | **0.0333** | -0.0095 |
| **learned router** (leave-one-noise-type-out) | **0.0370** | -0.0058 |

Headroom between always-enhance and the oracle is +0.0095 WER. The router captures **61 %** of it, with a paired clip bootstrap 95 % CI on the router-minus-always difference of [-0.0119, -0.0003] — excludes zero.

Oracle decision per noise type: `bus`=enhance, `cafe`=passthrough, `living`=passthrough, `office`=passthrough, `psquare`=passthrough.

## Folds

| held-out noise | n | routed to passthrough | oracle choice |
|---|---|---|---|
| bus | 164 | 23 (14 %) | enhance |
| cafe | 164 | 75 (46 %) | passthrough |
| living | 166 | 143 (86 %) | passthrough |
| office | 164 | 2 (1 %) | passthrough |
| psquare | 166 | 56 (34 %) | passthrough |
