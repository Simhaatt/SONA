# Condition-gated enhancement (R1)

Remedy for the ASR degradation. The decision is *which signal the recogniser consumes*; the human listener gets the enhanced signal either way.

| routing | corpus WER | vs always-enhance |
|---|---|---|
| always enhance (current) | 0.0876 | — |
| never enhance | 0.0711 | -0.0165 |
| **oracle, per noise type** (upper bound, not deployable) | **0.0711** | -0.0165 |
| **learned router** (leave-one-noise-type-out) | **0.0730** | -0.0146 |

Headroom between always-enhance and the oracle is +0.0165 WER. The router captures **89 %** of it, with a paired clip bootstrap 95 % CI on the router-minus-always difference of [-0.0215, -0.0083] — excludes zero.

Oracle decision per noise type: `bus`=passthrough, `cafe`=passthrough, `living`=passthrough, `office`=passthrough, `psquare`=passthrough.

## Folds

| held-out noise | n | routed to passthrough | oracle choice |
|---|---|---|---|
| bus | 164 | 13 (8 %) | passthrough |
| cafe | 164 | 71 (43 %) | passthrough |
| living | 166 | 0 (0 %) | passthrough |
| office | 164 | 56 (34 %) | passthrough |
| psquare | 166 | 49 (30 %) | passthrough |
