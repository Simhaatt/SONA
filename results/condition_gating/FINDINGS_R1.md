# R1 — condition-gated enhancement: findings

Remedy for the ASR negative result. Script: `scripts/condition_gated_enhancement.py`.
Corpus-level WER (Σ errors / Σ reference words) on the 824-clip VoiceBank test set.

## Result

| routing | corpus WER | vs always-enhance |
|---|---|---|
| always enhance (current behaviour) | 0.0428 | — |
| **never enhance** | **0.0345** | **−0.0083** |
| oracle, per noise type (upper bound, not deployable) | 0.0333 | −0.0095 |
| learned router, leave-one-noise-type-out | 0.0370 | −0.0058 |

The router captures **61 % of the oracle headroom**, and its improvement over always-enhance is
statistically supported: paired clip bootstrap 95 % CI **[−0.0119, −0.0003]**, excluding zero.

Per-fold behaviour is coherent — the router was never shown the held-out noise type, and still
routed `bus` mostly to the enhancer (only 14 % to passthrough, matching the oracle's `enhance`) and
`living` mostly away from it (86 % to passthrough, matching `passthrough`).

## The honest recommendation is simpler than the router

The oracle enhances **only** `bus`, and `bus`'s advantage is small: the gap between never-enhance
(0.0345) and the oracle (0.0333) is just 0.0012 WER. Almost all of the achievable gain comes from
*not enhancing*, not from routing cleverly.

So the deployment recommendation is blunt:

> **Do not put the enhancer in the ASR path.** Feed the recogniser the unprocessed signal and the
> human listener the enhanced one. Enhancement is worth +9.9 dB SI-SDRi and +0.009 STOI to a
> person; it costs +24 % relative WER to a recogniser. These are different consumers with
> different optima, and the system should stop pretending one signal serves both.

The router is worth reporting because it shows the decision is *learnable from the noisy audio
alone* and generalises to an unseen noise type — but on this corpus it does not beat simply
switching the enhancer off for ASR. We say so rather than presenting the router as the winner.

## Why this is on-thesis

The aggregate metric said "enhancement costs 24 % WER". Disaggregating showed the cost is
concentrated in four of five noise types, with one type actually benefiting. Acting on the
disaggregated view — by routing, or more simply by removing the enhancer from the ASR path —
recovers the loss. The remedy is only visible once the aggregate is broken apart, which is the
paper's argument in miniature.

## Limitations

- Five noise types is a small number of folds; the leave-one-out estimate is coarse.
- The router optimises WER only. A deployment that cares about both listeners would need a
  cost-weighted objective, which we have not fitted.
- Labels are derived from a single ASR model (Whisper `small.en`). The queued `tiny.en` run tests
  whether the routing decision is stable across recognisers; if the noise types that hurt differ by
  ASR model, the recommendation to keep the enhancer out of the ASR path gets *stronger*, not
  weaker.
