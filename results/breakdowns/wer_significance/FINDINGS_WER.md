# WER significance — findings (E3, 2026-08-20)

Two standard ways of aggregating WER over the same 824 clips, and they **disagree about which
conditions are affected and whether the aggregate effect exists at all**. That disagreement is not
a nuisance to resolve quietly; it is the paper's thesis reproducing itself inside a single leg.

Scripts: `scripts/wer_condition_significance.py` (per-clip), `scripts/wer_corpus_level.py`
(corpus-level). Baseline is `passthrough` throughout; positive difference = enhancement made WER
**worse**.

---

## 1. What each analysis says about GTCRN

| comparison | per-clip mean of rates | corpus-level (Σerr/Σwords) |
|---|---|---|
| **overall** | +11.1 %, p = 0.270 — **null** | **+24.1 %, CI excludes 0 — significant** |
| **bus** | **−41.7 %, p = 0.021** (q = 0.095) | −28.0 %, CI includes 0 — **not significant** |
| **cafe** | +43.8 %, p = 0.038 (q = 0.095) | **+57.9 %, CI excludes 0 — significant** |
| living / office / psquare | not significant | not significant |
| **SNR 2.5 dB** | +33.7 %, p = 0.095 | **+43.1 %, CI excludes 0 — significant** |

## 2. The corrected claim

The previously recorded finding — *"cuts WER 41 % in bus against a 44 % rise in cafe"* — was based
on point estimates only and **does not survive testing as stated**. Two changes are required:

1. **Drop the bus improvement.** It is significant per-clip at nominal p, but does not survive
   BH-FDR within the 5-noise-type family (q = 0.095), and the better-powered corpus-level analysis
   does not find it at all. There is no defensible claim that enhancement improves ASR in `bus`.
2. **Keep, and strengthen, the degradation.** Enhancement significantly *hurts* ASR overall
   (+24.1 % relative), and the damage is concentrated in **cafe babble** (+57.9 %) and at the
   **lowest SNR** (2.5 dB, +43.1 %).

Corrected statement for the paper:

> Enhancement significantly degrades ASR accuracy overall (+24 % relative corpus WER), and the
> degradation is concentrated: babble noise and the lowest-SNR bin account for essentially all of
> it, while the remaining three noise types show no significant effect. The apparent improvement in
> `bus` noise visible in per-clip averages is not statistically supported.

This is *still* a concentration result, and the classical-baseline finding is untouched and much
stronger — see §4.

## 3. Why the two analyses disagree, and why it matters

Per-clip averaging treats a 3-word utterance and a 36-word utterance as equal evidence. Corpus-level
weights by reference length, which is what the aggregate is supposed to mean.

The corpus has **6 183 reference words over 824 clips, median 7 words per clip**. Utterances are
short and Whisper transcribes most of them perfectly under both methods, so only **129 / 824**
pairs differ at all. Inside one noise condition the effective sample falls to **16–37 clips**,
which is why the per-clip test is underpowered against effects that look large.

The sign flip on `bus` follows directly: the per-clip improvement sits on short clips that
corpus-level aggregation correctly down-weights.

**The methodological point is reportable in its own right:** on short-utterance corpora, per-clip
WER averaging and corpus-level WER can disagree about *both* the existence of an aggregate effect
and the sign of a per-condition effect. Papers reporting "WER" almost never say which they used.

## 4. The classical-baseline finding is unaffected and very strong

Robust under both analyses and after FDR correction — 19 / 40 comparisons survive BH, and every
one of them is a classical method degrading WER:

| method | condition | per-clip diff | q |
|---|---|---|---|
| wiener | office | +243.8 % | 0.0000 |
| wiener | overall | +127.6 % | 0.0000 |
| wiener | cafe | +121.3 % | 0.0000 |
| wiener | snr 7.5 | +200.0 % | 0.0000 |
| spectral subtraction | snr 7.5 | +95.5 % | 0.0022 |

Classical baselines are worse than doing nothing on ASR in **every** condition tested, at effect
sizes an order of magnitude beyond the neural model's. This remains the most robust claim in the
leg.

## 5. Remedy — how to recover the power that was lost

The effect sizes are large; the problem is purely that too few clips differ. Three fixes, in order
of cost:

1. **Report corpus-level WER as primary** (done here, zero cost). Recovers the aggregate result
   the per-clip test could not detect.
2. **Add MS-SNSD to the WER evaluation** — 1 100 further clips with noise-type labels, roughly
   doubling n and adding noise categories VoiceBank does not cover. Needs one Whisper pass.
3. **Use a smaller ASR model as a second operating point.** Whisper is near-ceiling on this
   clean-ish read speech, which is the direct cause of the zero-inflation. A weaker model raises
   the error floor and multiplies the effective sample. Report both, since "does enhancement help
   ASR" is genuinely ASR-model-dependent — and that dependence is itself on-thesis.

Fix 1 is applied. Fixes 2 and 3 need a Whisper pass and are queued for the Kaggle batch alongside
the timing benchmark.
