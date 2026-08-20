# How to frame this paper

The single hardest decision in this project is not what to measure — that is done — but what the
paper is *about*. Get the framing wrong and a set of strong results reads as a grab bag. This file
records the framing, why it was chosen over the alternatives, and how each result earns its place.

---

## 1. The thesis

> **Compression and evaluation damage in audio systems is *concentrated*, not diffuse — and the
> aggregate metrics the field reports are structurally blind to it.**

Every result is an instance. The instances are not analogies; each is a case where an aggregate
number and its disaggregation disagree about what is happening.

| instance | the aggregate says | the disaggregation says |
|---|---|---|
| weight quantization | spread bits by weight count | 0.133 % of weights carry 98.7 % of loss |
| file size | fewer bits ⇒ smaller file | quantizing *more* produced a *larger* file |
| embedding quantization | Recall@10 lost ~50 % | task accuracy lost ~4 % |
| ASR evaluation | corpus WER +24 % | almost all of it is babble and the lowest SNR bin |
| WER aggregation itself | per-clip and corpus-level WER | disagree on both existence and sign |
| mask representation | SI-SDR: clean win | PESQ: significant loss |

## 2. Why merge the two drafts rather than publish separately

Both standalone papers have a specific weakness, and the merge removes both.

**AudioVQ alone** — its central claim now overlaps with Dimitropoulos & Mamoulis (June 2026), who
showed Recall@k understates deployment quality for ANN *search*. Standalone, that is a serious
novelty problem. As one leg of a broader thesis it is no longer load-bearing on its own, and the
overlap becomes corroboration from an independent direction.

**Quantization alone** — one architecture, one runtime. "n = 1" is damning for a standalone claim
and unremarkable as one instance of a pattern demonstrated four other ways.

## 3. Centre of gravity: speech enhancement

Studies 1 and 2 share one system (GTCRN on VoiceBank-DEMAND). Study 3 (embedding retrieval) enters
as *"does this hold beyond weights?"* — clearly subordinate, roughly two pages.

This is deliberate. Speech enhancement and embedding retrieval are different communities, and a
speech reviewer can find a retrieval study out of place. Keeping the centre of gravity in speech,
and letting the framework section carry the unification, is what stops the paper reading as two
papers stapled together.

**If it still reads as bolted on after drafting, cut Study 3 to two pages and keep AudioVQ as the
separate 5-page paper that already exists and compiles.** That decision gate is real; take it.

## 4. Make the thesis falsifiable

Position papers die when the thesis cannot be wrong. State the falsification condition explicitly
in the framework section:

> The thesis predicts that where a compression or evaluation aggregate is reported, disaggregating
> it will reveal that the effect is carried by a small part of the system. It is falsified by a
> case where damage is genuinely uniform across components, or where disaggregation changes
> nothing about the conclusion.

Then note honestly that we did not observe such a case, and that we looked — the classical-baseline
result *is* uniform (worse than passthrough in all 5 noise types and all 4 SNRs, no exceptions),
and we report it as the one place where the aggregate tells the whole story. **That is the control
case, and including it is what makes the thesis testable rather than unfalsifiable.**

## 5. Framing each result

### Study 1 — weight quantization

**Lead with the negative.** "We set out to test whether perceptual sensitivity ranking beats
MSE-family ranking. It does not — ρ = +1.000 against mask-MSE." Opening with the refuted hypothesis
buys enormous credibility for the positive result that follows, because it shows the sweep was
exploratory rather than a search for a headline.

**Then the discovery.** The stem exemption was found *while* sweeping, not hypothesised. Say that.

**The size inversion is the best single detail in the paper.** Quantizing more made the file bigger
(172 vs 29 QDQ nodes). It is concrete, surprising, mechanistically explained, and invisible to any
bits-per-weight analysis. Give it its own paragraph.

**Hessian column.** Whatever E2 returns, it is load-bearing: agreement makes the negative result
comprehensive; disagreement is a finding about HAWQ-V2's per-parameter normalisation under extreme
parameter-count imbalance (97 vs 166 080 params).

### Study 2 — evaluation metrics

**The WER story must be told as corrected.** The earlier point-estimate claim — "41 % cut in bus
against a 44 % rise in cafe" — did not survive testing. The `bus` improvement fails FDR correction
(q = 0.095) and does not appear at all in the better-powered corpus-level analysis.

Frame it as: enhancement significantly degrades ASR overall (+24 % relative corpus WER), and the
degradation is concentrated in babble and the lowest SNR bin.

**The per-clip vs corpus-level disagreement is a gift.** Two standard aggregations of the same 824
clips disagree about whether an aggregate effect exists *and* about the sign of a per-condition
effect. Papers reporting "WER" almost never say which they used. This is the thesis reproducing
itself inside a single leg — foreground it rather than resolving it silently.

**Complex mask.** SI-SDR says clean win; PESQ says significant loss. Only real PESQ surfaced it —
an SI-SDR-only evaluation would have reported a false win. This is the cleanest single
demonstration in the paper that one metric is not enough.

### Study 3 — embedding quantization

**Concede the prior claim in the first sentence.** Then draw the distinction: search approximation
returns slightly worse *true* vectors; quantization *destroys* representation content. Stability
under the first does not imply stability under the second.

**E1 (1/Ratio@k) is the strongest card here.** At 1 bit/coord on WavLM, Recall@10 = 0.483 while
1/Ratio@10 = 0.970 — and semantic retention is 96.5 %. A purely geometric, **label-free** metric
lands within half a point of the task-based measurement that recall misses by 48 points. That is a
better result than the one originally planned, because it gives practitioners something to compute
without needing downstream labels.

### Remedies — what turns diagnosis into contribution

A paper that only says "your metrics are wrong" is a complaint. Each failure mode gets a
countermeasure, and each countermeasure is *validated*, not proposed:

| failure | remedy | evidence |
|---|---|---|
| perceptual ≈ MSE ranking | don't build a PESQ-in-the-loop allocator | ρ = +1.000, allocations differ by 0.0003 dB |
| metrics flattered by quantization noise | rank on \|deviation from FP32\| | R3: 4/32 cells beat FP32, all on MS-SNSD, none in-domain |
| SI-SDR/PESQ conflict | report a metric set spanning phase sensitivity | complex-mask A/B + MOS |
| enhancement degrades ASR | keep the enhancer out of the ASR path | R1: 0.0428 → 0.0345 |
| recall misleads | use 1/Ratio@k, it is label-free | E1: 0.970 vs 96.5 % semantic |
| underpowered per-clip WER | corpus-level aggregation | E3b recovers the effect the per-clip test missed |

**The R1 framing needs care.** The learned router captures 61 % of headroom and beats
always-enhance significantly — but plain never-enhance beats the router. Say so. The honest
recommendation is the blunt one: *different consumers, different optima — give the recogniser the
unprocessed signal and the listener the enhanced one.* Presenting the router as the winner when the
simpler thing works better is exactly the overclaiming this paper argues against.

## 6. The generalizable contribution

The most citable output is not any single number. It is a **reporting protocol** — the rules the
paper's own failures imply:

1. Report bits per *vector*, not per coordinate, when dimensions are padded.
2. Report which WER aggregation you used; report both when utterances are short.
3. For PTQ, report deviation from the source model, not absolute scores; if a quantizer beats the
   source model, treat that metric as compromised on that corpus.
4. Report effective sample size for zero-inflated metrics, not just clip counts.
5. Correct for multiplicity when reporting per-condition breakdowns.
6. State CPU and machine load beside any absolute timing.

Six rules, each derived from a specific failure this paper documents. That section survives past
the specific systems studied, and it is what a future paper cites.

## 7. Title

Working title:

> **Where the Loss Lives: Compression and Evaluation Damage in Audio Systems Is Concentrated, and
> Aggregate Metrics Cannot See It**

Alternatives if that is too long for the venue:
- *Where the Loss Lives: Concentrated Damage in Audio Compression and Evaluation*
- *0.13 % of the Weights, 99 % of the Loss: On Aggregate Metrics in Audio Systems*

## 8. Venue

**IEEE Access** is the safest fit — broad enough for the retrieval leg, no page-limit fight for a
12–16 page paper, and it takes methodological/analysis contributions.

**EURASIP JASMP** is a better-respected home if the retrieval study is cut and the paper becomes
pure speech.

Do not target a 4-page conference. The paper's value is cumulative across legs, and cutting it to
four pages destroys exactly what makes it strong.
