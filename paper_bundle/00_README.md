# Paper bundle

Everything needed to write the paper, in six documents. Assembled 2026-08-20, venue note added
2026-08-21.

**Format decision:** write in **IEEE format (IEEEtran)** now; venue is deferred. Safe order — the
conference/journal switch is one documentclass option, so nothing is locked. See `05_VENUE.md`.

Read them in this order:

| # | file | what it is |
|---|---|---|
| 00 | `00_README.md` | this file — orientation, status, what is and is not settled |
| 01 | [`01_PAPER_DETAILS.md`](01_PAPER_DETAILS.md) | methods, data, splits, hyperparameters, statistics, provenance — what Methods and Experimental Setup need |
| 02 | [`02_ALL_RESULTS.md`](02_ALL_RESULTS.md) | every result table in the project, ~1900 lines, **generated** — never hand-edit |
| 03 | [`03_FRAMING.md`](03_FRAMING.md) | what the paper is about, why the two drafts merge, how each result earns its place |
| 04 | [`04_WRITING_GUIDE.md`](04_WRITING_GUIDE.md) | how research papers are written, and how not to sound machine-generated |
| 05 | [`05_VENUE.md`](05_VENUE.md) | venue costs and the decision: **write in IEEE format now, pick the venue later** |

`02_ALL_RESULTS.md` is produced by `scripts/build_paper_bundle.py`. Re-run it after any experiment
and the compendium is current by construction — that is the whole point, so numbers cannot drift
away from the artefacts that produced them.

```bash
python scripts/build_paper_bundle.py
```

---

## The paper in one sentence

> Compression and evaluation damage in audio systems is **concentrated**, not diffuse — and the
> aggregate metrics the field reports are structurally blind to it.

One journal paper (IEEE Access or EURASIP JASMP, 12–16 pp) merging what were two separate drafts.
Centre of gravity is speech enhancement; the embedding-retrieval study is subordinate. Full
reasoning in `03_FRAMING.md`.

---

## What is settled

Frozen, significance-tested, and machine-audited. **Do not re-run these to try to improve them.**

- Classical baselines are worse than doing nothing in **all 5 noise types and all 4 SNRs**, no
  exceptions. The most robust claim in the project, and the paper's control case: the one place
  where the aggregate does tell the whole story.
- Stem exemption: **0.133 % of weights carry 98.7 % of INT8 loss**; the mixed model is
  statistically indistinguishable from FP32 (p = 0.25) at 3.2× compression, significant in all 9
  conditions, and *smaller* than uniform INT8 (916.9 vs 950.7 KB).
- Perceptual and MSE-family sensitivity rankings coincide (ρ = +1.000 vs mask-MSE). Hypothesis
  refuted; reported as such.
- Geometry–meaning dissociation: ~50 % neighbour loss costs ~4 % task accuracy on 3/3 models.
- **1/Ratio@10 = 0.970 where Recall@10 = 0.483** and semantic retention is 96.5 % — a label-free
  geometric metric lands within half a point of the task measurement.
- Complex mask: significantly better SI-SDR and STOI, significantly **worse** PESQ (−0.016,
  p = 0.005).
- Dither diagnostic: 4/32 cells beat FP32, all on MS-SNSD, **none in-domain**.
- INT8 is **slower** than FP32 on both CPUs tested — 1.19–1.39× on Intel Xeon, 1.39–1.46× on AMD
  Ryzen, both on verified-idle machines. Not a one-machine artifact.
- **R2 — the stem exemption is smaller, faster *and* FP32-quality.** Mixed (29 QDQ nodes) beats
  uniform INT8 (172 nodes) at every thread count: 916.9 vs 950.7 KB, 1.13–1.32× vs 1.19–1.39×
  slowdown, and SI-SDR 18.691 against FP32's 18.690 (uniform: 18.569). The graph overhead causing
  the slowdown is itself concentrated — the thesis appearing in a third quantity after loss and
  file size.
- **Enhancement degrades ASR, on three independent measurements**: +24.1 % (VoiceBank/`small.en`),
  +23.1 % (VoiceBank/`tiny.en`), +46.6 % (MS-SNSD/`small.en`) relative corpus WER, all with CIs
  excluding zero.
- **E2 — three criterion families agree.** Perceptual (PESQ), output-error (SI-SDR, mask-MSE) and
  second-order curvature (full HAWQ-V2 Ω) all rank `stem` most sensitive and `head` least. The
  choice of sensitivity criterion does not matter, which makes the negative result comprehensive.
- **E4 — the stem finding replicates on a second model.** Complex-mask GTCRN: stem most sensitive,
  head least, stem carries **92.1 %** of INT8 damage (vs 98.7 %), and the size inversion holds
  (mixed 917.4 KB < uniform 952.2 KB).
- **E1b — 1/Ratio@k is ~13× better calibrated than recall** (MAE 0.033 vs 0.428 against measured
  task retention) while ranking methods identically. Recall preserves *ordering* but not
  *calibration*.
- **The dissociation is near-total at aggressive compression.** Uniform 1-bit loses 95–99 % of true
  neighbours while retaining 89–95 % of downstream task accuracy.

## What changed on 2026-08-20 — read this before quoting older docs

Two claims were **corrected** by properly-powered testing. Older documents in the repo still carry
the superseded versions.

**1. The WER condition story.** The previous claim — *"cuts WER 41 % in bus against a 44 % rise in
cafe"* — rested on point estimates and does not survive testing. The `bus` improvement fails
BH-FDR correction (q = 0.095) and does not appear at all in the better-powered corpus-level
analysis. The corrected claim: **enhancement significantly degrades ASR overall (+24 % relative
corpus WER), concentrated in babble and the lowest SNR bin; the apparent `bus` improvement is not
statistically supported.** See `results/breakdowns/wer_significance/FINDINGS_WER.md`.

**2. My own HAWQ-V2 implementation was wrong first time.** The first pass ranked layers by
*average* Hessian trace alone, which is not the HAWQ-V2 criterion — HAWQ-V2 ranks by
`Ω = (tr(H)/n)·‖Q(w)−w‖²`, and the dropped factor is exactly what carries parameter count. On a
model whose layers span 97 to 166 080 parameters, the incomplete version *inverts* the ranking and
produced a striking-but-false conclusion that HAWQ-V2 misranks. The corrected criterion agrees with
the empirical damage ordering. Reporting the first version would have been a strawman. See
`PERCEPTUAL_MP/results/hessian/FINDINGS_HESSIAN.md`.

**3. "Concentration" in the WER leg was partly a power artifact — softened after the Kaggle
batch.** On VoiceBank the degradation looked confined to babble and the lowest SNR bin. With 5.5×
the effective sample (MS-SNSD, 704/1100 differing pairs vs 129/824), **8 of 9 noise types are
individually significant**. Enhancement hurts ASR broadly, not narrowly; the VoiceBank
per-condition nulls were lack of power, not evidence of unaffected conditions. The only exception
is `Typing` — impulsive noise, where the enhancer barely acts at all.

This one runs *against* the paper's usual direction: here the aggregate was right and our own
disaggregation misled us. A paper about metrics misleading should be first to say so. See
`results/kaggle_phase_a/FINDINGS_KAGGLE.md`.

**4. The `bus` improvement is definitively an artifact.** It flips sign between recognisers on
identical audio: −28 % with `small.en`, **+37 % with `tiny.en`**, neither significant.

**5. Two WER aggregations disagree.** Per-clip averaging and corpus-level WER disagree about
whether an aggregate effect exists *and* about the sign of a per-condition effect, on the same 824
clips. This is not a nuisance — it is the paper's thesis reproducing itself inside one leg, and it
should be foregrounded.

## Remedies — the part that makes this a contribution rather than a complaint

Each failure mode has a **validated** countermeasure. None is merely proposed.

| failure | remedy | evidence |
|---|---|---|
| perceptual ≈ MSE ranking | skip the PESQ-in-the-loop allocator | ρ = +1.000; allocations differ by 0.0003 dB |
| metrics flattered by quantization noise | rank on \|deviation from FP32\| | R3: 4/32 cells beat FP32, none in-domain |
| SI-SDR/PESQ conflict | report a metric set spanning phase sensitivity | complex-mask A/B (MOS deferred — the effect is below audibility, which is itself the point) |
| enhancement degrades ASR | **keep the enhancer out of the ASR path** — no routing needed | R1 under `tiny.en`: the oracle picks passthrough for **all five** noise types, so oracle and never-enhance coincide (0.0711) |
| recall misleads | use 1/Ratio@k — label-free, needs no downstream labels | E1b: MAE 0.033 vs recall's 0.428 across 36 points |
| underpowered per-clip WER | aggregate at corpus level | E3b recovers the effect the per-clip test missed |

On R1, note the honest reading: the learned router captures 61–89 % of headroom and significantly
beats always-enhance, but **plain never-enhance beats the router** — and under `tiny.en` the oracle
selects passthrough in every condition, so there is no condition where enhancement helps ASR at
all. More data made the recommendation *simpler*, not more elaborate. The routing experiment still
earns its place by showing the decision is learnable from noisy audio alone and generalises to an
unseen noise type, but it is reported as a negative for routing, not a win.

## What is still open

Nothing now blocks drafting. The remaining compute is optional strengthening.

| item | blocking? | note |
|---|---|---|
| **MOS listening test** | **no — deferred to future work** | Built, validated, and ready (`mos_test/RUNBOOK.md`), but the effects are ~19x below what 20 raters can resolve, so it cannot pick a winner. Held for a larger-journal version, where the payoff is the *null*: significance at n=824 outrunning audibility |
| ~~R2 mixed-allocation timing~~ | **done** | Ran on Kaggle's idle Xeon. The free win landed: mixed is faster than uniform INT8 at every thread count |
| ~~WER on MS-SNSD and `tiny.en`~~ | **done** | Effective sample up 1.7× and 5.5×; softened the concentration claim, settled the `bus` artifact |
| ~~E4 second architecture~~ | **done** | Replicated: stem dominant on the complex-mask GTCRN too (92.1 % of damage), size inversion intact. Scope caveat: same backbone, different output head — a replication across *output representations*, not architecture families |
| per-channel quantization | no | the one negative with a real chance (~30 %) of legitimately flipping |
| on-device measurement | no | never done; argue from size and latency only |
| **author names** | **yes, before submission** | `SPARC FYP Team` is a placeholder in both manuscripts |

## Standing rules

1. **Do not re-run experiments to change signs.** Numbers are frozen at n = 824 with p-values to
   1e-51. A result at p = 0.005 does not flip by re-running; it flips only by changing the method,
   which makes it a different experiment that must be reported as one.
2. **Negative results stay.** They are the evidence for the thesis, not weaknesses to remove.
3. **Every remedy must be validated, not suggested.**
4. **Corrections found by us are strengths.** The scene classifier's 0.844 → 0.748 under grouped
   CV, and the `bus` effect vanishing under FDR, are both cases of catching our own optimism. A
   reviewer catching them would be fatal; us catching them is credibility.
5. **State hardware and load beside any absolute timing.**

## Existing manuscripts

Two compiling LaTeX papers already exist and are the raw material for the merge:

- `AudioVQ/paper/recall_wrong_yardstick.pdf` — 5 pp, 16 refs
- `PERCEPTUAL_MP/paper/stem_exempt_quantization.pdf` — 4 pp, 6 refs

Both are IEEEtran conference format. The merged journal paper reuses their sections, so start from
these rather than a blank file.
