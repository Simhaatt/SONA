# Paper bundle

Everything needed to write the paper, in five documents. Assembled 2026-08-20.

Read them in this order:

| # | file | what it is |
|---|---|---|
| 00 | `00_README.md` | this file — orientation, status, what is and is not settled |
| 01 | [`01_PAPER_DETAILS.md`](01_PAPER_DETAILS.md) | methods, data, splits, hyperparameters, statistics, provenance — what Methods and Experimental Setup need |
| 02 | [`02_ALL_RESULTS.md`](02_ALL_RESULTS.md) | every result table in the project, ~1900 lines, **generated** — never hand-edit |
| 03 | [`03_FRAMING.md`](03_FRAMING.md) | what the paper is about, why the two drafts merge, how each result earns its place |
| 04 | [`04_WRITING_GUIDE.md`](04_WRITING_GUIDE.md) | how research papers are written, and how not to sound machine-generated |

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
- INT8 is 1.39–1.46× **slower** than FP32 on x86, re-measured at 6 % machine load.
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

**3. Two WER aggregations disagree.** Per-clip averaging and corpus-level WER disagree about
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
| enhancement degrades ASR | **keep the enhancer out of the ASR path** | R1: 0.0428 → 0.0345 |
| recall misleads | use 1/Ratio@k — label-free, needs no downstream labels | E1b: MAE 0.033 vs recall's 0.428 across 36 points |
| underpowered per-clip WER | aggregate at corpus level | E3b recovers the effect the per-clip test missed |

On R1, note the honest reading: the learned router captures 61 % of headroom and significantly
beats always-enhance, but **plain never-enhance beats the router**. The recommendation is the blunt
one — different consumers, different optima. Presenting the router as the winner would be exactly
the overclaiming this paper argues against.

## What is still open

Nothing now blocks drafting. The remaining compute is optional strengthening.

| item | blocking? | note |
|---|---|---|
| **MOS listening test** | **no — deferred to future work** | Built, validated, and ready (`mos_test/RUNBOOK.md`), but the effects are ~19x below what 20 raters can resolve, so it cannot pick a winner. Held for a larger-journal version, where the payoff is the *null*: significance at n=824 outrunning audibility |
| R2 mixed-allocation timing | no | queued in `sona_phase_a_kaggle.ipynb`; possible free win if fewer QDQ nodes means less slowdown |
| WER on MS-SNSD, and with `tiny.en` | no | queued on Kaggle; both attack the same zero-inflation power problem |
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
