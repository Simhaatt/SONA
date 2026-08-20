# Final plan — one paper from all the work

Written 2026-08-20. Supersedes the paper-related parts of `NEXT_STEPS_PLAN.md` (expired).
Read `PROJECTS_STATUS.md` for the state of the underlying results.

**Hardware reality: essentially none of this needs the T4.** The remaining work is ONNX
quantization, retrieval over cached embeddings, and statistics — all CPU-bound. Total core compute
is **7–9 hours on the local machine**.

**Nothing blocks drafting.** The MOS listening test was deferred to future work on 2026-08-20 (see
Phase D), so the critical path is now writing, not measuring.

---

## 1. The paper

**One journal paper**, merging both drafts. Target: IEEE Access (safest fit) or EURASIP JASMP.
12–16 pages.

### Thesis

> Compression and evaluation damage in audio systems is **concentrated**, not diffuse — and the
> aggregate metrics the field reports are structurally blind to it.

The merge only works because both existing drafts are instances of this. A 50/50 staple of
"quantization study + retrieval study" would score worse than either alone; the framework section
has to do the unifying work.

### The five instances

| leg | aggregate metric says | disaggregated reality |
|---|---|---|
| Weight quantization | spread bits evenly by weight count | 0.133 % of weights carry 98.7 % of loss |
| File size | fewer bits ⇒ smaller file | quantizing *more* made it *bigger* (172 vs 29 QDQ nodes) |
| Embedding quantization | Recall@k: lost ~50 % | task accuracy: lost ~4 % |
| ASR evaluation | corpus WER +24 % — enhancement hurts | almost all of it is `cafe` babble (+57.9 %) and the 2.5 dB bin (+43.1 %) |
| WER aggregation itself | per-clip vs corpus-level | disagree on both existence and sign, same 824 clips |
| Mask representation | SI-SDR: clean win, p < 1e-6 | PESQ: significant *loss*, p = 0.005 |

### Why merging is strictly better than either alone

- AudioVQ's central claim now overlaps with [Dimitropoulos & Mamoulis 2026]. Standalone that is a
  serious novelty problem; as one leg of a broader thesis it stops being fatal.
- The quantization paper's single architecture is damning as a standalone claim, but fine as one
  instance of a pattern demonstrated four other ways.

### Structure

1. Introduction — the thesis, stated falsifiably (say what result would refute "concentration")
2. Related work
3. Framework: concentration vs. aggregation
4. **Study 1** — weight quantization (stem exemption)
5. **Study 2** — evaluation metrics (WER conditions, complex-mask trade-off)
6. **Study 3** — embedding quantization (geometry–meaning), explicitly subordinate
7. **Remedies** — a validated countermeasure per failure mode
8. Reporting protocol — the generalizable contribution
9. Limitations

(Human validation is deferred — see Phase D. Its slot reopens for a larger-journal version.)

**Centre of gravity is speech enhancement.** Studies 1 and 2 share one system (GTCRN/VoiceBank).
Study 3 enters as "does this hold beyond weights?", not as a co-equal half.

### Cut from this paper

Scene classifier as a standalone result (but see R1 — it returns as a component), online
adaptation / trust-region guard, heavier-architecture ablations (compress to one line). All stay in
the FYP thesis.

---

## 2. Already done

- Tree consolidated; 46,078 files verified migrated, 0 missing
- Citation pass complete (`AudioVQ/REFERENCES.md`), one claim narrowed as a result
- RTF re-measured on an idle machine; INT8 slowdown replicates at 1.39–1.46×
- Two LaTeX manuscripts compiling to PDF with formal bibliographies
- Both number audits passing (19 + 25 claims) and wired to run on demand

---

## 3. Experiments

### Phase A — quick wins and reviewer-objection closers (~2.5 h, one sitting)

| # | experiment | cost | why |
|---|---|---|---|
| **R2** | Run `PERCEPTUAL_MP/onnx/phase3/frozen_mixed.onnx` through `scripts/bench_onnx_edge.py` | 20 min | **Run this first.** Mixed has 29 QDQ nodes vs uniform's 172. If fewer nodes means less slowdown, the mixed model is smaller *and* faster than uniform INT8 — a clean positive derived from the slowdown negative. Possible free win; shapes the framing. |
| **E3** | Per-condition WER significance (Wilcoxon + bootstrap CI) | 15 min | Leg 4 is currently point estimates while legs 1–2 have full stats. Inconsistent rigour is an easy reviewer hit. **Note:** `per_sample_wer.csv` has `noise_type=unknown` on all 4120 rows — join against `results/vbd_test_conditions.csv` by `uid` at analysis time (it covers all 824: 431 p257 + 393 p232). |
| **R3** | Formalize the dither diagnostic across all quantizer/corpus pairs | 30 min | Turns the MS-SNSD caveat into a reusable rule: *if a quantizer beats FP32 itself, it is dithering, not quality.* Generalizes beyond this paper. |
| **E1** | 1/Ratio@k on the embedding leg | 30 min | Highest value per minute in the whole plan. Dimitropoulos propose this as *the fix*; computing it either corroborates them or shows it **also** fails under quantization — turning the novelty threat into a contribution. Cached embeddings, pure numpy. |
| **E2** | Hessian-trace ranking (HAWQ-V2 proxy) | 1 h | *The* hole in Study 1. Currently compares perceptual vs two output-error rankings, never the field-standard curvature proxy. Hutchinson estimator, ~100 iterations on the dev split. Takes the negative from "two metrics agree" to "three agree, including the one everyone uses." |

**Also on day one: start MOS recruiting.** It cannot be compressed by working harder.

### Phase B — the flagship remedy (~2 h)

**R1 — condition-gated enhancement.** The WER negative has a treatment sitting in the repo: route
enhancement by condition instead of applying it blindly. Resurrects the scene classifier as a
load-bearing component.

Two disciplines built in from the start:

- **Do not map scene labels to conditions.** The classifier emits `speech/music/traffic/fan/crowd`;
  VoiceBank conditions are `bus/cafe/living/office/psquare`. Predict the *decision* (enhance or
  not) directly from the same 34 features in `src/controller/scene.py::features`.
- **Use leave-one-noise-type-out CV** — train on 4 types, test on the held-out one. With n=824 and
  5 types, per-clip CV would overfit trivially. This is the same discipline that corrected the
  scene classifier from 0.844 to 0.748, so the paper stays internally consistent.

Report **oracle first, then achieved**: "condition-gated routing has X headroom; a feature-based
router captures Y of it." Partial capture is still a complete, honest story.

### Phase C — the risk experiment (~3 h)

**E4 — stem sweep on a second architecture.** Kills the n=1 objection. Uses the existing
`sona-final-month/COMPLEX_MASK_AB/ckpt_complex_gtcrn` checkpoint, so **no training is needed** —
ONNX export, 8 configs × 200 dev clips (~70 s/config at measured RTF), then 824-clip final on 3
configs.

**This can weaken the headline claim, and that is accepted going in.** If the stem is not dominant
in the second model, "0.133 % of weights" becomes model-specific. Run it anyway: the fallback
finding — *which layer to protect is model- and runtime-specific, so measure* — is still
publishable, already matches the Rusci et al. contrast in Study 1, and a reviewer discovering this
is far worse than reporting it.

### Phase D — human validation — **DEFERRED to future work (decided 2026-08-20)**

> **Not required for this submission.** Kept built and validated for a larger-journal version.
>
> The reason is arithmetic. The differences the test would have to resolve are far below what a
> ~20-rater study can detect: complex vs magnitude differ by **0.016 PESQ** (PESQ reports on a
> MOS-like scale) against a detection floor of ~0.3 MOS, and mixed vs FP32 by **0.0014 dB**
> SI-SDR. Every comparison of interest would return null. The test cannot pick a winner.
>
> The assessment also changed once **E1** landed: it adjudicates a metric disagreement using
> *downstream task accuracy* — objective, label-based, no humans — showing Recall@10 off by 0.238
> against 0.0033 for 1/Ratio@10. The paper therefore already demonstrates the "use a better
> arbiter" move without a listening test, which is what MOS was mainly there to provide.
>
> **What it would still buy, when run:** the null is the finding. SI-SDR at p < 1e-6 and PESQ at
> p = 0.005 on a difference no listener can hear is the sharpest available statement of the thesis
> — *significance at n = 824 outruns audibility*. Plus TOST would convert "mixed ≈ FP32" from
> p = 0.25 (a failure to reject) into a positive equivalence result.
>
> **Risk to weigh before running it later:** if uniform INT8 also comes out equivalent, the stem
> exemption is perceptually moot. The result survives as a *fidelity* and *file size* claim, which
> is already how it is framed, but that would have to be said plainly.
>
> Everything is ready: 50 clips over both arms, app verified serving blind, paired + TOST analysis
> validated end to end on 15 simulated raters against a planted ground truth. See
> `mos_test/RUNBOOK.md`. Needs 18–20 raters × ~15 min.

The two comparisons it covers, for when it runs:

- **magnitude vs complex mask** — exactly where SI-SDR and PESQ contradict each other. Either
  outcome resolves a real disagreement.
- **FP32 vs mixed vs uniform INT8** — the paper claims mixed is indistinguishable from FP32 at
  p = 0.25. If listeners also cannot tell them apart, that is a far stronger claim than any
  p-value, and the natural headline.

App is built and browser-verified; 29 clips selected. Extend `mos_test/clip_manifest.csv` to cover
both comparisons.

### Optional — only if a reviewer asks

| experiment | cost | verdict |
|---|---|---|
| Per-channel / finer precision axis | 6–8 h local | **The one negative with a real chance of legitimately flipping (~30 %).** The current negative is scoped to "INT8, 6 layers" — a space *forced by runtime limits*, not chosen. Either outcome helps: it flips and you gain a positive, or it holds and the negative gains much broader scope. |
| Cross-runtime check (TFLite / OpenVINO) | 3–5 h, **timebox** | Tests whether "GRU unquantizable / INT8 slower" is a finding or an ORT quirk. Conversions are fiddly and may fail on this graph. |
| Scale sweep 10k → 23k vectors | ~6 h | Extends the trend only ~0.4 decade; does not settle the 10⁶ question. |
| QAT or PESQ-aligned retraining | 8–9 h **T4** | The only item genuinely needing GPU. Off-thesis for a paper about metric blindness. Skip. |

---

## 4. Sequence

| when | work | output |
|---|---|---|
| **Day 1** | Phase A (R2 → E3 → R3 → E1 → E2) | Five reviewer objections closed; possible free win from R2 |
| **Day 2** | Phase B (R1) | Flagship remedy: oracle + router numbers |
| **Day 3** | Phase C (E4) | n=2 architectures, or a redirected framing |
| **Week 2** | Merge the two `.tex` files into one journal-class document; write framework + remedies + protocol sections | Full draft |
| **Week 3** | Buffer; final pass | Submission-ready draft |

Extend both `audit_paper_numbers.py` scripts to cover every new number as it is produced, so no
stale figure can reach a reviewer.

---

## 5. Decision gates

- **If R2 shows mixed is also faster** → promote it to a headline claim in Study 1; the abstract
  changes from "smaller" to "smaller and faster."
- **If E4 shows the stem is not dominant elsewhere** → reframe Study 1 from "the stem is special"
  to "concentration is real but its location is model-specific — measure, do not assume." The
  thesis survives intact; only Study 1's headline changes.
- **If R1's router captures little of the oracle headroom** → report the oracle as headroom and the
  router as a negative, which is *still on-thesis*. Do not tune the router against the test set.
- **If Study 3 still reads as bolted on after drafting** → cut it to a two-page section and keep
  AudioVQ as the separate 5-page paper that already exists and compiles.

---

## 6. Standing rules

- **Do not re-run experiments to change signs.** Numbers are frozen at n=824 with p-values to
  1e-51. A result at p = 0.005 does not flip by re-running; it flips only by changing the method,
  which makes it a different experiment that must be reported as one.
- **Negative results stay.** They are the evidence for the thesis, not weaknesses to remove. The
  scene classifier's 0.748 in particular is a *correction* of an optimistic 0.844 — restoring the
  higher number means restoring texture leakage.
- **Every remedy must be validated, not suggested.** "Future work should…" gets punished.
- **Author line is a placeholder.** `SPARC FYP Team` and the project email are in both manuscripts;
  real names and affiliation go in before this goes anywhere.
