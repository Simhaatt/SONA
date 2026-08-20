# Perceptually-Guided Mixed-Precision Quantization — Implementation Plan

Status: proposed, not started. Written 2026-07-24.
Gate: do NOT start until WER / ONNX-INT8 / DeepFilterNet2 / MOS / stats / rewrite workstreams land.
The thesis deadline outranks this entirely. This is the first thing to cut if anything slips.

---

## 1. Hypothesis

Standard post-training quantization calibrates to minimize **activation/weight MSE**. For a speech
enhancement model the objective that actually matters is **perceptual** (PESQ/STOI), which weights
frequency bands extremely non-uniformly. So MSE-calibrated quantization may be systematically
misallocating its error budget for an audio model.

**Testable claim:** ranking the network's layers by *perceptual* sensitivity to quantization gives
a different order than ranking them by *MSE* sensitivity — and allocating bits by the perceptual
ranking gives better PESQ/STOI at matched model size.

**Null result (equally publishable):** the two rankings coincide, so MSE-guided allocation is
already perceptually optimal for compact conv-recurrent enhancers. This fits the paper's existing
honest-negative identity ("heavier architectures don't help", "complex masking trades PESQ for
fidelity"). Report it as such — do not bury it.

Derivation from TurboQuant (arXiv 2504.19874): its second structural finding is that an
MSE-optimal quantizer is *biased for inner-product estimation*, so a different objective needs a
different quantizer. This is the same mismatch in a different domain. Cite it for the insight, not
the algorithm — TurboQuant itself does not apply here (no KV cache, no vector DB, no memory
bottleneck).

---

## 2. Prior art — check before claiming novelty

Sensitivity-guided mixed precision is **established** (HAWQ and successors use Hessian-trace
sensitivity; there is a large NAS-for-quantization literature). The increment claimed here is
narrow and must be stated narrowly:

> using a *perceptual audio metric* as the per-layer sensitivity signal, on a compact
> conv-recurrent speech enhancer, and showing the resulting allocation differs from MSE-guided.

Do a literature pass on: HAWQ / HAWQ-V2, mixed-precision PTQ for speech enhancement, quantized
RNNoise / DeepFilterNet variants. If someone has already done perceptual-metric bit allocation for
SE, drop the novelty claim and reframe as a replication on a new architecture.

---

## 3. Assets that already exist (reuse, don't rebuild)

| Asset | Path | Use |
|---|---|---|
| Locked GTCRN ckpt | `results/kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/extracted/ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt` | frozen model under study |
| ONNX export + static QDQ | `scripts/export_onnx.py` | extend with per-layer precision map |
| Edge benchmark | `scripts/bench_onnx_edge.py` | RTF / latency / size |
| Metric suite | `src/eval/metrics.py` | si_sdr, stoi, estoi, lsd, pesq, wer |
| Eval harness | `src/eval/evaluate.py`, `report.py` | `evaluate_method`, `aggregate`, CSV/LaTeX |
| Bootstrap CI pattern | `scripts/analyze_mos.py::_bootstrap_ci` | reuse for sensitivity CIs |
| WSL PESQ env | `scripts/score_pesq_wsl.py` (`/opt/pesqenv/bin/python`) | PESQ, unavailable natively |

**Verified quantizable layers — there are exactly 7:**

```
/stem/stem.0/Conv        stem, 3->96 ch
/blocks.0/conv/Conv      gated TF block, dilation 1
/blocks.1/conv/Conv      gated TF block, dilation 2
/blocks.2/conv/Conv      gated TF block, dilation 4
/blocks.3/conv/Conv      gated TF block, dilation 8
/band_gru/GRU            band-wise recurrent decoder
/head/MatMul             output projection
```

`onnxruntime.quantization.quantize_static` exposes `nodes_to_quantize`, `nodes_to_exclude`,
`per_channel`, `weight_type`, `activation_type` — verified present in ORT 1.20.1. Per-layer
control is available in tooling that already works.

---

## 4. Data splits — leakage discipline

This project has already been bitten once by leaky evaluation (scene classifier: per-clip CV
inflated accuracy 0.81 → 0.75 under grouped CV). Do not repeat it. Bit allocation is a **model
selection** decision, so selecting it on the test set and then reporting on the test set is
selection leakage.

Verified speaker inventory:

- `vbd_local/train_manifest.csv` — 5000 clips, **13 speakers**
  (p226, p227, p228, p230, p231, p233, p236, p239, p243, p244, p250, p254, p256)
- `vbd_local/test_manifest.csv` — 824 clips, **2 speakers** (p232, p257)

Split to use — **speaker-disjoint at every level**:

| Split | Source | Size | Used for |
|---|---|---|---|
| **Calibration** | 11 train speakers (exclude p254, p256) | 64 clips | QDQ activation calibration only |
| **Dev** | train speakers **p254, p256** only | ~200 clips (subsample) | sensitivity ranking + allocation search |
| **Test** | `test_manifest.csv` (p232, p257) | 824 clips | final numbers, touched **once** |
| **External** | `external_data/ms_snsd_sona/test.csv` | 1100 clips | generalization, touched **once** |

Rules: calibration and dev never share a speaker. The test set is not looked at until the
allocation is frozen. Write the frozen allocation to a file with a timestamp before running test.

Deliverable: `scripts/make_quant_splits.py` → emits `calib_manifest.csv`, `dev_manifest.csv`.

---

## 5. Phase 0 — Feasibility probe (~2 hours). **Do this before anything else.**

Four things can kill this study. Check all four before writing the real pipeline.

**0a. Is there enough degradation budget to decompose?  ← THE BIG RISK**

Current uniform INT8 costs only **−0.124 dB SI-SDR** (18.692 → 18.568). If the *total* budget is
that small, per-layer contributions are ~0.02 dB each and will sit below run-to-run noise. The
study would have no measurable signal at INT8.

Measure: FP32 vs uniform-INT8 on the dev split, per-clip paired, with bootstrap CI on ΔSTOI and
ΔSI-SDR. **Decision rule:** if the uniform-INT8 drop is not statistically distinguishable from
zero on ~200 dev clips, INT8 is the wrong operating point — move the whole study to INT4 (or
INT4/INT8 mixed), where degradation is large enough to differentiate layers. If INT4 is also
unsupported (0c), **abandon** — there is nothing to measure.

**0b. Does `nodes_to_quantize` give real per-layer control?**

Export with `nodes_to_quantize=['/head/MatMul']` only. Confirm: (i) it succeeds, (ii) the output
differs from FP32, (iii) it differs from uniform INT8. If per-layer selection is silently ignored,
the whole design fails.

**0c. Does ORT 1.20.1 support INT4 for these op types?**

`QuantType.QInt4` exists in recent ORT but Conv/GRU INT4 QDQ support is uncertain. Test one
Conv layer at INT4. **Fallback if unsupported:** make the precision axis per-layer
∈ {FP32, INT8} — i.e. "which layers must stay FP32" — which is still a valid, fully-supported
mixed-precision study, just with a smaller dynamic range. Note that the fallback interacts badly
with 0a; if INT8 degradation is unmeasurable AND INT4 is unsupported, stop.

**0d. Does ORT have a quantized GRU kernel?**

In QDQ format the op stays float and ORT fuses it into a quantized kernel *only if one exists*.
There is likely **no INT8 GRU kernel**, meaning GRU quantization costs quality with **no speed or
size benefit** in practice. Verify by timing GRU-only-quantized vs FP32. If confirmed, this is
itself a reportable finding ("the recurrent decoder is not profitably quantizable on this
runtime") and the GRU should be pinned to FP32 in all allocations.

**Also fix before trusting any RTF number:** the model-forward RTF measured 0.480 (1 thread,
20 clips) in one run and 0.121 in the earlier `bench_onnx_edge` run on the same machine and
thread count. That 4× gap is unexplained — likely background load. Re-run both cleanly, on an
idle machine, with warmup iterations discarded, before any RTF goes in the paper.

---

## 6. Phase 1 — Per-layer sensitivity (~1 day)

New: `scripts/quant_sensitivity.py`

For each of the 7 layers L (plus an all-FP32 reference and an all-INT8 reference):

1. Export a variant with **only L quantized** (`nodes_to_quantize=[L]`), calibrated on the
   calibration split.
2. Run over the dev split through the existing harness.
3. Record per-clip: STOI, ESTOI, SI-SDR, LSD, mask-domain MSE.

Compute for each layer, as **paired per-clip deltas vs the FP32 reference**, with 95% bootstrap
CIs (reuse `_bootstrap_ci` from `analyze_mos.py`):

- **Perceptual sensitivity:** ΔSTOI, ΔESTOI (native locally), ΔPESQ (WSL, confirmatory)
- **MSE-family sensitivity (the comparison baselines):**
  - `w_mse[L]` = ‖W − Q(W)‖²_F, computed directly from the checkpoint, no inference needed
  - ΔSI-SDR (signal fidelity — the MSE-like audio metric)
  - Δ mask-MSE = ‖M_fp32 − M_quant‖² (very cheap: no ISTFT, no reference needed)

**Output:** a sensitivity table + a rank-correlation analysis. The headline number is
**Spearman ρ between the perceptual ranking and each MSE-family ranking.** Low ρ ⇒ the rankings
genuinely differ ⇒ the hypothesis has legs. ρ ≈ 1 ⇒ null result, write it up as such and stop.

Cost: 7 exports + ~9 dev evaluations. Minutes each on 200 clips.

---

## 7. Phase 2 — Bit allocation search (~1 day, mostly unattended)

New: `scripts/quant_allocate.py`

With only 7 layers the search space is small enough to be **exhaustive**, which removes any
"did the search heuristic bias the result" objection:

- Primary axis: each layer ∈ {FP32, INT8} → 2⁷ = **128 configurations**
- If Phase 0c passes, extend to {FP32, INT8, INT4} → 3⁷ = 2187; use greedy descent from the
  Phase-1 ranking instead of exhaustive, then exhaustively confirm the top-10 neighbourhood.
- Secondary axis on the winner only: `per_channel` ∈ {True, False}

Estimated cost: ~200 dev clips ≈ 1000 s audio; at RTF ≈ 0.12 that is ~2 min/config ⇒ 128 configs
≈ 4 h unattended. Halve the dev set to ~100 clips for the sweep if that is too slow, then
re-score the top-10 on the full dev set.

For each config record: model size (bytes), mean STOI/ESTOI/SI-SDR/LSD on dev, and RTF.

**Then build the Pareto front of size (and RTF) vs perceptual quality, and compare four
allocation strategies at matched size:**

1. **Uniform INT8** — the current baseline (0.95 MB, −0.124 dB SI-SDR)
2. **MSE-guided** — allocate by the `w_mse` / ΔSI-SDR ranking
3. **Perceptually-guided** — allocate by the ΔSTOI / ΔPESQ ranking
4. **FP32** — quality upper bound (2.84 MB)

The scientific result is whether (3) dominates (2) on the size-vs-PESQ/STOI Pareto front.

**Freeze the winning allocation to a JSON file with a timestamp before touching test data.**

---

## 8. Phase 3 — Validation (~half day)

Frozen allocation only, run once:

- **VoiceBank-DEMAND test**, all 824 clips — full metric suite. Paired Wilcoxon signed-rank +
  bootstrap CIs vs uniform INT8 and vs FP32, matching the statistics already used in
  `scripts/significance.py`.
- **MS-SNSD external**, 1100 clips (`external_data/ms_snsd_sona/test.csv`) — confirms the
  allocation is not overfit to VoiceBank noise. This matters: the project's earlier online-adapter
  result looked good on VoiceBank and **failed to reproduce on MS-SNSD**. Expect the same scrutiny
  here.
- **PESQ** via the WSL `pesqenv` path (`scripts/score_pesq_wsl.py` pattern) — required, since the
  central claim is perceptual.
- **RTF / latency / model size** via `bench_onnx_edge.py` at 1 and 2 threads.
- **WER** via the now-working `scripts/eval_wer.py` — does perceptual allocation also preserve
  downstream ASR? Cheap to add, strengthens the story.

---

## 9. Phase 4 — Paper artifacts

- **Table:** per-layer sensitivity (ΔSTOI / ΔPESQ / ΔSI-SDR / w_mse) with CIs and rank correlations
- **Figure:** Pareto front — model size vs PESQ, four allocation strategies
- **Figure:** per-layer sensitivity bar chart, perceptual vs MSE ranking side by side
- **Table:** final frozen allocation on VoiceBank-824 + MS-SNSD-1100, vs uniform INT8 and FP32
- **Prose:** one paragraph in Related Work on PTQ / mixed precision; a Limitations entry noting
  edge results remain *simulated*, not Raspberry Pi measurements

---

## 10. Deliverables

```
scripts/make_quant_splits.py     speaker-disjoint calib/dev manifests
scripts/quant_sensitivity.py     Phase 1
scripts/quant_allocate.py        Phase 2 exhaustive/greedy search
scripts/quant_validate.py        Phase 3 frozen-allocation validation
scripts/export_onnx.py           EXTEND: --precision-map / --nodes-to-quantize
results/perceptual_mp/           sensitivity.csv, sweep.csv, allocation.json, final/
tests/test_quant_alloc.py        smoke tests (tiny model, no dataset)
```

---

## 11. Risks and kill criteria

| Risk | Likelihood | Kill criterion |
|---|---|---|
| INT8 degradation too small to decompose (§0a) | **High** | Uniform-INT8 ΔSTOI CI includes 0 on dev ⇒ move to INT4; if INT4 unsupported, **abandon** |
| INT4 unsupported for Conv/GRU in ORT 1.20.1 | Medium | Fall back to FP32/INT8 axis; combined with above, may end the study |
| Perceptual and MSE rankings coincide | Medium | Not a kill — write up as a negative result, ~1 day to document |
| No quantized GRU kernel (§0d) | High | Not a kill — pin GRU to FP32, report as a finding |
| Prior art already covers it (§2) | Medium | Reframe as replication on a new architecture; drop novelty claim |
| Gains vanish on MS-SNSD | Medium | Report honestly as dataset-sensitive, as with the online adapter |
| Eats thesis time | **High** | Hard stop — thesis wins, no exceptions |

**Total effort if everything works: ~3 focused days.** Phase 0 alone is ~2 hours and decides
whether the remaining ~3 days are worth spending. Do not skip it.

---

## 12. Honest positioning

This is a **systems/empirical contribution**, not a theoretical one. TurboQuant has
information-theoretic lower-bound proofs from Google Research; this is a measured bit-allocation
study on one frozen 722 K-parameter model. Positioned that way it is a solid fit for IEEE Access
or EURASIP JASMP (the targets already recommended). Positioned as a theoretical advance it will
not survive review.
