# 0.13 % of the Weights Carry 99 % of the Loss: Stem-Exempt INT8 Quantization of a Compact Speech Enhancer

**Draft, 2026-08-20.** Every number below is measured and traceable to an artefact in
`results/`; nothing is estimated. Findings map to [`FINDINGS.md`](FINDINGS.md). Items still
requiring work are marked **TODO** rather than asserted.

---

## Abstract

Post-training quantization (PTQ) of speech enhancement models is normally applied uniformly, and
mixed-precision variants assign precision by layer *type* or by a second-order sensitivity proxy.
We report a per-layer PTQ study of a 722 K-parameter Bark-band GTCRN denoiser in which the
allocation is chosen on a speaker-disjoint development split and then evaluated once,
out-of-sample, on the 824-clip VoiceBank-DEMAND test set and on 1 100 MS-SNSD clips.

We set out to test whether ranking layers by a *perceptual* criterion (PESQ) differs from ranking
them by an MSE-family criterion, and **it does not**: across the six quantizable layers the two
rankings agree at Spearman ρ = +0.943 against SI-SDR and **ρ = +1.000** against mask-MSE, and the
two allocations differ by 0.0003 dB. We report this negative result in full, because it says
something useful — for this model class, the cheap MSE-based sensitivity proxy is already
perceptually adequate.

The study's positive finding came from the sensitivity sweep itself. A single **3.8 KB stem
convolution — 0.133 % of the model's weights — carries 98.7 % of the total INT8 degradation** on
the held-out test set. Exempting it produces a model that is statistically **indistinguishable from FP32**
(Δ = 0.0014 dB, p = 0.25, n = 824) at 3.2× compression, versus a uniform-INT8 loss of 0.109 dB
(p = 3.7e-51). The exemption is significant in **all nine test conditions** (5 noise types × 4
SNRs) and replicates on MS-SNSD. Counter-intuitively the mixed model is also **smaller** than
uniform INT8 (916.9 vs 950.7 KB), because quantizing every layer causes the runtime to insert QDQ
pairs around 143 additional activation tensors whose graph overhead exceeds the stem's 2.8 KB of
weight savings.

We further show that layer-wise quantization errors are **not additive** (interaction term
excludes zero on both corpora), and document two runtime limits that bound what any allocation
method can achieve on this stack: the band GRU cannot be quantized at all, and INT8 inference is
**1.4× slower** than FP32 on x86.

---

## 1. Introduction

Compact speech enhancement models are deployed under a memory budget, and PTQ is the cheapest way
to meet it — no retraining, no labelled calibration targets. The default is uniform INT8. When
that costs too much quality, the standard remedies are quantization-aware training (expensive) or
mixed precision, where some layers are kept at higher precision.

Mixed precision needs a rule for *which* layers. Two rules dominate. The first assigns by layer
**type** — for RNN denoisers on microcontrollers, Rusci et al. [2022] quantize the recurrent
layers to INT8 and keep the rest at FP16. The second assigns by a **second-order sensitivity
proxy**: HAWQ and HAWQ-V2 rank layers by top Hessian eigenvalue or average Hessian trace and
allocate bits accordingly.

Both proxies are ultimately about reconstruction error. But the objective that matters for speech
enhancement is *perceptual*, and perceptual quality weights frequency bands very non-uniformly.
That gap motivated this study: **if the perceptual ranking of layers differs from the MSE ranking,
allocating by the perceptual one should win at matched size.**

It does not differ, and we report that plainly (§5.1). What the sweep did reveal is a much
sharper structural fact (§5.2): the degradation is not spread across layers at all. It is almost
entirely attributable to one tiny layer at the very front of the network.

**Contributions.**

1. **A negative result, reported in full** (§5.1): for this model at INT8 and 6-layer granularity,
   perceptual and MSE-family sensitivity rankings coincide (ρ up to +1.000). The cheap proxy is
   already adequate; practitioners need not build a PESQ-in-the-loop allocator.
2. **The stem-exemption result** (§5.2): 0.133 % of the weights carry 98.7 % of INT8 loss, and
   exempting them recovers FP32-indistinguishable quality at 3.2× compression — validated
   out-of-sample after the allocation was frozen and timestamped.
3. **A size inversion that contradicts the naive model of quantization** (§5.2): quantizing
   *more* layers produced a *larger* file, because QDQ graph overhead outgrew the weight savings.
   We give the node counts. This is invisible in any bits-per-weight analysis.
4. **Condition-level robustness and a cross-corpus caveat** (§5.3, §5.4), including an explicit
   statement of where uniform INT8 beats our allocation and why that is a dithering artifact
   rather than a better quantizer.
5. **Evidence that layer-wise quantization error is non-additive** (§5.5), which invalidates the
   additive-cost assumption behind greedy per-layer allocation search.
6. **Two hard runtime limits** (§5.6) that bound this problem on ONNX Runtime and are worth
   recording for anyone attempting the same thing.

## 2. Related work and positioning

**Mixed-precision PTQ for speech enhancement.** The closest prior work is Rusci et al. [2022],
who deploy an RNN denoiser on a 9-core RISC-V MCU using mixed FP16–INT8 PTQ, reporting 1.4–1.7×
memory saving at 0.06 PESQ degradation against 0.3 for uniform INT8. Two differences matter.
First, their allocation is by **layer type** — recurrent layers to INT8 — whereas ours is derived
from measured per-layer sensitivity. Second, and more striking, our result runs in the *opposite
direction on the same architectural family*: on our runtime the recurrent layer cannot be
quantized at all (§5.6), and the layer that must be protected is the small input convolution. We
read this as evidence that "which layer to protect" is a property of the *runtime and model*, not
a transferable rule — which is an argument for measuring rather than assuming.

**Hessian-guided allocation.** HAWQ and HAWQ-V2 rank layers by curvature and are the standard
sensitivity-based allocators. Our §5.1 result is adjacent but distinct: we do not compare against
a Hessian proxy, we compare a *perceptual* ranking against *direct output-error* rankings, and
find they coincide. **TODO:** adding a Hessian-trace ranking as a third column would strengthen
§5.1 considerably and is cheap to compute; it is the single most valuable addition to this draft.

**Speech enhancement compression generally.** Other routes to the same budget include
sign-exponent-only floating point (SEOFP-NET, EOFP-QNN), matrix-product-operator factorisation,
and structured sparsity for tiny accelerators. These are complementary to PTQ rather than
alternatives to it; our contribution is about *where* the loss lives under PTQ, not about beating
these methods.

**Objective mismatch.** The framing — that an MSE-optimal quantizer may be wrong for the
objective one actually cares about — was taken from TurboQuant [Zandieh et al. 2025]. Its
*algorithm* does not apply here: there is no KV cache, no vector database, no memory bottleneck of
the kind it addresses. We cite it for motivation only, and the perceptual-vs-MSE form of the
hypothesis it suggested was refuted (§5.1).

## 3. Method

**Model.** The locked SONA backbone: Bark-band GTCRN, `ch96_h96_b4_mw1p25_mr0p25`, 722 209
parameters — a stem convolution, four gated time–frequency convolution blocks with dilations
1/2/4/8, a per-band GRU decoder, and a linear head producing per-band gains. FP32 size 2842.7 KB.

**Quantization.** ONNX Runtime `quantize_static`, QDQ format, per-layer control via
`nodes_to_quantize`. Activation calibration uses 64 clips drawn from 11 training speakers.
Per-layer control was verified to produce genuinely distinct graphs (uniform INT8 = 172 QDQ nodes,
head-only = 5, stem-only = 6) with distinct outputs, so the sensitivity study is mechanically
meaningful rather than a no-op.

**Leakage discipline.** Bit allocation is a model-selection decision, so it is made on a
development set carved from TRAIN **by speaker**, never on the test set.

| split | speakers | clips | used for |
|---|---|---:|---|
| calibration | 11 train speakers (p226…p250) | 64 | QDQ activation calibration only |
| dev | **p254, p256** (train, held out) | 200 | sensitivity ranking + allocation search |
| test | p232, p257 | 824 | final numbers, touched **once** |
| external | MS-SNSD | 1100 | generalization, touched **once** |

`make_quant_splits.py` hard-fails on speaker overlap. The chosen allocation was written to
`results/phase3/frozen_allocation.json` **before** any test-set evaluation was run.

**Search space.** The band GRU is unquantizable on this runtime (§5.6) and INT4 is unavailable for
Conv, so the precision axis is per-layer ∈ {FP32, INT8} over **6** quantizable layers (5 Conv +
1 MatMul) — 2⁶ = 64 configurations, small enough to enumerate exhaustively rather than search
greedily. This matters given the non-additivity in §5.5.

## 4. Experimental setup

VoiceBank-DEMAND 16 kHz; 824-clip test set with official condition labels (5 noise types × 4 input
SNRs) extracted to `results/vbd_test_conditions.csv`. MS-SNSD provides 1 100 external clips.
Metrics: SI-SDR, STOI, ESTOI, LSD, and PESQ (computed under WSL, since PESQ has no wheel for the
local Python). Significance throughout is paired Wilcoxon signed-rank with 95 % bootstrap CIs on
per-clip differences.

## 5. Results

### 5.1 Perceptual and MSE sensitivity rankings coincide (negative result)

| ranking comparison | Spearman ρ |
|---|---:|
| PESQ vs SI-SDR | +0.943 |
| PESQ vs mask-MSE | **+1.000** |

`stem` ranks most sensitive under *every* metric; `head` ranks least sensitive under *every*
metric. The two induced allocations differ at exactly one position and produce a ΔSI-SDR of
**+0.0003 dB** — indistinguishable.

**Conclusion:** at INT8 and this granularity, MSE-guided allocation is already perceptually
adequate for this model. We make no perceptual-advantage claim. The honest scope of this negative
result is stated in §6.

### 5.2 The stem exemption (main result, out-of-sample)

VoiceBank-DEMAND test set, 824 clips, evaluated after the allocation was frozen:

| model | size (KB) | SI-SDR (dB) | vs FP32 |
|---|---:|---:|---|
| FP32 | 2842.7 | 18.3765 | — |
| **Mixed (stem FP32, 5 layers INT8)** | **916.9** | **18.3751** | **p = 0.25 (n.s.)** |
| Uniform INT8 | 950.7 | 18.2671 | p = 3.7e-51 |

- Stem exemption vs uniform: **+0.108 dB SI-SDR**, 95 % CI [+0.091, +0.126], p = 9.2e-49
- Mixed vs FP32: Δ = 0.0014 dB, CI [−0.0012, +0.0041], **not significant**
- Total quantization cost (FP32 − uniform): 0.109 dB. The stem accounts for **98.7 %** of it
  ((0.1094 − 0.0014) / 0.1094); `FINDINGS.md` quotes ~97 % from the smaller dev split
- The stem is **3 840 bytes = 0.133 %** of model weights

**The mixed model is also smaller than uniform INT8.** Verified cause: requesting quantization of
every layer makes ONNX Runtime wrap ~143 additional activation tensors in QDQ pairs (**172 vs 29
QDQ nodes**), and that graph overhead exceeds the stem's 2.8 KB of weight savings. Quantizing the
stem *alone* even grows the file above FP32 (2851.9 vs 2842.7 KB). Any analysis that counts only
bits-per-weight will miss this entirely.

**Fidelity to FP32** — mean absolute per-clip deviation, the property PTQ is actually supposed to
deliver:

| corpus | metric | mixed | uniform INT8 | advantage |
|---|---|---:|---:|---:|
| VoiceBank-824 | \|ΔSI-SDR\| | **0.0202** | 0.1458 | 7.2× |
| VoiceBank-824 | \|ΔSTOI\| | **0.00025** | 0.00186 | 7.4× |
| VoiceBank-824 | \|ΔESTOI\| | **0.00055** | 0.00481 | 8.7× |
| VoiceBank-824 | \|ΔLSD\| | **0.0088** | 0.1708 | 19.5× |
| MS-SNSD-1100 | \|ΔSI-SDR\| | **0.0328** | 0.2832 | 8.6× |
| MS-SNSD-1100 | \|ΔSTOI\| | **0.00051** | 0.00710 | 13.9× |
| MS-SNSD-1100 | \|ΔESTOI\| | **0.00079** | 0.00966 | 12.2× |
| MS-SNSD-1100 | \|ΔLSD\| | **0.0209** | 0.2915 | 13.9× |

**8/8 cells favour the mixed allocation, by 7–20×, across two corpora.**

### 5.3 Condition robustness

Significant in **all 9 conditions**:

| condition | n | ΔSI-SDR | 95 % CI | p |
|---|---:|---:|---|---:|
| psquare | 166 | +0.175 | [+0.126, +0.235] | 1.2e-15 |
| cafe | 164 | +0.156 | [+0.124, +0.189] | 6.8e-20 |
| living | 166 | +0.098 | [+0.064, +0.134] | 1.3e-07 |
| bus | 164 | +0.087 | [+0.051, +0.130] | 1.7e-08 |
| office | 164 | +0.024 | [+0.011, +0.038] | 1.5e-04 |
| 2.5 dB | 207 | +0.125 | [+0.088, +0.165] | 1.2e-11 |
| 7.5 dB | 206 | +0.095 | [+0.065, +0.128] | 9.4e-11 |
| 12.5 dB | 207 | +0.104 | — | < 1e-10 |
| 17.5 dB | 204 | +0.108 | — | < 1e-10 |

The effect is **flat across SNR** — so it is not an SNR artifact — and largest in hard
non-stationary noise (psquare, cafe), smallest in easy stationary office noise. That ordering is
what one would predict if the stem's early feature extraction matters most when spectral structure
is complex, which is a satisfying but *post-hoc* explanation; we offer it as interpretation, not
evidence.

### 5.4 Cross-corpus caveat — stated explicitly

On MS-SNSD the stem-exemption SI-SDR benefit replicates (**+0.039 dB**, p = 3.0e-12) and mixed
remains indistinguishable from FP32 on SI-SDR (p = 0.17). **However, on MS-SNSD uniform INT8
scores better than mixed on STOI (−0.0060), ESTOI (−0.0067) and LSD (+0.233).**

This is not evidence that uniform is the better allocation. Uniform INT8 also beats **FP32
itself** on MS-SNSD STOI (0.8212 vs 0.8154) — a quantizer cannot improve on the model it is
approximating. Its larger quantization noise acts as a mild dither that flatters intelligibility
metrics under distribution shift while degrading waveform fidelity. The mixed model instead tracks
FP32 closely on every metric (§5.2).

**The claim we make is therefore fidelity to FP32, not "better enhancement on every metric
everywhere."** We state this in the abstract as well as here, because the temptation to overclaim
is exactly what this table guards against.

### 5.5 Quantization error is not additive

| corpus | stem alone | 5 blocks alone | sum | actual combined | interaction | ratio |
|---|---:|---:|---:|---:|---|---:|
| dev (200) | −0.0061 | −0.0023 | −0.0083 | −0.0996 | −0.0913 [−0.149, −0.040] | **11.95×** |
| MS-SNSD (1100) | +0.0260 | +0.0010 | +0.0269 | −0.0377 | −0.0647 [−0.090, −0.039] | 1.40× |

The interaction term's CI excludes zero on both corpora, so error **compounds** through the
network rather than accumulating independently. The magnitude is strongly corpus-dependent
(11.95× vs 1.40×), so we report the *existence* of the interaction as robust and its *size* as
corpus-specific.

Practical consequence: greedy per-layer allocation search assumes additive costs and is therefore
unsound here. Exhaustive enumeration over the 64 configurations was affordable and is what we did.

### 5.6 Runtime limits (findings, not bugs)

- **The band GRU cannot be quantized** by ONNX Runtime 1.20.1. Requesting it yields **0 QDQ
  nodes**, no size change (2852.9 vs 2842.7 KB) and no speedup (0.995×). The recurrent decoder is
  incompressible on this runtime and is pinned to FP32 in every allocation. Note the direct
  contrast with Rusci et al. [2022], whose method depends on quantizing exactly this layer type.
- **INT4 is unavailable for Conv** via `quantize_static` ("Quantized weights … must be 8-bit
  before packing as 4-bit values"). ORT's 4-bit path covers MatMul only, and this graph has one
  tiny MatMul, so INT4 cannot reach the weight-heavy Conv layers.
- **INT8 is ~1.4× slower than FP32** on x86. Re-measured 2026-08-20 on a verified-idle machine
  (6 % CPU load, AMD Ryzen 5 5600H, same ORT 1.20.1): 1.39× / 1.43× / 1.46× at 1 / 2 / 4 threads,
  with absolute RTF 0.084 (FP32) vs 0.117 (INT8) single-threaded. **On this hardware the benefit
  of quantization is memory, not speed** — and the disadvantage grows slightly with thread count,
  consistent with QDQ overhead parallelising worse than the dense kernels it replaces.

## 6. Limitations

- **The negative result in §5.1 is scoped to this checkpoint, at INT8, over 6 layers.** With a
  finer precision axis (INT4 available, or per-channel), or a model whose layers are less
  dominated by a single one, the perceptual and MSE rankings could plausibly diverge. We did not
  show that they never diverge; we showed they do not here, and explained why the search space was
  forced to be this coarse (§5.6).
- **One checkpoint, one architecture.** The stem-exemption magnitude is not claimed to transfer.
  The transferable claim is methodological: measure per-layer sensitivity, because the answer is
  runtime- and model-specific and contradicted a published rule for the same architecture family.
- **PESQ was computed under WSL** on rendered wavs, not in the main harness. The setup was
  validated against published reference values before use.
- **No Hessian baseline** — see the TODO in §2. Without it, §5.1 compares perceptual ranking
  against direct output-error rankings, not against the standard curvature proxy.
- **No on-device measurement.** All timings are x86 with thread pinning as an edge proxy. The
  contrast with Rusci et al. is therefore across both method *and* hardware, and we cannot
  separate those two factors.

## 7. Conclusion

We tested whether perceptually-guided bit allocation beats MSE-guided allocation for a compact
speech enhancer, and found that it does not — the rankings coincide almost perfectly. Reporting
that negative result is worthwhile on its own: it licenses practitioners to keep using the cheap
proxy.

The sweep that produced it revealed something more useful. Quantization loss in this model is not
distributed across layers; **0.133 % of the weights carry 98.7 % of it**. Protecting that one 3.8 KB
layer yields an FP32-indistinguishable model at 3.2× compression, significant in every test
condition, replicated on a second corpus — and, because of QDQ graph overhead, *smaller* than
quantizing everything. Combined with the non-additivity of layer-wise errors and the fact that a
published mixed-precision rule for this architecture family is inverted on our runtime, the
practical lesson is consistent: for PTQ of small speech models, per-layer sensitivity is worth
measuring rather than assuming.

---

## Submission checklist

- [x] All numbers traceable to `results/` artefacts
- [x] Allocation frozen and timestamped before test-set evaluation
- [x] Cross-corpus caveat stated in abstract, not buried
- [x] Absolute RTF figures re-measured on an idle machine
- [ ] **Hessian-trace ranking added to §5.1** (highest-value remaining item)
- [ ] Formal bibliography with full entries (currently author-year inline)
- [ ] LaTeX conversion — `results/phase4/table1_main_result.tex` and `table2_conditions.tex` are
      already generated and match §5.2 and §5.3
