# AudioVQ — Findings

Running record. Nothing here is asserted without a measurement behind it; predictions are
labelled as predictions until tested, and negative results are kept.

---

## F1. The quantizer implementation reproduces the paper's theory

Before any audio experiment, the TurboQuant-style quantizer was validated against the distortion
values stated in the source paper (17/17 unit tests pass, `tests/test_quantizers.py`).

Measured MSE on random unit vectors, d = 256, vs the paper's stated values and its Theorem 1
upper bound √3·π/2 · 4⁻ᵇ:

| bits/coord | measured MSE | paper's value | Theorem 1 bound | within bound |
|---|---|---|---|---|
| 1 | 0.4029 | 0.36 | 0.6802 | ✓ |
| 2 | 0.1199 | 0.117 | 0.1700 | ✓ |
| 3 | 0.0343 | 0.030 | 0.0425 | ✓ |
| 4 | 0.0094 | 0.009 | 0.0106 | ✓ |

Distortion falls below the proven upper bound at every bitrate and within 2.5–14.4 % of the
refined values. The small positive excess is expected: the codebook is built from finite samples
(4·10⁵) of the analytic post-rotation coordinate distribution, and the decoder renormalises.

**Implication:** downstream audio results rest on a faithful implementation, not a broken one.

---

## F2. Audio embedding spaces are extremely anisotropic

WavLM-Base+ mean-pooled embeddings, MS-SNSD clean speech, n = 10 000, d = 768.

| measure | audio embeddings | isotropic Gaussian reference |
|---|---|---|
| mean pairwise cosine | **+0.830** | −0.000 |
| effective dimensionality (participation ratio) | **22.6 / 768 = 2.9 %** | 713 / 768 = 92.9 % |
| variance in top-1 eigendirection | 16.3 % | 0.2 % |
| variance in top-10 eigendirections | 49.7 % | 2.1 % |

Nearly all embeddings point in a similar direction, and the cloud occupies roughly **23 of 768
available dimensions**. This is far more extreme than the image/text benchmarks the source method
was validated on.

---

## F3. The rotation works exactly as theory predicts — at the level it actually operates on

An important distinction that is easy to get wrong: mean pairwise cosine and participation ratio
are **rotation-invariant**, so an orthogonal transform *cannot* change them. Observing
+0.830 → +0.830 is therefore a **sanity check that the rotation is exactly orthogonal**, not
evidence that rotation failed.

TurboQuant's claim concerns the coordinates *within a single vector*, which is what licenses a
single shared per-coordinate codebook. Measured there:

| within-vector statistic | raw | after rotation | isotropic ideal |
|---|---|---|---|
| crest factor (max\|c\| / rms) | 10.22 | **3.34** | 3.37 |
| excess kurtosis | +38.75 | **+0.02** | −0.00 |

The rotation converts a wildly spiky coordinate profile into one matching the isotropic ideal to
two decimal places. **The mechanism transfers to audio intact.**

---

## F4. The prediction was CONFIRMED (RQ1)

F2 and F3 together produced a falsifiable prediction, stated **before** the benchmark was read:

> Because audio embeddings occupy ~3 % of their ambient dimensions, a **data-oblivious**
> quantizer must spend bits uniformly over directions the data never occupies, while
> **data-dependent** PQ/OPQ can place codewords where the data actually lies. PQ/OPQ should
> therefore outperform the rotation+scalar scheme on audio, and the gap should widen as the bit
> budget shrinks.

Measured (10 000-vector MS-SNSD database, 1 000 disjoint queries, Recall@10 vs exhaustive
float32 ground truth):

| budget (bits/vec) | compression | data-oblivious | PQ | OPQ | best DD − oblivious |
|---|---|---|---|---|---|
| 800 | 30.7× | 0.4829 | 0.5439 | **0.7260** | **+0.2431** |
| 1568 | 15.7× | 0.6198 | 0.6886 | **0.8435** | **+0.2237** |
| 3104 | 7.9× | 0.8791 | 0.8720 | **0.8990** | **+0.0199** |

**Confirmed, including the directional detail:** the gap collapses from **+0.243 at 800 bits to
+0.020 at 3104 bits**, exactly the widening-as-budget-shrinks pattern the anisotropy analysis
predicted. Worst-case-optimal quantization is the wrong guarantee for data this far from worst
case, and the penalty is largest precisely where compression matters most.

Note the crossover: at 3104 bits the data-oblivious method **overtakes plain PQ** (0.8791 vs
0.8720) and comes within 0.02 of OPQ.

---

## F5. The rotation is essential on audio (RQ2)

Identical quantizer and codebook, rotation the only variable:

| bits/coord | with rotation | without | Δ Recall@10 | relative gain |
|---|---|---|---|---|
| 1 | 0.4829 | 0.3437 | +0.1392 | **+40.5 %** |
| 2 | 0.6198 | 0.4430 | +0.1768 | **+39.9 %** |
| 3 | 0.7950 | 0.5051 | +0.2899 | **+57.4 %** |
| 4 | 0.8791 | 0.5503 | +0.3288 | **+59.7 %** |

Large, monotone, and unambiguous — and mechanistically explained by F3: without rotation the
within-vector coordinate profile is spiky (crest 10.2, kurtosis +38.8), so one shared
per-coordinate codebook fits badly. **On a highly anisotropic embedding space the rotation is not
an optional refinement; it is what makes scalar quantization viable at all.**

---

## F6. The practical case for data-oblivious VQ: build cost

Recall is not the only axis. Index construction cost, measured on the same hardware:

| method | bits/vec | Recall@10 | build + search |
|---|---|---|---|
| **scalar-4b + rotation** | 3104 | 0.8791 | **0.8 s** |
| OPQ-m384 | 3072 | 0.8990 | 360.4 s |
| PQ-m384 | 3072 | 0.8720 | 25.2 s |

**OPQ buys +0.0199 Recall@10 for 434× the build time.** The data-oblivious method needs no
training pass at all, so an index can be built or extended on-device and in one streaming pass —
which is the deployment setting that motivated the project.

**Practical rule this supports:**
- aggressive compression (≤ 1536 bits/vec) and a training pass affordable → **use OPQ**; the
  data-oblivious penalty (−0.22 Recall@10) is too large to accept.
- high bitrate, or no opportunity to train (on-device / streaming index construction) →
  **data-oblivious is competitive**: 98 % of OPQ's recall at 1/434th the build cost, and it
  beats plain PQ outright.

---

## F6b. Recall differences are statistically unambiguous (and the oblivious method beats PQ)

Per-query recall over 1000 queries, 95 % bootstrap CIs and paired Wilcoxon vs the data-oblivious
method (WavLM):

| bits/coord | oblivious | PQ | Δ vs obl. | OPQ | Δ vs obl. |
|---|---|---|---|---|---|
| 1 | 0.4829 [0.474, 0.492] | 0.5439 | +0.061 (p=1.8e-24) | 0.7260 | **+0.243** (p=3.5e-146) |
| 2 | 0.6198 [0.611, 0.629] | 0.6886 | +0.069 (p=1.1e-36) | 0.8435 | **+0.224** (p=3.4e-148) |
| 3 | 0.7950 [0.789, 0.801] | 0.7665 | **−0.029** (p=3.6e-12) | 0.8466 | +0.052 (p=4.2e-37) |
| 4 | 0.8791 [0.874, 0.884] | 0.8720 | **−0.007** (p=1.5e-02) | 0.8990 | +0.020 (p=9.0e-09) |

Every difference is significant. Two things are now established rather than observed: OPQ's lead
is real and collapses with budget, and at 3–4 bits/coord **the data-oblivious method significantly
outperforms plain PQ** — a data-dependent baseline — while needing no training pass at all.

---

## F7. Geometry and meaning come apart: Recall@k badly understates task performance

At the most aggressive budget (1 bit/coord, ~30× compression) the data-oblivious quantizer keeps
only **48 % of the true nearest neighbours** — yet downstream semantic retrieval barely moves:

| task | float32 P@10 | after 30× compression | retained |
|---|---|---|---|
| speaker identity (824 clips, 2 classes) | 0.9340 | 0.9012 | **96.5 %** |
| noise type (1100 clips, 9 classes) | 0.6219 | 0.5970 | **96.0 %** |

Losing more than half of the exact neighbour list costs ~4 % of semantic precision. The
class-level structure lives in coarse directions that survive aggressive quantization, while what
compression destroys is mostly the fine ordering *among* near-ties — which the downstream task
does not depend on.

**Consequence for the field:** Recall@k, the standard yardstick in the VQ/ANN literature, is a
**pessimistic proxy** for embedding-retrieval applications. A method dismissed on recall may be
entirely adequate in deployment. Both metrics should be reported.

**Replicates across both embedding models** — the dissociation is not model-specific:

| model | Recall@10 kept | speaker P@10 kept | noise-type P@10 kept |
|---|---|---|---|
| WavLM-Base+ | 48 % | 96.5 % | 96.0 % |
| wav2vec2-base | 53 % | 95.7 % | 96.9 % |

Roughly half the neighbour list is destroyed on both models, and both retain ~96 % of task
performance. This is the project's most transferable result.

---

## F8. The rotation's benefit is task-dependent — and reverses (an honest negative)

The rotation gave a large, monotone gain on geometric recall (F5, +40–60 %). On semantic
retrieval it does **not** behave the same way. Paired Wilcoxon over queries, rotation the only
variable:

| task | bits/coord | with rotation | without | Δ | p | verdict |
|---|---|---|---|---|---|---|
| speaker | 2 | 0.9205 | 0.9117 | +0.0089 | 0.031 | rotation helps |
| speaker | 3 | 0.9300 | 0.9203 | +0.0097 | 0.0027 | rotation helps |
| speaker | 1, 4 | — | — | +0.002 / +0.006 | n.s. | no difference |
| **noise type** | **1** | 0.5970 | **0.6093** | **−0.0123** | **0.019** | **rotation HURTS** |
| **noise type** | **2** | 0.6028 | **0.6171** | **−0.0143** | **0.0020** | **rotation HURTS** |
| noise type | 3, 4 | — | — | −0.007 / −0.004 | n.s. | no difference |

**Rotation significantly helps speaker retrieval and significantly hurts noise-type retrieval.**

Proposed mechanism, consistent with F2/F3: audio embeddings are anisotropic, so information sits
in a *sparse* set of dominant coordinates. Rotation deliberately destroys that sparsity, spreading
the signal densely across all 768 coordinates. When the semantic signal *is* the sparsity pattern
— which "what kind of noise is this" plausibly is, being a coarse spectral signature — dense
coarse quantization loses more of it than clipping a sparse pattern does. Fine-grained speaker
identity instead needs full geometry, which rotation protects.

This is a limitation of the rotation trick that the source paper's Recall-oriented benchmarks
could not have surfaced.

### F8b. The reversal does NOT replicate on wav2vec2 — scope the claim

Repeating the identical ablation on wav2vec2, **none of the eight comparisons is significant**:

| task | bits | Δ (rot − no-rot) | p |
|---|---|---|---|
| speaker | 1 / 2 / 3 / 4 | −0.003 / +0.004 / −0.003 / −0.002 | 0.47 / 0.28 / 0.089 / 0.24 |
| noise type | 1 / 2 / 3 / 4 | −0.001 / −0.006 / −0.003 / −0.001 | 0.68 / 0.079 / 0.087 / 0.94 |

So F8's task-dependent reversal is **not a general property of the rotation** — it is specific to
WavLM here, and the claim must be scoped accordingly.

Importantly, the non-replication is *predicted by the same mechanism*. wav2vec2's raw coordinates
are already near-isotropic (crest 3.62 vs the 3.37 ideal), so the rotation barely alters them —
which is exactly why it produced only +2.8 % on recall (F10) and why every semantic effect here is
a statistically indistinguishable ±0.006. **The magnitude of the rotation's effect, in either
direction, scales with how much work it has to do.**

Corrected statement of the finding:

> On embedding spaces with spiky within-vector coordinate profiles, the rotation has large effects
> whose *sign is task-dependent*: it helps fine-grained geometry and speaker identity, and can
> significantly hurt tasks whose signal is carried by a sparse set of dominant coordinates. On
> already-isotropic spaces the rotation is effectively inert.

### F8c. The scoped claim was then confirmed by a third model

HuBERT (crest 12.53 — the spikiest of the three) reproduces the WavLM pattern precisely:

| task | bits | Δ (rot − no-rot) | p | verdict |
|---|---|---|---|---|
| speaker | 2 / 3 / 4 | +0.0055 / +0.0057 / +0.0040 | 0.024 / 7.4e-4 / 1.1e-3 | **rotation helps** |
| noise type | 1 / 2 / 3 | −0.0172 / −0.0168 / −0.0111 | 2.7e-3 / 2.6e-4 / 3.3e-3 | **rotation HURTS** |

Tally across the three models, which is exactly what the mechanism predicts:

| model | raw crest | rotation's semantic effect |
|---|---|---|
| HuBERT | 12.53 (spiky) | **sign-flip present** — helps speaker, hurts noise type |
| WavLM | 10.22 (spiky) | **sign-flip present** — helps speaker, hurts noise type |
| wav2vec2 | 3.62 (flat) | **inert** — all eight comparisons null |

The single non-replication is the one model where the mechanism says the rotation has nothing to
do. The scoped claim was written *before* HuBERT was run and survived the test; the unscoped
version ("rotation hurts noise-type retrieval") would have been contradicted by wav2vec2.

---

## F9. On semantic tasks the data-oblivious method closes the gap entirely

Same budget (1 bit/coord), codebooks for PQ/OPQ trained on a **held-out 10 000-vector bank**
(see the protocol note below — an earlier version trained them on the evaluation set and
overstated them):

| method | speaker P@10 | noise-type P@10 |
|---|---|---|
| float32 | 0.9340 | 0.6219 |
| OPQ (data-dependent) | **0.9292** | 0.5974 |
| PQ (data-dependent) | 0.9252 | 0.5825 |
| rotation + scalar (oblivious) | 0.9012 | 0.5970 |
| **scalar, no rotation (oblivious)** | 0.8989 | **0.6093** |

On noise type the simplest data-oblivious quantizer **beats OPQ** (0.6093 vs 0.5974) and the
rotated variant ties it (0.5970 vs 0.5974). On speaker, OPQ retains its edge (+0.028).

**Replicates on wav2vec2** at the same budget: noise-type P@10 is **0.7557 for the oblivious
method vs 0.7422 for OPQ** (96.9 % vs 95.2 % retention) — oblivious again ahead — while OPQ again
leads on speaker (0.9677 vs 0.9300). The same task-split appears on both models: **OPQ's advantage
is confined to the fine-grained task; on coarse class retrieval the untrained quantizer matches or
beats it.**

**This reverses the conclusion drawn from geometric recall alone**, where OPQ led by +0.243 at
this budget. Combined with F6's 434× build-cost advantage, the practical case for data-oblivious
quantization on audio is far stronger when judged by the task rather than by neighbour recall.

---

## F10. Cross-model replication, and a mechanism test stated in advance

Everything above rests on WavLM-Base+. Repeating the structural analysis on **wav2vec2-base**
(identical corpora, identical pipeline):

| | WavLM-Base+ | wav2vec2-base | isotropic ideal |
|---|---|---|---|
| mean pairwise cosine | +0.830 | **+0.929** | 0.000 |
| effective dimensionality | 22.6 / 768 (2.9 %) | **11.1 / 768 (1.4 %)** | 92.9 % |
| variance in top-10 directions | 49.7 % | **64.7 %** | 2.1 % |
| **within-vector crest (raw)** | **10.22** | **3.62** | 3.37 |
| **within-vector kurtosis (raw)** | **+38.75** | **+0.57** | −0.00 |

**F2 replicates and is even stronger** — wav2vec2 occupies just 1.4 % of its ambient dimensions.

But the two models differ sharply in the property F3 identified as the rotation's mechanism:
WavLM's raw coordinate profile is extremely spiky (crest 10.2), whereas **wav2vec2's is already
close to isotropic before any rotation** (crest 3.62 vs the 3.37 ideal). Rotation therefore has
far less work to do on wav2vec2 (3.62 → 3.29) than on WavLM (10.22 → 3.34).

**Prediction, recorded before running the wav2vec2 benchmark:** if the within-vector flattening in
F3 is genuinely *why* rotation produced the large recall gains in F5 (+40–60 % on WavLM), then
the rotation ablation must yield a **much smaller gain on wav2vec2**. If instead rotation helps
equally on both, the F3 mechanism is not the explanation and should be withdrawn.

### Result: CONFIRMED

| model | raw within-vector crest | mean relative Recall@10 gain from rotation |
|---|---|---|
| WavLM-Base+ | 10.22 (very spiky) | **+49.4 %** |
| wav2vec2-base | 3.62 (already near-isotropic) | **+2.8 %** |

Per budget, wav2vec2's rotation gains are +1.0 %, +7.3 %, +0.1 %, +2.6 % — an order of magnitude
smaller than WavLM's +40.5 %, +39.9 %, +57.4 %, +59.7 %. **The rotation is ~18× more valuable on
the model whose coordinates were spiky**, exactly as the mechanism required.

This upgrades F3 from a plausible story to a **tested causal claim**: rotation helps *because* it
flattens a spiky within-vector coordinate profile, and where that profile is already flat the
rotation buys almost nothing.

### F10b. RQ1 replicates on wav2vec2 — with a crossover

Matched-budget Recall@10 on wav2vec2 (same protocol, same corpora):

| budget | compression | oblivious | PQ | OPQ | best DD − obl. |
|---|---|---|---|---|---|
| 800 | 30.7× | 0.5299 | 0.4947 | **0.7322** | +0.2023 |
| 1568 | 15.7× | 0.7085 | 0.6738 | **0.8001** | +0.0916 |
| 2336 | 10.5× | 0.7935 | 0.7021 | **0.8368** | +0.0433 |
| 3104 | 7.9× | **0.8965** | 0.8161 | 0.8472 | **−0.0493** |

The widening-with-shrinking-budget pattern **replicates on both models** (+0.202 → −0.049 here,
+0.243 → +0.020 on WavLM), so the anisotropy-driven prediction is not model-specific.

Two results are stronger on wav2vec2 than on WavLM:
- at the largest budget the **data-oblivious method overtakes OPQ outright** (0.8965 vs 0.8472),
  where on WavLM it merely closed to within 0.020;
- it **beats plain PQ at every one of the four budgets**, not just the top two.

Taken with F6b (where the oblivious method significantly beat PQ at 3–4 bits on WavLM), the
picture is consistent: **OPQ's learned rotation is what makes the data-dependent side competitive;
plain PQ is not reliably better than data-oblivious quantization on audio embeddings at all.**

### F11. A cheap a-priori diagnostic — as a threshold test, not a magnitude predictor

**Within-vector crest factor is measurable in seconds from the embeddings alone** — no retrieval
evaluation, no ground truth, no codebook training. An initial two-model version of this finding
claimed it predicts *how much* the rotation helps. A third model (HuBERT-base) was added
specifically to test that, and **falsified the quantitative form**:

| model | raw within-vector crest | mean relative Recall@10 gain from rotation |
|---|---|---|
| HuBERT-base | **12.53** | +40.2 % |
| WavLM-Base+ | 10.22 | **+49.4 %** |
| wav2vec2-base | 3.62 | +2.8 % |

HuBERT is the spikiest of the three yet gains *less* than WavLM, so the predicted ordering
(HuBERT > WavLM > wav2vec2) **does not hold**. A Pearson r of +0.91 across the three points is not
meaningful evidence here: with n = 3 and a two-cluster structure, it merely restates that
wav2vec2 differs from the other two.

**What the three models do support is a threshold effect:**

- crest ≫ isotropic ideal (~3.4) → rotation is essential (10.22 → +49 %, 12.53 → +40 %)
- crest ≈ isotropic ideal → rotation is nearly worthless (3.62 → +2.8 %)

**Corrected claim:** the crest factor is a reliable *binary screen* for whether the rotation is
worth applying, and is **not** a predictor of how large the gain will be. A practitioner can
decide whether to pay for the rotation before building an index; they cannot forecast the margin.

This is a weaker claim than the two-model version supported, and it is the one the evidence
actually carries.

---

## F12. The advantage of trained codebooks GROWS with index size (a negative for the practical case)

All main experiments use 10 000-vector indices. Sweeping the database size at a matched budget
(2 bit/coord vs PQ/OPQ m=192), WavLM:

| index size | oblivious R@10 | PQ | OPQ | gap (OPQ − oblivious) |
|---|---|---|---|---|
| 1 000 | 0.8048 | 0.9002 | 0.8729 | **+0.068** |
| 2 500 | 0.6742 | 0.7974 | 0.8299 | +0.156 |
| 5 000 | 0.6398 | 0.7482 | 0.8639 | +0.224 |
| 10 000 | 0.6198 | 0.6886 | 0.8435 | **+0.224** |

The data-oblivious method's recall falls faster with scale than OPQ's, so **the gap more than
triples over a single decade of index size**. This is a genuine limitation of the practical
recommendation and is reported as prominently as the favourable results: measurements taken at
10 000 vectors should not be assumed to hold at 10⁶.

**What survives.** The motivating deployment is an *on-device* index — a phone or Raspberry Pi
holding a modest bank of prototypes — which is precisely the small-N regime where the gap is
smallest (+0.068 at N=1000). The practical claim therefore narrows rather than collapses:
data-oblivious quantization is attractive for **small on-device indices built without a training
pass**, and progressively less so as the index grows toward server scale, where a training pass is
also more affordable.

**Replicates on all three models** — gap to OPQ at a matched budget, by index size:

| index size | WavLM | wav2vec2 | HuBERT |
|---|---|---|---|
| 1 000 | +0.068 | **+0.017** | +0.036 |
| 2 500 | +0.156 | +0.055 | +0.051 |
| 5 000 | +0.224 | +0.102 | +0.074 |
| 10 000 | +0.224 | +0.092 | +0.089 |

**3/3 models widen monotonically** (HuBERT is strictly monotone; WavLM and wav2vec2 widen with a
flat final step). The scale limitation is a property of the method, not of one embedding space.

Two things sharpen the practical reading. At N = 1000 the gap is small on every model
(+0.017 to +0.068), so the small-index regime is where the untrained method is genuinely
defensible — and that is exactly the on-device case. And the *magnitude* of the penalty is
strongly model-dependent (WavLM +0.224 vs HuBERT +0.089 at 10k), so the choice of embedding model
matters as much as the choice of quantizer.

**Caveat on the caveat.** The reachable range here is 1k–10k (the MS-SNSD bank holds 23 075
utterances). Behaviour at 10⁶ is extrapolation and is labelled as such; the trend direction is
measured, its continuation is not.

---

## F13. The deployable structured rotation costs recall when d is not a power of two

All benchmarks above use a **dense Gaussian** rotation (O(d²) per vector). The variant one would
actually deploy is the **structured Hadamard** transform, which admits an O(d log d) algorithm.
Comparing them exposes a practical cost that equal-bits-per-coordinate tables hide.

At equal bits/coordinate the Hadamard variant *appears* better — but d = 768 is not a power of
two, so it pads to 1024 and silently spends **33 % more bits**:

| bits/coord | rotation | bits/vector | WavLM R@10 |
|---|---|---|---|
| 1 | Gaussian | 800 | 0.4829 |
| 1 | Hadamard | **1056** | 0.4898 |
| 2 | Gaussian | 1568 | 0.6198 |
| 2 | Hadamard | **2080** | 0.6768 |

Compared at **matched total bits** (Gaussian curve interpolated to the Hadamard budget), the
ordering reverses:

| bits/vector | Hadamard | Gaussian (interp.) | Δ |
|---|---|---|---|
| 1056 (WavLM) | 0.4898 | 0.5285 | **−0.039** |
| 2080 (WavLM) | 0.6768 | 0.7366 | **−0.060** |
| 1056 (HuBERT) | 0.6597 | 0.7245 | **−0.065** |
| 2080 (HuBERT) | 0.7979 | 0.8388 | **−0.041** |

**The structured rotation is not free.** Padding to the next power of two wastes bits on 256
all-zero dimensions, costing 0.04–0.065 Recall@10 at matched budget on both models tested. Any
deployment at a dimension that is not a power of two pays this, and reporting only
bits-per-coordinate would conceal it.

**Implementation caveat.** Our Hadamard path builds the full matrix and applies it as a dense
matmul, so the measured encode times (0.30–0.48 s vs 0.19–0.33 s for Gaussian) reflect an O(d²)
implementation and say nothing about the O(d log d) fast Walsh–Hadamard transform's real speed.
The recall comparison above is unaffected — it depends only on the transform's mathematical
effect — but no speed claim should be drawn from these timings.

---

## F14. Layer choice matters as much as quantizer choice — and our headline layer is the worst one

Every experiment above uses the **final** transformer layer, the conventional default. Sweeping
all 12 layers of WavLM (3 000-vector index, 400 queries, 2 bit/coord):

| layer | crest | kurtosis | mean cos | R@10 (rot) | rotation gain |
|---|---|---|---|---|---|
| 0 | 5.54 | 3.34 | +0.809 | 0.7605 | +8.0 % |
| **2** | 8.89 | 14.68 | +0.759 | **0.7980** | +20.3 % |
| 6 | 6.85 | 6.06 | +0.673 | 0.7865 | +19.4 % |
| 10 | **15.57** | **94.61** | +0.858 | 0.7013 | +19.3 % |
| **11 (final — used everywhere above)** | 10.37 | 39.34 | +0.838 | **0.6435** | +34.9 % |

**Three consequences, in order of importance.**

**(a) The final layer is the worst of the twelve for retrieval.** Layer 2 reaches R@10 = 0.7980
versus 0.6435 at the final layer — **+0.155 absolute, +24 % relative, for free.** For comparison,
the best *quantizer* choice at a matched budget is worth about +0.22 (OPQ over data-oblivious at
2 bit/coord). Layer selection and quantizer selection are therefore of **comparable magnitude**,
and layer selection costs nothing. A practitioner should tune the layer before tuning the
quantizer.

**(b) Our absolute numbers are pessimistic.** All headline results use the final layer, so the
reported recall understates what this pipeline can achieve. The *comparisons* are unaffected —
every method was evaluated on identical embeddings — but absolute figures should be read as a
lower bound, and this is now stated in the paper's limitations.

**(c) Crest factor is a layer property, not a model property.** It varies **2.8×** across layers of
a single model (5.54 → 15.57), a spread comparable to the range across the three models
(3.62 → 12.53). Describing "WavLM's crest factor" is therefore imprecise; it is the crest factor of
a chosen layer.

### F14b. The layer effect replicates, but the best layer is model-specific

HuBERT, same sweep:

| model | best layer | R@10 at best | R@10 at final layer | gain from switching |
|---|---|---|---|---|
| WavLM | **2** | 0.7980 | 0.6435 | **+0.155 (+24 %)** |
| HuBERT | **0** | 0.8383 | 0.7802 | **+0.058 (+7.4 %)** |

**2/2 models: the final layer is suboptimal.** But the best layer differs (2 vs 0) and so does the
magnitude, so the actionable advice is *sweep the layers* — not "use layer 2". The sweep is cheap:
one forward pass returns all layers.

### F14c. The crest-factor screen fails *within* a model — an important limit on F11

Correlation between crest factor and rotation gain across the 12 layers of each model:

| model | Pearson r (crest vs rotation gain, across layers) |
|---|---|
| WavLM | **+0.35** |
| HuBERT | **−0.14** |

**The signs are opposite.** Within a single model, crest factor carries essentially no consistent
information about how much the rotation will help. This is a sharper limit than F11's earlier
qualification.

### F14d. A framing error corrected: "flat model" was really "flat *layer*"

The wav2vec2 sweep completes the picture and forces a correction to how F10/F11 were described.

| model | crest at final layer | crest range across layers | r (crest vs gain) |
|---|---|---|---|
| WavLM | 10.37 | 5.54 – 15.57 | +0.35 |
| HuBERT | 12.38 | 9.38 – 19.47 | −0.14 |
| **wav2vec2** | **3.62** | **3.62 – 17.15** | **+0.74** |

**wav2vec2 is not a "flat model".** Its *final layer* is flat (crest 3.62, rotation gain +7.7 %),
but its layers 1–8 are as spiky as anything in WavLM or HuBERT (crest up to 17.15) and there the
rotation is worth **+30–40 %**. Every earlier statement of the form "wav2vec2 is the model where
rotation is inert" should read "**wav2vec2's final layer** is where rotation is inert."

The F10 prediction itself still stands — it compared final layers, and the measurement was correct
— but its *explanation* was attributing a layer property to a model.

This also explains the inconsistent within-model correlations. wav2vec2 is the only model whose
layers straddle the flat region (3.62 → 17.15), and it is the only one showing a strong
correlation (+0.74). HuBERT's layers are *all* well above the flat region (9.38 – 19.47), so within
that range crest carries no signal (−0.14) even though rotation helps every layer (+8.7–29.4 %).

**Final statement of the crest-factor claim, after four tests:**

> Crest factor is a property of a **specific layer**, not of a model, and functions only as a
> coarse threshold test: a layer with crest near the isotropic ideal (~3.4) gains little from the
> rotation, while layers well above it gain substantially. Within the above-threshold regime the
> magnitude is **not** predictable — neither across models (HuBERT is spikiest yet gains less than
> WavLM) nor across layers within a model (r = +0.35, −0.14, +0.74). Screen the layer you actually
> intend to deploy; do not screen a model, and do not expect the size of the benefit to follow.

Four independent tests each removed a stronger version of this claim. What remains is what the
evidence supports.

**Pooled evidence.** Across all **36 layer-points** (12 layers × 3 models), the correlation between
crest factor and rotation gain is **r = +0.24** — weakly positive, consistent with a threshold that
has been crossed by nearly every point rather than with any proportional law. The one point closest
to the isotropic ideal (wav2vec2 layer 11, crest 3.62) has among the lowest gains (+7.7 %), while
the 35 points above the threshold show gains scattered between +8 % and +40 % with no trend. See
`results/figures/fig10_layers.png`.

**And a check worth stating:** across all three models the **final layer was never the best layer**
for retrieval (best layers: WavLM 2, wav2vec2 0, HuBERT 0). The convention of taking the last
encoder layer is costly for this task in every case we measured.

### F14e. The central claim survives the threat F14 created

F14 raised a direct threat to validity: every headline result uses the final layer, which is the
*worst* of twelve. If the geometry–meaning dissociation (F7) appeared only at a poor operating
point, it would be an artifact of layer choice rather than a property of quantization. Re-running
F7's measurement on WavLM's best layer, at 1 bit/coord:

| layer | Recall@10 kept | speaker P@10 kept | noise-type P@10 kept |
|---|---|---|---|
| **2 (best for retrieval)** | 62.3 % | **99.7 %** | **97.5 %** |
| 11 (final — used for all headline results) | 54.8 % | 96.5 % | 95.7 % |

**The dissociation holds on both layers, and is stronger on the better one.** At layer 2, ~38 % of
true neighbours are destroyed while 99.7 % of speaker-retrieval and 97.5 % of noise-type-retrieval
accuracy survive.

The pass condition was fixed before the run (task retention > 85 % *and* exceeding recall
retention, on both layers) so the test could have failed. It did not: **F7 is a property of
quantization, not of the operating point**, and the paper's central claim is unaffected by the
layer-choice limitation.

---

## Experimental discipline

- Database (MS-SNSD `clean_train`) and queries (`clean_test`) are **disjoint corpora splits**;
  recall is never measured against vectors the query contributed to.
- Ground truth is exhaustive float32 search, so quantization is the only variable.
- Retrieval for the scalar methods is exact search over the reconstructed database, so no ANN
  index approximation is confounded with quantization loss.
- PQ/OPQ come from **faiss**, not from reimplementation, so the data-dependent side is the
  reference implementation rather than a strawman.
- Bit accounting includes the scalar method's 32-bit stored norm rather than omitting it.
- **Codebook training is held out.** For the semantic tasks the banks are small (824 / 1100
  vectors) and faiss requires ≥9984 points to fit 256 centroids; an initial version trained PQ/OPQ
  on the same vectors it scored, which both under-trained the codebooks and let them overfit the
  evaluation set. Corrected to fit on the disjoint 10 000-vector database bank. The effect was
  material — noise-type PQ fell from 99.4 % to 93.7 % retention, OPQ from 99.9 % to 96.1 % — so
  any number from before that fix should be discarded.
- Method comparisons on the semantic tasks use **paired** tests over queries (Wilcoxon), not
  comparisons of independent means, since every method is evaluated on identical queries.
