# Recall Is the Wrong Yardstick: Evaluating Data-Oblivious Vector Quantization on Audio Embeddings

**Draft.** Every number is measured; unverified items are marked TODO rather than asserted.
Findings F1–F11 map to [`FINDINGS.md`](FINDINGS.md).

---

## Abstract

Retrieval over audio embeddings underpins query-by-example search, acoustic-scene matching and
noise-prototype lookup, and on edge hardware the embedding index — not the model — is usually the
memory bottleneck. Data-oblivious vector quantization compresses vectors with no training pass at
all, making it attractive for indices built or extended on-device, but it is evaluated almost
exclusively by geometric recall against uncompressed nearest neighbours.

We evaluate data-oblivious quantization (random rotation followed by per-coordinate optimal scalar
quantization) on audio embeddings across **three self-supervised speech models** — WavLM-Base+,
wav2vec2-base and HuBERT-base — using 10 000-vector indices from MS-SNSD and VoiceBank-DEMAND.

Our central result is a **dissociation between geometry and meaning under lossy compression**.
At ~30× compression the quantizer loses roughly half of the true nearest neighbours, yet retains
**~96 % of downstream speaker- and noise-type-retrieval accuracy** on every model tested. Recall@k
— the metric the quantization literature optimises and reports — substantially understates
deployment performance, and a method dismissed on recall may be entirely adequate in practice.
Concurrent work reaches the same conclusion from the *search-approximation* side, showing
downstream quality is stable while Recall@k falls [Dimitropoulos & Mamoulis 2026]; we establish it
for the distinct and harsher case where the representation itself is destroyed, on audio, with a
measured effect size replicated across three backbones.

We also give a mechanistic account of the random rotation. It works by flattening a spiky
*within-vector* coordinate profile (crest factor 10.2 → 3.3 against an isotropic ideal of 3.4), a
property that varies sharply across embedding models. This produced a cross-model prediction,
recorded in advance and confirmed: rotation is worth +49 % mean Recall@10 on WavLM but only
+2.8 % on wav2vec2, whose coordinates are already near-isotropic. A third model showed the crest
factor is a reliable **binary screen** for whether rotation is worth applying, but *not* a
predictor of how large the gain will be — we report the weaker claim the evidence supports.

Practically, the untrained quantizer reaches 98 % of OPQ's recall at **1/434th** the index build
cost, beats plain product quantization outright on two of three models, and matches or exceeds OPQ
on coarse semantic retrieval — making it a defensible choice for on-device audio indices despite
losing the standard recall benchmark.

---

## 1. Introduction

**Positioning.** This work sits at the intersection of two literatures that have not been
connected. The speech community has established that self-supervised audio representations are
strongly anisotropic and that their layers differ substantially [Pasad et al. 2021; layer-wise
cross-model analyses]. The vector-quantization community has established that data-oblivious
rotation-plus-scalar schemes degrade on anisotropic embeddings [Qdrant 2025]. **Neither has asked
what the first fact implies for the second**, and neither evaluates audio embeddings under
data-oblivious quantization. We do not claim to have discovered either half.

**Contributions.**

1. **The first evaluation of data-oblivious VQ on audio embeddings**, across three self-supervised
   models and two corpora. Published benchmarks of these methods cover text, image and ads
   embeddings and LLM KV caches; audio is untested (§2). Existing audio-embedding compression work
   uses binary hashing, LSH and PQ — all data-*dependent* or task-trained.
2. **A quantified geometry–meaning dissociation under quantization** (§5.3): ~50 % neighbour loss
   costs ~4 % of downstream task accuracy, replicated on all three models and verified not to be
   an artifact of the operating point. **This is the paper's central claim**, stated at the scope
   the evidence supports. We do *not* claim to be first to observe that Recall@k understates
   deployment quality: [Dimitropoulos & Mamoulis 2026] show the same for ANN *search*
   approximation on general corpora and propose 1/Ratio@k in its place. Our contribution is the
   harder case and the number — search approximation returns *slightly worse true vectors*, while
   quantization *permanently destroys representation content*, so stability under the former does
   not imply stability under the latter — measured here on audio, which neither literature covers,
   with the effect size and its cross-model replication. That two literatures reach this
   conclusion by different routes is corroboration, not competition.
3. **A mechanistic account of the random rotation** — the within-vector *crest factor*, a statistic
   the speech-anisotropy literature does not use (it reports cosine similarity and isotropy) —
   validated by an advance cross-model prediction (§5.2) and then narrowed, over four successive
   tests, to a coarse per-layer threshold rather than the quantitative predictor first claimed.
4. **An honest map of what does and does not replicate** (§5.4): the geometry–meaning dissociation
   holds on 3/3 models, the budget trend cleanly on 2/3, and the task-dependent rotation reversal
   on 2/2 of the models where the mechanism predicts an effect (inert on the third) — reported as
   such rather than generalised.
5. **Three practical costs that standard reporting conceals** (§5.6–5.8): the advantage of trained
   codebooks *grows* with index size on 3/3 models, so results at 10⁴ vectors do not transfer to
   10⁶; the deployable structured rotation loses 0.04–0.065 Recall@10 at matched bits when the
   dimension is not a power of two, invisible in bits-per-coordinate tables; and layer choice —
   known to matter for task accuracy — turns out to matter comparably for *quantized retrieval*,
   with the conventional final layer worst on 3/3 models.

## 2. Related work and positioning

**What is already known, and which we do not claim.** Three relevant facts are established.

*(i)* Data-oblivious rotation-plus-scalar quantization degrades on *anisotropic* embeddings: the
underlying analysis assumes vectors uniformly distributed on the sphere and does not extend to
spaces where a few directions carry most of the variance [Qdrant 2025 — an industry technical
report, not peer-reviewed; we mark it as such wherever it is load-bearing].

*(i-b)* The data-oblivious regime is not a one-method field. RaBitQ [Gao & Long, SIGMOD 2024]
randomly rotates a bi-valued codebook and derives an unbiased distance estimator with a proven
error bound, and was the first practically deployable method to attain the asymptotically optimal
space–accuracy trade-off. We situate the method under test alongside it rather than presenting the
regime as unexplored. Like TurboQuant, RaBitQ is evaluated on text and image ANN benchmarks.

*(ii)* Self-supervised speech representations **are** strongly anisotropic — wav2vec2 embeddings
are reported as occupying a narrow cone with layerwise cosine similarities of 0.9–1.0 in upper
layers, consistent with the +0.83 to +0.93 we measure at final layers (§5.1).

*(iii)* Layers of these models differ substantially in what they encode, and layer-wise analysis is
an established subfield [Pasad et al. 2021; cross-model layer analyses].

We claim none of these. Our anisotropy and layer measurements (§5.1, §5.8) **reproduce known
properties in service of a question neither literature has asked**: what those properties imply for
data-oblivious quantization of audio indices.

**Audio embedding compression already exists**, via binary speaker embedding, audio-fingerprint
hashing, LSH and product quantization. All are data-dependent, task-trained, or both; the
training-free regime we study is not covered.

**On evaluation metrics.** That Recall@k is a pessimistic proxy is *not* our discovery.
[Dimitropoulos & Mamoulis 2026] show that downstream indicators — classification accuracy, RAG
answer quality — stay stable while Recall@k drops substantially in ANN *search*, and propose the
inverse approximation ratio 1/Ratio@k instead. Their setting differs from ours in the way that
matters: they hold the vectors exact and vary how hard the index is searched, so a missed
neighbour is merely a *slightly more distant* true vector. Under quantization the vectors
themselves are damaged and the information is gone. Stability under the first does not entail
stability under the second, and we test the second. On the compression side, CoRECT
[arXiv:2510.19340] benchmarks eight compression families on text retrieval up to 100 M passages
and finds the best method varies by model — the closest existing work to §5.6, and likewise
audio-free.

**What is not covered by prior work.** Published benchmarks of these quantizers use arXiv titles,
DBpedia entities, Wikipedia, product catalogues, CLIP image and ads embeddings — **no audio**. The
audio side is no better covered from its own direction: MAEB [arXiv:2602.16008] evaluates 50+
audio embedding models over 30 tasks but measures **uncompressed** embedding quality only, with no
quantization, index-memory or on-device dimension. Existing audio embedding compression is
data-dependent or task-trained throughout — e.g. jointly learned embeddings and balanced hash
codes for query-by-example [arXiv:2211.11060]. Analyses in this literature use quantile-based
per-coordinate rescaling but do not characterise within-vector crest factor or kurtosis
[Qdrant 2025].

**A distinction worth stating explicitly.** wav2vec2 and related models use product quantization
*inside their own training objective* (Gumbel-softmax codebooks over encoder features). That is
unrelated to compressing utterance embeddings for a retrieval index, which is our subject.

*Citation pass completed 2026-08-20; full bibliography, per-claim audit and the one claim it
forced us to narrow (contribution 2) are recorded in [`REFERENCES.md`](REFERENCES.md). One
database-level sweep (DBLP / ACL Anthology / ISCA Archive) remains before submission.*

## 3. Method

Random rotation (Hadamard or dense Gaussian) followed by a per-coordinate Lloyd–Max quantizer
derived from the analytic post-rotation coordinate distribution; norms stored in float32 and
counted in the bit budget. Baselines: faiss PQ and OPQ (trained on held-out data), uniform scalar
quantization, and the identical quantizer with rotation disabled as the ablation.

**Implementation validation (F1).** Measured MSE distortion falls below the source method's proven
upper bound at every bitrate (b=1: 0.403 vs 0.680; b=4: 0.0094 vs 0.0106), within 2.5–14.4 % of its
refined values. 34 unit tests cover the quantizers and the retrieval evaluation, including
brute-force verification of the top-k and recall routines.

## 4. Experimental setup

Indices of 10 000 MS-SNSD utterances with 1 000 disjoint queries; semantic tasks use
VoiceBank-DEMAND speaker labels (824 clips) and MS-SNSD noise-type labels (1 100 clips). Ground
truth is exhaustive float32 search. PQ/OPQ codebooks are fitted on a held-out 10 000-vector bank,
never on the vectors scored. Comparisons use paired Wilcoxon tests and bootstrap CIs over queries.

## 5. Results

### 5.1 Audio embedding spaces are extremely anisotropic (F2)

| | WavLM | wav2vec2 | HuBERT | isotropic ref. |
|---|---|---|---|---|
| mean pairwise cosine | +0.830 | +0.929 | +0.853 | −0.000 |
| effective dimensionality | 2.9 % | 1.4 % | 2.1 % | 92.9 % |

All three occupy 1–3 % of their 768 ambient dimensions — far from the uniform-on-sphere case the
method is optimal for. This confirms a known limitation (§2) in a new domain.

### 5.2 Why the rotation works, and when it does not (F3, F5, F10, F11)

| | HuBERT | WavLM | wav2vec2 | ideal |
|---|---|---|---|---|
| within-vector crest (raw → rotated) | 12.53 → 3.40 | 10.22 → 3.34 | 3.62 → 3.29 | 3.37 |
| within-vector kurtosis (raw) | +52.97 | +38.75 | +0.57 | −0.00 |
| **mean relative Recall@10 gain from rotation** | **+40.2 %** | **+49.4 %** | **+2.8 %** |

The prediction that rotation would be near-worthless on wav2vec2 was recorded before that
benchmark ran, with the explicit condition that equal gains would falsify the mechanism.
*(Figure `fig4_mechanism.png`: the within-vector profile before and after rotation against the
isotropic ideal. Figure `fig2_rotation_ablation.png`: the ablation by bit-width. Figure
`fig8_mechanism_crossmodel.png`: rotation benefit against raw crest factor across models.)*

HuBERT was then added to test whether crest factor predicts the *magnitude* of the gain. **It does
not** — HuBERT is the spikiest yet gains less than WavLM, inverting the predicted ordering. We
therefore report the crest factor as a **binary screen** separating "rotation essential"
(crest ≫ 3.4) from "rotation near-worthless" (crest ≈ 3.4), and explicitly not as a quantitative
predictor.

### 5.3 Geometry and meaning come apart — the central result (F7)

At 1 bit/coordinate (~30× compression):

| model | Recall@10 kept | speaker P@10 kept | noise-type P@10 kept |
|---|---|---|---|
| WavLM | 48 % | 96.5 % | 96.0 % |
| wav2vec2 | 53 % | 95.7 % | 96.9 % |
| HuBERT | 70 % | **98.6 %** | **97.3 %** |

Roughly half the neighbour list is destroyed while ~96 % of task accuracy survives. What
compression removes is largely the fine ordering *among near-ties*, which the downstream task does
not depend on. Since prior evaluation of these methods is recall-only, this gap between benchmark
and deployment has not previously been visible.

*(Figure `fig5_geometry_vs_meaning.png`: recall retention collapsing with bit-width while both
semantic curves stay near 1.0 — the paper's central figure.)*

**Not an artifact of the operating point.** §5.8 shows the final layer — used for all results above
— is the *worst* of twelve for retrieval, which raises the possibility that the dissociation only
appears at a poor operating point. Repeating the measurement on WavLM's best layer (layer 2) shows
the effect is if anything **stronger** there: 62.3 % of neighbours retained against **99.7 %** of
speaker-retrieval and **97.5 %** of noise-type-retrieval accuracy. The dissociation is a property of
quantization, not of the layer chosen.

### 5.4 What replicates, and what does not (F6b, F8b, F10b)

| finding | replication |
|---|---|
| geometry–meaning dissociation | **3/3 models** |
| data-dependent advantage widens as budget shrinks | **2/3 cleanly** (WavLM, wav2vec2 Spearman −1.00; HuBERT non-monotone, ρ = −0.60) |
| rotation's task-dependent sign-flip, **as scoped to spiky spaces** | **2/2 spiky models** (WavLM, HuBERT); inert on the one flat model (wav2vec2), as the mechanism predicts |

The sign-flip deserves comment as a methodological point. Observed first on WavLM, it failed to
replicate on wav2vec2; rather than generalise or discard it, we scoped it to spaces where the
rotation does substantial work (crest ≫ 3.4) and noted that wav2vec2 is inert by construction.
HuBERT — run afterwards, and the spikiest model of the three — reproduced the WavLM pattern
exactly: rotation significantly *helps* speaker retrieval (p ≤ 0.024 at 2–4 bits) and
significantly *hurts* noise-type retrieval (p ≤ 3.3e-3 at 1–3 bits). The scoped claim was recorded
before that test and survived it; the unscoped version would have been contradicted.

Matched-budget Recall@10, best data-dependent minus data-oblivious:

| bits/vec | WavLM | wav2vec2 | HuBERT |
|---|---|---|---|
| 800 | +0.243 | +0.202 | +0.040 |
| 1568 | +0.224 | +0.092 | +0.089 |
| 2336 | +0.052 | +0.043 | −0.037 |
| 3104 | +0.020 | −0.049 | −0.012 |

*(Figure `fig1_recall_vs_bits.png`: operating curves for all methods at Recall@10 and @100.
Figure `fig7_crossmodel_rq1.png`: the same comparison side by side across the three models.
Figure `fig6_rotation_task_dependent.png`: the task-dependent sign of the rotation effect with
significance marked.)*

The advantage is strongly model-dependent: the untrained method wins 0/4 budgets on WavLM, 1/4 on
wav2vec2 and 2/4 on HuBERT. It also significantly beats *plain PQ* at 3–4 bits on WavLM
(−0.029, p=3.6e-12) and at every budget on wav2vec2 — OPQ's learned rotation, not product
quantization itself, is what makes the data-dependent side competitive.

### 5.5 Cost (F6)

At ~3100 bits/vector OPQ buys +0.0199 Recall@10 for **434×** the index build time (360.4 s vs
0.8 s), and requires a training pass that a streaming or on-device index cannot always afford.
*(Figure `fig3_recall_vs_buildcost.png`: retrieval quality against index build time, log scale.)*

### 5.6 The advantage of trained codebooks grows with index size (F12)

Sweeping database size at a matched budget (2 bit/coord vs OPQ m=192). Gap to OPQ:

| index size | WavLM | wav2vec2 | HuBERT |
|---|---|---|---|
| 1 000 | +0.068 | **+0.017** | +0.036 |
| 2 500 | +0.156 | +0.055 | +0.051 |
| 5 000 | +0.224 | +0.102 | +0.074 |
| 10 000 | +0.224 | +0.092 | +0.089 |

**The gap widens with index size on all three models** — by up to 3.3× over a single decade. This
qualifies the practical case directly: our headline comparisons are measured at 10 000 vectors and
**should not be assumed to hold at 10⁶**. The favourable regime is small on-device indices — the
deployment that motivates training-free construction in the first place, and where the gap is
smallest on every model (+0.017 to +0.068) — while at server scale a training pass is both more
necessary and more affordable. We report this as prominently as the results favouring the method.

*(Figure `fig9_scale.png`: data-oblivious recall falling with index size, and the gap to OPQ widening.)*

The penalty's magnitude is also strongly model-dependent (WavLM +0.224 vs HuBERT +0.089 at 10k),
so for this deployment the choice of embedding model matters as much as the choice of quantizer.

### 5.7 The deployable structured rotation is not free (F13)

Our benchmarks use a dense Gaussian rotation; the variant one would deploy is the structured
Hadamard transform. Because d = 768 is not a power of two, the Hadamard path pads to 1024 and
spends **33 % more bits** — which a bits-per-*coordinate* table hides. At matched bits per
*vector* (Gaussian curve interpolated to the Hadamard budget) the ordering reverses:

| bits/vector | Hadamard | Gaussian (interp.) | Δ |
|---|---|---|---|
| 1056 (WavLM) | 0.4898 | 0.5285 | **−0.039** |
| 2080 (WavLM) | 0.6768 | 0.7366 | **−0.060** |
| 1056 (HuBERT) | 0.6597 | 0.7245 | **−0.065** |
| 2080 (HuBERT) | 0.7979 | 0.8388 | **−0.041** |

Padding costs 0.04–0.065 Recall@10 on both models. Any deployment at a dimension that is not a
power of two pays this, and we note the reporting convention that conceals it: **bits per vector,
not bits per coordinate, is the comparable unit.**

*Implementation caveat:* our Hadamard path applies the full matrix as a dense matmul, so its
measured encode times reflect an O(d²) implementation and support no claim about the O(d log d)
fast Walsh–Hadamard transform. The recall comparison is unaffected, depending only on the
transform's mathematical effect.

### 5.8 Layer choice rivals quantizer choice (F14)

All results above use the final encoder layer, the conventional default. Sweeping all 12 WavLM
layers (3 000-vector index, 2 bit/coord) shows this is the **worst** layer of the twelve for
retrieval: layer 2 attains R@10 = 0.7980 against 0.6435 at the final layer — **+0.155 absolute,
+24 % relative, at zero cost.** For scale, the best *quantizer* choice at a matched budget is worth
about +0.22. Layer selection is therefore of comparable value to quantizer selection and is free;
practitioners should tune it first.

**This replicates on HuBERT**, where layer 0 is best (0.8383 vs 0.7802 at the final layer,
+7.4 %). Both models penalise the conventional default, but the optimal layer differs (2 vs 0) and
so does the magnitude, so the actionable advice is to *sweep the layers* rather than adopt a fixed
one. The sweep is cheap: a single forward pass returns every layer.

Two further consequences. Our absolute recall figures, all measured at the final layer, are a
**lower bound** on what this pipeline achieves (method *comparisons* are unaffected, since every
method saw identical embeddings). And the within-vector crest factor varies **2.8×** across layers
of one model (5.54 → 15.57), a spread comparable to the range across the three models we study —
so crest factor is a property of a *layer*, not of a model.

**A correction this sweep forces.** wav2vec2's *final* layer has crest 3.62 — the "flat" case of
§5.2 — but its layers 1–8 reach crest 17.15, as spiky as anything in WavLM or HuBERT, and there the
rotation is worth +30–40 %. wav2vec2 is therefore not a flat *model*; it has a flat *final layer*.
The §5.2 prediction was correct as measured (it compared final layers), but its explanation
attributed a layer property to a model, and we correct that here.

This also resolves the within-model correlations, which at first appear contradictory
(+0.35 WavLM, −0.14 HuBERT, +0.74 wav2vec2). wav2vec2 is the only model whose layers straddle the
flat region (3.62 → 17.15) and the only one with a strong correlation; HuBERT's layers are *all*
well above it (9.38 → 19.47), so crest carries no signal there even though rotation helps every
layer. The behaviour is a threshold, not a proportionality.

We therefore state the surviving claim precisely: **crest factor is a property of a specific layer,
not of a model, and works only as a coarse threshold test** — layers near the isotropic ideal gain
little from the rotation, layers well above it gain substantially, and within that regime the
magnitude is not predictable in either direction. Screen the layer you intend to deploy.

## 6. Limitations

- Three embedding models, one language, English read/near-field speech only.
- **All headline results use the final encoder layer**, which §5.8 shows is the worst of twelve
  for retrieval on WavLM. Absolute recall figures are therefore a lower bound; method comparisons
  are unaffected because every method was evaluated on identical embeddings. We verified that the
  central result is not an artifact of this choice (§5.3, layer-2 check), but re-running the full
  program on a better layer is left to future work.
- Indices of 10 000 vectors. A scale sweep (§5.6) shows the data-dependent advantage *growing*
  with index size over the reachable 1k–10k range, so results here should not be assumed to hold
  at 10⁶; the trend direction is measured, its continuation is not.
- Build-cost timings come from one CPU and were not measured on an idle machine.
- Two semantic label sets from two corpora; broader downstream tasks untested.
- **No on-device measurement** — the edge motivation is argued from index size and build cost, not
  demonstrated on edge hardware.
- Several ingredients are prior work (§2) and we claim only their combination: the anisotropy
  limitation of data-oblivious VQ, the anisotropy of speech representations, layer-wise variation
  in those representations, and audio-embedding compression via hashing/PQ are each established
  independently. Our contribution is the connection and its consequences, not any single component.
- **Citation pass incomplete.** Related work was surveyed by targeted search rather than a
  systematic review; the claims of novelty in §1 should be re-verified against a full literature
  search before submission, particularly for work combining audio embeddings with training-free
  quantization.

## 7. Conclusion

Data-oblivious vector quantization loses to trained codebooks on audio embeddings by the standard
recall benchmark, for a reason already documented in the literature and here measured on audio for
the first time. But the standard benchmark is the wrong yardstick for the applications these
indices serve: half the neighbour list can be destroyed while ~96 % of downstream task accuracy
survives, on every model we tested. Once that is measured, and index build cost is counted, the
untrained method is competitive or better in exactly the setting that motivated it.
