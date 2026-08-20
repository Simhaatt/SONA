# References and citation audit

Written 2026-08-20, closing the `TODO: formal citation pass` left in `PAPER_DRAFT.md` §2.

Every entry below was located and its scope checked against what the draft claims about it. The
audit section at the end records **one finding that requires a claim change** before submission.

---

## 1. Bibliography

### Data-oblivious quantization — the method under test

**[Zandieh 2025]** A. Zandieh, M. Daliri, M. Hadian, V. Mirrokni. *TurboQuant: Online Vector
Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874, 2025. Google Research;
subsequently ICLR 2026.
→ The source method. Random rotation induces a concentrated Beta distribution on coordinates;
near-independence in high dimensions then permits per-coordinate optimal scalar quantization,
followed by a 1-bit QJL pass on the residual for unbiased inner products. Validated on LLM KV
caches and standard image/text ANN benchmarks. Cite for the method and for its distortion bound
(matched to within ≈2.7× of the information-theoretic lower bound), which §3's F1 validation
checks against.

**[Gao 2024]** J. Gao, C. Long. *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
Error Bound for Approximate Nearest Neighbor Search.* Proc. ACM Management of Data (SIGMOD), 2024.
doi:10.1145/3654970.
→ **Missing from the current draft and should be added.** RaBitQ is the other major
*data-oblivious* quantizer with a proven error bound: it randomly rotates a bi-valued codebook and
derives an unbiased distance estimator, and was the first practically deployable method achieving
the asymptotically optimal space–accuracy trade-off. It belongs in §2 as the nearest neighbour of
the method under test, and its absence currently overstates how sparse the data-oblivious
literature is. Like TurboQuant, it is evaluated on text/image ANN benchmarks — **not audio** — so
adding it does not weaken the paper's novelty claim; it strengthens the framing.

**[Qdrant 2025]** Qdrant. *TurboQuant in Qdrant.* Technical article,
`qdrant.tech/articles/turboquant-quantization/`, 2025.
→ **Grey literature — mark it as such in the bibliography.** It is an engineering report, not
peer-reviewed, and the draft currently leans on it for three separate load-bearing statements
(§2(i), the "no audio in published benchmarks" claim, and the crest-factor gap). It does support
them: it states plainly that the rotation proof assumes vectors uniformly distributed on the
sphere and "does not extend to anisotropic embeddings", it benchmarks 10 datasets that are all
text/image (arXiv titles, DBpedia, Wikipedia, product catalogues, CLIP image and ads embeddings),
and it reports a coordinate-variance ratio of 233.5× on DBpedia while never characterising
*within-vector* crest factor or kurtosis. Keep the citations, but attribute them to an industry
technical report rather than implying a peer-reviewed source.

### Data-dependent baselines

**[Jégou 2011]** H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor
Search.* IEEE TPAMI 33(1), 2011.

**[Ge 2013]** T. Ge, K. He, Q. Ke, J. Sun. *Optimized Product Quantization for Approximate Nearest
Neighbor Search.* CVPR, 2013.

→ The PQ and OPQ baselines used via faiss. Both are currently referenced only by name in §3 and
need proper entries.

### Evaluation methodology — the paper's central claim

**[Dimitropoulos 2026]** D. Dimitropoulos, N. Mamoulis. *ANN Search: Recall What Matters.*
arXiv:2606.04522, submitted 3 June 2026.
→ **The most important find of this pass. It is prior work on the paper's central claim and is
currently uncited.** It argues that Recall@k overstates the true cost of approximation, shows that
downstream indicators (classification accuracy, RAG quality) stay stable while Recall@k drops
substantially, and proposes the inverse approximation ratio 1/Ratio@k as a replacement metric.
See the audit in §2 below for exactly how our claim must be narrowed — and for why the audio and
quantization results survive it.

**[Fröbe 2025]** CoRECT: *A Framework for Evaluating Embedding Compression Techniques at Scale.*
arXiv:2510.19340, 2025.
→ Benchmarks eight families of embedding compression on **text** retrieval at up to 100 M
passages, and finds the best compression method varies by model. Relevant to §5.6 (contribution 5:
the advantage of trained codebooks grows with index size). It is the closest existing work on the
scale question and should be cited there; it is text-only and does not test audio, so §5.6's
audio finding stands, but the draft can no longer imply that scale-dependence of compression
choice is itself unobserved.

### Audio representation properties — reproduced, not claimed

**[Pasad 2021]** A. Pasad, J.-C. Chou, K. Livescu. *Layer-wise Analysis of a Self-supervised
Speech Representation Model.* arXiv:2107.04734 / ASRU, 2021.
→ Establishes the layer-wise gradient (acoustic → phonetic → word-level) via CCA probing. Backs
§2(iii) and the framing of §5.8.

**[Pasad 2023]** A. Pasad, B. Shi, K. Livescu. *Comparative Layer-wise Analysis of Self-supervised
Speech Models.* arXiv:2211.03929 / ICASSP, 2023.
→ The cross-model version (wav2vec2 / HuBERT / WavLM); the correct citation for the draft's
current placeholder "cross-model layer analyses". Directly supports §5.8's premise that layer
properties differ by pre-training objective.

**[Liu 2024]** *Orthogonality and Isotropy of Speaker and Phonetic Information in Self-supervised
Speech Representations.* arXiv:2406.09200, 2024.
→ Measures isotropy of speaker vs phonetic subspaces in SSL speech representations. The most
directly relevant support for §5.1 and the cleanest peer-reviewed anchor for the anisotropy claim
the draft currently attributes loosely.

**[Anisotropy 2026]** *Assessing the Impact of Anisotropy in Neural Representations of Speech.*
arXiv:2506.11096, 2026.
→ Recent, and important for §2's "we do not claim to have discovered audio anisotropy" position:
it confirms anisotropy in speech representations is an actively studied property with known
downstream consequences. Cite it precisely where the draft disclaims novelty.

### Backbone models

**[Chen 2022]** S. Chen et al. *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
Speech Processing.* arXiv:2110.13900 / IEEE JSTSP, 2022.
**[Baevski 2020]** A. Baevski, H. Zhou, A. Mohamed, M. Auli. *wav2vec 2.0: A Framework for
Self-Supervised Learning of Speech Representations.* NeurIPS, 2020.
**[Hsu 2021]** W.-N. Hsu et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked
Prediction of Hidden Units.* IEEE/ACM TASLP, 2021.

### Audio embedding compression — the "not covered" claim

**[Singh 2022]** *Simultaneously Learning Robust Audio Embeddings and Balanced Hash Codes for
Query-by-Example.* arXiv:2211.11060, 2022.
→ The concrete instance behind §2's "binary hashing … all data-dependent or task-trained".
Jointly *learns* continuous embeddings and binary codes end-to-end, reporting 2× and 1.4× speedups
over LSH and PQ. Exactly the data-dependent regime we contrast against; cite it by name rather
than gesturing at "audio-fingerprint hashing".

**[MAEB 2026]** *MAEB: Massive Audio Embedding Benchmark.* arXiv:2602.16008, 2026.
→ 50+ models over 30 tasks. Checked specifically for this audit: it evaluates **uncompressed**
embedding quality only — no quantization, no index memory, no on-device deployment. It is
therefore evidence *for* the paper's gap claim and worth citing as such.

---

## 2. Audit findings

### Finding A — contribution 2 must be narrowed (blocking)

The draft says of the geometry–meaning dissociation: *"Prior evaluation of these methods is
**exclusively** geometric-recall based, so this failure of the standard metric has not been visible
from either literature."*

[Dimitropoulos 2026], posted 3 June 2026, makes the general form of that argument first: recall
drops substantially while downstream indicators stay stable. As written, our contribution 2 is
**not defensible** and a reviewer who knows the ANN literature will catch it.

What survives, and is still a real contribution:

| the draft's claim | status after this pass |
|---|---|
| Recall@k understates deployment performance | **prior work** — [Dimitropoulos 2026]. Cite, do not claim. |
| …under *quantization*, as opposed to index-search approximation | **ours.** They vary search effort on an exact-vector index; we vary the *representation*. Lossy compression destroys information permanently; a truncated graph traversal does not. Different mechanism, and not implied by their result. |
| …on *audio* embeddings | **ours.** Their datasets are text/general; no audio anywhere in this literature. |
| the specific magnitude (~50 % neighbour loss → ~4 % task loss), replicated on 3/3 models | **ours.** They report stability qualitatively; we give the effect size, on three backbones, with the operating-point control. |

Recommended rewrite of contribution 2: claim the dissociation **for lossy quantization on audio
embeddings, with a measured effect size**, and cite [Dimitropoulos 2026] as independent
corroboration of the general phenomenon from the search-approximation side. That framing is
stronger than the current one, because a second literature independently reaching the same
conclusion by a different route is evidence the finding is real rather than a quirk of our setup.

### Finding B — RaBitQ omission (should fix)

§2 characterises the data-oblivious regime as essentially uncovered. [Gao 2024] is a SIGMOD paper
in exactly that regime with a proven optimal space–accuracy trade-off. Omitting it overstates the
gap. Adding it costs nothing: RaBitQ is likewise evaluated on text/image ANN benchmarks only.

### Finding C — grey-literature attribution (should fix)

Three load-bearing statements in §2 rest on [Qdrant 2025], an industry technical article. The
statements check out, but the draft should say what the source is.

### Finding D — claims that survived unchanged (verified)

- **No prior work applies data-oblivious VQ to audio embeddings.** Searched across the
  quantization literature (TurboQuant, RaBitQ, PQ/OPQ, CoRECT), the audio-embedding-benchmark
  literature (MAEB, DASB), and audio-compression work (binary hashing, LSH, PQ, neural codecs).
  Audio embedding *compression* exists but is uniformly data-dependent or task-trained.
  **Contribution 1 stands.**
- **The within-vector crest factor is not used in the speech-anisotropy literature.** That
  literature reports mean pairwise cosine similarity and isotropy scores, both *across*-vector
  statistics. **Contribution 3's novelty stands.**
- **wav2vec2's internal Gumbel-softmax product quantization is unrelated to index compression.**
  The distinction the draft draws in §2 is correct and worth keeping — it is the most likely
  reviewer confusion.

### Remaining gap

This pass was a targeted search across the quantization, ANN-evaluation, speech-representation and
audio-retrieval literatures, not an exhaustive systematic review with a recorded protocol. Before
submission, one database-level sweep (DBLP + ACL Anthology + ISCA Archive, terms: *vector
quantization*, *embedding compression*, *audio retrieval*, *data-oblivious*) should be run and
recorded, chiefly to catch anything appearing between June and August 2026 — the Dimitropoulos
find shows this area is moving fast enough that a two-month-old gap matters.
