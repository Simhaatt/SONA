# Every detail needed to write the paper

Methods, data, splits, hyperparameters and provenance. Numbers live in
[`02_ALL_RESULTS.md`](02_ALL_RESULTS.md); this file is what the Methods and Experimental Setup
sections need. Framing decisions are in [`03_FRAMING.md`](03_FRAMING.md).

---

## 1. Systems under study

### 1.1 SONA / GTCRN speech enhancer (Studies 1 and 2)

Locked checkpoint `gtcrn_ch96_h96_b4_mw1p25_mr0p25`. **Not retrained since July 2026**; every
result here is evaluation, analysis or deployment work on the frozen weights.

**Architecture** — Bark-band GTCRN, **722 209 parameters**, FP32 ONNX size **2842.7 KB**:

| component | detail | params |
|---|---|---|
| stem | single conv, `stem.0` | 1 152 |
| blocks 0–3 | gated time–frequency conv, dilations 1 / 2 / 4 / 8, ch = 96 | 166 080 each |
| band GRU | per-band causal GRU decoder, hidden = 96 | 55 872 |
| head | linear → per-band gain logits | 97 |

**Signal path** — STFT → Bark-band features → model → per-band gain → apply → ISTFT/overlap-add.
Every method in the benchmark shares this front-end, the same per-band gain output space, and the
same evaluation harness; only the gain-estimation step differs. That equal-footing design is the
reason the comparison is publishable, and it belongs in the Methods section explicitly.

**STFT / bands**

| parameter | value |
|---|---|
| sample rate | 16 000 Hz |
| n_fft | 512 |
| hop | 128 |
| win_length | 512 |
| center | true |
| bands | 32, Bark scale |
| noise estimator | minimum statistics |

**Training (for reproducibility; not re-run)**

| parameter | value |
|---|---|
| epochs | 50 |
| batch size | 8 |
| optimiser LR | 3e-4 |
| weight decay | 1e-5 |
| crop | 2.0 s |
| loss | SI-SDR + 1.25 · log-magnitude + 0.25 · multi-resolution magnitude |
| seed | 1234 |
| hardware | Kaggle T4 |

**Complex-mask variant** — identical backbone, only the output representation differs: a 2-channel
head predicting real and imaginary parts of a bounded complex ratio mask (`bound · tanh`,
bound = 2). **722 306 params, i.e. +97 over the magnitude model.** Two engineering details that
must be reported because they materially affected the comparison:

1. *Identity initialisation.* A tanh complex head starts near mask = 0 (silence), whereas the
   sigmoid magnitude head starts near gain 0.5. The complex head is initialised (bias → real = 1,
   imag = 0; head weight scaled 0.1) to start as near pass-through. Scaling 0.01 was measured and
   crippled learning; 0.1 was chosen.
2. *Scale-sensitive SNR term.* SI-SDR is scale-invariant, so the complex mask drifted output level
   and tanked scale-sensitive SNR (−6.6 dB) while SI-SDR looked fine. Adding `snr_weight = 0.3`
   anchored the level. **Applied identically to both models**, so it is not a thumb on the scale.

### 1.2 AudioVQ retrieval (Study 3)

Embeddings from pretrained **WavLM-Base+**, **wav2vec2-base**, **HuBERT-base** (torchaudio),
mean-pooled over time, 768-d. Indices of **10 000 MS-SNSD utterances** with **1 000 disjoint
queries**. Ground truth is exhaustive float32 search over the same database.

Quantizers compared at matched bits/vector: rotation + per-coordinate Lloyd–Max (the data-oblivious
method under test), the same quantizer with rotation disabled (ablation), uniform scalar, and
faiss PQ / OPQ trained on a **held-out 10 000-vector bank, never on the vectors scored**.

---

## 2. Data and splits

### 2.1 VoiceBank-DEMAND

| split | speakers | clips | used for |
|---|---|---|---|
| train | 28 speakers | ~11.5 k | original training (frozen) |
| **calibration** | 11 train speakers (p226–p250) | 64 | QDQ activation calibration only |
| **dev** | **p254, p256** (train, held out) | 200 | quantization sensitivity + allocation search |
| **test** | **p232, p257** | **824** | final numbers, touched once |

Local 16 kHz copy at `vbd_local/` (3.0 GB, gitignored, rebuildable via
`sona-final-month/scripts/prep_voicebank_local.py`). Manifests: 5 000 train rows, 824 test rows,
all paths verified resolvable.

Official condition labels for the test set are in `results/vbd_test_conditions.csv` —
**5 noise types × 4 SNRs**, covering all 824 clips (431 p257 + 393 p232):

| noise types | SNRs |
|---|---|
| bus, cafe, living, office, psquare | 2.5, 7.5, 12.5, 17.5 dB |

### 2.2 MS-SNSD (external)

1 100 clip test set at `external_data/ms_snsd_sona/test.csv`, with `noise_type` and `snr_db`
columns. Noise categories: AirConditioner, Babble, Cafeteria, Munching, Neighbor, ShuttingDoor,
Typing, AirportAnnouncements, Traffic and others. SNRs −5 to +10 dB.

### 2.3 Leakage discipline — state this explicitly in the paper

Bit allocation is a **model-selection decision**, so it was chosen on a dev set carved from TRAIN
**by speaker**, never on test. `make_quant_splits.py` hard-fails on speaker overlap. The chosen
allocation was written to disk with a timestamp (`phase3/frozen_allocation.json`,
`frozen_at: 2026-08-14 18:49:13`) **before** any test-set evaluation ran.

Verified during the 2026-08-20 consolidation: re-running `make_quant_splits.py` regenerates both
manifests **bit-identically**, so the discipline survived a full path rewrite.

---

## 3. Quantization method (Study 1)

ONNX Runtime **1.20.1**, `quantize_static`, QDQ format, per-layer control via `nodes_to_quantize`.

**Search space.** The band GRU is unquantizable on this runtime and INT4 is unavailable for Conv,
so the precision axis is per-layer ∈ {FP32, INT8} over **6 quantizable layers** (5 Conv + 1
MatMul) — **2⁶ = 64 configurations, enumerated exhaustively** rather than searched greedily. That
choice matters given error non-additivity (§4).

**Per-layer control verified real**, not a no-op: uniform INT8 = 172 QDQ nodes, head-only = 5,
stem-only = 6, with outputs differing from FP32 and from each other.

**Frozen allocation** (`phase3/frozen_allocation.json`):

```
INT8:  blocks.0/1/2/3 conv, head MatMul
FP32:  stem conv (exempted), band_gru (unquantizable)
```

**Runtime limits — report as findings, not bugs**

| limit | evidence |
|---|---|
| band GRU cannot be quantized | requesting it yields **0 QDQ nodes**, no size change (2852.9 vs 2842.7 KB), no speedup (0.995×) |
| INT4 unavailable for Conv | `quantize_static` fails: "Quantized weights must be 8-bit before packing as 4-bit"; ORT's 4-bit path covers MatMul only, and this graph has one tiny MatMul |
| INT8 slower than FP32 | 1.39× / 1.43× / 1.46× at 1 / 2 / 4 threads, idle machine |

Note the direct contrast with Rusci et al. (2022), whose mixed FP16–INT8 method for RNN denoisers
depends on quantizing exactly the recurrent layers that are incompressible here. Frame as: *which
layer to protect is a property of runtime and model, not a transferable rule.*

---

## 4. Statistical methods

Use these consistently and say so once in the paper rather than per-table.

| method | where | detail |
|---|---|---|
| paired Wilcoxon signed-rank | all per-clip comparisons | two-sided |
| bootstrap CI | all effect sizes | 10 000 resamples, seed 2026, percentile method |
| **paired** clip bootstrap | corpus-level WER, R1 routing | one index draw per replicate applied to both arms |
| **Benjamini–Hochberg FDR** | per-condition breakdowns | applied **within each (method, split) family** — the 5 noise types or 4 SNR bins a reader interprets together, not across all 40 tests |
| Spearman ρ | sensitivity ranking agreement | 6 quantizable layers |
| leave-one-noise-type-out CV | R1 router | held-out noise type never seen in training |
| grouped CV (by noise source file) | scene classifier | prevents texture leakage |

**Effective sample size must be reported for WER.** Per-clip WER is zero-inflated: 6 183 reference
words over 824 clips, median 7 words per clip, and only **129/824** pairs differ between GTCRN and
passthrough. Wilcoxon discards ties, so per-condition effective n falls to **16–37**. Report n_eff
alongside n or the power is implied rather than shown.

---

## 5. Hardware and timing provenance

Any absolute timing must carry machine and load. This was a correction: the July RTF figures were
taken on a loaded machine and were not quotable.

| | local | Kaggle |
|---|---|---|
| CPU | AMD Ryzen 5 5600H, 12 logical | Intel Xeon (recorded per run) |
| GPU | none | T4 |
| load at RTF measurement | **6 %** (2026-08-20 idle run) | recorded in `machine.json` |
| ORT | 1.20.1 pinned | 1.20.1 pinned |

Re-measured 2026-08-20: FP32 RTF 0.084, INT8 0.117 single-thread. Relative orderings were stable
across load in every run; only absolute values moved (~25–30 % lower on the idle machine).

---

## 6. Reproducibility

No hardcoded machine paths remain. Corpora resolve from environment variables with tree-relative
fallbacks derived from each file's own location:

```
AUDIOVQ_MS_SNSD / AUDIOVQ_VBD_TEST_CLEAN / AUDIOVQ_NOISY_MANIFEST   (AudioVQ)
SONA_VBD_TRAIN_MANIFEST / SONA_VBD_TEST_MANIFEST                     (PERCEPTUAL_MP)
```

**Verification status** (2026-08-20): 75 tests pass in the SONA repo, 34 in AudioVQ,
`sona-final-month/scripts/check_all.py` green on all 6 stages, both paper-number audits pass
(19 + 25 claims), and quantization splits regenerate bit-identically.

### Scripts producing each result

| result | script |
|---|---|
| benchmark tables | `sona-final-month/scripts/build_final_tables.py` |
| significance battery | `sona-final-month/scripts/significance.py` |
| quantization phases 0–4 | `PERCEPTUAL_MP/scripts/phase{0,1,2,3,4}*.py` |
| **E2 Hessian ranking** | `PERCEPTUAL_MP/scripts/hessian_ranking.py` |
| **R3 dither diagnostic** | `PERCEPTUAL_MP/scripts/dither_diagnostic.py` |
| **E1 1/Ratio@k** | `AudioVQ/scripts/ratio_at_k.py` |
| **E3 per-clip WER significance** | `scripts/wer_condition_significance.py` |
| **E3b corpus-level WER** | `scripts/wer_corpus_level.py` |
| **R1 condition gating** | `scripts/condition_gated_enhancement.py` |
| ONNX timing | `scripts/bench_onnx_edge.py` |
| claim audits | `AudioVQ/scripts/audit_paper_numbers.py`, `PERCEPTUAL_MP/scripts/audit_paper_numbers.py` |
| this bundle | `scripts/build_paper_bundle.py` |

---

## 7. Provenance and honest attribution

**TurboQuant (arXiv:2504.19874)** motivated the *objective-mismatch* framing in Study 1 — that an
MSE-optimal quantizer can be wrong for the objective you care about. Its **algorithm does not
apply**: no KV cache, no vector database, no memory bottleneck of that kind. Cite for motivation
only. The perceptual-vs-MSE form of the hypothesis it suggested was **refuted**.

In Study 3, TurboQuant is the method under test rather than an inspiration.

**Prior work we explicitly do not claim** (all in `AudioVQ/REFERENCES.md`):
- Recall@k understating deployment quality — Dimitropoulos & Mamoulis 2026
- anisotropy of speech representations — Pasad et al., and the isotropy literature
- layer-wise variation in SSL speech models — Pasad et al. 2021, 2023
- data-oblivious VQ degrading on anisotropic embeddings — Qdrant 2025 (industry technical report,
  not peer-reviewed; label it as such)
- RaBitQ as the other major data-oblivious quantizer with a proven bound — Gao & Long, SIGMOD 2024

---

## 8. Open items

| item | status |
|---|---|
| **MOS listening test** | app built and browser-verified, 29 clips selected, analysis script written — **needs human raters** |
| R2 mixed-allocation timing | queued for Kaggle (`sona_phase_a_kaggle.ipynb`) |
| WER on MS-SNSD (+1100 clips) | queued for Kaggle — power remedy |
| WER with `tiny.en` | queued for Kaggle — power remedy |
| E4 second architecture | complex-GTCRN checkpoint exists; sweep not yet run |
| per-channel quantization | optional; the one negative with a real chance of flipping |
| Pi / on-device measurement | never done; argue from size and latency only |
| author names | **`SPARC FYP Team` is a placeholder** — replace before submission |
