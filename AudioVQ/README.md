# AudioVQ — Data-Oblivious Vector Quantization for On-Device Audio Retrieval

**Standalone project. Independent of the SONA speech-enhancement work** (`../noise cancellation`),
though it targets a capability that system's roadmap identified as future work: an on-device
retrieval memory over audio/noise prototypes.

---

## Motivation

Retrieval over audio embeddings underpins query-by-example search, acoustic-scene matching,
speaker verification against enrolled templates, and noise-prototype lookup for adaptive
enhancement. On edge hardware the embedding index — not the model — is often the memory
bottleneck: one million 768-d float32 embeddings is ~3 GB, which no phone or Raspberry Pi will
hold.

Vector quantization solves this, but the dominant methods (Product Quantization and its variants)
are **data-dependent**: they learn k-means codebooks over a training sample, requiring a
calibration pass and re-training whenever the distribution shifts. **Data-oblivious** quantizers
need no training at all — they apply instantly to any vector — which matters when the index is
built or extended on-device.

TurboQuant (Zandieh et al., arXiv:2504.19874) is a recent data-oblivious scheme: randomly rotate
each vector so its coordinates become Beta-distributed and nearly independent, then apply an
optimal scalar quantizer per coordinate, with a 1-bit QJL pass on the residual for unbiased inner
products. It was validated on **LLM KV caches and standard image/text ANN benchmarks**.

**It has not been evaluated on audio embeddings, and there is reason to expect audio to behave
differently.** Self-supervised audio representations (wav2vec2, WavLM, HuBERT) are known to be
strongly **anisotropic** — variance concentrated in a few directions carrying energy, channel and
speaker information. TurboQuant's random rotation exists precisely to destroy such structure, so
audio is a case where it could help *more* than on the already-isotropic embeddings it was tested
on — or fail, if the anisotropy is so severe that per-coordinate scalar quantization is a poor fit.

## Research questions

- **RQ1** How does data-oblivious rotation+scalar VQ compare with data-dependent PQ/OPQ for audio
  embedding retrieval **at matched bits per vector**?
- **RQ2** How much does the random rotation itself contribute? (ablation: identical quantizer with
  and without rotation)
- **RQ3** Do the theoretical MSE and inner-product distortion bounds hold empirically on audio
  embeddings, or does audio anisotropy break the concentration assumptions?
- **RQ4** What is the practical recall-vs-memory operating point for on-device audio retrieval,
  and does the data-oblivious method's zero-training-cost advantage survive contact with real
  recall requirements?

## Evaluation protocol

**Primary metric — geometry preservation (standard ANN protocol).** Ground truth is exhaustive
search over the *uncompressed* embeddings; we report Recall@{1,10,100} of those true neighbours
after quantization. This needs no semantic labels and measures exactly what a quantizer is
supposed to preserve.

**Secondary — downstream semantic retrieval.** Using labels already present in the corpora:
speaker identity (VoiceBank-DEMAND speaker IDs) and noise type (MS-SNSD `noise_type`).

**Distortion diagnostics.** MSE and inner-product error vs the paper's predicted bounds, plus an
explicit anisotropy measurement of the embedding space to test RQ3's premise.

**Methods compared** (all at matched bits/vector):
| method | data-dependent? | role |
|---|---|---|
| TurboQuant-style (rotation + optimal scalar) | no | method under test |
| same quantizer, rotation removed | no | RQ2 ablation |
| Product Quantization (faiss) | yes | standard baseline |
| Optimized PQ (faiss, learned rotation) | yes | strong baseline |
| uniform scalar quantization | no | naive floor |
| float32 exhaustive | — | upper bound |

## Data

Reuses audio already on disk (no new downloads):
- **MS-SNSD** — 23,075 clean train + 1,100 clean test utterances; 128+51 noise recordings, each
  labelled with noise type.
- **VoiceBank-DEMAND** — ~5,800 utterances across 15 speakers with speaker IDs.

Embeddings from pretrained **WavLM-Base+ / wav2vec2 / HuBERT** (torchaudio), mean-pooled over time.

## Layout

```
src/        quantizers, embedding extraction, retrieval evaluation
scripts/    pipeline entry points
embeddings/ cached .npy embedding banks (gitignored)
results/    metrics, tables, figures
tests/      unit tests (correctness of quantizers, recall computation)
```

## Status

Core experiments complete on **two embedding models** (WavLM-Base+, wav2vec2-base). Full detail in
[`FINDINGS.md`](FINDINGS.md); headline results:

1. **Implementation validated against theory** — measured distortion falls below TurboQuant's
   Theorem 1 upper bound at every bitrate (34 unit tests pass across quantizers and retrieval).
2. **Audio embeddings are extremely anisotropic** — mean pairwise cosine +0.83 (WavLM) / +0.93
   (wav2vec2); the cloud occupies 2.9 % / 1.4 % of its 768 dimensions. Both far from the
   worst case data-oblivious VQ is optimal for.
3. **RQ1 answered, prediction confirmed on both models** — trained codebooks (OPQ) win, and the
   gap widens as the budget shrinks (+0.24 → +0.02 on WavLM; +0.20 → −0.05 on wav2vec2, where the
   untrained method overtakes OPQ at the largest budget).
4. **The rotation's value is mechanistically explained and predicted across models** — it works by
   flattening a spiky within-vector coordinate profile, so it gives +49 % mean recall on WavLM
   (raw crest 10.2) but only +2.8 % on wav2vec2 (raw crest 3.6, already near-isotropic).
5. **Geometry and meaning come apart** — at ~30× compression roughly half the true neighbours are
   lost while ~96 % of downstream retrieval performance survives, on both models. **Recall@k is a
   pessimistic proxy for deployment.**
6. **Practical outcome** — the untrained quantizer reaches 98 % of OPQ's recall at **1/434th** the
   index build cost, beats plain PQ outright on wav2vec2 at every budget, and matches or beats OPQ
   on coarse semantic retrieval.

Reported honestly: the perceptual-vs-fidelity reversal seen on WavLM (F8) **did not replicate** on
wav2vec2 and has been scoped accordingly rather than generalised.

## Reproducing

```bash
python scripts/extract_embeddings.py --model wavlm_base_plus   # ~13k clips, GPU
python scripts/analyze_anisotropy.py --model wavlm_base_plus
python scripts/run_retrieval_benchmark.py --model wavlm_base_plus
python scripts/run_semantic_retrieval.py --model wavlm_base_plus
python scripts/make_figures.py && python scripts/make_crossmodel_artifacts.py
python -m pytest tests/ -v
```
