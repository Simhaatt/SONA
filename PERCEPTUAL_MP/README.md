# PERCEPTUAL_MP — perceptually-guided mixed-precision quantization

Self-contained experiment folder. Full design lives in `../PERCEPTUAL_MP_QUANT_PLAN.md`.

**Hypothesis.** Standard PTQ calibrates to minimize activation/weight MSE, but the objective that
matters for speech enhancement is perceptual (PESQ/STOI), which weights frequency bands very
non-uniformly. So ranking layers by *perceptual* sensitivity may differ from ranking them by
*MSE* sensitivity — and allocating bits by the perceptual ranking may win at matched model size.

**Provenance.** Derived from the *objective-mismatch* insight in TurboQuant (arXiv 2504.19874),
not from its algorithm. TurboQuant itself does not apply to SONA: no KV cache, no vector database,
no memory bottleneck. Cite for the insight only.

```
splits/     speaker-disjoint calib + dev manifests (built from TRAIN only)
scripts/    quant_lib.py, make_quant_splits.py, phase0_probe.py
onnx/       exported FP32 / quantized variants
results/    phase0 findings, per-clip deltas
```

## Data splits — leakage discipline

Bit allocation is a **model-selection** decision, so it is chosen on a dev set carved from TRAIN
by speaker. The 824-clip test set is not touched until the allocation is frozen.

| Split | Speakers | Clips | Used for |
|---|---|---|---|
| calibration | 11 train speakers (p226…p250) | 64 | QDQ activation calibration only |
| dev | **p254, p256** (train, held out) | 200 | sensitivity ranking + allocation search |
| test | p232, p257 | 824 | final numbers, touched **once** |
| external | MS-SNSD | 1100 | generalization, touched **once** |

Verified speaker-disjoint by `make_quant_splits.py`, which hard-fails on overlap.

## Phase 0 findings — structural (deterministic, settled)

**0b — per-layer control WORKS.** `nodes_to_quantize` produces genuinely distinct graphs:
uniform INT8 = 172 QDQ nodes; head-only = 5; stem-only = 6. Outputs differ from FP32 and from
each other. The sensitivity study is mechanically possible.

**0d — the band GRU is NOT quantizable on this runtime.** Requesting it yields **0 QDQ nodes**,
no size change (2852.9 vs 2842.7 KB) and no speedup (0.995×). ORT 1.20.1 has no INT8 GRU path
here. The GRU is therefore permanently FP32 and is **excluded from the study**, leaving **6
quantizable layers** (5 Conv + 1 MatMul) — so exhaustive allocation is 2⁶ = 64 configs, not 128.
This is a reportable finding, not a bug: on this runtime the recurrent decoder cannot be
compressed at all.

**0c — INT4 is unavailable via `quantize_static` for Conv.** Fails with
*"Quantized weights … must be 8-bit before packing as 4-bit values"*. ORT's 4-bit path
(`MatMul4BitsQuantizer`) covers MatMul only, and this graph has exactly one MatMul (the tiny
head), so INT4 cannot reach the Conv layers holding most of the weight. Consequence: the
precision axis is restricted to per-layer ∈ {FP32, INT8} unless ORT is upgraded.

**RTF measurements — resolved 2026-08-20.** The July figures were not trustworthy: the model-forward
RTF measured 0.121 in one run and 0.434–0.472 in another on the same machine and thread count,
because the machine was **not idle** (21 % CPU, multiple browser processes). The *relative*
ordering was stable across both runs — notably **INT8 is SLOWER than FP32** here.

Re-measured on an idle machine (6 % CPU load, same ORT 1.20.1, 100 clips, 1/2/4 threads):
`../results/onnx/edge_bench_idle_2026-08-20/`. The slowdown replicates at **1.39–1.46×**, and
absolute RTFs settled ~25–30 % below the July numbers (fp32 1-thread **0.084**, int8 1-thread
**0.117**). Absolute RTF figures may now enter the paper provided the machine and load are stated.

## Phase 0 — the open question (0a)

Whether there is enough degradation budget to **decompose per layer**. Uniform INT8 costs only
~−0.08 dB SI-SDR; on a 4-clip smoke test, single-layer effects were ~10–40× smaller and not
individually distinguishable from zero. The 200-clip run tightens the CIs ~7× and decides it.

Decision rule: if no single quantizable layer shows a measurable effect at n=200, the per-layer
ranking has no signal at INT8 — and since INT4 is unavailable (0c), the study should **stop**
rather than manufacture a ranking from noise.
