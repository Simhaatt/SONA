# PERCEPTUAL_MP — Findings

All results below are measured, not estimated. Selection was done on a speaker-disjoint dev split
carved from TRAIN (p254/p256); the VoiceBank test set (p232/p257, 824 clips) and MS-SNSD (1100
clips) were touched only after the allocation was frozen and timestamped
(`results/phase3/frozen_allocation.json`).

---

## 1. The original hypothesis: NOT SUPPORTED (honest negative)

**Hypothesis:** ranking layers by *perceptual* sensitivity (PESQ) would differ from ranking them
by *MSE-family* sensitivity (SI-SDR, mask-MSE), so perceptually-guided bit allocation would beat
MSE-guided allocation.

**Result:** the rankings agree almost perfectly across the 6 quantizable layers.

| comparison | Spearman ρ |
|---|---|
| PESQ vs SI-SDR | +0.943 |
| PESQ vs mask-MSE | **+1.000** |

`stem` ranks #1 most sensitive under *every* metric; `head` ranks #6 under *every* metric. At the
allocation level the two orderings differ at exactly one position and produce a ΔSI-SDR of
**+0.0003 dB** — indistinguishable.

**Conclusion:** for this checkpoint, at INT8 / 6-layer granularity, perceptual and MSE-family
sensitivity converge. MSE-guided allocation is already perceptually adequate. Report as a
negative result; do not claim a perceptual advantage.

---

## 2. Main positive result: exempt the stem from INT8

Discovered while sweeping the size/quality curve, then verified out-of-sample.

### 2a. Fidelity to FP32 — the robust, cross-corpus claim

Post-training quantization's job is to reproduce the source model. Mean **absolute per-clip
deviation from FP32** (lower = more faithful):

| corpus | metric | mixed (stem FP32) | uniform INT8 | advantage |
|---|---|---|---|---|
| VoiceBank-824 | \|ΔSI-SDR\| | **0.0202** | 0.1458 | **7.2×** |
| VoiceBank-824 | \|ΔSTOI\| | **0.00025** | 0.00186 | 7.4× |
| VoiceBank-824 | \|ΔESTOI\| | **0.00055** | 0.00481 | 8.7× |
| VoiceBank-824 | \|ΔLSD\| | **0.0088** | 0.1708 | **19.5×** |
| MS-SNSD-1100 | \|ΔSI-SDR\| | **0.0328** | 0.2832 | **8.6×** |
| MS-SNSD-1100 | \|ΔSTOI\| | **0.00051** | 0.00710 | 13.9× |
| MS-SNSD-1100 | \|ΔESTOI\| | **0.00079** | 0.00966 | 12.2× |
| MS-SNSD-1100 | \|ΔLSD\| | **0.0209** | 0.2915 | 13.9× |

**8/8 cells favour the mixed allocation, by 7–20×, across two corpora.** This is the headline.

### 2b. VoiceBank-DEMAND test set (824 clips, out-of-sample)

| model | size | SI-SDR | vs FP32 |
|---|---|---|---|
| FP32 | 2842.7 KB | 18.3765 | — |
| **Mixed (stem FP32, 5 layers INT8)** | **916.9 KB** | **18.3751** | **p = 0.25 (n.s.)** |
| Uniform INT8 | 950.7 KB | 18.2671 | p = 3.7e-51 |

- Exempting the stem vs uniform: **+0.108 dB SI-SDR**, 95% CI [+0.092, +0.126], **p = 9.2e-49**
- Mixed vs FP32: **statistically indistinguishable** (p = 0.25)
- The stem is **3840 bytes = 0.133%** of model weights, yet carries ~97% of the degradation
  (that figure is from the dev split; computed on the held-out 824-clip test set it is
  **98.7%** — (0.1094 − 0.0014) / 0.1094 — which is the number the paper draft quotes)

**The mixed model is also smaller than uniform INT8** (916.9 vs 950.7 KB). Cause verified: asking
ONNX Runtime to quantize every layer makes it wrap ~143 additional activation tensors in QDQ
pairs (**172 vs 29 QDQ nodes**), and that graph overhead exceeds the stem's 2.8 KB of weight
savings. Quantizing the stem *alone* even grows the file (2851.9 KB vs FP32's 2842.7 KB).

### 2c. Condition robustness (VoiceBank test)

Significant in **all 9 conditions** — 5 noise types × 4 SNRs:

| condition | ΔSI-SDR | p |
|---|---|---|
| psquare | +0.175 | 1.2e-15 |
| cafe | +0.156 | 6.8e-20 |
| living | +0.098 | 1.3e-07 |
| bus | +0.087 | 1.7e-08 |
| office | +0.024 | 1.5e-04 |
| 2.5 / 7.5 / 12.5 / 17.5 dB | +0.125 / +0.095 / +0.104 / +0.108 | all < 1e-10 |

Flat across SNR (not an SNR artifact); largest in hard non-stationary noise, smallest in easy
stationary office noise — consistent with the stem's early feature extraction mattering most
when spectral structure is complex.

### 2d. Cross-corpus caveat — state this explicitly

On MS-SNSD the *stem-exemption* SI-SDR benefit replicates (**+0.039 dB**, p = 3.0e-12) and mixed
remains indistinguishable from FP32 on SI-SDR (p = 0.17). **However**, on MS-SNSD uniform INT8
scores *better* than mixed on STOI (−0.0060), ESTOI (−0.0067) and LSD (+0.233).

This is not evidence that uniform is a better allocation: uniform INT8 also beats **FP32 itself**
on MS-SNSD STOI (0.8212 vs 0.8154). Its larger quantization noise acts as a mild dither that
flatters intelligibility metrics under distribution shift while degrading waveform fidelity. The
mixed model instead tracks FP32 closely on every metric (§2a). Frame the claim as *fidelity*, not
as "better enhancement on every metric everywhere."

---

## 3. Non-additive quantization error

Layer-wise degradations do **not** add:

| corpus | stem alone | 5 blocks alone | sum | actual combined | interaction | ratio |
|---|---|---|---|---|---|---|
| dev (200) | −0.0061 | −0.0023 | −0.0083 | −0.0996 | −0.0913 [−0.149, −0.040] | **11.95×** |
| MS-SNSD (1100) | +0.0260 | +0.0010 | +0.0269 | −0.0377 | −0.0647 [−0.090, −0.039] | 1.40× |

The interaction term's CI excludes zero on both, so error compounds through the network rather
than accumulating independently. The magnitude is strongly corpus-dependent (11.95× vs 1.40×), so
report the *existence* of the interaction as robust and its *size* as corpus-specific.

---

## 4. Runtime limitations found (report as findings, not bugs)

- **The band GRU cannot be quantized** by ONNX Runtime 1.20.1: requesting it yields 0 QDQ nodes,
  no size change, and ~8.5% *slower* inference. The recurrent decoder is incompressible on this
  runtime; it is pinned to FP32 in all allocations.
- **INT4 is unavailable for Conv layers** via `quantize_static` ("must be 8-bit before packing as
  4-bit"). ORT's 4-bit path covers MatMul only, and this graph has a single tiny MatMul, so INT4
  cannot reach the weight-heavy Conv layers. The precision axis is therefore {FP32, INT8}.
- **INT8 is ~1.4× SLOWER than FP32** on this x86 CPU. The compression benefit is *size*, not
  speed, on this hardware. **Re-measured on an idle machine 2026-08-20** (6 % CPU load, ORT
  pinned to the same 1.20.1) — see `../results/onnx/edge_bench_idle_2026-08-20/`. The finding
  replicates and the ratio is now tight and monotone in thread count: **1.39× / 1.43× / 1.46×**
  at 1 / 2 / 4 threads. Absolute RTFs fell ~25–30 % from the July figures (fp32 1-thread
  0.115 → **0.084**; int8 1-thread 0.151 → **0.117**), confirming the earlier numbers were
  load-inflated. The July run's *relative* orderings were already correct, so nothing above is
  overturned — the absolute figures are now safe to publish, with machine and load stated.

---

## 5. Provenance

Motivated by the objective-mismatch insight in TurboQuant (arXiv 2504.19874) — that an
MSE-optimal quantizer can be wrong for the objective you actually care about. TurboQuant's
*algorithm* does not apply here (no KV cache, no vector database, no memory bottleneck); only the
framing carried over, and the perceptual-vs-MSE form of that hypothesis was **refuted** (§1).
Cite for motivation only.
