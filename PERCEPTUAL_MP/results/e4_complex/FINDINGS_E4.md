# E4 — stem exemption on a second architecture: findings

Script: `PERCEPTUAL_MP/scripts/e4_second_architecture.py`. Complex-mask GTCRN, 200 dev clips
(p254/p256, speaker-disjoint from TRAIN). Same runtime, same calibration split, same protocol as
the magnitude model.

**This experiment was run knowing it could weaken the paper's headline.** It did not.

---

## 1. Result — the finding replicates

| | magnitude GTCRN | complex GTCRN |
|---|---|---|
| most sensitive layer | **stem** | **stem** |
| least sensitive layer | **head** | **head** |
| stem's share of INT8 degradation | **98.7 %** | **92.1 %** |
| uniform INT8 ΔSI-SDR | −0.109 dB | −0.051 dB |
| mixed (stem FP32) ΔSI-SDR | −0.0014 dB (n.s.) | +0.0041 dB |
| uniform INT8 size | 950.7 KB | 952.2 KB |
| **mixed size** | **916.9 KB** | **917.4 KB** |

Per-layer sensitivity, mean absolute ΔSI-SDR against FP32:

| layer | mean abs ΔSI-SDR |
|---|---:|
| **stem** | **0.0176** |
| block0_d1 | 0.0124 |
| block1_d2 | 0.0086 |
| block3_d8 | 0.0086 |
| block2_d4 | 0.0083 |
| head | 0.0067 |

Three things transfer:

1. **The stem dominates.** Most sensitive of the six on both models, and the head is least
   sensitive on both.
2. **The exemption works.** Removing the stem from the INT8 set takes the loss from −0.051 dB to
   +0.004 dB.
3. **The size inversion replicates.** The mixed model is *smaller* than uniform INT8 (917.4 vs
   952.2 KB) on this architecture too, for the same reason — quantizing every layer makes the
   runtime insert QDQ pairs around many more activation tensors, and that graph overhead exceeds
   the stem's weight savings.

## 2. What this does and does not license

**Does:** the claim moves from one checkpoint to two models, so "the stem carries the damage" is no
longer a single-checkpoint observation. The reviewer objection that motivated this experiment is
answered.

**Does not:** both models share the *same backbone* — stem, four gated TF-conv blocks, band GRU —
and differ only in the output head (2-channel complex ratio mask, +97 parameters). This is a
replication across **output representations**, not across architecture families. It does not show
the finding holds for a differently-shaped denoiser.

State it that way in the paper. The honest scope is: *robust to the output representation, untested
across backbone families.* A genuinely independent test would need something like DeepFilterNet2,
whose enhanced outputs are already in the repo but which was never put through the per-layer sweep.

## 3. One thing our own diagnostic flags

The mixed model scores **+0.0041 dB above FP32** on this dev split, with a bootstrap CI of
[+0.0000, +0.0080] that only marginally excludes zero.

By the dither rule established in R3 — *a lossy approximation cannot be better than the model it
approximates, so a variant beating FP32 means the metric is compromised on that data* — this cell
should be flagged rather than reported as an improvement. The effect is tiny and the CI touches
zero, so the correct reading is "indistinguishable from FP32", the same conclusion as the magnitude
model, **not** "better than FP32".

Applying our own rule against our own favourable result is the consistent thing to do, and it is
worth a sentence in the paper for exactly that reason.

## 4. Caveats

- Dev split only (200 clips). The magnitude model's headline numbers come from the 824-clip test
  set; this replication was not run on test, so the two columns in §1 are not measured on the same
  data. Comparable in direction, not directly comparable in magnitude.
- The complex checkpoint was trained on a 5 000-clip subset, not the full corpus, so its absolute
  quality is below the locked model. The *relative* per-layer pattern is what transfers.
- ONNX export for this model required pinning the legacy TorchScript exporter: torch 2.13's dynamo
  path emits `Split` with a `num_outputs` attribute that ONNX Runtime 1.20.1 cannot load during
  static quantization. Recorded in `scripts/export_onnx.py` behind `--dynamo`.
