# Kaggle Phase A batch — findings

Run 2026-08-20 on Kaggle. **Intel Xeon @ 2.00 GHz, 4 cores, 1-min load average 0.41** (idle) —
against the local AMD Ryzen 5 5600H, so the timing results double as a cross-architecture check.
ONNX Runtime pinned to 1.20.1 as everywhere else.

---

## R2 — the stem exemption is smaller, faster, *and* FP32-quality

| backend | threads | mean RTF | slowdown vs FP32 | mean SI-SDR | size |
|---|---:|---:|---:|---:|---:|
| FP32 | 1 | 0.0894 | — | 18.690 | 2842.7 KB |
| uniform INT8 | 1 | 0.1244 | 1.392× | 18.569 | 950.7 KB |
| **mixed (stem FP32)** | 1 | **0.1181** | **1.322×** | **18.691** | **916.9 KB** |
| FP32 | 2 | 0.0819 | — | 18.690 | 2842.7 KB |
| uniform INT8 | 2 | 0.0986 | 1.204× | 18.569 | 950.7 KB |
| **mixed** | 2 | **0.0944** | **1.153×** | **18.691** | **916.9 KB** |
| FP32 | 4 | 0.0837 | — | 18.690 | 2842.7 KB |
| uniform INT8 | 4 | 0.0994 | 1.188× | 18.569 | 950.7 KB |
| **mixed** | 4 | **0.0948** | **1.133×** | **18.691** | **916.9 KB** |

**The prediction held.** Mixed carries 29 QDQ nodes against uniform INT8's 172, and it is faster at
every thread count — by 4–5 % consistently. So exempting the stem gives a model that is
simultaneously:

- **smaller** than uniform INT8 (916.9 vs 950.7 KB),
- **faster** than uniform INT8 (1.13–1.32× vs 1.19–1.39× slowdown),
- **quality-identical to FP32** (18.691 vs 18.690 SI-SDR, against uniform's 18.569).

This converts the study's "INT8 is slower than FP32" negative into a positive: the graph overhead
that causes the slowdown is itself concentrated, and the same exemption that recovers the quality
also recovers part of the speed. It is the concentration thesis showing up in a third quantity
after loss and file size.

**Cross-architecture replication.** INT8 is slower than FP32 on both CPUs tested — 1.19–1.39× on
Intel Xeon, 1.39–1.46× on AMD Ryzen. The finding is not an artifact of one machine. Absolute RTFs
differ, as expected, and both are now measured on verified-idle machines.

---

## W2 / W3 — the power fixes worked, and they change a claim

Effective sample — clips whose WER actually differs between GTCRN and passthrough:

| evaluation | effective n | vs original |
|---|---|---|
| VoiceBank, `small.en` (original) | 129 / 824 = 15.7 % | — |
| VoiceBank, `tiny.en` | 216 / 824 = 26.2 % | 1.7× |
| **MS-SNSD, `small.en`** | **704 / 1100 = 64.0 %** | **5.5×** |

### The aggregate degradation is now firmly established

Enhancement significantly raises corpus WER on **three independent measurements**:

| evaluation | Δ corpus WER | relative | CI excludes 0 |
|---|---:|---:|---|
| VoiceBank / `small.en` | +0.0083 | +24.1 % | yes |
| VoiceBank / `tiny.en` | +0.0165 | +23.1 % | yes |
| MS-SNSD / `small.en` | +0.0641 | +46.6 % | yes |

The two VoiceBank figures agree closely (+24.1 % and +23.1 %) despite different recognisers, which
is strong evidence the effect is real and not an artifact of one ASR model.

### A claim that must be softened: "concentration" was partly a power artifact

The earlier reading — *degradation concentrated in babble and the lowest SNR bin* — does not
survive better power. On MS-SNSD, **8 of 9 noise types are individually significant**:

| noise | relative Δ | significant |
|---|---:|---|
| CopyMachine | +76.1 % | yes |
| Babble | +44.2 % | yes |
| AirportAnnouncements | +78.4 % | yes |
| ShuttingDoor | +94.3 % | yes |
| Munching | +80.4 % | yes |
| AirConditioner | +67.6 % | yes |
| VacuumCleaner | +41.8 % | yes |
| Neighbor | +12.4 % | yes |
| **Typing** | +23.2 % | **no** |

With 5.5× the effective sample, enhancement hurts ASR **nearly everywhere**, not in a couple of
conditions. The VoiceBank per-condition nulls were largely a lack of power, not evidence that those
conditions are unaffected.

Corrected statement:

> Enhancement significantly degrades ASR accuracy, replicated across two corpora and two
> recognisers (+23 % to +47 % relative). The degradation is broad rather than concentrated: with
> adequate statistical power, 8 of 9 MS-SNSD noise types show it individually. The only exception
> is `Typing` — impulsive noise, where the enhancer barely acts at all (+0.2 dB SI-SDRi in the
> objective benchmark), so there is little processing to do damage.

Note this is the opposite direction to the rest of the paper: here the *aggregate* was right and the
disaggregation initially misled us, because the per-condition tests were underpowered. That is worth
stating plainly — a paper about metrics misleading should be first to report when its own
disaggregation was the misleading step.

### The `bus` improvement is definitively an artifact

Across three measurements of the same clips it does not hold a sign:

| measurement | `bus` relative Δ | significant |
|---|---:|---|
| VoiceBank `small.en`, per-clip | −41.7 % (helps) | fails FDR (q = 0.095) |
| VoiceBank `small.en`, corpus | −28.0 % (helps) | no |
| VoiceBank `tiny.en`, corpus | **+37.2 % (hurts)** | no |

A sign flip between recognisers on identical audio settles it. The earlier correction was right and
now has independent support.

---

## R1 remedy re-tested — and it gets simpler

Re-running condition-gated enhancement against the `tiny.en` labels:

| routing | corpus WER |
|---|---:|
| always enhance | 0.0876 |
| learned router (leave-one-noise-type-out) | 0.0730 |
| **never enhance** | **0.0711** |
| oracle, per noise type | 0.0711 |

**The oracle chooses passthrough for all five noise types.** Under this recogniser there is no
condition where enhancement helps ASR, so the oracle and never-enhance coincide exactly.

The router still beats always-enhance significantly (CI [−0.0215, −0.0083], 89 % of headroom
captured) but cannot beat simply switching the enhancer off. Combined with the MS-SNSD result, the
recommendation is now unambiguous and needs no routing machinery at all:

> **Keep the enhancer out of the ASR path.** Give the recogniser the unprocessed signal and the
> human listener the enhanced one. They are different consumers with different optima.

More data made the recommendation *simpler*, not more elaborate. The routing experiment was still
worth running — it establishes that the decision is learnable from noisy audio alone and
generalises to an unseen noise type — but it is reported as a negative for routing, not a win.
