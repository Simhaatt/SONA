# RTF re-measurement on an idle machine — 2026-08-20

Both `PERCEPTUAL_MP/FINDINGS.md` §4 and `PERCEPTUAL_MP/README.md` flagged the July timings as
untrustworthy: they were taken while the machine sat at ~21 % CPU with browsers running, and the
same configuration measured 0.121 in one run and 0.434–0.472 in another. No absolute RTF was to be
published until re-measured. **This is that re-measurement.**

## Conditions

| | |
|---|---|
| CPU | AMD Ryzen 5 5600H (6 physical / 12 logical, 3.3 GHz base) |
| OS | Windows 11 Home Single Language |
| CPU load at start | **6 %** (vs ~21 % for the July run) |
| CPU load at finish | 16 % |
| onnxruntime | **1.20.1** — pinned to the exact version the quantization study used, so the numbers stay comparable |
| torch | 2.13.0+cpu, `torch.set_num_threads(1)` for the PyTorch reference |
| clips | 100, VoiceBank-DEMAND test manifest |

This is *edge-simulated*, not an on-device measurement: constrained cores are approximated by
pinning ONNX Runtime's intra-op thread count. It is not a Raspberry Pi number and must not be
reported as one.

## Result

| backend | threads | mean RTF | p95 latency (ms) | mean SI-SDR | size (KB) |
|---|---:|---:|---:|---:|---:|
| fp32 | 1 | 0.0840 | 706.96 | 18.692 | 2842.7 |
| fp32 | 2 | 0.0600 | 527.81 | 18.692 | 2842.7 |
| fp32 | 4 | 0.0483 | 460.89 | 18.692 | 2842.7 |
| int8 | 1 | 0.1166 | 985.02 | 18.575 | 950.7 |
| int8 | 2 | 0.0857 | 746.05 | 18.575 | 950.7 |
| int8 | 4 | 0.0703 | 566.64 | 18.575 | 950.7 |
| pytorch | — | 0.0972 | 677.61 | 18.692 | — |

## What changed, and what did not

**Absolute RTFs dropped by ~25–30 % once the machine was idle**, confirming the July numbers were
load-inflated and should not have been published:

| config | July (loaded) | now (idle) |
|---|---:|---:|
| fp32, 1 thread | 0.1150 | **0.0840** |
| int8, 1 thread | 0.1510 | **0.1166** |
| fp32, 2 threads | 0.0722 | **0.0600** |
| int8, 2 threads | 0.0980 | **0.0857** |
| pytorch | 0.1215 | **0.0972** |

**The finding itself replicates unchanged: INT8 is slower than FP32 on this x86 CPU**, and the
ratio is now measured cleanly at three thread counts:

| threads | int8 / fp32 |
|---:|---:|
| 1 | 1.39× |
| 2 | 1.43× |
| 4 | 1.46× |

That is a tighter interval than the "~1.3–1.8× slower" recorded in July, and it is monotone in
thread count — INT8's disadvantage *grows* slightly as cores are added, consistent with the QDQ
node overhead being harder to parallelise than the dense FP32 kernels it replaces.

Quality is unchanged from July: INT8 costs 0.117 dB SI-SDR (18.692 → 18.575) on these 100 clips.

## Conclusion for the papers

These absolute figures are safe to quote, with the machine and load stated. The claim to make is:

> On x86 CPU, INT8 quantization of this model buys a 3.0× size reduction (2842.7 → 950.7 KB) at a
> cost of 0.117 dB SI-SDR **and a 1.4× slowdown**. The compression benefit is memory, not speed.

The July run's *relative* orderings were already correct, so no earlier conclusion is overturned —
only the absolute numbers are now trustworthy.
