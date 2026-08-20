# Consolidated status — all research tracks

**Last updated 2026-08-20.** This file is the single entry point for the whole body of work.
Everything now lives under `D:\noise cancellation`; the old `F:\Rohit\FYP\` (and the `E:\Rohit\FYP\`
paths written into older docs) is retired. Nothing outstanding remains on those drives.

Hardware from here on: **Kaggle T4 for anything needing a GPU, local CPU for everything else.**
No local GPU.

---

## Repository layout after consolidation

```
D:\noise cancellation\
├── src/ tests/ configs/ scripts/     SONA core: models, eval harness, Layer A/B controller
├── web_demo/                         Flask demo (port 7860)  — see project memory
├── results/                          every experiment's summary/per-clip artefacts
├── vbd_local/                        local 16 kHz VoiceBank-DEMAND (3.0 GB, gitignored)
├── external_data/                    MS-SNSD corpus + SONA-mixed test set
├── PERCEPTUAL_MP/                    mixed-precision quantization study  (Track 2)
├── sona-final-month/                 results consolidation, complex-mask A/B,
│                                     scene classifier, realtime, FYP report  (Track 3)
├── AudioVQ/                          data-oblivious VQ for audio retrieval  (Track 4)
└── mos_test/                         blinded MOS listening-test app
```

Doc map:

| file | what it is |
|---|---|
| `PROJECTS_STATUS.md` | **this file** — current state of everything |
| `SONA_STATUS.md` | chronological run log for the SONA core (to 2026-07-07) |
| `NEXT_STEPS_PLAN.md` | the Jul-20 → Aug-16 plan; **expired**, kept for provenance |
| `PERCEPTUAL_MP_QUANT_PLAN.md` / `PERCEPTUAL_MP/FINDINGS.md` | quantization design + results |
| `PERCEPTUAL_MP/PAPER_DRAFT.md` | **quantization paper draft** (new 2026-08-20) |
| `sona-final-month/COMPLEX_MASK_FINDINGS.md` | complex vs magnitude mask A/B |
| `sona-final-month/SCENE_CLASSIFIER_FINDINGS.md` | scene classifier real-audio evaluation |
| `AudioVQ/FINDINGS.md` / `AudioVQ/PAPER_DRAFT.md` | the primary paper |
| `AudioVQ/REFERENCES.md` | **bibliography + citation audit** (new 2026-08-20) |

Verified after the move: **75 tests pass in the SONA repo** (`tests/` + `web_demo/tests`) and
**34 in AudioVQ**. Every corpus manifest was rewritten to `D:` paths and all 13 500+ referenced
audio files were confirmed present on disk.

**Migration completeness audit (2026-08-20):** all **46 078** files under the old FYP tree were
checked for a counterpart here — **0 missing**. The 44 files that differ in size are the ones
deliberately edited during the move (re-rooted scripts, rewritten manifests, updated docs, new
paper drafts). Deliberately not copied: `.venv_dfn2/` (1.2 GB virtualenv, reproducible from
`requirements.txt`; all 825 DeepFilterNet2 enhanced clips and its scored summaries are here),
`sona-final-month/.git` (initialised but **zero commits**, so no history existed), and
`__pycache__`/`.pytest_cache`. The archival `vbd16k.zip` (2.3 GB Kaggle training set) and
`DS_10283_2791.zip` (16 GB) are both present.

---

## Track 1 — SONA core (speech enhancement)

Locked backbone `gtcrn_ch96_h96_b4_mw1p25_mr0p25`, 722 K params. Not retrained since July; all
later work is evaluation, analysis and deployment on the frozen checkpoint.

### The benchmark (824-clip VoiceBank-DEMAND test set)

| method | SI-SDRi | STOI | ESTOI | LSD | RTF |
|---|---:|---:|---:|---:|---:|
| passthrough | 0.007 | 0.921 | 0.787 | 10.500 | 0.006 |
| hard threshold | −3.694 | 0.767 | 0.573 | 13.775 | 0.008 |
| soft threshold | −5.815 | 0.728 | 0.532 | 15.282 | 0.008 |
| spectral subtraction | −0.251 | 0.884 | 0.734 | 8.864 | 0.008 |
| Wiener | −1.927 | 0.824 | 0.677 | 10.871 | 0.008 |
| **learned GTCRN (ours)** | **9.927** | **0.930** | **0.832** | **6.469** | 0.009 |

**Every classical baseline is worse than doing nothing**, on every metric — and the breakdowns
confirm this holds in all 5 noise types and all 4 SNRs without exception. That is the paper's
most robust claim: it is not an averaging artifact.

### Significance battery (`sona-final-month/SIGNIFICANCE/`)

Paired Wilcoxon + 10 k-resample bootstrap CIs on all 824 clips. GTCRN beats the attention
baseline on SI-SDRi by **+0.845 dB** (CI [0.753, 0.942], p < 1e-6) and beats every heavier
variant tried — Mamba-GTCRN (+0.083), SONA-G governed (+0.039), SONA-R residual (+0.059). A clean
negative-results ablation: **added capacity never helped.**

Strict-guarded online adaptation vs frozen: differences are significant but the effect size is
~0.003 dB — i.e. the guard works exactly as designed (provably non-degrading) and buys nothing.
Frame it as *a safety governor for test-time adaptation*, not as a win.

### Where we stand against published work

| method | PESQ | STOI | ESTOI | SI-SDRi |
|---|---:|---:|---:|---:|
| noisy | 1.968 | 0.921 | 0.787 | — |
| **GTCRN (ours, 722 K)** | ~2.52 | 0.930 | 0.832 | 9.93 |
| DeepFilterNet2 (reproduced here) | 3.057 | 0.944 | 0.862 | 7.262 |
| MetricGAN+ (published) | 3.134 | 0.932 | 0.832 | — |

The DeepFilterNet2 reproduction landed at **PESQ 3.057 vs the published 3.08** on all 824 clips,
which validates the harness. Honest positioning: our model is *efficient* (722 K params, RTF
~0.009) and leads on SI-SDRi, but is **not PESQ-competitive** with larger PESQ-optimized models.
Say so plainly in the paper — the size/fidelity trade-off is the contribution, not a PESQ win.

### ASR robustness (WER, Whisper, 824 clips)

| method | WER | bus | cafe | living | office | psquare |
|---|---:|---:|---:|---:|---:|---:|
| passthrough | **0.043** | 0.034 | 0.063 | 0.044 | 0.020 | 0.051 |
| GTCRN | 0.047 | **0.020** | 0.091 | 0.051 | 0.022 | 0.053 |
| complex GTCRN | 0.047 | 0.022 | 0.080 | 0.061 | 0.019 | 0.052 |
| spectral subtraction | 0.071 | 0.057 | 0.115 | 0.061 | 0.056 | 0.068 |
| Wiener | 0.097 | 0.081 | 0.140 | 0.103 | 0.070 | 0.092 |

**Report this honestly: on aggregate WER, enhancement slightly *hurts* (0.047 vs 0.043).** The
finding is that the effect is strongly condition-dependent — a 41 % relative *cut* in `bus` noise
(0.034 → 0.020) against a 44 % *rise* in `cafe` babble (0.063 → 0.091). Enhancement tuned for
human listening is not automatically good for a downstream ASR model. Classical methods degrade
WER everywhere, which is consistent with the perceptual table.

### Deployment artefacts

ONNX export FP32 2.84 MB → INT8 0.95 MB, both in `results/onnx/gtcrn/`. Re-measured on an idle
machine 2026-08-20 (`results/onnx/edge_bench_idle_2026-08-20/`): INT8 is **1.4× slower** than FP32
(RTF 0.117 vs 0.084 single-thread) while costing 0.117 dB SI-SDR. On this x86 CPU the compression
benefit is size only.

### Open

- **MOS listening test needs human raters.** The app (`mos_test/`) is built and browser-verified,
  clips selected, analysis script written. Only participants are missing.
- ~~RTF figures were taken on a loaded machine.~~ **Done 2026-08-20** — re-measured at 6 % CPU
  load with ORT pinned to 1.20.1. The INT8-slower-than-FP32 finding replicates (1.39–1.46× across
  1/2/4 threads); absolute RTFs fell ~25–30 % and are now safe to quote with machine and load
  stated. See `results/onnx/edge_bench_idle_2026-08-20/README.md`.
- Edge deployment is argued from size/latency; never measured on a Pi.

---

## Track 2 — Perceptual mixed-precision quantization (`PERCEPTUAL_MP/`)

Selection done on a speaker-disjoint dev split (p254/p256) carved from TRAIN; the 824-clip test
set and MS-SNSD were touched only after the allocation was frozen and timestamped.

**The original hypothesis was refuted, and that is reported as such.** Perceptual (PESQ) and
MSE-family sensitivity rankings agree almost perfectly across the 6 quantizable layers
(ρ = +0.943 vs SI-SDR, **ρ = +1.000** vs mask-MSE). Perceptually-guided allocation buys +0.0003 dB.

**The real result is the stem exemption.** The 3.8 KB stem convolution — **0.133 % of weights** —
carries **98.7 %** of INT8 quantization loss on the held-out test set (`FINDINGS.md` quotes ~97 %
from the smaller dev split). Exempting it yields a model statistically
indistinguishable from FP32 (p = 0.25) at 3.2× compression, significant in **all 9 conditions**,
and *smaller than uniform INT8* (916.9 vs 950.7 KB — quantizing everything makes ONNX Runtime
insert 143 extra QDQ nodes). Fidelity to FP32 improves by **7–20× on 8/8 metric-corpus cells**.

Caveat to state explicitly: on MS-SNSD uniform INT8 scores better on STOI/ESTOI/LSD — but it also
beats *FP32 itself* there, so it is a dithering artifact. Claim **fidelity to FP32**, never "better
on every metric".

Runtime findings worth reporting: the band GRU cannot be quantized at all by ORT 1.20.1 (0 QDQ
nodes), and INT4 is unavailable for Conv layers.

**Written up 2026-08-20** as `PERCEPTUAL_MP/PAPER_DRAFT.md` — a full conference-length draft with
the negative result reported in front, the closest prior work identified (Rusci et al. 2022, whose
layer-type rule is *inverted* on our runtime), and `scripts/audit_paper_numbers.py` checking all
25 asserted numbers against their source artefacts on every run. One substantive gap remains: a
Hessian-trace ranking as a third column in §5.1, which is cheap and would materially strengthen
the negative result.

---

## Track 3 — Final-month package (`sona-final-month/`)

Software-only analysis package. `python scripts/check_all.py` rebuilds tables, runs the
significance battery, regenerates figures, and runs the realtime + scene self-tests in one go.

**Complex ratio mask A/B** — identical GTCRN backbone, only the output representation changes
(+97 params). On 824 clips the complex mask significantly improves SI-SDRi (+0.313 dB, p < 1e-6)
and STOI (+0.0020, p < 1e-35) but **significantly lowers PESQ** (−0.016, p = 0.005). Only real
PESQ surfaced the trade-off — an SI-SDR-only evaluation would have reported a false clean win.
This is a genuinely instructive methodological result.

**Scene classifier** — the shipped heuristic scored **0.469** on real audio (vs 0.767 on its
synthetic self-test); two hand-tuned windows were measurably inverted. Retuning reached 0.714, and
swapping the *decision rule* while holding the same 8 features fixed reached 0.844 — proving the
rule, not the features, was the bottleneck. Then honest re-evaluation under **grouped** CV (no
noise texture on both sides of a split) plus a richer 34-feature set settled at
**0.748 ± 0.074**, and that is the number shipped. The 1,716-parameter MLP in
`src/controller/scene_model.py` needs neither scikit-learn nor torch at inference.

The self-correction from 0.844 → 0.748 is a strength, not a retraction: grouped CV is what a
reviewer would demand.

**Also here:** realtime block-streaming demo with COLA-verified overlap-add, the FYP technical
report (`report/main.tex` + PDF), and three results review decks.

---

## Track 4 — AudioVQ (primary paper) (`AudioVQ/`)

*Recall Is the Wrong Yardstick: Evaluating Data-Oblivious Vector Quantization on Audio
Embeddings.* Draft in `PAPER_DRAFT.md`, all 15 findings in `FINDINGS.md`.

Evidence: 3 embedding models (WavLM, wav2vec2, HuBERT) · 36 layer-points · 10 figures · 3 LaTeX
tables · 34 unit tests · 19 paper claims auto-audited against source artefacts.

**Central claim (F7): geometry and meaning come apart.** At ~30× compression roughly half the true
nearest neighbours are destroyed while **~96 % of downstream retrieval accuracy survives**, on all
three models. Prior evaluation of these methods is recall-only, so this failure of the standard
metric was not visible. Not an artifact of layer choice — it is *stronger* on the best layer
(99.7 % speaker retention at 62.3 % recall).

Supporting: trained codebooks (OPQ) win with the gap widening as budget shrinks (clean on 2/3
models); the rotation works by flattening a spiky within-vector coordinate profile (crest
10.2 → 3.3, isotropic ideal 3.4) — an advance cross-model prediction that was confirmed; the
untrained quantizer reaches 98 % of OPQ's recall at **1/434th** the build time; and the
conventional final layer is the **worst of twelve** on all three models.

Five claims were deliberately narrowed after tests that could have preserved the stronger version
(crest-factor prediction falsified 4×; the anisotropy observation credited to prior work rather
than claimed as ours). That discipline is itself worth foregrounding in the write-up.

### Citation pass — done 2026-08-20, and it forced a claim change

Full bibliography and per-claim audit in `AudioVQ/REFERENCES.md`. Three claims verified intact
(no prior data-oblivious VQ on audio; crest factor unused in the speech-anisotropy literature; the
wav2vec2-internal-PQ distinction is correct). Two gaps fixed (RaBitQ [SIGMOD 2024] added as the
other major data-oblivious quantizer; the Qdrant source relabelled as an industry technical
report). And one finding that matters:

> **[Dimitropoulos & Mamoulis, arXiv:2606.04522, 3 June 2026] — *ANN Search: Recall What Matters* —
> already argues the general form of our central claim**: downstream quality stays stable while
> Recall@k drops substantially. It is prior work, it was uncited, and "prior evaluation is
> *exclusively* recall-based" was not defensible as written.

The draft has been rewritten accordingly. What survives is stronger for being scoped: they vary
*search effort over exact vectors*, we vary *the representation itself* — quantization destroys
information permanently, so their stability result does not imply ours. Plus audio, which neither
literature covers, plus the measured effect size across three backbones. Two literatures reaching
the same conclusion by different routes is corroboration, not competition — and that is now how
the paper frames it.

### Open

- One database-level sweep (DBLP / ACL Anthology / ISCA Archive) before submission, chiefly to
  catch work from June–August 2026. The Dimitropoulos find shows a two-month gap matters here.
- 10⁶-scale behaviour is extrapolated from 10⁴ (the bank holds 23 k utterances).
- No on-device measurement.

---

## Reproducibility

No hardcoded machine paths remain in any script. Corpora resolve from environment variables with
tree-relative fallbacks derived from each file's own location, so the tree survives being moved
or copied to another drive:

```
AUDIOVQ_MS_SNSD / AUDIOVQ_VBD_TEST_CLEAN / AUDIOVQ_NOISY_MANIFEST   (AudioVQ)
SONA_VBD_TRAIN_MANIFEST / SONA_VBD_TEST_MANIFEST                     (PERCEPTUAL_MP)
```

`AudioVQ/src/config.py::require()` fails loudly on a missing corpus rather than silently scoring an
empty file list. `vbd_local/` and `AudioVQ/embeddings/` are gitignored and regenerable —
`sona-final-month/scripts/prep_voicebank_local.py` and `AudioVQ/scripts/extract_embeddings.py`
respectively.

## If picking this up later

1. **AudioVQ is the primary paper and is empirically complete.** The citation pass is done and
   the claim it forced has been narrowed. Remaining: one database-level sweep, then submission
   (IEEE Access / EURASIP JASMP are reasonable fits).
2. **The quantization paper has a full draft** with a passing number audit. The one substantive
   addition worth making is a Hessian-trace ranking in §5.1.
3. **SONA's other publishable results** — the condition-robust classical-baseline finding and the
   complex-mask PESQ/SI-SDR trade-off — still need drafting. All their numbers are frozen and
   significance-tested.
4. **Do not re-run experiments to "improve" numbers.** Every headline is frozen. The remaining
   work is written, not computed.
5. Both number audits should be run before anything is circulated:
   `python AudioVQ/scripts/audit_paper_numbers.py` (19 claims) and
   `python PERCEPTUAL_MP/scripts/audit_paper_numbers.py` (25 claims).
