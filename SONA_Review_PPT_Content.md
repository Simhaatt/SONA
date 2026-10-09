# SONA — Review Presentation Content (slide by slide)

**Project:** AI-Based Adaptive Noise Cancellation for Speech Enhancement (SONA)
**Team:** A. Rohit Daniel · Simhaa T. T. · Vijayaraghavan A.
**Guide:** Dr. G. R. Venkatakrishnan
**Department:** Electrical and Electronics Engineering, Sri Sivasubramaniya Nadar College of Engineering, Chennai
**Review scope:** Phase 1 (software) complete → Phase 2 (hardware) plan
**Suggested length:** 38 main slides + backup · 20–25 min talk + live demo + Q&A

> How to use this file: each slide has **On slide** (what goes on the slide, keep it short),
> **Visual** (what to show and where the file is), and **Say** (speaker notes). Anything in
> `[FILL IN: …]` must be completed by the team — mostly conference submission details that are
> not recorded in the repository.

---

## Presentation flow and timing

| Part | Slides | Time | Presenter (suggested) |
|---|---|---:|---|
| A. Opening: problem, objectives, scope | 1–4 | 2 min | [FILL IN] |
| B. Background and gap | 5–6 | 1.5 min | [FILL IN] |
| C. Development journey | 7–10 | 3 min | [FILL IN] |
| D. System design and method | 11–17 | 4 min | [FILL IN] |
| E. Results | 18–24 | 4 min | [FILL IN] |
| F. Further research studies | 25–27 | 2 min | [FILL IN] |
| G. Application and live demo | 28–31 | 4 min (incl. demo) | [FILL IN] |
| H. Publications: conference paper and journal work | 32–35 | 3 min | [FILL IN] |
| I. Phase 2: hardware plan | 36–37 | 2 min | [FILL IN] |
| J. Limitations, conclusion, Q&A | 38–40 | 1.5 min + Q&A | [FILL IN] |

Design notes for the deck: one idea per slide; big numbers for results; keep tables to ≤ 6 rows
on a slide (move the rest to backup); use the same colour for "learned model" and for "classical"
on every chart (orange for neural, blue for classical matches the explainer page and web app).

---

# PART A — OPENING

## Slide 1 — Title

**On slide**
- SONA: AI-Based Adaptive Noise Cancellation for Speech Enhancement
- Phase 1: Software system · Phase 2: Hardware (plan)
- Team names, guide, department, college, review name and date `[FILL IN: Review N, date]`

**Visual:** the hero waveform strip from the explainer page (noisy left half → clean right half),
or the before/after spectrogram pair from the web app.

**Say:** "Our project removes background noise from speech using a small neural network. Today we
present Phase 1 — the complete software system, its evaluation and the research it produced — and
our plan for Phase 2, the hardware device."

---

## Slide 2 — The problem

**On slide**
- Speech is almost never recorded in silence: fans, traffic, babble, typing, machines
- Noise reduces intelligibility and causes listening fatigue
- Affects phone/video calls, hearing aids, voice assistants, recordings
- Classical filters assume *stationary* noise — real noise isn't

**Visual:** spectrogram of the vacuum-cleaner clip (noisy) — noise covers the whole frequency range.
Source: web app → Library → "Vacuum cleaner" → Spectrogram tab, or `SONA_Project_Explainer.html` §4.

**Say:** "In a spectrogram you can see the noise filling every frequency while the voice sits
underneath it. The job is to keep the voice and remove everything else — in real time."

---

## Slide 3 — Objectives

**On slide**
1. Build a compact, learned speech-enhancement model that runs in real time
2. Compare it fairly against classical DSP methods under one identical pipeline
3. Validate on audio from a different dataset (generalisation)
4. Build a usable system: scene-aware controller + interactive application
5. Prepare the model and streaming engine for embedded hardware (Phase 2)

**Say:** "Every objective maps to a section of this talk; objectives 1–4 are complete, objective 5
is the bridge into Phase 2."

---

## Slide 4 — Scope: Phase 1 vs Phase 2

**On slide**

| | Phase 1 — Software (complete) | Phase 2 — Hardware (next) |
|---|---|---|
| Input | Audio files: library clips and uploads | Live microphone |
| Processing | Laptop CPU / Kaggle GPU | Embedded board (e.g. Raspberry Pi 5) |
| Output | Enhanced WAV, analytics | Clean speech in headphones, live |
| Focus | Model, evaluation, controller, application | Latency, streaming, power, device |

**Say:** "Phase 1 answers *does the method work and how well*. Phase 2 answers *can it run live on
a small device*."

---

# PART B — BACKGROUND

## Slide 5 — Classical methods and why they fail

**On slide**
- **Spectral subtraction** (Boll, 1979): subtract an estimated noise spectrum
- **Wiener filter** (Lim & Oppenheim; Ephraim–Malah MMSE): gain from estimated SNR
- **Hard / soft thresholding** (Donoho, 1995): zero or shrink low-energy bins
- Four structural weaknesses:
  1. Thresholds / noise estimates tuned by hand
  2. Decide from amplitude only — no idea which frequencies carry speech
  3. Assume stationary noise
  4. Remove weak speech along with noise

**Say:** "These are the textbook methods. We implemented all four on the same front-end as our
model, so we can measure exactly how they behave on real noise."

---

## Slide 6 — Learned enhancement and the gap we address

**On slide**
- RNNoise (2018): DSP features + small RNN, real-time
- DCCRN (2020), DeepFilterNet2 (2022): high quality, larger models
- MetricGAN+ (2021): optimises a PESQ surrogate
- **GTCRN (ICASSP 2024):** competitive quality at ultra-low compute → our backbone
- **Gap:** most papers compare methods with *different* front-ends and pipelines, and rarely test
  on a second corpus → differences cannot be attributed to the model alone

**Say:** "Our contribution is as much methodological as architectural: one shared front-end, one
evaluation harness, honest reporting including negative results."

---

# PART C — DEVELOPMENT JOURNEY

## Slide 7 — Development timeline (Phase 1)

**On slide**

| When | Milestone |
|---|---|
| Jul, week 1 | Shared STFT front-end, 4 classical baselines, evaluation harness, first neural models (Mask GRU, Mask TCN, causal Transformer); online-adaptation experiments on Kaggle GPUs |
| Jul, weeks 1–2 | GTCRN pilot → 16-configuration sweep → loss tuning → full 824-clip evaluation; MS-SNSD external validation; three heavier variants tested and rejected; **model locked** |
| Jul, weeks 3–4 | Statistical significance battery, complex-mask A/B (5,000-clip training), learned scene classifier, controller modes + EQ, real-time streaming engine, first web demo, conference paper |
| August | Quantisation study, speech-recognition (WER) study, embedding-compression study, journal paper bundle |
| October | Web application v2 (9 engines, quality metrics, compare, real-time simulation), redesigned multi-page interface, portable package |

**Visual:** horizontal timeline graphic (5 milestones).

**Say:** "Development was iterative: every stage produced a frozen result before the next began, so
no number in this presentation depends on a later change."

---

## Slide 8 — Development methodology

**On slide**
- **Fair-comparison principle:** every method shares the front-end, output space and metrics
- **Frozen results:** once a run is evaluated, its numbers are not re-run to "improve" them
- **Statistics:** paired Wilcoxon signed-rank + 10,000-sample bootstrap CIs; FDR correction for
  per-condition breakdowns
- **Held-out testing:** test noises unseen in training; plus a second corpus (MS-SNSD)
- **Negative results kept:** failed ideas are reported, not hidden
- **Self-correction:** e.g. scene-classifier accuracy re-measured with grouped CV (0.844 → 0.748)

**Say:** "Several of our own early claims did not survive proper testing; we corrected them before
anyone else had to. That discipline is what makes the remaining claims trustworthy."

---

## Slide 9 — Tools and tech stack

**On slide**

| Area | Tools |
|---|---|
| Language / ML | Python, PyTorch, NumPy, SciPy |
| Audio | soundfile, librosa, custom STFT/ISTFT (perfect reconstruction) |
| Metrics | SI-SDR, SNR, STOI/ESTOI (pystoi), LSD, PESQ (pesq), Whisper WER |
| Training compute | Kaggle NVIDIA T4 GPU |
| Deployment | ONNX / ONNX Runtime (FP32, INT8, mixed precision) |
| Application | Flask backend, vanilla JS frontend, Web Audio API |
| Quality | pytest — ~60 core unit tests + 29 web-application tests |
| Writing | LaTeX (Springer LNCS, IEEEtran), TikZ figures |

**Say:** "Everything runs locally; training used free Kaggle GPUs."

---

## Slide 10 — Datasets and data pipeline

**On slide**
- **VoiceBank-DEMAND** (training + main test): noisy/clean speech pairs, 16 kHz
  - Locked model trained on **2,000** pairs; tested on **824** clips (unseen speakers and noise types)
- **MS-SNSD** (external test): **1,100** clips, different speakers/microphones, 9 noise types
  (air conditioner, babble, typing, vacuum, door slams, airport announcements, copy machine, …),
  SNR −5 to +10 dB
- Pipeline: resample to 16 kHz mono → manifest CSVs → 2-second random crops for training →
  full clips for testing

**Say:** "Using two corpora is what lets us say whether the model learned speech or just learned
VoiceBank."

---

# PART D — SYSTEM DESIGN AND METHOD

## Slide 11 — System architecture

**On slide (diagram)**
Noisy audio → Frame + window → STFT → Band features → **Learned model** → Per-band gain mask →
Apply to magnitude (keep phase) → ISTFT + overlap-add → **Controller (scene → mode → EQ /
compressor / limiter)** → Enhanced audio

**Visual:** paper Fig. 1 (`sona-final-month/report/main.tex`, TikZ "End-to-end real-time pipeline")
or explainer §5 diagram.

**Say:** "Two layers: Layer A removes noise; Layer B shapes the result for the situation."

---

## Slide 12 — How a computer sees sound (STFT front-end)

**On slide**
- 16 kHz audio → 512-sample (32 ms) Hann windows, 128-sample (8 ms) hop, 75 % overlap
- FFT per frame → 257 frequency bins → spectrogram
- 257 bins grouped into **32 Bark bands** (matches human hearing resolution)
- 3 features per band per frame: log energy, temporal delta, spectral flux
- Exact reconstruction under overlap-add (verified by unit test)

**Visual:** explainer §2 diagram (waveform → frames → spectrogram → 32 bands).

---

## Slide 13 — The core idea: a mask

**On slide**
- Model predicts a gain M(k,t) ∈ [0,1] for every band k and frame t
- Enhanced = ISTFT( M · |X| · e^{j∠X} ) — noisy phase reused
- 1 = keep (speech), 0 = remove (noise)
- Classical methods compute M from a formula; SONA **learns** M from data

**Visual:** explainer §3 diagram (noisy × mask = clean).

**Say:** "All the intelligence goes into predicting the mask; everything around it is exact maths."

---

## Slide 14 — Fair benchmark design

**On slide**
- One shared front-end, one 32-band mask output, one evaluation harness
- Only the gain estimator is swapped:
  - Classical: Wiener, spectral subtraction, soft threshold, hard threshold
  - Neural: GTCRN ★, complex GTCRN, causal Transformer, Mask GRU, Mask TCN
- ⇒ any score difference is attributable to the estimator alone

**Visual:** explainer §5 "swappable box" diagram.

---

## Slide 15 — GTCRN architecture

**On slide**
- Input: band features (3 × T × 32)
- Stem: 1×3 conv → 96 channels, GELU
- 4 gated time-frequency blocks: 3×3 conv, time dilation 1, 2, 4, 8 (causal), GLU gate
  `v ⊙ σ(g)`, group norm, residual
- Band-wise GRU (hidden 96): memory along time for each band
- Linear head + sigmoid → gain M(k,t)
- **722,209 parameters · 2.8 MB**

**Visual:** paper Fig. 2 (GTCRN backbone) or explainer §6 diagram.

**Say:** "Dilations double each block, so four small filters see about 30 frames — a quarter of a
second — of history."

---

## Slide 16 — Training and model selection

**On slide**
- Loss: **L = −SI-SDR + 1.25 · L_mag + 0.25 · L_multi-res** (STFT sizes 256/512/1024)
- AdamW, lr 3×10⁻⁴, weight decay 10⁻⁵, batch 8, 50 epochs, 2 s crops, Kaggle T4
- Selection path: pilot → **16-config sweep** (channels 64/96, hidden 96/128, blocks 4/5,
  magnitude weight 0.5/0.75) → loss-focus runs (adding multi-resolution term) → locked
  `ch96 · h96 · 4 blocks · mag 1.25 · mr 0.25`

**Say:** "The multi-resolution term catches artefacts that a single STFT size misses."

---

## Slide 17 — Variants tested (ablation design)

**On slide**
- Same backbone and front-end, only the head changes:
  (a) attention mask · (b) Mamba-GTCRN (state-space memory) · (c) governed dual-head ·
  (d) residual refiner · (e) complex-ratio mask
- Plus **guarded test-time (online) adaptation** — the project's original research idea

**Visual:** paper Fig. 3 (comparison architectures).

---

# PART E — RESULTS

## Slide 18 — Main result: learned vs classical (824 VoiceBank-DEMAND clips)

**On slide** (paper Table 1)

| Method | SI-SDRi ↑ | SNRi ↑ | STOI ↑ | ESTOI ↑ | LSD ↓ |
|---|---:|---:|---:|---:|---:|
| Noisy input | 0.00 | 0.00 | 0.921 | 0.787 | 10.50 |
| Hard threshold | −3.69 | −2.37 | 0.767 | 0.573 | 13.78 |
| Soft threshold | −5.82 | −4.16 | 0.728 | 0.532 | 15.28 |
| Spectral subtraction | −0.25 | 0.30 | 0.884 | 0.734 | 8.86 |
| Wiener filter | −1.93 | −1.11 | 0.824 | 0.677 | 10.87 |
| **GTCRN (ours)** | **+9.93** | **+5.41** | **0.930** | **0.832** | **6.47** |

**Say:** "This is the central result. Every classical filter makes the speech *worse* than doing
nothing — negative SI-SDR improvement. Our model gains almost 10 dB and improves intelligibility.
This held in all 5 noise types and all 4 SNR levels — no exceptions."

---

## Slide 19 — Comparing learned architectures

**On slide** (paper Table 2, 824 clips)

| Model | SI-SDRi | PESQ | STOI | ESTOI | LSD |
|---|---:|---:|---:|---:|---:|
| *Original (no processing)* | *0.00* | *1.97* | *0.921* | *0.787* | *10.50* |
| Attention mask | 9.08 | 2.50 | **0.932** | **0.834** | **5.97** |
| Mamba-GTCRN | 9.84 | 2.50 | 0.928 | 0.831 | 6.62 |
| Governed dual-head | 9.89 | 2.45 | 0.930 | 0.831 | 6.64 |
| Residual-refined | 9.87 | 2.52 | 0.929 | 0.831 | 6.51 |
| **GTCRN + MR loss** | **9.93** | **2.52** | 0.930 | 0.832 | 6.47 |

**Say:** "Adding capacity did not help — the plain GTCRN has the best SI-SDR and PESQ. The attention
model is slightly better on STOI and LSD, which is why we report a full metric set rather than one
number."

---

## Slide 20 — Generalisation: external validation on MS-SNSD

**On slide**

| Model | VoiceBank SI-SDRi | MS-SNSD SI-SDRi | MS-SNSD PESQ | MS-SNSD STOI | MS-SNSD ESTOI |
|---|---:|---:|---:|---:|---:|
| *Original (no processing)* | *0.00* | *0.00* | *—* | ***0.818*** | *0.668* |
| GTCRN | +9.93 | **+3.15** | **1.40** | 0.816 | **0.690** |
| Attention | +9.08 | +2.59 | 1.32 | 0.804 | 0.664 |

- Speech quality still improves on a corpus never seen in training (+3.15 dB); GTCRN generalises
  better than attention
- **But intelligibility does not improve on unfamiliar audio:** STOI stays at the original level
  (0.818 → 0.816); only the extended measure (ESTOI) rises slightly (0.668 → 0.690)
- → more varied training data is the clearest future work

**Say:** "Comparing against the original audio is what exposes this: the cleaned speech is cleaner,
but not easier to understand, on recordings unlike the training data. We report it rather than hide it."
*(Original MS-SNSD scores: computed over all 1,100 clips with the same metric code; PESQ was not
available locally.)*

**Visual:** `sona-final-month/figures/ms_snsd_external.png`

---

## Slide 21 — Positioning against published systems (perceptual quality)

**On slide** (paper Table 3)

| Method | Params | PESQ | STOI |
|---|---:|---:|---:|
| Noisy input | — | 1.97 | 0.921 |
| **Ours (GTCRN)** | **0.72 M** | 2.49 | 0.930 |
| DCCRN† | 3.7 M | 2.68 | — |
| DeepFilterNet2† | 2.3 M | 3.08 | — |
| MetricGAN+ | 1.9 M | 3.13 | 0.932 |

† reported in source paper; others measured in our harness. Our harness reproduces the standard
noisy (1.97) and MetricGAN+ (3.13) values → measurement is valid.

**Say:** "We are honest about this: our compact model is 3–5× smaller but below PESQ-optimised
state of the art. Closing that gap with perceptual training objectives is future work."
*(Note: 2.49 is the magnitude model in the 5k-clip A/B run; the locked model scores 2.52 on the
same 824 clips — see Slide 19.)*

---

## Slide 22 — Complex mask: a measured trade-off

**On slide** (paper Table 4, identical backbone, 824 clips, paired Wilcoxon)

| Metric | Magnitude | Complex | Δ | p |
|---|---:|---:|---:|---:|
| SI-SDRi | 9.73 | 10.04 | **+0.31** | < 10⁻⁶ |
| STOI | 0.929 | 0.931 | **+0.002** | < 10⁻⁶ |
| PESQ | 2.494 | 2.478 | **−0.016** | 0.005 |

**Say:** "Correcting phase improves waveform fidelity but slightly lowers perceptual score. An
SI-SDR-only evaluation would have reported a false win — this is why we measure several metrics."

**Visual:** `sona-final-month/figures_analysis/phase_analysis.png`

---

## Slide 23 — Online (test-time) adaptation

**On slide**
- Idea: refine a small low-rank adapter per clip at inference, maximising a reference-free
  quality estimate (TorchAudio-SQUIM) — no clean reference needed
- Unguarded: hurts quality (MS-SNSD PESQ 1.323 → 1.307)
- With trust region + strict output guard: safe, but gain ≈ 0
  (VoiceBank: SI-SDRi 9.082 → 9.079, PESQ 2.501 → 2.503)
- Conclusion: a safety-bounded refinement, not a large improvement; available in the app as an
  experimental switch that reports whether the guard accepted the update

**Visual:** `sona-final-month/figures/online_adaptation.png`

---

## Slide 24 — Real-time performance

**On slide**
- GPU workstation: **RTF ≈ 0.009** (1 s of audio in ~9 ms, > 100× real time)
- Single laptop CPU thread: **RTF ≈ 0.08–0.10** (~10× real time)
- Block streaming (COLA overlap-add): algorithmic latency = block length (32–256 ms options)
- Streaming each block independently costs quality (vacuum clip: 4.2 dB at 32 ms → 6.5 dB at
  256 ms vs 9.2 dB whole-file) → fixed in Phase 2 by stateful frame-by-frame inference

**Visual:** `sona-final-month/figures/rtf_comparison.png` + explainer §13 latency-vs-quality chart.

---

# PART F — FURTHER RESEARCH STUDIES

## Slide 25 — Model compression (quantisation)

**On slide**
- INT8 post-training quantisation: 2.84 MB → 0.95 MB
- Per-layer study found damage is **concentrated**: the first (stem) layer is **0.133 %** of weights
  but carries **98.7 %** of the INT8 quality loss
- "Stem-exempt" mixed precision: SI-SDR 18.375 dB vs FP32 18.377 dB (p = 0.25, no difference),
  smaller than uniform INT8 (916.9 vs 950.7 KB) and faster than it
- Replicated on the complex-mask model (stem = 92.1 % of damage)
- Relevance: directly informs the Phase 2 on-device model

---

## Slide 26 — Enhancement vs speech recognition

**On slide**
- Fed enhanced audio to Whisper ASR: corpus word error rate **3.45 % → 4.28 % (+24 % relative)**
- Replicated with a second recogniser (+23 %) and on MS-SNSD (+47 %)
- Lesson: audio tuned for human ears can remove cues a recogniser uses
- Remedy validated: give the recogniser the unprocessed signal, the listener the enhanced one

---

## Slide 27 — Scene classifier (with corrected evaluation)

**On slide**
- 7 scenes: speech, music, traffic, fan, crowd, mixed, unknown
- First version: hand-tuned heuristic (0.47 accuracy on real audio)
- Learned MLP with **1,716 parameters**
- Accuracy **0.748 ± 0.074** under grouped cross-validation (speakers/sessions never seen)
- (Per-clip CV had suggested 0.844 — we corrected it ourselves)

---

# PART G — APPLICATION AND DEMO

## Slide 28 — Layer B controller

**On slide**
- Scene classifier → picks one of 7 modes: speech, strong speech, music, podcast, gaming, hearing,
  studio
- Mode sets: denoise strength, safety floor, 8-band EQ, compressor, limiter
- Scene memory: save your preferred settings per scene; restored automatically
- Optional 50/60 Hz mains-hum notch filter

**Visual:** explainer §11 diagram.

---

## Slide 29 — The web application

**On slide**
- **Enhance:** sample library (10 real test clips with clean references) or file upload;
  optional settings panel with 9 engines, sound modes, hum filter, scene memory
- **Results:** instant A/B (original / enhanced / removed / clean), live visualiser,
  before/after spectrograms, quality scores vs the clean reference, session history
- **Compare:** every method on the same clip, ranked, each playable
- **Real-time:** block-streaming simulation with delay, per-block compute vs deadline, RTF
- **About:** model cards and every benchmark table
- Works on audio files; live microphone input is Phase 2 (hardware)
- Local only — audio never leaves the machine; one-click package for any laptop

**Visual:** 4 screenshots (Enhance, Results with spectrograms, Compare, Real-time) — take fresh ones
from `python web_demo/app.py` or reuse the ones in `SONA_Project_Explainer.html` §12.

---

## Slide 30 — Live demo (script)

**On slide:** "Live demo" + the URL `http://localhost:7860`

**Demo script (≈ 3 min):**
1. Enhance → Library → **Vacuum cleaner** → Enhance. The app opens Results.
2. Play, press **1 / 2 / 4** to switch original / enhanced / clean while playing.
3. Point at the before/after spectrograms: the noise floor disappears.
4. Headline numbers: **+9.2 dB speech quality gain**, intelligibility (STOI) 0.771 → 0.800.
5. Compare → Run comparison → all 9 methods ranked; play spectral subtraction vs GTCRN.

**Backup if the demo fails:** the explainer page §4 has the same clips embedded offline.

---

## Slide 31 — Software engineering and testing

**On slide**
- Modular codebase: `dsp/`, `baselines/`, `models/`, `train/`, `eval/`, `controller/`
- Unit tests guarantee: STFT→ISTFT exact reconstruction, band partition of unity, EQ unity =
  identity, every adapter is identity at reset, end-to-end enhancement improves a real test clip
  by > 3 dB
- Web API validated: upload checks, safe IDs, error handling; 29 automated tests
- Reproducible: configs in YAML, frozen result tables, regenerable sample library

---

# PART H — PUBLICATIONS

## Slide 32 — Conference paper

**On slide**
- **Title:** *Real-Time Adaptive Audio Noise Cancellation Using Learning-Based Frequency-Domain
  Processing on Edge Hardware*
- **Authors:** A. Rohit Daniel, Simhaa T. T., Vijayaraghavan A. — Guide: Dr. G. R. Venkatakrishnan
- **Format:** Springer LNCS
- **Conference:** `[FILL IN: conference name, year]`
- **Submitted:** `[FILL IN: date]` · **Paper ID:** `[FILL IN]` · **Status:** `[FILL IN: under
  review / accepted / presented]`
- **Contributions:**
  1. Fair shared-front-end benchmark: all classical filters degrade real noise; learned model +9.9 dB
  2. Compact 722K-parameter GTCRN with multi-resolution spectral objective
  3. Ablation of five architectural variants + guarded test-time adaptation
  4. External validation on MS-SNSD and a real-time interactive system

**Visual:** first page of the paper PDF (`sona-final-month/report/main.pdf`).

---

## Slide 33 — How the conference paper was prepared and submitted

**On slide (process)**
1. **Freeze results** — all numbers taken from frozen, significance-tested benchmark runs
   (`sona-final-month/FINAL_TABLES/`)
2. **Structure** — Introduction · Related work · Proposed method · Experimental setup · Results ·
   Ablation studies · System implementation · Conclusion
3. **Figures** — pipeline, GTCRN backbone and variant diagrams drawn in TikZ; web-app screenshot
4. **Tables** — main comparison (824 clips), learned architectures, PESQ vs published systems,
   complex-mask A/B with p-values
5. **Template** — Springer LNCS (`llncs` class), page limit `[FILL IN]`
6. **Internal review** — guide review, claims checked against result files
7. **Submission** — via `[FILL IN: portal, e.g. conference CMS]` on `[FILL IN: date]`
8. **Next** — `[FILL IN: reviews expected / camera-ready deadline / presentation date]`

**Say:** "Every number in the paper traces to a result file in our repository, so the paper can be
regenerated if any reviewer asks."

---

## Slide 34 — Journal work: "Where the Loss Lives"

**On slide**
- **Thesis:** damage from compression and evaluation in audio systems is **concentrated**, not
  spread evenly — and the aggregate metrics the field reports cannot see it
- **Study 1 — weight quantisation:** 0.133 % of weights carry 98.7 % of INT8 loss; perceptual
  and MSE-based sensitivity rankings agree (ρ = +1.000) — hypothesis refuted and reported
- **Study 2 — evaluation metrics:** enhancement raises ASR word error rate by 24 %; per-clip and
  corpus-level WER disagree; complex mask wins SI-SDR but loses PESQ
- **Study 3 — embedding compression:** at ~30× compression, Recall@10 keeps ~50 % of neighbours
  but task accuracy stays ~96 %; a label-free metric (1/Ratio@k) is ~13× better calibrated than
  recall
- **Control case:** classical baselines are uniformly worse than doing nothing — the one place
  the aggregate tells the whole story (makes the thesis falsifiable)
- **Contribution beyond numbers:** a validated remedy per failure + a 6-rule reporting protocol

**Visual:** the "aggregate says / disaggregation says" table:

| Instance | Aggregate says | Disaggregation says |
|---|---|---|
| Weight quantisation | spread bits by weight count | 0.133 % of weights carry 98.7 % of loss |
| File size | fewer bits ⇒ smaller file | quantising more made a larger file |
| Embedding compression | Recall@10 lost ~50 % | task accuracy lost ~4 % |
| Mask representation | SI-SDR: clean win | PESQ: significant loss |

---

## Slide 35 — Journal status and plan

**On slide**
- Format: IEEEtran (journal), 12–16 pages
- Two companion manuscripts already written and compiling:
  - *0.13 % of the Weights Carry 99 % of the Loss: Stem-Exempt INT8 Quantization of a Compact
    Speech Enhancer* (4 pp)
  - *Recall Is the Wrong Yardstick: Evaluating Data-Oblivious Vector Quantization on Audio
    Embeddings* (5 pp)
- All experiments complete and frozen; merged manuscript in preparation
- Target: **arXiv preprint → IEEE/ACM Transactions on Audio, Speech, and Language Processing
  (TASLP)** — peer-reviewed, no publication charge
- Pending before submission: final author list on manuscripts, merged draft, optional MOS
  listening test (tool built; deferred because effect sizes are below audibility)
- `[FILL IN: any update since — e.g. arXiv ID, submission date]`

---

# PART I — PHASE 2: HARDWARE

## Slide 36 — Phase 2 goal and architecture

**On slide**
- Goal: a standalone device — speak into a mic, hear clean speech in headphones, live
- Phase 1 works on audio files; **all microphone input (recording and live) moves to Phase 2**
- Microphone → audio codec (ADC, I²S/USB) → **embedded board** (capture ring buffer → streaming
  GTCRN, one 8 ms frame at a time with state kept → controller + overlap-add → playback) → codec
  DAC → headphones
- Phone connects over Wi-Fi to the board's web UI for modes and strength; battery powered

| Component | Role | What matters |
|---|---|---|
| Embedded board (e.g. Raspberry Pi 5) | model + controller + audio loop | 4-core ARM, RTF < 0.5 |
| Audio codec / USB interface | analog ↔ digital | low latency, 16 kHz |
| Microphone | capture | close-talk MEMS or lapel |
| Headphones | output | closed-back (no feedback into mic) |
| Power bank | portability | ≥ 3 h under load |
| Enclosure, button, LED | usability | mode switching, status |

**Visual:** explainer §14 hardware diagram.

---

## Slide 37 — Phase 2 milestones and targets

**On slide**

| Milestone | Work |
|---|---|
| H1 Bring-up | Board + codec + mic + headphones; measure raw mic-to-ear latency (loopback) |
| H2 Model on board | ONNX export, ARM benchmark at 1–4 threads, stem-exempt mixed precision |
| H3 Frame-by-frame streaming | Cache conv history + GRU state, per-frame normalisation, fine-tune → streamed quality ≈ whole-file |
| H4 Device & controls | Real-time audio loop with overrun monitoring, on-board controller, button, Wi-Fi UI |
| H5 Evaluation | Latency, CPU, temperature, battery, dropouts, real-room recordings, listening test |

**Targets:** mic-to-ear latency < 50 ms · RTF < 0.5 on board · 0 dropouts in 30 min ·
streamed SI-SDRi ≈ +9.9 dB (whole-file quality) · ≥ 3 h battery.

`[FILL IN: Phase 2 schedule — month per milestone]`

**Say:** "H3 is the key technical step: today each streaming block is processed from scratch, which
costs 3–5 dB. Carrying the model's memory between 8 ms frames removes that loss."

---

# PART J — CLOSING

## Slide 38 — Limitations and what we learned

**On slide**
- PESQ below large state-of-the-art models (2.52 vs 3.1) — compact model trade-off
- Large drop on unfamiliar audio (+9.9 → +3.2 dB on MS-SNSD) and no intelligibility gain there
  (STOI 0.818 original vs 0.816); weak on impulsive noise (typing)
- No subjective listening test yet (tool built)
- Not yet measured on embedded hardware → Phase 2
- Lessons: simple models can win; one metric is never enough; test on a second corpus; correct your
  own claims early

---

## Slide 39 — Conclusion

**On slide**
- Classical filters make real noisy speech worse; a 722K-parameter learned filter gains **+9.9 dB**
  SI-SDR and improves intelligibility
- Fair, shared-front-end methodology; honest negative and trade-off results
- Complete software system: controller, real-time engine, web application, portable package
- One conference paper `[FILL IN: status]` and journal work in preparation
- Phase 2: the same model, live, on an embedded device

---

## Slide 40 — Thank you / Questions

**On slide:** Thank you · team names · guide · contact emails
(`rohit2310086@ssn.edu.in`, `simhaa2310510@ssn.edu.in`, `vijayaraghavan2310125@ssn.edu.in`)

---

# BACKUP SLIDES

## B1 — Metric definitions

| Metric | Meaning | Range / direction |
|---|---|---|
| SI-SDR(i) | closeness to clean speech, scale-invariant (improvement over input) | dB, higher better |
| SNR(i) | signal-to-noise ratio (improvement) | dB, higher better |
| PESQ | ITU-T P.862 perceptual quality | 1.0–4.5, higher better |
| STOI / ESTOI | intelligibility | 0–1, higher better |
| LSD | log-spectral distance | dB, lower better |
| RTF | processing time ÷ audio duration | < 1 = faster than real time |

## B2 — Full hyperparameters (locked model)

16 kHz · STFT 512 / hop 128 / Hann · 32 Bark bands · 3 features · stem 96 ch · 4 gated blocks
(3×3, dilation 1/2/4/8, dropout 0.1) · GRU hidden 96 · AdamW lr 3e-4, wd 1e-5 · batch 8 ·
50 epochs · 2 s crops · loss −SI-SDR + 1.25 mag + 0.25 multi-res · 2,000 training pairs ·
seed 1234 · 722,209 parameters

## B3 — Statistical testing

Paired Wilcoxon signed-rank on per-clip scores; 10,000-sample bootstrap 95 % CIs;
Benjamini–Hochberg FDR correction for per-condition breakdowns; effect sizes reported beside
p-values.

## B4 — Classical baselines by noise type and SNR

Worse than doing nothing in all 5 noise types × 4 SNR levels.
Source: `results/breakdowns/` and `sona-final-month/SIGNIFICANCE/significance.md`.

## B5 — Quantisation details

ONNX Runtime QDQ, 6 quantisable layers, 64 configurations searched on a speaker-disjoint dev
split; allocation frozen before test evaluation. Mixed: 29 QDQ nodes vs uniform 172 → smaller and
faster. Source: `PERCEPTUAL_MP/FINDINGS.md`.

## B6 — Web application API

`/api/engines`, `/api/samples`, `/api/sample`, `/api/upload`, `/api/reference`, `/api/classify`,
`/api/enhance`, `/api/compare`, `/api/stream`, `/api/research`, audio and download routes,
`/api/preferences`. Source: `web_demo/README.md`.

---

# LIKELY QUESTIONS AND ANSWERS

| Question | Answer |
|---|---|
| Why do classical methods score negative? | They decide from amplitude alone under a stationary-noise assumption; on real non-stationary noise they remove weak speech and distort it below the untouched input. Held in every noise type and SNR. |
| Why GTCRN and not a bigger model? | Best SI-SDRi of everything we tested, 722K params; heavier variants (Mamba, governed, residual) did not beat it. Size matters for Phase 2. |
| Your PESQ is lower than DeepFilterNet2 / MetricGAN+. | Yes — they are 3–5× larger and/or optimise PESQ directly. We position honestly as efficient, not state-of-the-art PESQ; perceptual-objective training is future work. |
| Is it really real time? | RTF ≈ 0.009 on GPU, ≈ 0.08–0.10 on one CPU thread — both far below 1. Latency is set by block size; Phase 2 targets < 50 ms mic-to-ear. |
| Why does quality drop when streaming? | Each block is enhanced independently today, so the model loses its memory at block boundaries. Phase 2 milestone H3 keeps state across 8 ms frames. |
| How do you know it generalises? | External MS-SNSD corpus: speech quality still improves (+3.15 dB, better than attention), but intelligibility stays at the original level (STOI 0.818 → 0.816). That is the honest limit, and the reason more varied training data is our first future-work item. |
| Did online adaptation work? | Unguarded it hurt; guarded it is safe but ≈ 0 gain. We report it as a negative result — and the app shows the guard's decision live. |
| Are the differences statistically significant? | Paired Wilcoxon + bootstrap CIs on 824 clips; e.g. complex vs magnitude p < 10⁻⁶ (SI-SDR) and p = 0.005 (PESQ). |
| Why no listening (MOS) test? | Tool built and validated; the effects between our top variants are smaller than 20 raters can resolve, so it is deferred to the journal version. |
| What is new in your work? | A fair shared-front-end benchmark, honest negative/trade-off results, the stem-exemption quantisation finding, and a complete real-time system. |
| Which board for Phase 2 and why? | A 4-core ARM board such as Raspberry Pi 5: our CPU RTF leaves large headroom; the model is 2.8 MB (≈ 0.92 MB mixed-precision). |

---

# ASSET INVENTORY (where to find every figure)

| Asset | Path |
|---|---|
| Pipeline / GTCRN / variants diagrams (TikZ) | `sona-final-month/report/main.tex` (compiled in `main.pdf`) |
| Conference paper PDF | `sona-final-month/report/main.pdf`, `FYP_Technical_Report.pdf` |
| VoiceBank metrics chart | `sona-final-month/figures/voicebank_metrics.png` |
| MS-SNSD external chart | `sona-final-month/figures/ms_snsd_external.png` |
| Online adaptation chart | `sona-final-month/figures/online_adaptation.png` |
| RTF comparison | `sona-final-month/figures/rtf_comparison.png` |
| Architecture progression | `sona-final-month/figures/architecture_progression.png` |
| Spectrogram comparison / error spectrogram / phase analysis | `sona-final-month/figures_analysis/` |
| Web app screenshot (paper) | `sona-final-month/report/web_demo.png` |
| Explainer page (diagrams, audio, charts) | `SONA_Project_Explainer.html` |
| Final result tables (CSV/MD) | `sona-final-month/FINAL_TABLES/` |
| Significance tests | `sona-final-month/SIGNIFICANCE/significance.md` |
| Journal bundle | `paper_bundle/00_README.md` … `05_VENUE.md` |
| Companion manuscripts | `PERCEPTUAL_MP/paper/stem_exempt_quantization.pdf`, `AudioVQ/paper/recall_wrong_yardstick.pdf` |
| Live demo | `python web_demo/app.py` → `http://localhost:7860` |
