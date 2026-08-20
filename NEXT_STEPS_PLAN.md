# SONA — Final-Month Plan (2026-07-20 → submission)

> **EXPIRED — kept for provenance.** This plan covered Jul 20 → Aug 16 2026 and that window has
> closed. Most of it was executed and lives in `sona-final-month/`; the parts that were cut
> (Raspberry Pi, RL governor) stayed cut, and the ONNX export it deferred was done after all.
> For the current state of the work read `PROJECTS_STATUS.md`.
> Note the path advice in Week 1 item 4 is inverted relative to today: the tree was consolidated
> back onto `D:
oise cancellation` on 2026-08-20.

Constraints this plan is built around:

- ~1 month left. Priorities in order: thesis/report quality, journal paper, working live demo.
- Hardware: laptop/PC mic only. Raspberry Pi / TFLite edge work is CUT from this cycle
  (moved to "future work" in the thesis).
- Layer A is LOCKED: `gtcrn_ch96_h96_b4_mw1p25_mr0p25` (GTCRN + multi-resolution spectral
  loss), externally validated on MS-SNSD. No new architectures. No more Kaggle training runs
  unless a result-verification rerun is strictly needed.
- `vbd16k.zip` = VoiceBank-DEMAND 16 kHz training dataset (already used on Kaggle). Keep it
  as the archival copy of the training data; no action needed.

Guiding rule for the month: **stop generating results, start consolidating and packaging
them.** Every remaining task must end up either in the thesis, the paper, or the viva demo.

---

## Week 1 (Jul 20–26): Freeze scope + consolidate results

Goal: one authoritative results package that the thesis, paper, and viva all draw from.

1. **Master results table script** — write `scripts/build_final_tables.py` that walks
   `results/*/extracted/summary.csv` (and the non-extracted `summary.csv` runs) and emits:
   - `results/FINAL_TABLES/voicebank_full824.csv|md` — passthrough/classical → attention →
     GTCRN → Mamba/SONA-G/SONA-R, all full-824 rows.
   - `results/FINAL_TABLES/ms_snsd_external.csv|md` — attention vs GTCRN, 1100 clips.
   - `results/FINAL_TABLES/online_adaptation_story.csv|md` — frozen vs trust-region vs
     strict-guard rows on both datasets.
2. **Statistical significance** — the `per_sample.csv` files already exist for the key runs.
   Add `scripts/significance.py`: paired Wilcoxon signed-rank + 95% bootstrap CIs on
   per-clip SI-SDRi/PESQ/STOI for (a) GTCRN vs attention, (b) GTCRN vs Mamba/SONA-G/SONA-R,
   (c) frozen vs strict-guarded online. Examiners and reviewers both ask for this; it is
   cheap and uses only local CPU.
3. **Figures** — matplotlib scripts for: metric bar charts (VoiceBank + MS-SNSD), the
   architecture-search progression, spectrogram before/after examples (use the web-demo
   sample + 2–3 MS-SNSD clips at different SNRs), and an RTF/latency comparison.
4. **Repo hygiene** — fix the `D:\noise cancellation` → `E:\Rohit\FYP\noise cancellation`
   path drift in `SONA_STATUS.md`, `web_demo` docs, and any hardcoded script paths; run the
   full local pytest suite (43 + 17) and record the pass count; back up `results/` and the
   locked checkpoint somewhere off-machine (Drive/OneDrive).

Gate at end of week 1: final tables + significance numbers exist and are frozen. After this
point no result may change.

## Week 2 (Jul 27–Aug 2): Thesis core chapters + realtime demo build

Thesis track (majority of the week):

1. **Methodology chapter** — the strongest asset is the fair-benchmark design: shared STFT
   front-end, shared per-band gain space, shared eval harness. Write this first.
   Then: GTCRN architecture, multi-resolution spectral loss, the OnlineAdapter with
   trust-region + guards, Layer B controller design.
2. **Results chapter** — drop in the Week-1 tables/figures. Structure the narrative honestly
   and it becomes a strength, not a weakness:
   - Classical → neural: large gains (first result).
   - Backbone search: GTCRN + MR loss wins VoiceBank AND external MS-SNSD; heavier variants
     (Mamba-GTCRN, SONA-G, SONA-R) all fail the full-824 gate — a clean negative-results
     ablation examiners like.
   - Online adaptation: unguarded adaptation is dataset-sensitive and degrades externally;
     the strict guard makes it provably non-degrading but with ~zero effect size. Frame as
     "a safety governor for test-time adaptation" — a real finding, not a failure.
3. **Discussion/limitations** — effect-size honesty, scene classifier is heuristic, edge
   deployment deferred, RL governor deferred.

Demo track (parallel, timeboxed to ~2 days):

4. **Realtime mic pipeline** — `scripts/realtime_demo.py` using `sounddevice` duplex
   streams; reuse `web_demo/engine.py` as the session wrapper. The model is causal and RTF
   is ~0.006–0.06 on CPU, so block-based streaming (e.g., 32 ms hop, overlap-add state kept
   across blocks) is feasible. Measure and print end-to-end latency and RTF live.
   **Fallback:** the offline web demo already works and is demo-worthy on its own — if
   realtime is not stable within the timebox, ship the web demo + a pre-recorded realtime
   screen capture and move on. Do not let this eat thesis time.
5. **Scene-classifier sanity pass** (open item 7 from SONA_STATUS) — run
   `scripts/classify_scene.py` on ~10 real recordings (your voice, music, fan, street);
   tune thresholds only if labels are embarrassing; record the confusion notes for the
   thesis Layer B section.

## Week 3 (Aug 3–9): Finish thesis + paper draft

1. **Thesis remaining chapters** — introduction, literature review (RNNoise, DeepFilterNet,
   GTCRN original, test-time adaptation, non-intrusive quality estimation/SQUIM),
   system/implementation chapter (Layer B controller, web demo with screenshots, test
   suite), conclusion + future work (RL governor, Mamba retry with more data, edge/TFLite,
   preference learning).
2. **Full thesis assembly + first proofread pass.** Send to supervisor for feedback by end
   of week — this is the deadline that matters; everything else flexes around it.
3. **Paper draft (parallel, from thesis material)** — 6–8 page conference-style draft
   (ICASSP/Interspeech format is the right shape even if targeting a journal later):
   - Story: compact Bark-band GTCRN + multi-resolution spectral loss, benchmarked on an
     equal-footing harness, externally validated on MS-SNSD; plus the guarded test-time
     adaptation analysis showing why unguarded TTA fails under dataset shift.
   - The negative TTA result + safety-guard framing is the novelty hook; small-model
     external validation is the credibility hook.

## Week 4 (Aug 10–16): Polish, viva prep, buffer

1. Incorporate supervisor feedback into the thesis; final formatting, references, appendix
   (repro instructions, config listings, test output).
2. **Viva/demo rehearsal** — scripted 5-minute demo: bundled noisy sample → enhance → A/B →
   scene memory → (if built) live mic. Prepare offline backups of every audio example in
   case of stage fright hardware.
3. Slide deck for the presentation (reuse Week-1 figures).
4. Finish the paper draft to "supervisor-reviewable" state; submission itself can happen
   after the FYP deadline.
5. Buffer: nothing new starts in this week.

---

## Explicitly cut / deferred (record in thesis as future work)

- Raspberry Pi / ONNX / TFLite export (no hardware, no time).
- RL update/no-update governor (Layer B roadmap item; action space only just stabilized).
- Retraining or new architecture variants of any kind.
- Provisional patent draft — carry the claim sketch in the design doc forward, but drafting
  it competes directly with the thesis; revisit after submission.

## Risks

- Realtime demo instability → hard 2-day timebox + offline web demo fallback (already
  tested, 17 passing tests).
- Significance tests contradict a headline claim → run them in Week 1, not Week 3, so the
  narrative can adapt.
- Path drift / machine failure → Week-1 backup of `results/`, checkpoints, and `vbd16k.zip`.
- Supervisor turnaround → thesis draft out by end of Week 3, not Week 4.
