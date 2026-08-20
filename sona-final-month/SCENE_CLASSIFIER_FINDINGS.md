# Scene Classifier — Real-Audio Evaluation & Learned Upgrade

Layer B item (plan Week 2 / SONA_STATUS open item 7): score the scene classifier
on real audio and fix it if the labels are embarrassing. They were. This is the
full evidence trail, including the one place a learned model was justified.

## 1. Real-audio evaluation (was: synthetic self-test only)

Built a labelled real-audio set from MS-SNSD, whose filenames encode the true
noise type (`scripts/build_scene_manifest.py`). Only categories with an
unambiguous SONA scene label were scored:

| scene   | MS-SNSD sources                                   | clips |
|---------|---------------------------------------------------|------:|
| fan     | AirConditioner, VacuumCleaner, WasherDryer        | 38    |
| crowd   | Babble, Cafeteria, Restaurant, …                  | 25    |
| speech  | clean_test/clnsp*.wav                             | 30    |
| traffic | Traffic, Car, Bus, Metro, Station                 | 5     |

`music` has no MS-SNSD source — reported as a coverage gap, not counted as a miss.

**Shipped heuristic on real pure-source audio: 0.469 accuracy** (vs 0.767 on the
old synthetic self-test). Two hand-tuned windows were measurably *inverted*:
- `crowd` rewarded `high_ratio ∈ [0.12, 0.55]`, but real babble measures ~0.03.
- `speech` carried a `−0.5·high` penalty, though real speech has the *highest*
  HF energy (~0.14) of any class.

## 2. Threshold retune (heuristic, no learning)

The classifier is a deterministic hand-tuned scorer with no learned weights, so
fine-tuning / distillation do not apply — the lever is the `_tri()` windows.
Re-seated them on measured per-class quantiles and refined by balanced-accuracy
coordinate search under stratified 5-fold CV (`scripts/tune_scene_thresholds.py`).

| set                                   | accuracy (CV)     |
|---------------------------------------|-------------------|
| shipped heuristic                     | 0.469             |
| data-seeded windows (no search)       | 0.694             |
| tuned heuristic (pure sources)        | **0.714 ± 0.025** |

Most of the gain is from correct seeding, not fitting — so it is not overfit.
Applied to `src/controller/scene.py`: crowd recall 0.04→0.76, fan 0.29→0.66.

## 3. The train/serve mismatch (why pure-source tuning was not enough)

Production input is *noisy speech*, not pure noise or clean speech. Evaluated on
320 speech+noise mixtures across 0–15 dB SNR (`scripts/build_scene_mixtures.py`):

- tuned heuristic on mixtures (all SNR):        **0.409**
- tuned heuristic on noise-dominant (≤5 dB):    **0.490 ± 0.072 CV**

At high SNR the speech masks the noise character, so the noise-environment label
becomes genuinely ambiguous — an inherent ceiling for 8 spectral features under a
hand-written rule. The heuristic barely beats the 0.422 majority-class baseline.

## 4. Where a learned model IS justified

Holding the *same 8 features* fixed and swapping only the decision rule, under an
identical 5-fold protocol (`scripts/compare_scene_learned.py`), on the
noise-dominant mixtures:

| decision rule        | accuracy (CV)     | balanced |
|----------------------|-------------------|----------|
| majority baseline    | 0.422             | 0.250    |
| **tuned heuristic**  | **0.490**         | 0.539    |
| logistic regression  | 0.717             | 0.656    |
| random forest (200)  | 0.800             | 0.728    |
| **MLP (32,16)**      | **0.844 ± 0.052** | 0.804    |

**The decision rule, not the feature set, was the bottleneck (+0.35).** This is
the one place in the SONA pipeline where replacing a hand-tuned rule with a
learned one is evidence-backed rather than assumed.

## 5. Honest re-evaluation: grouped CV + richer features (supersedes §4)

The 0.844 in §4 used **per-clip** StratifiedKFold. With only ~5 traffic and a
handful of fan/crowd noise textures, mixtures of the *same* texture can land in
both train and test — the model can memorise a texture rather than learn the
scene. The honest protocol groups every split by **noise source file**
(`StratifiedGroupKFold`), so no texture is ever on both sides.

Under that protocol, and comparing the shipped 8 features against a **richer
34-feature set** (temporal spread + deltas + finer spectral bands, see
`scripts/scene_feature_lib.py`), searched over 10 model families
(`scripts/scene_search.py`) on 611 clips / 108 source groups:

| features | best model | honest grouped-CV acc | optimistic per-clip |
|----------|-----------|----------------------:|--------------------:|
| base 8   | random forest | 0.748             | 0.796               |
| **rich 34** | **MLP(32,16)** | **0.773**       | 0.895               |

Averaged over **5 seeds × 5 folds** for stability (`finalize_scene_model.py`),
the finalists settle to:

| model            | honest acc        | balanced |
|------------------|-------------------|----------|
| MLP(32,16)       | **0.748 ± 0.074** | 0.705    |
| logreg (balanced)| 0.708             | 0.753    |
| svm-rbf          | 0.742             | 0.727    |
| logreg+MLP vote  | 0.754             | 0.729    |

**The honest, texture-leakage-free, multi-seed accuracy is ~0.75** — not the
0.813 reported in §4, which was optimistic (per-clip, single seed, smaller set).
The correction is the point: grouped CV is what a reviewer would demand.

## 6. Shipped model

The MLP(32,16) on the 34 rich features (within 1 std of the best ensemble, and
the only finalist with a clean plain-numpy export) is shipped as **1,716
parameters** (`finalize_scene_model.py` → `src/controller/scene_model.py`):

- **Honest grouped-CV accuracy: 0.748 ± 0.074** (5 seeds × 5 folds, 611 clips,
  108 source groups). The product `SceneClassifier.features` computes the exact
  34 features the model was trained on (verified bit-identical to the training
  extractor, max diff 0.0).
- Inference needs no scikit-learn and no torch — stays local and edge-deployable.
- `SceneClassifier` uses it by default and **falls back to the tuned heuristic**
  if `scene_model.py` is absent (`use_learned=False`). Both paths are unit-tested.
- GTCRN denoising itself (Layer A, locked) was re-verified objectively:
  +5.8–6.3 dB SI-SDRi on AirConditioner, +3.4–4.1 dB on Babble, +0.2 dB on
  Typing (impulsive noise is an inherent spectral-masking limitation).

## Reproduce

```
python scripts/build_scene_manifest.py
python scripts/build_scene_mixtures.py
python scripts/eval_scene_classifier.py --manifest recordings/manifest_msnsd.csv --out-dir SCENE_EVAL_REAL
python scripts/tune_scene_thresholds.py --manifest recordings/manifest_mixtures_lowsnr.csv --auto-seed
python scripts/compare_scene_learned.py --cache SCENE_EVAL_LOWSNR/feature_cache.json --heuristic-cv 0.49
python scripts/train_scene_mlp.py
```

## Honest limitations (for the thesis)

- Ground truth is the MS-SNSD noise *category*, not human scene judgement.
- `music` is untested end-to-end (no source in MS-SNSD).
- `traffic` support is thin (5 pure clips, 40 mixtures) — its per-class numbers
  are the least reliable.
- Mixture SNRs use synthetic mixing at the same SNR MS-SNSD itself uses; real
  rooms add reverberation not modelled here.
