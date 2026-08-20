#!/usr/bin/env python3
"""Re-seat the scene classifier's `_tri()` windows on real feature distributions.

Plan item (Week 2, item 5): "tune thresholds only if labels are embarrassing."
On the 98-clip MS-SNSD real-audio set the shipped heuristic scores 0.469, so a
retune is warranted. The classifier is a deterministic hand-tuned scorer with no
learned weights (`src/controller/scene.py`), so fine-tuning / distillation do not
apply — the correct lever is the threshold windows themselves.

Method
------
* Features are extracted once and cached (audio decode is the slow part).
* The scorer is re-implemented here with the SAME structure as `scene.py`, but
  with its constants exposed as a parameter vector.
* Optimisation is randomised coordinate search on **balanced accuracy**
  (macro-recall), not raw accuracy, so the 5-clip `traffic` class cannot be
  ignored in favour of the 38-clip `fan` class.
* Generalisation is reported by **stratified k-fold cross-validation**: each fold
  tunes on k-1 folds and scores the held-out fold, so the headline number is not
  fitted on the data it is measured on. A full-data fit is also emitted, for
  pasting into `scene.py`, with its (optimistic) in-sample score clearly labelled.

    python scripts/tune_scene_thresholds.py --manifest recordings/manifest_msnsd.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

FEATURES = ["centroid_hz", "flatness", "rolloff_hz", "zcr",
            "low_ratio", "mid_ratio", "high_ratio", "modulation"]

# Parameter vector: (weight, lo, hi) per term, grouped by label. Seeded from the
# measured medians in SCENE_EVAL_REAL/feature_stats.json rather than the shipped
# constants, since several shipped windows sit off the real distribution.
SEED: dict[str, list] = {
    "speech": [
        ["mid_ratio", 1.0, 0.30, 0.72],
        ["modulation", 1.3, 0.55, 1.30],
        ["centroid_hz", 1.2, 700.0, 1800.0],
        ["zcr", 0.6, 0.09, 0.22],
        ["high_ratio", 1.2, 0.06, 0.24],
    ],
    "music": [
        ["low_ratio", 0.8, 0.15, 0.55],
        ["rolloff_hz", 0.8, 1200.0, 4500.0],
        ["modulation", 0.7, 0.15, 0.60],
        ["high_ratio", 0.5, 0.05, 0.30],
    ],
    "fan": [
        ["modulation", 1.8, 0.00, 0.42],
        ["low_ratio", 1.0, 0.32, 0.78],
        ["centroid_hz", 0.9, 200.0, 620.0],
        ["mid_ratio", 0.7, 0.16, 0.48],
    ],
    "traffic": [
        ["centroid_hz", 1.4, 60.0, 330.0],
        ["mid_ratio", 1.3, 0.00, 0.24],
        ["low_ratio", 0.9, 0.35, 0.80],
        ["modulation", 0.6, 0.35, 0.95],
    ],
    "crowd": [
        ["mid_ratio", 1.5, 0.56, 0.88],
        ["high_ratio", 1.0, 0.00, 0.085],
        ["modulation", 0.9, 0.32, 0.80],
        ["centroid_hz", 0.8, 430.0, 900.0],
    ],
}
FLOOR = {"mixed": 0.62, "unknown": 0.10}
LABELS = ["speech", "music", "traffic", "fan", "crowd", "mixed", "unknown"]


def auto_seed(rows, n_terms: int = 4, labels: tuple = ("speech", "traffic", "fan", "crowd")):
    """Derive initial `_tri` windows from per-class quantiles.

    For each label the most *discriminative* features are chosen — those whose
    class median sits furthest from the pooled median relative to the pooled
    spread — and the window is seeded to that class's [p25, p75]. This keeps the
    seed principled (it describes where the class actually lives) instead of
    hand-guessed, and lets the same tuner re-seat itself on any manifest.
    """
    by: dict[str, list] = {}
    for feats, lab in rows:
        by.setdefault(lab, []).append(feats)
    pooled = {k: np.array([f[k] for f, _ in rows], dtype=float) for k in FEATURES}

    seed: dict[str, list] = {}
    for lab in labels:
        if lab not in by:
            continue
        ranked = []
        for k in FEATURES:
            v = np.array([f[k] for f in by[lab]], dtype=float)
            spread = float(np.percentile(pooled[k], 75) - np.percentile(pooled[k], 25)) or 1.0
            sep = abs(float(np.median(v)) - float(np.median(pooled[k]))) / spread
            # p10..p90 (not p25..p75): the triangular kernel zeroes anything
            # outside the window, so a 50% window discards half the class
            ranked.append((sep, k, float(np.percentile(v, 10)), float(np.percentile(v, 90))))
        ranked.sort(reverse=True)
        terms = []
        for sep, k, p25, p75 in ranked[:n_terms]:
            if p75 <= p25:  # degenerate (constant feature) — widen slightly
                p75 = p25 + max(abs(p25) * 0.1, 1e-3)
            terms.append([k, round(1.0 + min(sep, 1.5), 2), max(0.0, p25), p75])
        seed[lab] = terms
    return seed


def _tri(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    c, h = (lo + hi) * 0.5, (hi - lo) * 0.5
    return max(0.0, 1.0 - abs(float(x) - c) / h)


def score(feats: dict, params: dict) -> str:
    scores = dict(FLOOR)
    for label, terms in params.items():
        scores[label] = sum(w * _tri(feats[k], lo, hi) for k, w, lo, hi in terms)
    return max(scores.items(), key=lambda kv: kv[1])[0]


def balanced_accuracy(rows, params) -> float:
    per: dict[str, list] = {}
    for feats, true in rows:
        per.setdefault(true, []).append(int(score(feats, params) == true))
    if not per:
        return 0.0
    return float(np.mean([np.mean(v) for v in per.values()]))


def accuracy(rows, params) -> float:
    if not rows:
        return 0.0
    return float(np.mean([int(score(f, params) == t_) for f, t_ in rows]))


def clone(params):
    return {k: [list(t) for t in v] for k, v in params.items()}


def optimise(rows, seed_params, iters: int, rng: random.Random):
    best = clone(seed_params)
    best_score = balanced_accuracy(rows, best)
    for _ in range(iters):
        cand = clone(best)
        label = rng.choice(list(cand))
        term = rng.randrange(len(cand[label]))
        key, w, lo, hi = cand[label][term]
        span = max(hi - lo, 1e-6)
        which = rng.random()
        if which < 0.34:
            w = max(0.0, w * rng.uniform(0.75, 1.3))
        elif which < 0.67:
            lo = lo + rng.uniform(-0.25, 0.25) * span
        else:
            hi = hi + rng.uniform(-0.25, 0.25) * span
        # every feature here (ratios, zcr, modulation, flatness, Hz) is
        # non-negative, so a negative window bound is meaningless
        lo = max(0.0, lo)
        if hi <= lo:
            continue
        cand[label][term] = [key, w, lo, hi]
        s = balanced_accuracy(rows, cand)
        if s > best_score:
            best, best_score = cand, s
    return best, best_score


def stratified_folds(rows, k: int, rng: random.Random):
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    folds: list[list] = [[] for _ in range(k)]
    for label, items in by.items():
        items = items[:]
        rng.shuffle(items)
        for i, it in enumerate(items):
            folds[i % k].append(it)
    return folds


def load_rows(manifest: Path, repo: Path, cache: Path):
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [(d["feats"], d["label"]) for d in data]
    sys.path.insert(0, str(repo.resolve()))
    from src.controller import SceneClassifier  # type: ignore
    from src.utils.audio_io import load_audio, DEFAULT_SR  # type: ignore

    clf = SceneClassifier(sr=DEFAULT_SR)
    out, rows = [], []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            wav, _ = load_audio(row["path"], sr=DEFAULT_SR, mono=True)
            feats = clf.features(np.asarray(wav, dtype=np.float32))
            feats = {k: float(v) for k, v in feats.items()}
            out.append({"feats": feats, "label": row["label"]})
            rows.append((feats, row["label"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return rows


def emit_code(params) -> str:
    lines = []
    for label, terms in params.items():
        parts = []
        for key, w, lo, hi in terms:
            fmt = "%.1f" if abs(lo) > 20 or abs(hi) > 20 else "%.3f"
            parts.append(f"{w:.2f} * _tri({key_map(key)}, {fmt % lo}, {fmt % hi})")
        lines.append(f"        {label} = (\n            " +
                     "\n            + ".join(parts) + "\n        )")
    return "\n".join(lines)


def key_map(k: str) -> str:
    return {"centroid_hz": "centroid", "rolloff_hz": "rolloff",
            "low_ratio": "low", "mid_ratio": "mid", "high_ratio": "high",
            "modulation": "mod", "flatness": "flatness", "zcr": "zcr"}[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "SCENE_EVAL_REAL")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-seed", action="store_true",
                    help="derive seed windows from per-class quantiles of THIS "
                         "manifest instead of the hard-coded SEED")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows = load_rows(args.manifest, args.repo, args.out_dir / "feature_cache.json")
    print(f"loaded {len(rows)} clips")

    base = auto_seed(rows) if args.auto_seed else SEED
    if args.auto_seed:
        print("auto-seeded windows from per-class quantiles:")
        for lab, terms in base.items():
            print(f"  {lab:8s} " + ", ".join(f"{k}[{lo:.3g},{hi:.3g}]"
                                             for k, _w, lo, hi in terms))

    base_bal = balanced_accuracy(rows, base)
    print(f"seed (pre-search) balanced acc: {base_bal:.4f}  acc: {accuracy(rows, base):.4f}")

    # ---- honest generalisation: stratified k-fold CV ----
    # NOTE: with --auto-seed the seed is derived per-fold from TRAIN only, so the
    # held-out fold never informs the windows it is scored against.
    folds = stratified_folds(rows, args.folds, rng)
    cv_acc, cv_bal = [], []
    for i in range(args.folds):
        test = folds[i]
        train = [r for j, f in enumerate(folds) if j != i for r in f]
        fold_seed = auto_seed(train) if args.auto_seed else SEED
        tuned, _ = optimise(train, fold_seed, args.iters, random.Random(args.seed + i))
        cv_acc.append(accuracy(test, tuned))
        cv_bal.append(balanced_accuracy(test, tuned))
        print(f"  fold {i+1}/{args.folds}: held-out acc {cv_acc[-1]:.4f}  bal {cv_bal[-1]:.4f}")

    print(f"\nCROSS-VALIDATED (honest) accuracy: {np.mean(cv_acc):.4f} "
          f"+/- {np.std(cv_acc):.4f}")
    print(f"CROSS-VALIDATED balanced accuracy: {np.mean(cv_bal):.4f} "
          f"+/- {np.std(cv_bal):.4f}")

    # ---- full-data fit, for shipping into scene.py ----
    final, final_bal = optimise(rows, base, args.iters * 2, random.Random(args.seed + 99))
    print(f"\nfull-data fit (IN-SAMPLE, optimistic): acc {accuracy(rows, final):.4f}  "
          f"bal {final_bal:.4f}")

    (args.out_dir / "tuned_params.json").write_text(json.dumps(final, indent=2),
                                                    encoding="utf-8")
    (args.out_dir / "tuned_scene_code.py.txt").write_text(emit_code(final), encoding="utf-8")
    summary = {
        "n_clips": len(rows),
        "shipped_accuracy": 0.4694,
        "cv_accuracy_mean": float(np.mean(cv_acc)),
        "cv_accuracy_std": float(np.std(cv_acc)),
        "cv_balanced_mean": float(np.mean(cv_bal)),
        "cv_balanced_std": float(np.std(cv_bal)),
        "full_fit_accuracy_in_sample": accuracy(rows, final),
        "folds": args.folds,
        "iters": args.iters,
    }
    (args.out_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    print(f"\nwrote tuned params + code to {args.out_dir}")


if __name__ == "__main__":
    main()
