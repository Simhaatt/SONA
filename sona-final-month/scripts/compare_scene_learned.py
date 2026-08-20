#!/usr/bin/env python3
"""Does a learned scene classifier beat the tuned heuristic on the SAME features?

Motivation
----------
Threshold retuning lifted the heuristic from 0.469 to 0.714 on *pure* sources,
but on deployment-realistic speech+noise mixtures it saturates near 0.49 CV.
Before concluding "the scene classifier is weak", we need to know whether the
ceiling belongs to the *features* or to the *hand-tuned decision rule*.

This script holds the feature extractor fixed (the exact 8 features from
`SceneClassifier.features`) and swaps only the decision rule, under an identical
stratified k-fold protocol. If a small learned model clears the heuristic by a
wide margin, the rule is the bottleneck and replacing it is justified; if not,
the features are the bottleneck and no amount of fitting will help.

Models are deliberately small — this runs on CPU inside the realtime budget and
keeps the "no cloud, edge-deployable" property of the system intact.

    python scripts/compare_scene_learned.py --cache SCENE_EVAL_LOWSNR/feature_cache.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["centroid_hz", "flatness", "rolloff_hz", "zcr",
            "low_ratio", "mid_ratio", "high_ratio", "modulation"]


def build_models(seed: int):
    return {
        "majority baseline": DummyClassifier(strategy="most_frequent"),
        "logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
        "random forest (200)": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1),
        "MLP (32,16)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000,
                          random_state=seed)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True,
                    help="feature_cache.json written by tune_scene_thresholds.py")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heuristic-cv", type=float, default=None,
                    help="tuned-heuristic CV accuracy, for the comparison line")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.cache.read_text(encoding="utf-8"))
    X = np.array([[d["feats"][k] for k in FEATURES] for d in data], dtype=float)
    y = np.array([d["label"] for d in data])
    print(f"{len(y)} clips, {X.shape[1]} features, classes: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = {}
    for name, model in build_models(args.seed).items():
        accs, bals, preds, trues = [], [], [], []
        for tr, te in skf.split(X, y):
            m = build_models(args.seed)[name]
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            accs.append(accuracy_score(y[te], p))
            bals.append(balanced_accuracy_score(y[te], p))
            preds.extend(p)
            trues.extend(y[te])
        results[name] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
        }
        print(f"  {name:22s} acc {np.mean(accs):.4f} +/- {np.std(accs):.4f}   "
              f"balanced {np.mean(bals):.4f}")
        results[name]["_preds"], results[name]["_trues"] = preds, trues

    best = max(results, key=lambda k: results[k]["acc_mean"] if k != "majority baseline" else -1)
    print(f"\nbest learned model: {best}  ({results[best]['acc_mean']:.4f})")
    if args.heuristic_cv is not None:
        d = results[best]["acc_mean"] - args.heuristic_cv
        print(f"tuned heuristic:    {args.heuristic_cv:.4f}")
        print(f"delta:              {d:+.4f}  "
              f"({'features are the bottleneck' if d < 0.05 else 'decision rule was the bottleneck'})")

    labs = sorted(set(y))
    cm = confusion_matrix(results[best]["_trues"], results[best]["_preds"], labels=labs)
    print(f"\npooled out-of-fold confusion — {best} (rows=true, cols=pred)")
    print("          " + "".join(f"{l[:8]:>9s}" for l in labs))
    for i, l in enumerate(labs):
        print(f"{l:>9s} " + "".join(f"{v:9d}" for v in cm[i]))

    if args.out:
        clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in results.items()}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(clean, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
