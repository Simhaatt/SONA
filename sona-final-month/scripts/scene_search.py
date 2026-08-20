#!/usr/bin/env python3
"""Search for the best honest scene classifier: features x models x CV protocol.

Answers "did we get the best results?" with evidence rather than the first model
that beat the heuristic. Three axes are varied:

  features  base 8 (shipped)          vs  rich (temporal + finer spectral)
  model     logreg / svm / rf / extra-trees / gbrt / xgboost / mlp (tuned)
  CV        per-clip StratifiedKFold  vs  StratifiedGroupKFold by noise source

The grouped protocol is the HONEST one: mixtures built from the same noise
texture are kept on the same side of the split, so a model cannot win by
memorising a texture that also appears in the test fold. The per-clip number is
reported alongside only as the optimistic upper bound.

Class weights are balanced (fan clips outnumber traffic ~6:1), and the score is
macro-balanced accuracy so the small traffic class cannot be ignored.

    python scripts/scene_search.py            # full search
    python scripts/scene_search.py --quick    # fewer configs
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

import scene_feature_lib as fl

warnings.filterwarnings("ignore")


def extract(manifests, cache: Path, sr, repo: Path):
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data
    import soundfile as sf
    rows = []
    for m in manifests:
        rows.extend(fl.load_manifest_rows(Path(m)))
    print(f"extracting features from {len(rows)} clips ...")
    out = []
    for i, (path, label, group) in enumerate(rows):
        wav, srr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if srr != sr:
            idx = np.linspace(0, len(wav) - 1, int(len(wav) * sr / srr))
            wav = np.interp(idx, np.arange(len(wav)), wav).astype(np.float32)
        feats = fl.compute_all(wav, sr=sr)
        out.append({"feats": {k: float(v) for k, v in feats.items()},
                    "label": label, "group": group})
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def build_models(quick: bool, seed: int):
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def sc(est):
        return make_pipeline(StandardScaler(), est)

    models = {
        "logreg": sc(LogisticRegression(max_iter=3000, class_weight="balanced",
                                        C=1.0, random_state=seed)),
        "svm_rbf": sc(SVC(kernel="rbf", C=8.0, gamma="scale",
                          class_weight="balanced", probability=False,
                          random_state=seed)),
        "rf": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     min_samples_leaf=1, n_jobs=-1, random_state=seed),
        "extratrees": ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                           n_jobs=-1, random_state=seed),
        "mlp_32_16": sc(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=4000,
                                      alpha=1e-3, random_state=seed)),
        "mlp_64_32": sc(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=4000,
                                      alpha=1e-3, random_state=seed)),
    }
    if not quick:
        models["gbrt"] = GradientBoostingClassifier(random_state=seed)
        try:
            from xgboost import XGBClassifier
            models["xgboost"] = XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, n_jobs=-1,
                eval_metric="mlogloss", random_state=seed)
        except Exception:
            pass
        models["svm_rbf_c32"] = sc(SVC(kernel="rbf", C=32.0, gamma="scale",
                                       class_weight="balanced", random_state=seed))
        models["mlp_96_48_24"] = sc(MLPClassifier(hidden_layer_sizes=(96, 48, 24),
                                                  max_iter=5000, alpha=3e-3,
                                                  random_state=seed))
    return models


def encode(data, keys):
    X = np.array([[d["feats"].get(k, 0.0) for k in keys] for d in data], dtype=float)
    y = np.array([d["label"] for d in data])
    g = np.array([d["group"] for d in data])
    return X, y, g


def cv_score(model_factory, X, y, groups, mode, folds, seed):
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if mode == "group":
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split = splitter.split(X, y)

    accs, bals = [], []
    for tr, te in split:
        # xgboost needs integer-encoded labels
        m = clone(model_factory) if hasattr(model_factory, "get_params") else model_factory
        classes = sorted(set(y))
        cmap = {c: i for i, c in enumerate(classes)}
        try:
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
        except (ValueError, TypeError):
            yi = np.array([cmap[c] for c in y])
            inv = {i: c for c, i in cmap.items()}
            m.fit(X[tr], yi[tr])
            p = np.array([inv[i] for i in m.predict(X[te])])
        accs.append(accuracy_score(y[te], p))
        bals.append(balanced_accuracy_score(y[te], p))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(bals))


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--manifests", nargs="+", default=[
        str(root / "recordings" / "manifest_msnsd.csv"),
        str(root / "recordings" / "manifest_mixtures.csv"),
    ])
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out-dir", type=Path, default=root / "SCENE_SEARCH")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = extract(args.manifests, args.out_dir / "feature_cache_rich.json",
                   fl.DEFAULT_SR, args.repo)
    from collections import Counter
    print(f"{len(data)} clips  classes={dict(Counter(d['label'] for d in data))}  "
          f"groups={len(set(d['group'] for d in data))}")

    base_keys = fl.BASE_KEYS
    rich_keys = fl.rich_key_order(data[0]["feats"])
    print(f"base features: {len(base_keys)}   rich features: {len(rich_keys)}")

    results = []
    models = build_models(args.quick, args.seed)
    for feat_name, keys in (("base8", base_keys), ("rich", rich_keys)):
        X, y, g = encode(data, keys)
        for mname, model in models.items():
            for mode in ("group", "clip"):
                acc, std, bal = cv_score(model, X, y, g, mode, args.folds, args.seed)
                results.append({"features": feat_name, "model": mname,
                                "cv": mode, "acc": acc, "std": std, "bal": bal})
                tag = "HONEST" if mode == "group" else "optim."
                print(f"  {feat_name:5s} {mname:14s} {tag:7s} "
                      f"acc {acc:.4f}+/-{std:.4f}  bal {bal:.4f}")

    # rank by honest (grouped) accuracy
    honest = [r for r in results if r["cv"] == "group"]
    honest.sort(key=lambda r: r["acc"], reverse=True)
    print("\n==== TOP 8 by HONEST grouped-CV accuracy ====")
    for r in honest[:8]:
        print(f"  {r['acc']:.4f}  bal {r['bal']:.4f}  {r['features']:5s} {r['model']}")

    best = honest[0]
    print(f"\nBEST HONEST: {best['features']} + {best['model']} "
          f"-> {best['acc']:.4f} (bal {best['bal']:.4f})")
    (args.out_dir / "search_results.json").write_text(
        json.dumps({"results": results, "best": best,
                    "base_keys": base_keys, "rich_keys": rich_keys}, indent=2),
        encoding="utf-8")
    print(f"wrote {args.out_dir/'search_results.json'}")


if __name__ == "__main__":
    main()
