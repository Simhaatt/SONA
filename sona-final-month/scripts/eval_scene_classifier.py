"""Evaluate the SONA scene classifier on a folder of labelled recordings.

Plan item (Week 2): sanity-check the deterministic feature-based scene
classifier on real speech/music/noise clips and record where it confuses
classes, so the Layer B section can report accuracy and (if needed) retuned
thresholds.

Manifest CSV (header required): path,label
    path  = wav/flac/... file (absolute or relative to the manifest)
    label = one of the SONA scene labels (speech/music/traffic/fan/crowd/mixed/unknown)

    python scripts/eval_scene_classifier.py --manifest recordings/manifest.csv \
        --out-dir SCENE_EVAL

Outputs: confusion.csv, per_file.csv, report.md (accuracy, per-class P/R/F1).
Use --self-test to validate the harness with a synthetic dataset + dummy
classifier (no audio, no torch, no repo import needed).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def load_manifest(path: Path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "path" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise SystemExit("manifest must have columns: path,label")
        for r in reader:
            rows.append((r["path"].strip(), r["label"].strip().lower()))
    return rows


def get_real_classifier(repo: Path):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.controller import SceneClassifier  # type: ignore
    from src.utils.audio_io import load_audio, DEFAULT_SR  # type: ignore
    clf = SceneClassifier(sr=DEFAULT_SR)

    def predict(file_path: str) -> str:
        wav, _ = load_audio(file_path, sr=DEFAULT_SR, mono=True)
        return clf.predict(wav, sr=DEFAULT_SR).label

    return predict, list(SceneClassifier.labels)


def confusion_and_metrics(pairs, labels):
    """pairs: list[(true, pred)]. Returns (confusion dict, metrics dict, accuracy)."""
    labels = list(dict.fromkeys(labels))
    conf = {t: {p: 0 for p in labels} for t in labels}
    correct = 0
    for true, pred in pairs:
        if true not in conf:
            conf[true] = {p: 0 for p in labels}
        if pred not in conf[true]:
            conf[true][pred] = 0
        conf[true][pred] += 1
        correct += int(true == pred)
    all_labels = list(dict.fromkeys(labels + list(conf.keys())))
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for true, pred in pairs:
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1
    metrics = {}
    for lab in all_labels:
        p = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        r = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        metrics[lab] = {"precision": round(p, 4), "recall": round(r, 4),
                        "f1": round(f1, 4), "support": tp[lab] + fn[lab]}
    accuracy = correct / len(pairs) if pairs else 0.0
    return conf, metrics, accuracy


def write_outputs(out_dir: Path, per_file, conf, metrics, accuracy):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(conf.keys())
    with open(out_dir / "per_file.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "true", "pred", "correct"])
        for path, true, pred in per_file:
            w.writerow([path, true, pred, int(true == pred)])
    with open(out_dir / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["true\\pred"] + labels)
        for t in labels:
            w.writerow([t] + [conf[t].get(p, 0) for p in labels])
    lines = ["# Scene classifier evaluation\n",
             f"Overall accuracy: **{accuracy:.4f}** ({len(per_file)} clips)\n",
             "## Per-class metrics\n",
             "| label | precision | recall | f1 | support |",
             "| --- | --- | --- | --- | --- |"]
    for lab, m in metrics.items():
        lines.append(f"| {lab} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |")
    lines += ["", "## Confusion (rows = true, cols = predicted)\n",
              "| true\\pred | " + " | ".join(labels) + " |",
              "| --- | " + " | ".join("---" for _ in labels) + " |"]
    for t in labels:
        lines.append(f"| {t} | " + " | ".join(str(conf[t].get(p, 0)) for p in labels) + " |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"accuracy={accuracy:.4f} over {len(per_file)} clips -> {out_dir}")


def self_test(out_dir: Path):
    labels = ["speech", "music", "traffic", "fan", "crowd", "mixed", "unknown"]
    import random
    rng = random.Random(0)
    truth = [rng.choice(labels) for _ in range(60)]

    def dummy_predict(path: str) -> str:
        true = path.split("::")[1]
        return true if rng.random() < 0.8 else rng.choice(labels)  # 80% accurate

    per_file = [(f"clip{i}::{t}", t, dummy_predict(f"clip{i}::{t}"))
                for i, t in enumerate(truth)]
    conf, metrics, acc = confusion_and_metrics([(t, p) for _, t, p in per_file], labels)
    write_outputs(out_dir, per_file, conf, metrics, acc)
    assert 0.5 <= acc <= 1.0
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "SCENE_EVAL")
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parents[2])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.out_dir)
        return
    if not args.manifest:
        raise SystemExit("--manifest is required (or use --self-test)")

    rows = load_manifest(args.manifest)
    predict, labels = get_real_classifier(args.repo.resolve())
    base = args.manifest.resolve().parent
    per_file, pairs = [], []
    for rel, true in rows:
        fp = (base / rel) if not Path(rel).is_absolute() else Path(rel)
        try:
            pred = predict(str(fp))
        except Exception as exc:
            print(f"  ! {fp}: {exc}", file=sys.stderr)
            pred = "unknown"
        per_file.append((str(fp), true, pred))
        pairs.append((true, pred))
    conf, metrics, acc = confusion_and_metrics(pairs, labels)
    write_outputs(args.out_dir, per_file, conf, metrics, acc)


if __name__ == "__main__":
    main()
