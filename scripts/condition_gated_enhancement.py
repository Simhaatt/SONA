"""R1 -- condition-gated enhancement: a remedy for the ASR degradation.

The negative result. Enhancement significantly raises corpus WER overall (+24 % relative), and the
per-noise breakdown shows why: of the five VoiceBank noise types, only `bus` is improved; the other
four are all made worse, `cafe` badly.

The remedy. Do not put the enhancer in the ASR path unconditionally. Decide per clip, from the
noisy audio alone, whether to send the enhanced or the unprocessed signal to the recogniser. The
enhanced signal still goes to the human listener either way -- enhancement is worth +9.9 dB SI-SDRi
and +0.009 STOI for a person, and nothing here argues against that. What the data argues against is
letting a recogniser consume it blindly.

Three routings are scored:

  always      current behaviour: enhance everything (the negative result)
  never       passthrough everything
  oracle      per-noise-type decision taken from the test data itself -- an upper bound, NOT a
              deployable method, reported to show the available headroom
  router      per-clip decision from the 34 scene features of the noisy signal, evaluated under
              **leave-one-noise-type-out** cross-validation, so the held-out noise type is never
              seen in training

Leave-one-noise-type-out is deliberately the hard protocol. Splitting per clip would let the router
memorise a noise texture and report a number that would not survive deployment -- the same mistake
grouped CV corrected in the scene classifier (0.844 -> 0.748). A router that only works on noise
types it has already seen is not a remedy.

Usage:
    python scripts/condition_gated_enhancement.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

SEED = 2026
N_BOOT = 10000
ENHANCED = "neural_gtcrn"
BASELINE = "passthrough"


def corpus_wer(uids, wer_by_uid, words) -> float:
    num = sum(wer_by_uid[u] * words[u] for u in uids)
    den = sum(words[u] for u in uids)
    return num / den if den else float("nan")


def load_features(manifest: str, uids: set[str], cache: Path) -> dict[str, np.ndarray]:
    """34 scene features per clip, computed on the NOISY signal (what a deployment sees)."""
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {u: v for u, v in zip(d["uids"].tolist(), d["feats"])}

    import soundfile as sf

    from src.controller.scene import SceneClassifier

    sc = SceneClassifier()
    feats, keep = [], []
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        uid = Path(r["noisy"]).stem
        if uid not in uids:
            continue
        wav, sr = sf.read(r["noisy"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        f = sc.features(wav)
        feats.append([float(f[k]) for k in sorted(f)])
        keep.append(uid)
        if (i + 1) % 200 == 0:
            print(f"  features {i+1}/{len(rows)}", flush=True)
    arr = np.asarray(feats, dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, uids=np.array(keep), feats=arr)
    print(f"  cached features -> {cache}  shape={arr.shape}")
    return {u: v for u, v in zip(keep, arr)}


def fit_logreg(X: np.ndarray, y: np.ndarray, iters: int = 400, lr: float = 0.5) -> np.ndarray:
    """Tiny L2 logistic regression, class-balanced. No sklearn dependency at inference."""
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Xs.shape[1])
    pos = max(y.sum(), 1.0)
    neg = max(len(y) - y.sum(), 1.0)
    sw = np.where(y > 0, len(y) / (2 * pos), len(y) / (2 * neg))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Xs @ w))
        g = Xs.T @ (sw * (p - y)) / len(y) + 1e-3 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return np.r_[w, mu, sd]


def predict_logreg(model: np.ndarray, X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    w, mu, sd = model[:d + 1], model[d + 1:2 * d + 1], model[2 * d + 1:]
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Xs @ w))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wer", default=str(ROOT / "results" / "wer_full824" / "per_sample_wer.csv"))
    ap.add_argument("--transcripts",
                    default=str(ROOT / "results" / "wer_full824" / "clean_transcripts.csv"))
    ap.add_argument("--conditions", default=str(ROOT / "results" / "vbd_test_conditions.csv"))
    ap.add_argument("--manifest", default=str(ROOT / "vbd_local" / "test_manifest.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "condition_gating"))
    args = ap.parse_args()

    words = {r["uid"]: len(r["text"].split())
             for r in csv.DictReader(Path(args.transcripts).open(newline="", encoding="utf-8"))}
    wer: dict[tuple[str, str], float] = {}
    for r in csv.DictReader(Path(args.wer).open(newline="", encoding="utf-8")):
        wer[(r["method"], r["uid"])] = float(r["wer"])
    cond = {r["uid"]: r["noise_type"]
            for r in csv.DictReader(Path(args.conditions).open(newline="", encoding="utf-8"))}

    uids = sorted(u for u in cond
                  if (ENHANCED, u) in wer and (BASELINE, u) in wer and u in words)
    enh = {u: wer[(ENHANCED, u)] for u in uids}
    pas = {u: wer[(BASELINE, u)] for u in uids}
    noise = {u: cond[u] for u in uids}
    types = sorted(set(noise.values()))
    print(f"{len(uids)} clips, {len(types)} noise types")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- fixed routings
    always = corpus_wer(uids, enh, words)
    never = corpus_wer(uids, pas, words)

    # oracle: per noise type, whichever is better on the test data itself (upper bound)
    oracle_choice = {}
    for t in types:
        us = [u for u in uids if noise[u] == t]
        oracle_choice[t] = "enhance" if corpus_wer(us, enh, words) < corpus_wer(us, pas, words) \
            else "passthrough"
    oracle = corpus_wer(uids, {u: (enh if oracle_choice[noise[u]] == "enhance" else pas)[u]
                               for u in uids}, words)
    print(f"\nalways-enhance  {always:.4f}")
    print(f"never-enhance   {never:.4f}")
    print(f"oracle (per-type, upper bound) {oracle:.4f}   {oracle_choice}")

    # ---------------------------------------------------------------- learned router
    feats = load_features(args.manifest, set(uids), out / "scene_features_824.npz")
    fu = [u for u in uids if u in feats]
    X = np.asarray([feats[u] for u in fu])
    # label 1 = enhancement HURTS this clip (route to passthrough)
    y = np.asarray([1.0 if enh[u] > pas[u] else 0.0 for u in fu])
    print(f"router data: X={X.shape}  hurt={int(y.sum())}  not-hurt={int(len(y)-y.sum())}")

    routed: dict[str, float] = {}
    fold_rows = []
    for t in types:
        te = [i for i, u in enumerate(fu) if noise[u] == t]
        tr = [i for i, u in enumerate(fu) if noise[u] != t]
        model = fit_logreg(X[tr], y[tr])
        p = predict_logreg(model, X[te])
        for i, prob in zip(te, p):
            u = fu[i]
            routed[u] = pas[u] if prob >= 0.5 else enh[u]
        chose_pass = int((p >= 0.5).sum())
        fold_rows.append({"held_out_noise": t, "n": len(te),
                          "routed_to_passthrough": chose_pass,
                          "frac_passthrough": chose_pass / len(te),
                          "oracle_choice": oracle_choice[t]})
        print(f"  fold {t:9s} n={len(te):4d} -> passthrough for {chose_pass:4d} "
              f"({100*chose_pass/len(te):5.1f} %)   oracle says {oracle_choice[t]}")

    router = corpus_wer(fu, routed, words)
    always_f = corpus_wer(fu, enh, words)
    never_f = corpus_wer(fu, pas, words)
    oracle_f = corpus_wer(fu, {u: (enh if oracle_choice[noise[u]] == "enhance" else pas)[u]
                               for u in fu}, words)

    # paired clip bootstrap for router vs always-enhance
    rng = np.random.default_rng(SEED)
    w_arr = np.array([words[u] for u in fu], float)
    e_r = np.array([routed[u] for u in fu]) * w_arr
    e_a = np.array([enh[u] for u in fu]) * w_arr
    idx = rng.integers(0, len(fu), size=(N_BOOT, len(fu)))
    wb = w_arr[idx].sum(axis=1)
    bd = e_r[idx].sum(axis=1) / wb - e_a[idx].sum(axis=1) / wb
    lo, hi = np.percentile(bd, [2.5, 97.5])

    headroom = always_f - oracle_f
    captured = (always_f - router) / headroom if headroom > 0 else float("nan")

    summary = {
        "n_clips": len(fu),
        "corpus_wer": {"always_enhance": always_f, "never_enhance": never_f,
                       "oracle_per_type": oracle_f, "learned_router": router},
        "oracle_choice": oracle_choice,
        "headroom_vs_always": headroom,
        "router_gain_vs_always": always_f - router,
        "fraction_of_headroom_captured": captured,
        "router_vs_always_ci95": [float(lo), float(hi)],
        "router_significant": bool(hi < 0),
        "folds": fold_rows,
    }
    (out / "condition_gating.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Condition-gated enhancement (R1)", "",
        "Remedy for the ASR degradation. The decision is *which signal the recogniser consumes*; "
        "the human listener gets the enhanced signal either way.", "",
        "| routing | corpus WER | vs always-enhance |", "|---|---|---|",
        f"| always enhance (current) | {always_f:.4f} | — |",
        f"| never enhance | {never_f:.4f} | {never_f - always_f:+.4f} |",
        f"| **oracle, per noise type** (upper bound, not deployable) | **{oracle_f:.4f}** | "
        f"{oracle_f - always_f:+.4f} |",
        f"| **learned router** (leave-one-noise-type-out) | **{router:.4f}** | "
        f"{router - always_f:+.4f} |",
        "",
        f"Headroom between always-enhance and the oracle is {headroom:+.4f} WER. The router "
        f"captures **{100*captured:.0f} %** of it, with a paired clip bootstrap 95 % CI on the "
        f"router-minus-always difference of [{lo:+.4f}, {hi:+.4f}]"
        + (" — excludes zero." if hi < 0 else " — includes zero."),
        "",
        "Oracle decision per noise type: "
        + ", ".join(f"`{t}`={oracle_choice[t]}" for t in types) + ".",
        "",
        "## Folds", "",
        "| held-out noise | n | routed to passthrough | oracle choice |", "|---|---|---|---|",
    ]
    for f in fold_rows:
        lines.append(f"| {f['held_out_noise']} | {f['n']} | {f['routed_to_passthrough']} "
                     f"({100*f['frac_passthrough']:.0f} %) | {f['oracle_choice']} |")
    (out / "condition_gating.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nalways {always_f:.4f} | never {never_f:.4f} | oracle {oracle_f:.4f} | "
          f"router {router:.4f}")
    print(f"router captures {100*captured:.0f} % of headroom; CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
