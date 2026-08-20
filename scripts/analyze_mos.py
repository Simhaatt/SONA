"""Analyze MOS listening-test responses: per-condition MOS + CI, attention-check filtering,
and correlation against an objective intrusive metric (STOI; PESQ needs a compiled backend not
available in this local environment -- rerun this with `--metric pesq` on Kaggle if needed).

Usage:
    python scripts/analyze_mos.py --mos-dir mos_test --clean-manifest "E:\\...\\test_manifest.csv"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 1234):
    import numpy as np

    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) == 0:
        return None, None, None
    boot_means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mos-dir", default=str(ROOT / "mos_test"))
    ap.add_argument("--clean-manifest",
                    default=str(ROOT / "vbd_local" / "test_manifest.csv"))
    ap.add_argument("--drop-failed-attention-check", action="store_true", default=True)
    args = ap.parse_args()

    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    from src.eval.metrics import stoi_metric
    from src.utils.audio_io import load_audio

    mos_dir = Path(args.mos_dir)
    responses_path = mos_dir / "responses.csv"
    if not responses_path.exists():
        raise SystemExit(f"{responses_path} not found -- no ratings collected yet")

    with open(responses_path, newline="", encoding="utf-8") as fh:
        responses = list(csv.DictReader(fh))

    # ------------------------------------------------------------- attention-check filtering
    attn_by_rater: dict[str, list[int]] = {}
    for r in responses:
        if r["is_attention_check"] == "1":
            attn_by_rater.setdefault(r["rater_id"], []).append(int(r["rating"]))

    failed_raters = {rid for rid, ratings in attn_by_rater.items() if max(ratings) < 4}
    print(f"Raters: {len({r['rater_id'] for r in responses})} total, "
          f"{len(failed_raters)} failed the attention check "
          f"(rated the clean reference clip below 'Good')")
    if failed_raters:
        print("  failed:", sorted(failed_raters))

    kept = responses
    if args.drop_failed_attention_check:
        kept = [r for r in responses if r["rater_id"] not in failed_raters]
    kept = [r for r in kept if r["is_attention_check"] != "1"]
    n_raters_kept = len({r["rater_id"] for r in kept})
    print(f"Scoring {len(kept)} ratings from {n_raters_kept} raters "
          f"(attention-check clip excluded from condition scores)")

    if n_raters_kept < 10:
        print("\nNOTE: fewer than 10 raters -- report this as a PILOT listening study "
              "(indicative, not conclusive), per the plan's honesty standard.\n")

    # ------------------------------------------------------------------- per-condition MOS
    by_cond: dict[str, list[int]] = {}
    for r in kept:
        by_cond.setdefault(r["condition"], []).append(int(r["rating"]))

    print("\n=== Per-condition MOS (95% bootstrap CI) ===\n")
    print(f"{'condition':22s} {'n':>5s} {'mean':>7s} {'ci_lo':>7s} {'ci_hi':>7s}")
    mos_summary = []
    for cond, ratings in sorted(by_cond.items()):
        mean, lo, hi = _bootstrap_ci(ratings)
        print(f"{cond:22s} {len(ratings):5d} {mean:7.3f} {lo:7.3f} {hi:7.3f}")
        mos_summary.append({"condition": cond, "n": len(ratings), "mean_mos": mean,
                             "ci_lo": lo, "ci_hi": hi})

    with open(mos_dir / "mos_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["condition", "n", "mean_mos", "ci_lo", "ci_hi"])
        w.writeheader()
        w.writerows(mos_summary)

    # -------------------------------------------------------- correlate per-clip MOS vs. STOI
    manifest_path = mos_dir / "clip_manifest.csv"
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        clip_manifest = {row["clip_id"]: row for row in csv.DictReader(fh)}

    clean_by_uid: dict[str, str] = {}
    with open(args.clean_manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            clean_by_uid[Path(row["noisy"]).stem] = row["clean"]

    per_clip_mos: dict[str, list[int]] = {}
    for r in kept:
        per_clip_mos.setdefault(r["clip_id"], []).append(int(r["rating"]))

    mos_vals, stoi_vals = [], []
    for clip_id, ratings in per_clip_mos.items():
        row = clip_manifest.get(clip_id)
        if row is None or row["condition"] == "clean_attention_check":
            continue
        clean_path = clean_by_uid.get(row["uid"])
        clip_path = mos_dir / "audio" / f"{clip_id}.wav"
        if not clean_path or not clip_path.exists():
            continue
        clean, sr = load_audio(clean_path, sr=16000, mono=True)
        est, _ = load_audio(clip_path, sr=16000, mono=True)
        try:
            s = stoi_metric(clean, est, sr)
        except Exception:
            continue
        mos_vals.append(float(np.mean(ratings)))
        stoi_vals.append(s)

    if len(mos_vals) >= 3:
        r_p, p_p = pearsonr(mos_vals, stoi_vals)
        r_s, p_s = spearmanr(mos_vals, stoi_vals)
        print(f"\n=== MOS vs. STOI correlation (n={len(mos_vals)} clips) ===")
        print(f"Pearson  r={r_p:.3f} (p={p_p:.4f})")
        print(f"Spearman r={r_s:.3f} (p={p_s:.4f})")
    else:
        print("\nNot enough clips with both MOS and STOI to correlate yet.")

    print(f"\nWrote {mos_dir / 'mos_summary.csv'}")


if __name__ == "__main__":
    main()
