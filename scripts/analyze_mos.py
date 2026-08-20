"""Analyze MOS listening-test responses: per-condition MOS + CI, attention-check filtering,
and correlation against an objective intrusive metric (STOI; PESQ needs a compiled backend not
available in this local environment -- rerun this with `--metric pesq` on Kaggle if needed).

Usage:
    python scripts/analyze_mos.py --mos-dir mos_test --clean-manifest "E:\\...\\test_manifest.csv"
"""

from __future__ import annotations

import argparse
import csv
import os
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

    # ------------------------------------------------- paired comparisons + equivalence tests
    # Each rater heard every utterance under every condition, so pair within (rater, uid):
    # this removes rater severity and utterance difficulty from the error term.
    from scipy import stats as _st

    PAIRS = [
        # (A, B, question, equivalence?)
        ("gtcrn", "complex_gtcrn",
         "complex vs magnitude mask -- where SI-SDR and PESQ disagree", False),
        ("quant_fp32", "quant_mixed",
         "stem-exempt mixed vs FP32 -- the indistinguishability claim", True),
        ("quant_fp32", "quant_uniform_int8",
         "uniform INT8 vs FP32 -- should be audibly worse", False),
        ("quant_mixed", "quant_uniform_int8",
         "mixed vs uniform INT8 -- the allocation comparison", False),
        ("noisy", "gtcrn", "does enhancement help a listener at all", False),
    ]
    EQUIV_MARGIN = float(os.environ.get("MOS_EQUIV_MARGIN", "0.3"))

    idx: dict[tuple[str, str, str], int] = {}
    for r in kept:
        idx[(r["rater_id"], r["uid"], r["condition"])] = int(r["rating"])

    print("\n=== Paired comparisons (within rater and utterance) ===\n")
    print(f"margin for equivalence tests: +/-{EQUIV_MARGIN} MOS (declared, not fitted)\n")
    print(f"{'comparison':38s} {'n':>4s} {'mean diff':>10s} {'95% CI':>18s} {'p':>9s}  verdict")
    paired_rows = []
    for a, b, question, want_equiv in PAIRS:
        da, db = [], []
        for (rater, uid, cond), val in idx.items():
            if cond != a:
                continue
            other = idx.get((rater, uid, b))
            if other is not None:
                da.append(val)
                db.append(other)
        n = len(da)
        if n < 3:
            print(f"{a+' vs '+b:38s} {n:4d}   (too few paired observations)")
            continue
        d = np.asarray(da, dtype=float) - np.asarray(db, dtype=float)
        mean_d = float(np.mean(d))
        boot = np.array([np.mean(np.random.default_rng(1234 + i).choice(d, n, replace=True))
                         for i in range(2000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        try:
            p_val = float(_st.wilcoxon(da, db).pvalue) if np.any(d != 0) else float("nan")
        except ValueError:
            p_val = float("nan")

        row = {"a": a, "b": b, "question": question, "n_pairs": n, "mean_diff": mean_d,
               "ci95_low": float(lo), "ci95_high": float(hi), "wilcoxon_p": p_val}

        if want_equiv:
            # TOST: equivalence at alpha if the 90 % CI lies inside +/-margin
            sd = float(np.std(d, ddof=1)) if n > 1 else float("nan")
            se = sd / np.sqrt(n) if n > 1 and sd > 0 else float("nan")
            if se and se == se and se > 0:
                t_lo = (mean_d + EQUIV_MARGIN) / se
                t_hi = (mean_d - EQUIV_MARGIN) / se
                p_lo = 1 - _st.t.cdf(t_lo, df=n - 1)
                p_hi = _st.t.cdf(t_hi, df=n - 1)
                p_tost = float(max(p_lo, p_hi))
                c90 = _st.t.ppf(0.95, df=n - 1) * se
                eq_lo, eq_hi = mean_d - c90, mean_d + c90
                equivalent = bool(eq_lo > -EQUIV_MARGIN and eq_hi < EQUIV_MARGIN)
                row.update({"tost_p": p_tost, "ci90_low": float(eq_lo),
                            "ci90_high": float(eq_hi), "equivalent": equivalent,
                            "equiv_margin": EQUIV_MARGIN})
                verdict = ("EQUIVALENT" if equivalent else
                           "not shown equivalent (inconclusive, not 'different')")
            else:
                verdict = "equivalence not computable"
        else:
            verdict = ("A better" if (p_val == p_val and p_val < 0.05 and mean_d > 0) else
                       "B better" if (p_val == p_val and p_val < 0.05 and mean_d < 0) else
                       "no significant difference")

        paired_rows.append(row)
        p_s = "--" if p_val != p_val else (f"{p_val:.2e}" if p_val < 1e-3 else f"{p_val:.3f}")
        print(f"{a+' vs '+b:38s} {n:4d} {mean_d:+10.3f} "
              f"[{lo:+6.3f},{hi:+6.3f}] {p_s:>9s}  {verdict}")
        if want_equiv and "tost_p" in row:
            print(f"{'':38s}      TOST p={row['tost_p']:.4f}  "
                  f"90% CI [{row['ci90_low']:+.3f}, {row['ci90_high']:+.3f}] "
                  f"vs margin +/-{EQUIV_MARGIN}")

    if paired_rows:
        cols = sorted({k for r in paired_rows for k in r})
        with open(mos_dir / "mos_paired.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(paired_rows)

    # ---- power note, so an inconclusive equivalence result is read correctly
    n_pairs_typical = max((r["n_pairs"] for r in paired_rows), default=0)
    if n_pairs_typical:
        sd_guess = 1.0
        need = int(np.ceil((1.96 + 0.84) ** 2 * (sd_guess / EQUIV_MARGIN) ** 2))
        print(f"\npower: {n_pairs_typical} paired observations available; detecting a "
              f"{EQUIV_MARGIN} MOS difference at 80 % power needs ~{need} "
              f"(~{int(np.ceil(need / 7))} raters at 7 utterances each).")
        if n_pairs_typical < need:
            print("      -> under-powered; report equivalence results as inconclusive rather "
                  "than as evidence of sameness.")

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
