"""Phase 3: validate the FROZEN mixed-precision allocation on data never used for selection.

The allocation (stem FP32 + the other 5 quantizable layers INT8, GRU FP32 because ORT cannot
quantize it) was chosen on the dev split carved from TRAIN. This script touches, for the first
time:
    * VoiceBank-DEMAND test set, 824 clips (speakers p232/p257 -- disjoint from calib and dev)
    * MS-SNSD external, 1100 clips (entirely different corpus)

Reports paired per-clip bootstrap CIs + Wilcoxon signed-rank against BOTH baselines:
    uniform INT8 (does exempting the stem actually help?) and FP32 (is the mixed model lossless?)

Also re-tests the non-additivity/interaction result out-of-sample.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from scipy.stats import wilcoxon

import quant_lib as QL

DEFAULT_CKPT = (
    QL.REPO
    / "results/kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/extracted"
    / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt"
)

# The frozen allocation: everything quantizable EXCEPT the stem.
FROZEN_QUANTIZED = [
    "/blocks.0/conv/Conv", "/blocks.1/conv/Conv",
    "/blocks.2/conv/Conv", "/blocks.3/conv/Conv", "/head/MatMul",
]
FROZEN_FP32 = ["/stem/stem.0/Conv", "/band_gru/GRU"]


def fix_manifest(manifest: Path, out_dir: Path) -> Path:
    """Rewrite the stale 'D:\\noise cancellation' prefix documented in NEXT_STEPS_PLAN.md."""
    with open(manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return manifest
    repo_str = str(QL.REPO)
    changed = False
    for r in rows:
        for k in ("clean", "noisy"):
            if k in r and r[k] and not Path(r[k]).exists():
                fixed = r[k].replace("D:\\noise cancellation", repo_str)
                if Path(fixed).exists():
                    r[k], changed = fixed, True
    if not changed:
        return manifest
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = out_dir / "ms_snsd_resolved.csv"
    with open(fixed_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return fixed_path


def eval_backend(tf, samples, backend, label):
    from src.eval.metrics import lsd, si_sdr, stoi_metric

    is_ort = hasattr(backend, "run")
    rows = []
    with torch.no_grad():
        for s in samples:
            wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
            spec, feats = tf.analyze(wav)
            if is_ort:
                gain = torch.from_numpy(backend.run(["gain"], {"feats": feats.numpy()})[0])
            else:
                gain = backend(feats)
            enh = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()
            if s.clean is None:
                continue
            rows.append({
                "uid": s.uid, "method": label,
                "si_sdr": si_sdr(s.clean, enh),
                "stoi": stoi_metric(s.clean, enh, s.sr, extended=False),
                "estoi": stoi_metric(s.clean, enh, s.sr, extended=True),
                "lsd": lsd(s.clean, enh),
                "noise_type": s.noise_type, "snr_db": s.snr_db,
            })
    return rows


def paired(a_rows, b_rows, metric, la, lb):
    a = {r["uid"]: r[metric] for r in a_rows}
    b = {r["uid"]: r[metric] for r in b_rows}
    uids = [u for u in a if u in b]
    da = np.array([a[u] for u in uids]); db = np.array([b[u] for u in uids])
    d = da - db
    mean, lo, hi = QL.bootstrap_ci(list(d))
    try:
        _, p = wilcoxon(da, db)
    except Exception:
        p = float("nan")
    return {"comparison": f"{la} - {lb}", "metric": metric, "n": len(uids),
            "mean_delta": mean, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": float(p),
            "significant": bool(lo is not None and (lo > 0 or hi < 0))}


def main() -> None:
    torch.set_num_threads(2)
    onnx_dir = HERE.parent / "onnx" / "phase3"
    out_dir = HERE.parent / "results" / "phase3"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- freeze the allocation first
    frozen = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quantized_int8": FROZEN_QUANTIZED,
        "kept_fp32": FROZEN_FP32,
        "selected_on": "dev split (VoiceBank TRAIN speakers p254/p256), 200 clips",
        "rationale": "Phase 2b: exempting the 3.8KB stem recovers +0.097 dB SI-SDR "
                     "(p=4.2e-05) for +2.8KB; mixed model statistically indistinguishable "
                     "from FP32 (p=0.77) at 3.2x compression.",
    }
    with open(out_dir / "frozen_allocation.json", "w", encoding="utf-8") as fh:
        json.dump(frozen, fh, indent=2)
    print(f"FROZEN ALLOCATION recorded at {frozen['frozen_at']}")
    print(f"  INT8: {[QL.SHORT[n] for n in FROZEN_QUANTIZED]}")
    print(f"  FP32: {[QL.SHORT[n] for n in FROZEN_FP32]}\n")

    model, tf, _ = QL.load_model(DEFAULT_CKPT)
    calib = QL.load_samples(HERE.parent / "splits" / "calib_manifest.csv")
    calib_feats = QL.calib_features(tf, calib)

    fp32_path = HERE.parent / "onnx" / "phase0" / "model_fp32.onnx"
    mixed_path = onnx_dir / "frozen_mixed.onnx"
    if not mixed_path.exists():
        QL.quantize_variant(fp32_path, mixed_path, calib_feats, nodes_to_quantize=FROZEN_QUANTIZED)
    uniform_path = HERE.parent / "onnx" / "phase0" / "uniform_int8.onnx"

    backends = {
        "fp32": model,
        "mixed_stemfp32": QL.make_session(mixed_path),
        "uniform_int8": QL.make_session(uniform_path),
        "stem_only_int8": QL.make_session(HERE.parent / "onnx" / "phase0" / "only_stem.onnx"),
        "blocks5_int8": QL.make_session(HERE.parent / "onnx" / "phase2" / "step5_shared.onnx"),
    }
    sizes = {
        "fp32": QL.model_size_kb(fp32_path),
        "mixed_stemfp32": QL.model_size_kb(mixed_path),
        "uniform_int8": QL.model_size_kb(uniform_path),
    }

    datasets = {
        "voicebank_test824": (
            os.environ.get("SONA_VBD_TEST_MANIFEST",
                           str(QL.REPO / "vbd_local" / "test_manifest.csv")), 0),
        "ms_snsd_external": (str(fix_manifest(
            QL.REPO / "external_data" / "ms_snsd_sona" / "test.csv", out_dir)), 0),
    }

    all_results = {}
    for ds_name, (manifest, limit) in datasets.items():
        print("=" * 88)
        print(f"DATASET: {ds_name}")
        print("=" * 88)
        samples = QL.load_samples(manifest, limit=(limit or None))
        print(f"  {len(samples)} clips")
        if not samples:
            print("  SKIP (no samples resolved)")
            continue

        rows_by = {}
        for name, backend in backends.items():
            t0 = time.perf_counter()
            rows_by[name] = eval_backend(tf, samples, backend, name)
            print(f"  {name:16s} mean SI-SDR={np.mean([r['si_sdr'] for r in rows_by[name]]):8.4f} "
                  f"STOI={np.mean([r['stoi'] for r in rows_by[name]]):.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)

        # save per-clip for later stats / per-noise breakdowns
        flat = [r for rows in rows_by.values() for r in rows]
        with open(out_dir / f"{ds_name}_per_clip.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            w.writeheader(); w.writerows(flat)

        print(f"\n  PAIRED TESTS ({ds_name})")
        ds_stats = {}
        for a, b in [("mixed_stemfp32", "uniform_int8"), ("fp32", "mixed_stemfp32"),
                     ("fp32", "uniform_int8")]:
            print(f"    {a} vs {b}:")
            for metric in ("si_sdr", "stoi", "estoi", "lsd"):
                r = paired(rows_by[a], rows_by[b], metric, a, b)
                ds_stats[f"{a}_vs_{b}__{metric}"] = r
                flag = "*" if r["significant"] else " "
                print(f"      {metric:6s} delta={r['mean_delta']:+.5f} "
                      f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]{flag} p={r['wilcoxon_p']:.2e}")

        # ---------------------------------------------------- out-of-sample interaction re-test
        fp = {r["uid"]: r["si_sdr"] for r in rows_by["fp32"]}
        st = {r["uid"]: r["si_sdr"] for r in rows_by["stem_only_int8"]}
        bl = {r["uid"]: r["si_sdr"] for r in rows_by["blocks5_int8"]}
        un = {r["uid"]: r["si_sdr"] for r in rows_by["uniform_int8"]}
        uids = [u for u in fp if u in st and u in bl and u in un]
        d_stem = np.array([st[u] - fp[u] for u in uids])
        d_blk = np.array([bl[u] - fp[u] for u in uids])
        d_uni = np.array([un[u] - fp[u] for u in uids])
        inter = d_uni - (d_stem + d_blk)
        m, lo, hi = QL.bootstrap_ci(list(inter))
        ratio = abs(d_uni.mean() / (d_stem + d_blk).mean()) if (d_stem + d_blk).mean() else float("nan")
        print(f"\n  NON-ADDITIVITY ({ds_name}):")
        print(f"    stem-only {d_stem.mean():+.5f} | blocks5 {d_blk.mean():+.5f} | "
              f"sum {(d_stem+d_blk).mean():+.5f} | actual {d_uni.mean():+.5f}")
        print(f"    interaction {m:+.5f} [{lo:+.5f},{hi:+.5f}]"
              f"{'*' if (lo>0 or hi<0) else ' '}  ratio {ratio:.2f}x")
        ds_stats["interaction"] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "ratio": float(ratio),
                                    "d_stem": float(d_stem.mean()), "d_blocks": float(d_blk.mean()),
                                    "d_uniform": float(d_uni.mean())}
        all_results[ds_name] = ds_stats
        print()

    all_results["sizes_kb"] = sizes
    all_results["frozen_allocation"] = frozen
    with open(out_dir / "phase3_validation.json", "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"wrote {out_dir / 'phase3_validation.json'}")


if __name__ == "__main__":
    main()
