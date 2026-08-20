"""Phase 3b: MS-SNSD cross-corpus validation of the frozen mixed-precision allocation.

Split out from phase3_validate.py so the already-completed VoiceBank half
(voicebank_test824_per_clip.csv) is not recomputed. Evaluates the same 5 backends on the
1100-clip MS-SNSD external corpus -- a genuinely different noise distribution and recording
setup -- to test whether "exempt the stem from INT8" generalises beyond VoiceBank-DEMAND.

This matters because the project's earlier online-adaptation result looked good on VoiceBank and
FAILED to reproduce on MS-SNSD; the same scrutiny applies here.
"""

from __future__ import annotations

import csv
import json
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

FROZEN_QUANTIZED = [
    "/blocks.0/conv/Conv", "/blocks.1/conv/Conv",
    "/blocks.2/conv/Conv", "/blocks.3/conv/Conv", "/head/MatMul",
]


def eval_backend(tf, samples, backend, label, checkpoint_every=200, ckpt_path=None):
    """Evaluate, appending to disk periodically so a kill doesn't lose everything."""
    from src.eval.metrics import lsd, si_sdr, stoi_metric

    is_ort = hasattr(backend, "run")
    rows, buffer = [], []
    with torch.no_grad():
        for i, s in enumerate(samples):
            wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
            spec, feats = tf.analyze(wav)
            if is_ort:
                gain = torch.from_numpy(backend.run(["gain"], {"feats": feats.numpy()})[0])
            else:
                gain = backend(feats)
            enh = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()
            if s.clean is None:
                continue
            row = {
                "uid": s.uid, "method": label,
                "si_sdr": si_sdr(s.clean, enh),
                "stoi": stoi_metric(s.clean, enh, s.sr, extended=False),
                "estoi": stoi_metric(s.clean, enh, s.sr, extended=True),
                "lsd": lsd(s.clean, enh),
                "noise_type": s.noise_type, "snr_db": s.snr_db,
            }
            rows.append(row); buffer.append(row)
            if ckpt_path and len(buffer) >= checkpoint_every:
                _append(ckpt_path, buffer); buffer = []
    if ckpt_path and buffer:
        _append(ckpt_path, buffer)
    return rows


def _append(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def paired(a_rows, b_rows, metric, la, lb):
    a = {r["uid"]: r[metric] for r in a_rows}
    b = {r["uid"]: r[metric] for r in b_rows}
    uids = [u for u in a if u in b]
    da = np.array([a[u] for u in uids]); db = np.array([b[u] for u in uids])
    mean, lo, hi = QL.bootstrap_ci(list(da - db))
    try:
        _, p = wilcoxon(da, db)
    except Exception:
        p = float("nan")
    return {"comparison": f"{la} - {lb}", "metric": metric, "n": len(uids),
            "mean_delta": mean, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": float(p),
            "significant": bool(lo is not None and (lo > 0 or hi < 0))}


def main() -> None:
    torch.set_num_threads(3)
    out_dir = HERE.parent / "results" / "phase3"
    onnx_dir = HERE.parent / "onnx" / "phase3"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    per_clip_path = out_dir / "ms_snsd_per_clip.csv"

    model, tf, _ = QL.load_model(DEFAULT_CKPT)
    calib = QL.load_samples(HERE.parent / "splits" / "calib_manifest.csv")
    calib_feats = QL.calib_features(tf, calib)

    fp32_path = HERE.parent / "onnx" / "phase0" / "model_fp32.onnx"
    mixed_path = onnx_dir / "frozen_mixed.onnx"
    if not mixed_path.exists():
        QL.quantize_variant(fp32_path, mixed_path, calib_feats, nodes_to_quantize=FROZEN_QUANTIZED)

    backends = {
        "fp32": model,
        "mixed_stemfp32": QL.make_session(mixed_path),
        "uniform_int8": QL.make_session(HERE.parent / "onnx" / "phase0" / "uniform_int8.onnx"),
        "stem_only_int8": QL.make_session(HERE.parent / "onnx" / "phase0" / "only_stem.onnx"),
        "blocks5_int8": QL.make_session(HERE.parent / "onnx" / "phase2" / "step5_shared.onnx"),
    }

    manifest = out_dir / "ms_snsd_resolved.csv"
    samples = QL.load_samples(manifest)
    print(f"MS-SNSD external: {len(samples)} clips x {len(backends)} backends")

    # resume support: skip backends already fully written
    done_counts: dict[str, int] = {}
    if per_clip_path.exists():
        with open(per_clip_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                done_counts[r["method"]] = done_counts.get(r["method"], 0) + 1
        print(f"  resuming; already on disk: {done_counts}")

    rows_by = {}
    for name, backend in backends.items():
        if done_counts.get(name, 0) >= len(samples):
            print(f"  {name:16s} already complete, loading from disk")
            with open(per_clip_path, newline="", encoding="utf-8") as fh:
                rows_by[name] = [r for r in csv.DictReader(fh) if r["method"] == name]
            for r in rows_by[name]:
                for k in ("si_sdr", "stoi", "estoi", "lsd"):
                    r[k] = float(r[k])
            continue
        t0 = time.perf_counter()
        rows_by[name] = eval_backend(tf, samples, backend, name, ckpt_path=per_clip_path)
        print(f"  {name:16s} SI-SDR={np.mean([r['si_sdr'] for r in rows_by[name]]):8.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    print("\n" + "=" * 84)
    print("MS-SNSD CROSS-CORPUS PAIRED TESTS")
    print("=" * 84)
    stats = {}
    for a, b in [("mixed_stemfp32", "uniform_int8"), ("fp32", "mixed_stemfp32"),
                 ("fp32", "uniform_int8")]:
        print(f"\n{a} vs {b}:")
        for metric in ("si_sdr", "stoi", "estoi", "lsd"):
            r = paired(rows_by[a], rows_by[b], metric, a, b)
            stats[f"{a}_vs_{b}__{metric}"] = r
            flag = "*" if r["significant"] else " "
            print(f"  {metric:6s} delta={r['mean_delta']:+.5f} "
                  f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]{flag} p={r['wilcoxon_p']:.2e}")

    # out-of-sample interaction re-test
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
    print(f"\nNON-ADDITIVITY (MS-SNSD, out-of-corpus):")
    print(f"  stem {d_stem.mean():+.5f} | blocks5 {d_blk.mean():+.5f} | "
          f"sum {(d_stem+d_blk).mean():+.5f} | actual {d_uni.mean():+.5f}")
    print(f"  interaction {m:+.5f} [{lo:+.5f},{hi:+.5f}]{'*' if (lo>0 or hi<0) else ' '} "
          f"ratio {ratio:.2f}x")
    stats["interaction"] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "ratio": float(ratio)}

    with open(out_dir / "phase3b_ms_snsd.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nwrote {out_dir / 'phase3b_ms_snsd.json'}")


if __name__ == "__main__":
    main()
