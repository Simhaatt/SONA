"""Phase 0 feasibility probe for the perceptually-guided mixed-precision study.

Answers the four questions that can kill the study, before any real pipeline is written:

  0a  Is there enough degradation budget to DECOMPOSE per layer?   <-- the big risk
      Uniform INT8 costs only ~-0.12 dB SI-SDR overall. If that is not statistically
      distinguishable from zero on the dev split -- and if single-layer effects are smaller
      still -- there is no signal to rank layers by, and the study must move to INT4 or stop.
  0b  Does `nodes_to_quantize` give real per-layer control?
  0c  Does ORT support INT4 for these op types?
  0d  Is the GRU profitably quantizable (does a quantized kernel exist)?
  0e  Clean RTF measurement with warmup + repeats, to settle the 0.121 vs 0.480 discrepancy.

Usage:
    python PERCEPTUAL_MP/scripts/phase0_probe.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import torch

import quant_lib as QL

DEFAULT_CKPT = (
    QL.REPO
    / "results/kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07/extracted"
    / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt"
)


def count_qdq(path) -> int:
    import onnx

    m = onnx.load(str(path))
    return sum(1 for n in m.graph.node if n.op_type in ("QuantizeLinear", "DequantizeLinear"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--splits-dir", default=str(HERE.parent / "splits"))
    ap.add_argument("--onnx-dir", default=str(HERE.parent / "onnx" / "phase0"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "phase0"))
    ap.add_argument("--n-dev", type=int, default=200)
    args = ap.parse_args()

    torch.set_num_threads(1)
    onnx_dir, out_dir = Path(args.onnx_dir), Path(args.out_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = Path(args.splits_dir)

    findings: dict = {}
    print("=" * 78)
    print("PHASE 0 FEASIBILITY PROBE")
    print("=" * 78)

    model, tf, cfg = QL.load_model(args.checkpoint)
    calib_samples = QL.load_samples(splits / "calib_manifest.csv")
    dev_samples = QL.load_samples(splits / "dev_manifest.csv", limit=args.n_dev)
    print(f"\ncheckpoint : {Path(args.checkpoint).name}")
    print(f"calibration: {len(calib_samples)} clips   dev: {len(dev_samples)} clips")

    print("\n[setup] computing calibration features ...")
    calib_feats = QL.calib_features(tf, calib_samples)

    fp32_path = onnx_dir / "model_fp32.onnx"
    QL.export_fp32(model, tf, fp32_path)
    print(f"[setup] FP32 graph exported ({QL.model_size_kb(fp32_path):.1f} KB)")

    # ------------------------------------------------------------------ 0b per-layer control
    print("\n" + "-" * 78)
    print("0b  per-layer control via nodes_to_quantize")
    print("-" * 78)

    variants: dict[str, Path] = {}
    uniform_path = onnx_dir / "uniform_int8.onnx"
    QL.quantize_variant(fp32_path, uniform_path, calib_feats, nodes_to_quantize=None)
    variants["uniform_int8"] = uniform_path

    # Probe every weight-bearing layer, including the GRU -- confirming the GRU is inert is
    # part of what phase 0 is for.
    single, qdq_counts = {}, {}
    for node in QL.LAYERS:
        short = QL.SHORT[node]
        p = onnx_dir / f"only_{short}.onnx"
        QL.quantize_variant(fp32_path, p, calib_feats, nodes_to_quantize=[node])
        single[short] = p
        variants[f"only_{short}"] = p

    qdq_uniform = count_qdq(uniform_path)
    print(f"  uniform INT8      : {qdq_uniform:3d} QDQ nodes, {QL.model_size_kb(uniform_path):8.1f} KB")
    inert = []
    for short, p in single.items():
        n = count_qdq(p)
        qdq_counts[short] = n
        flag = "  <- INERT (not quantized)" if n == 0 else ""
        print(f"  only {short:10s}   : {n:3d} QDQ nodes, {QL.model_size_kb(p):8.1f} KB{flag}")
        if n == 0:
            inert.append(short)
    # Per-layer control works if the layers ORT *can* quantize each got their own small QDQ
    # subset. A layer ORT refuses to quantize at all (the GRU) is a runtime limitation, not a
    # failure of per-layer control.
    quantizable_short = [QL.SHORT[n] for n in QL.QUANTIZABLE]
    per_layer_works = all(0 < qdq_counts[s] < qdq_uniform for s in quantizable_short)

    # functional check: outputs must actually differ
    probe = torch.randn(1, 200, tf.n_bands, tf.n_feat_channels).numpy()
    s_fp32 = QL.make_session(fp32_path)
    out_fp32 = s_fp32.run(["gain"], {"feats": probe})[0]
    out_uni = QL.make_session(uniform_path).run(["gain"], {"feats": probe})[0]
    out_head = QL.make_session(single["head"]).run(["gain"], {"feats": probe})[0]
    d_uni = float(np.abs(out_fp32 - out_uni).max())
    d_head = float(np.abs(out_fp32 - out_head).max())
    d_head_uni = float(np.abs(out_head - out_uni).max())
    print(f"  max|fp32 - uniform| = {d_uni:.3e}")
    print(f"  max|fp32 - headonly|= {d_head:.3e}")
    print(f"  max|headonly - uni| = {d_head_uni:.3e}")
    ok_0b = per_layer_works and d_head > 0 and d_head_uni > 0
    print(f"  => per-layer control {'WORKS' if ok_0b else 'BROKEN'} "
          f"for the {len(quantizable_short)} quantizable layers")
    if inert:
        print(f"  => layers ORT refuses to quantize: {inert} (excluded from the study)")
    findings["0b_per_layer_control"] = {
        "pass": bool(ok_0b), "qdq_uniform": qdq_uniform, "qdq_single": qdq_counts,
        "inert_layers": inert,
        "max_abs_diff_fp32_vs_uniform": d_uni,
        "max_abs_diff_fp32_vs_headonly": d_head,
    }

    # ------------------------------------------------------------------------- 0c INT4 support
    print("\n" + "-" * 78)
    print("0c  INT4 support for these op types")
    print("-" * 78)
    int4_status, int4_err = "unsupported", None
    try:
        p4 = onnx_dir / "int4_probe.onnx"
        QL.quantize_variant(fp32_path, p4, calib_feats,
                            nodes_to_quantize=["/stem/stem.0/Conv"], weight_bits=4)
        try:
            s4 = QL.make_session(p4)
            s4.run(["gain"], {"feats": probe})
            int4_status = "works"
        except Exception as e:  # quantizes but no runtime kernel
            int4_status, int4_err = "quantized_but_no_kernel", str(e)[:200]
    except Exception as e:
        int4_status, int4_err = "export_failed", str(e)[:200]
    print(f"  INT4 weight quantization: {int4_status}")
    if int4_err:
        print(f"    -> {int4_err}")
    findings["0c_int4"] = {"status": int4_status, "error": int4_err}

    # ---------------------------------------------------------- 0a degradation budget on dev
    print("\n" + "-" * 78)
    print("0a  degradation budget -- is there signal to decompose?")
    print("-" * 78)
    print(f"  scoring {len(dev_samples)} dev clips per variant (this is the slow part) ...")

    t0 = time.perf_counter()
    ref_rows = QL.eval_variant(tf, dev_samples, model, "fp32")
    print(f"    fp32 reference done ({time.perf_counter()-t0:.0f}s)")

    all_delta_rows = []
    summary_0a = {}
    for label, path in variants.items():
        t0 = time.perf_counter()
        rows = QL.eval_variant(tf, dev_samples, QL.make_session(path), label)
        deltas = QL.paired_deltas(ref_rows, rows)
        for d in deltas:
            d["variant"] = label
        all_delta_rows += deltas

        entry = {}
        for key in ("d_stoi", "d_estoi", "d_si_sdr", "d_lsd", "mask_mse"):
            mean, lo, hi = QL.bootstrap_ci([d.get(key) for d in deltas])
            entry[key] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
            if key != "mask_mse" and mean is not None:
                # "signal" = CI excludes zero
                entry[key]["significant"] = bool(lo > 0 or hi < 0)
        summary_0a[label] = entry
        print(f"    {label:22s} done ({time.perf_counter()-t0:.0f}s)")

    print(f"\n  {'variant':22s} {'dSTOI':>22s} {'dSI-SDR dB':>22s} {'mask MSE':>10s}")
    for label, e in summary_0a.items():
        st, si, mm = e["d_stoi"], e["d_si_sdr"], e["mask_mse"]
        st_s = "*" if st.get("significant") else " "
        si_s = "*" if si.get("significant") else " "
        print(f"  {label:22s} "
              f"{st['mean']:+.5f} [{st['ci_lo']:+.5f},{st['ci_hi']:+.5f}]{st_s} "
              f"{si['mean']:+.3f} [{si['ci_lo']:+.3f},{si['ci_hi']:+.3f}]{si_s} "
              f"{mm['mean']:.3e}")
    print("  (* = 95% bootstrap CI excludes zero, i.e. a measurable effect)")

    uni = summary_0a["uniform_int8"]
    # The GRU is inert (ORT does not quantize it), so any "effect" measured for it is just
    # ONNX-vs-PyTorch float noise. Exclude it from the decomposability check.
    inert_labels = {f"only_{s}" for s in inert}
    singles_measurable = [
        lbl for lbl in summary_0a
        if lbl.startswith("only_") and lbl not in inert_labels
        and (summary_0a[lbl]["d_stoi"].get("significant")
             or summary_0a[lbl]["d_si_sdr"].get("significant"))
    ]
    findings["0a_budget"] = {
        "uniform_stoi_significant": bool(uni["d_stoi"].get("significant")),
        "uniform_sisdr_significant": bool(uni["d_si_sdr"].get("significant")),
        "single_layers_measurable": singles_measurable,
        "n_quantizable_probed": len(quantizable_short),
        "summary": summary_0a,
    }

    with open(out_dir / "0a_dev_deltas.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["variant", "uid", "d_si_sdr", "d_stoi", "d_estoi", "d_lsd", "mask_mse"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_delta_rows)

    # ------------------------------------------------------- 0d GRU profitably quantizable?
    print("\n" + "-" * 78)
    print("0d  is the GRU profitably quantizable?")
    print("-" * 78)

    def timed_rtf(path_or_model, n_rep: int = 7, n_frames: int = 300):
        x = torch.randn(1, n_frames, tf.n_bands, tf.n_feat_channels)
        audio_s = n_frames * tf.hop / tf.sr
        is_ort = hasattr(path_or_model, "run")
        for _ in range(3):  # warmup -- discarded
            if is_ort:
                path_or_model.run(["gain"], {"feats": x.numpy()})
            else:
                with torch.no_grad():
                    path_or_model(x)
        ts = []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            if is_ort:
                path_or_model.run(["gain"], {"feats": x.numpy()})
            else:
                with torch.no_grad():
                    path_or_model(x)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)) / audio_s, float(np.percentile(ts, 75) - np.percentile(ts, 25)) / audio_s

    rtf_fp32, iqr_fp32 = timed_rtf(QL.make_session(fp32_path, threads=1))
    rtf_gru, iqr_gru = timed_rtf(QL.make_session(single["band_gru"], threads=1))
    rtf_uni, iqr_uni = timed_rtf(QL.make_session(uniform_path, threads=1))
    sz_fp32 = QL.model_size_kb(fp32_path)
    sz_gru = QL.model_size_kb(single["band_gru"])
    print(f"  fp32          RTF {rtf_fp32:.4f} (IQR {iqr_fp32:.4f})  {sz_fp32:8.1f} KB")
    print(f"  GRU-only int8 RTF {rtf_gru:.4f} (IQR {iqr_gru:.4f})  {sz_gru:8.1f} KB")
    print(f"  uniform  int8 RTF {rtf_uni:.4f} (IQR {iqr_uni:.4f})  {QL.model_size_kb(uniform_path):8.1f} KB")
    gru_speedup = rtf_fp32 / rtf_gru if rtf_gru else float("nan")
    gru_size_ratio = sz_gru / sz_fp32
    gru_profitable = gru_speedup > 1.05 or gru_size_ratio < 0.9
    print(f"  GRU quantization speedup {gru_speedup:.3f}x, size ratio {gru_size_ratio:.3f}")
    print(f"  => GRU {'IS' if gru_profitable else 'is NOT'} profitably quantizable on this runtime")
    findings["0d_gru"] = {
        "rtf_fp32": rtf_fp32, "rtf_gru_only": rtf_gru, "rtf_uniform": rtf_uni,
        "speedup": gru_speedup, "size_ratio": gru_size_ratio,
        "profitable": bool(gru_profitable),
    }

    # ------------------------------------------------------------------- 0e clean RTF baseline
    print("\n" + "-" * 78)
    print("0e  clean RTF (warmup discarded, median of 7) -- settles 0.121 vs 0.480")
    print("-" * 78)
    rtf_torch, iqr_torch = timed_rtf(model)
    print(f"  pytorch  (1 thread) RTF {rtf_torch:.4f} (IQR {iqr_torch:.4f})")
    print(f"  onnx fp32(1 thread) RTF {rtf_fp32:.4f} (IQR {iqr_fp32:.4f})")
    print(f"  onnx int8(1 thread) RTF {rtf_uni:.4f} (IQR {iqr_uni:.4f})")
    findings["0e_rtf"] = {"pytorch": rtf_torch, "onnx_fp32": rtf_fp32, "onnx_int8": rtf_uni,
                          "iqr_pytorch": iqr_torch, "iqr_onnx_fp32": iqr_fp32}

    # ------------------------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    go = True
    reasons = []
    if not ok_0b:
        go = False
        reasons.append("per-layer quantization control does not work -> design is impossible")
    if inert:
        reasons.append(
            f"ORT does not quantize {inert} at all -> excluded from the study; the search space "
            f"is {len(quantizable_short)} layers, so exhaustive allocation is 2^{len(quantizable_short)}"
            f" = {2**len(quantizable_short)} configs")
    if not (uni["d_stoi"].get("significant") or uni["d_si_sdr"].get("significant")):
        go = False
        reasons.append("uniform INT8 degradation is NOT measurable on dev -> no budget to decompose")
    elif not singles_measurable:
        reasons.append(
            "uniform INT8 is measurable but NO single layer is -> per-layer ranking at INT8 has "
            "no signal; the study needs INT4 (see 0c) or it should stop")
        if int4_status != "works":
            go = False
            reasons.append("INT4 also unavailable -> nothing left to differentiate layers with")
    if not gru_profitable:
        reasons.append("GRU is not profitably quantizable -> pin it to FP32 (a reportable finding)")

    print(f"\n  PROCEED TO PHASE 1: {'YES' if go else 'NO'}\n")
    for r in reasons:
        print(f"   - {r}")
    if go and singles_measurable:
        print(f"   - single layers with measurable effect: {singles_measurable}")
    findings["verdict"] = {"proceed": bool(go), "reasons": reasons}

    with open(out_dir / "phase0_findings.json", "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2)
    print(f"\nwrote {out_dir / 'phase0_findings.json'}")
    print(f"wrote {out_dir / '0a_dev_deltas.csv'}")


if __name__ == "__main__":
    main()
