"""E4 -- does the stem-exemption result hold on a second architecture?

The headline of Study 1 is that one 3.8 KB stem convolution (0.133 % of weights) carries ~98.7 % of
INT8 quantization loss. That was measured on a single checkpoint, which is the most obvious
objection a reviewer can raise.

This repeats the per-layer sensitivity sweep on the complex-mask GTCRN: the *same* backbone
(stem + 4 gated TF-conv blocks + band GRU) with a different output representation -- a 2-channel
head predicting a bounded complex ratio mask instead of a real per-band gain, +97 parameters. Same
runtime, same calibration split, same dev clips, same protocol.

**This experiment can weaken the headline claim, and that is accepted going in.** If the stem is not
dominant here, "0.133 % of weights" becomes checkpoint-specific and the paper's claim shifts to the
methodological one -- *which layer to protect is a property of the model and runtime, so measure it
rather than assume a rule* -- which already matches the contrast drawn with Rusci et al., whose
method depends on quantizing exactly the recurrent layer that is incompressible here.

Everything runs on the speaker-disjoint dev split (p254/p256) carved from TRAIN. No test data.

Usage:
    py -3.12 PERCEPTUAL_MP/scripts/e4_second_architecture.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import quant_lib as QL  # noqa: E402


def eval_complex(tf, samples, backend, label: str, is_complex: bool) -> list[dict]:
    """Like quant_lib.eval_variant, but dispatches the mask-apply path on the model kind.

    The complex head emits a 2-channel real/imaginary mask that must be applied with
    apply_band_complex; using apply_band_gain would silently score a different system.
    """
    from src.eval.metrics import lsd, si_sdr, stoi_metric

    is_ort = hasattr(backend, "run")
    rows = []
    with torch.no_grad():
        for s in samples:
            wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
            spec, feats = tf.analyze(wav)
            if is_ort:
                gain_np = backend.run(["gain"], {"feats": feats.numpy()})[0]
                gain = torch.from_numpy(gain_np)
            else:
                gain = backend(feats)
                gain_np = gain.numpy()
            apply = tf.apply_band_complex if is_complex else tf.apply_band_gain
            enh = apply(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()

            row = {"method": label, "uid": s.uid, "_mask": gain_np}
            if s.clean is not None:
                row["si_sdr"] = si_sdr(s.clean, enh)
                row["stoi"] = stoi_metric(s.clean, enh, s.sr, extended=False)
                row["estoi"] = stoi_metric(s.clean, enh, s.sr, extended=True)
                row["lsd"] = lsd(s.clean, enh)
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(
        REPO / "sona-final-month" / "COMPLEX_MASK_AB" / "ckpt_complex_gtcrn" / "last.pt"))
    ap.add_argument("--fp32-onnx", default=str(
        REPO / "results" / "onnx" / "complex_gtcrn" / "model_fp32.onnx"))
    ap.add_argument("--dev-manifest", default=str(HERE.parent / "splits" / "dev_manifest.csv"))
    ap.add_argument("--calib-manifest", default=str(HERE.parent / "splits" / "calib_manifest.csv"))
    ap.add_argument("--calib-samples", type=int, default=30)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--onnx-dir", default=str(HERE.parent / "onnx" / "e4_complex"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "e4_complex"))
    args = ap.parse_args()

    model, tf, _cfg = QL.load_model(args.checkpoint)
    is_complex = bool(getattr(model, "is_complex", False))
    print(f"model is_complex = {is_complex}")
    if not is_complex:
        raise SystemExit("expected the complex-mask checkpoint; got a magnitude model")

    dev = QL.load_samples(args.dev_manifest, limit=args.limit)
    calib = QL.load_samples(args.calib_manifest, limit=args.calib_samples)
    calib_feats = QL.calib_features(tf, calib)
    print(f"dev clips: {len(dev)}   calibration clips: {len(calib)}")

    onnx_dir = Path(args.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    fp32 = Path(args.fp32_onnx)
    if not fp32.exists():
        raise SystemExit(f"missing FP32 export: {fp32}\nrun scripts/export_onnx.py first")

    # -------------------------------------------------------------- variants to build
    variants: dict[str, list[str] | None] = {}
    for node in QL.QUANTIZABLE:
        variants[f"only_{QL.SHORT[node]}"] = [node]
    variants["uniform_int8"] = None                                   # every quantizable layer
    stem_node = "/stem/stem.0/Conv"
    variants["mixed_stemfp32"] = [n for n in QL.QUANTIZABLE if n != stem_node]

    built: dict[str, Path] = {}
    for name, nodes in variants.items():
        path = onnx_dir / f"{name}.onnx"
        if not path.exists():
            print(f"  quantizing {name} ...", flush=True)
            QL.quantize_variant(fp32, path, calib_feats, nodes_to_quantize=nodes)
        built[name] = path
        print(f"  {name:20s} {QL.model_size_kb(path):9.1f} KB")

    # -------------------------------------------------------------- evaluate
    print("\nscoring FP32 reference ...", flush=True)
    ref = eval_complex(tf, dev, QL.make_session(fp32), "fp32", is_complex)

    rows_out = []
    per_clip = [{k: v for k, v in r.items() if k != "_mask"} for r in ref]
    for name, path in built.items():
        print(f"scoring {name} ...", flush=True)
        var = eval_complex(tf, dev, QL.make_session(path), name, is_complex)
        per_clip += [{k: v for k, v in r.items() if k != "_mask"} for r in var]
        deltas = QL.paired_deltas(ref, var)   # list of per-clip dicts: d_si_sdr, ..., mask_mse
        summary = {"variant": name, "size_kb": QL.model_size_kb(path), "n": len(dev)}
        metric_keys = sorted({k for d in deltas for k in d if k != "uid"})
        for key in metric_keys:
            arr = np.asarray([d[key] for d in deltas
                              if key in d and d[key] is not None and np.isfinite(d[key])],
                             dtype=float)
            if not arr.size:
                continue
            mean, lo, hi = QL.bootstrap_ci(arr)
            summary[key] = mean
            summary[f"{key}_abs"] = float(np.mean(np.abs(arr)))
            summary[f"{key}_ci_lo"] = lo
            summary[f"{key}_ci_hi"] = hi
        rows_out.append(summary)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in rows_out for k in r})
    cols = ["variant", "size_kb", "n"] + [c for c in cols if c not in ("variant", "size_kb", "n")]
    with (out / "e4_sensitivity.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    with (out / "e4_per_clip.csv").open("w", newline="", encoding="utf-8") as fh:
        pc_cols = ["method", "uid", "si_sdr", "stoi", "estoi", "lsd"]
        w = csv.DictWriter(fh, fieldnames=pc_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_clip)

    # -------------------------------------------------------------- the comparison that matters
    single = [r for r in rows_out if r["variant"].startswith("only_")]
    single.sort(key=lambda r: -abs(r.get("d_si_sdr_abs", 0.0)))
    uni = next((r for r in rows_out if r["variant"] == "uniform_int8"), None)
    mix = next((r for r in rows_out if r["variant"] == "mixed_stemfp32"), None)

    stem_share = float("nan")
    if uni and mix and uni.get("d_si_sdr") is not None:
        total = abs(uni["d_si_sdr"])
        remaining = abs(mix["d_si_sdr"]) if mix.get("d_si_sdr") is not None else float("nan")
        if total > 0:
            stem_share = 100.0 * (total - remaining) / total

    verdict = ("stem dominant -- replicates the magnitude-model finding"
               if single and single[0]["variant"] == "only_stem"
               else "stem NOT dominant -- the location of the damage is model-specific")

    summary_json = {
        "architecture": "complex_gtcrn",
        "most_sensitive_layer": single[0]["variant"] if single else None,
        "sensitivity_order": [r["variant"] for r in single],
        "uniform_int8_d_si_sdr": uni.get("d_si_sdr") if uni else None,
        "mixed_stemfp32_d_si_sdr": mix.get("d_si_sdr") if mix else None,
        "stem_share_of_degradation_pct": stem_share,
        "verdict": verdict,
    }
    (out / "e4_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    lines = ["# E4 -- stem exemption on a second architecture (complex-mask GTCRN)", "",
             "Same backbone, same runtime, same splits; only the output representation differs "
             "(2-channel complex ratio mask, +97 parameters). Dev split p254/p256, "
             f"{len(dev)} clips. Deltas are variant minus FP32.", "",
             "| variant | size KB | dSI-SDR | 95% CI | mean abs dSI-SDR |",
             "|---|---|---|---|---|"]
    for r in rows_out:
        lines.append(
            f"| {r['variant']} | {r['size_kb']:.1f} | {r.get('d_si_sdr', float('nan')):+.4f} | "
            f"[{r.get('d_si_sdr_ci_lo', float('nan')):+.4f}, "
            f"{r.get('d_si_sdr_ci_hi', float('nan')):+.4f}] | "
            f"{r.get('d_si_sdr_abs', float('nan')):.4f} |")
    lines += ["", "## Verdict", "",
              f"Most sensitive single layer: **{summary_json['most_sensitive_layer']}**",
              "",
              f"Sensitivity order (by mean |dSI-SDR|): "
              + " > ".join(r["variant"].replace("only_", "") for r in single),
              "",
              f"Stem's share of total INT8 degradation: **{stem_share:.1f} %**"
              if stem_share == stem_share else "Stem share not computable",
              "", f"**{verdict}**"]
    (out / "e4_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"most sensitive layer : {summary_json['most_sensitive_layer']}")
    print(f"order                : "
          + " > ".join(r["variant"].replace("only_", "") for r in single))
    print(f"stem share of damage : {stem_share:.1f} %")
    print(f"verdict              : {verdict}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
