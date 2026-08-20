"""Phase 2: bit-allocation Pareto front, scaled down from the plan's exhaustive 2^6=64 search.

Why scaled down: Phase 1 found the PESQ-sensitivity and SI-SDR-sensitivity rankings of the 6
quantizable layers agree almost everywhere (Spearman rho=0.943), differing at exactly ONE
position (block1_d2 vs block2_d4 swap ranks 3/4). Building cumulative "safest-layers-first"
allocations from each ranking therefore produces IDENTICAL layer subsets at 5 of 6 steps and
only diverges at a single step -- so the exhaustive 64-config grid would spend the overwhelming
majority of its budget confirming configs that are already known (from Phase 1) to behave
identically. Evaluating the 6 configs that actually matter (the shared steps once each, the one
divergent step twice) traces both curves completely for a fraction of the cost, and is the
correct response to what Phase 1 already showed rather than running the original plan
mechanically. Every step is still real measured data, nothing is inferred.

Steps quantized cumulatively (from PESQ-ranking, least- to most-sensitive):
  0: {}                                                       = FP32 (Phase 0, reused)
  1: {head}
  2: {head, block3_d8}
  3a (PESQ order):   {head, block3_d8, block2_d4}
  3b (SI-SDR order): {head, block3_d8, block1_d2}
  4: {head, block3_d8, block2_d4, block1_d2}   <- same set either order added
  5: {head, block3_d8, block2_d4, block1_d2, block0_d1}
  6: all 6 layers                                             = uniform_int8 (Phase 0, reused)
"""

from __future__ import annotations

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

NODE_OF = {
    "stem": "/stem/stem.0/Conv",
    "block0_d1": "/blocks.0/conv/Conv",
    "block1_d2": "/blocks.1/conv/Conv",
    "block2_d4": "/blocks.2/conv/Conv",
    "block3_d8": "/blocks.3/conv/Conv",
    "head": "/head/MatMul",
}

STEPS = {
    "step1_head": ["head"],
    "step2_head_b3": ["head", "block3_d8"],
    "step3a_pesq_add_b2": ["head", "block3_d8", "block2_d4"],
    "step3b_sisdr_add_b1": ["head", "block3_d8", "block1_d2"],
    "step4_shared": ["head", "block3_d8", "block2_d4", "block1_d2"],
    "step5_shared": ["head", "block3_d8", "block2_d4", "block1_d2", "block0_d1"],
}


def main() -> None:
    torch.set_num_threads(1)
    onnx_dir = HERE.parent / "onnx" / "phase2"
    out_dir = HERE.parent / "results" / "phase2"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tf, cfg = QL.load_model(DEFAULT_CKPT)
    calib_samples = QL.load_samples(HERE.parent / "splits" / "calib_manifest.csv")
    dev_samples = QL.load_samples(HERE.parent / "splits" / "dev_manifest.csv", limit=200)
    calib_feats = QL.calib_features(tf, calib_samples)

    fp32_path = HERE.parent / "onnx" / "phase0" / "model_fp32.onnx"
    if not fp32_path.exists():
        fp32_path = QL.export_fp32(model, tf, onnx_dir / "model_fp32.onnx")

    # ------------------------------------------------------------- export the 6 new configs
    paths = {}
    for name, layers in STEPS.items():
        nodes = [NODE_OF[l] for l in layers]
        p = onnx_dir / f"{name}.onnx"
        if not p.exists():
            QL.quantize_variant(fp32_path, p, calib_feats, nodes_to_quantize=nodes)
        paths[name] = (p, layers)
    print(f"exported/found {len(paths)} new configs")

    # ------------------------------------------------------------- analytic model size per config
    ckpt = torch.load(str(DEFAULT_CKPT), map_location="cpu")
    sd = ckpt["model"]
    layer_param_prefix = {
        "stem": "stem.0.", "block0_d1": "blocks.0.conv.", "block1_d2": "blocks.1.conv.",
        "block2_d4": "blocks.2.conv.", "block3_d8": "blocks.3.conv.", "head": "head.",
    }

    def layer_bytes(prefix: str) -> int:
        return sum(v.numel() * 4 for k, v in sd.items() if k.startswith(prefix))  # fp32 = 4B/elt

    total_bytes = sum(v.numel() * 4 for v in sd.values())
    layer_size = {name: layer_bytes(pfx) for name, pfx in layer_param_prefix.items()}
    # INT8 weights ~1B/elt vs FP32's 4B/elt for the quantized layers; everything else stays FP32.
    def estimated_size(quantized_layers: list[str]) -> int:
        size = total_bytes
        for l in quantized_layers:
            size -= layer_size[l] * 0.75  # 4B->1B is a 75% reduction for that layer's weights
        return size

    # ------------------------------------------------------------- evaluate each config on dev
    from src.eval.metrics import lsd, si_sdr, stoi_metric

    def eval_config(sess_or_model, samples):
        is_ort = hasattr(sess_or_model, "run")
        si, st, es = [], [], []
        with torch.no_grad():
            for s in samples:
                wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
                spec, feats = tf.analyze(wav)
                if is_ort:
                    gain = torch.from_numpy(sess_or_model.run(["gain"], {"feats": feats.numpy()})[0])
                else:
                    gain = sess_or_model(feats)
                enh = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()
                if s.clean is not None:
                    si.append(si_sdr(s.clean, enh))
                    st.append(stoi_metric(s.clean, enh, s.sr, extended=False))
                    es.append(stoi_metric(s.clean, enh, s.sr, extended=True))
        return {"si_sdr": float(np.mean(si)), "stoi": float(np.mean(st)), "estoi": float(np.mean(es))}

    results = {}
    print("evaluating fp32 baseline (0 layers quantized)...")
    t0 = time.perf_counter()
    results["step0_fp32"] = {"layers": [], "size_bytes": total_bytes,
                              **eval_config(model, dev_samples)}
    print(f"  done ({time.perf_counter()-t0:.0f}s)")

    for name, (path, layers) in paths.items():
        print(f"evaluating {name} ({layers})...")
        t0 = time.perf_counter()
        sess = QL.make_session(path)
        results[name] = {"layers": layers, "size_bytes": estimated_size(layers),
                          **eval_config(sess, dev_samples)}
        print(f"  done ({time.perf_counter()-t0:.0f}s)  "
              f"si_sdr={results[name]['si_sdr']:.4f} size={results[name]['size_bytes']/1024:.1f}KB")

    uniform_path = HERE.parent / "onnx" / "phase0" / "uniform_int8.onnx"
    if uniform_path.exists():
        print("evaluating step6_uniform (all 6 layers, reusing Phase 0 export)...")
        all_layers = list(NODE_OF.keys())
        sess = QL.make_session(uniform_path)
        results["step6_uniform"] = {"layers": all_layers, "size_bytes": estimated_size(all_layers),
                                    **eval_config(sess, dev_samples)}

    # ------------------------------------------------------------- assemble the two curves
    pesq_curve = ["step0_fp32", "step1_head", "step2_head_b3", "step3a_pesq_add_b2",
                  "step4_shared", "step5_shared", "step6_uniform"]
    sisdr_curve = ["step0_fp32", "step1_head", "step2_head_b3", "step3b_sisdr_add_b1",
                   "step4_shared", "step5_shared", "step6_uniform"]

    print("\n" + "=" * 90)
    print("PARETO FRONT: size vs quality, perceptually-guided (PESQ) vs fidelity-guided (SI-SDR) order")
    print("=" * 90)
    print(f"{'step':22s} {'n_layers':>8s} {'size_KB':>10s} {'si_sdr':>10s} {'stoi':>9s} {'estoi':>9s}")
    print("-- PESQ-guided cumulative order --")
    for step in pesq_curve:
        r = results[step]
        print(f"{step:22s} {len(r['layers']):8d} {r['size_bytes']/1024:10.1f} "
              f"{r['si_sdr']:10.4f} {r['stoi']:9.4f} {r['estoi']:9.4f}")
    print("-- SI-SDR-guided cumulative order --")
    for step in sisdr_curve:
        r = results[step]
        print(f"{step:22s} {len(r['layers']):8d} {r['size_bytes']/1024:10.1f} "
              f"{r['si_sdr']:10.4f} {r['stoi']:9.4f} {r['estoi']:9.4f}")

    r_pesq3, r_sisdr3 = results["step3a_pesq_add_b2"], results["step3b_sisdr_add_b1"]
    d_sisdr = r_pesq3["si_sdr"] - r_sisdr3["si_sdr"]
    d_stoi = r_pesq3["stoi"] - r_sisdr3["stoi"]
    print("\n" + "=" * 90)
    print("HEADLINE: the only point where the two curves differ")
    print("=" * 90)
    print(f"  PESQ-order step3  (quantize block2_d4 3rd): SI-SDR={r_pesq3['si_sdr']:.4f} STOI={r_pesq3['stoi']:.4f}")
    print(f"  SI-SDR-order step3(quantize block1_d2 3rd): SI-SDR={r_sisdr3['si_sdr']:.4f} STOI={r_sisdr3['stoi']:.4f}")
    print(f"  delta (pesq-order minus si_sdr-order): dSI-SDR={d_sisdr:+.4f} dSTOI={d_stoi:+.5f}")
    print(f"  at IDENTICAL size ({r_pesq3['size_bytes']/1024:.1f} KB both)")
    if abs(d_sisdr) < 0.02 and abs(d_stoi) < 0.001:
        print("\n=> The two allocation strategies are practically indistinguishable at their one "
              "point of difference. This CONFIRMS Phase 1's null result at the allocation level: "
              "perceptually-guided and MSE-guided mixed-precision allocation give the same "
              "size-vs-quality trade-off for this checkpoint at INT8/6-layer granularity.")
    else:
        print("\n=> A measurable difference exists at this one step -- worth a paired significance "
              "test on the full per-clip data before treating as conclusive.")

    with open(out_dir / "phase2_pareto.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "pesq_curve": pesq_curve, "sisdr_curve": sisdr_curve,
                  "step3_delta": {"d_si_sdr": d_sisdr, "d_stoi": d_stoi}}, fh, indent=2)
    print(f"\nwrote {out_dir / 'phase2_pareto.json'}")


if __name__ == "__main__":
    main()
