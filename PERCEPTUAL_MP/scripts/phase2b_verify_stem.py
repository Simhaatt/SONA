"""Phase 2b: rigorously verify the "keep the stem in FP32" finding with paired per-clip statistics.

Phase 2's Pareto front suggested that quantizing the 3.8 KB stem layer costs ~0.097 dB SI-SDR
while quantizing the other five layers (92% of the weights) costs ~0.002 dB -- i.e. excluding a
single tiny layer from quantization recovers almost all of the loss for +0.3% model size. Phase 2
only reported means, so this script re-evaluates the three relevant configs storing PER-CLIP
values, then runs paired bootstrap CIs and Wilcoxon signed-rank tests.

It also quantifies the INTERACTION: stem-only and blocks-only degradations are each tiny, yet
their combination is ~10x larger than their sum, which would mean quantization error compounds
non-linearly through the network rather than adding independently.

Configs (all calibrated on the calibration split, evaluated on the dev split):
    fp32          - no quantization (PyTorch reference)
    stem_only     - only /stem/stem.0/Conv quantized          (from Phase 0)
    blocks5       - the 5 non-stem layers quantized           (= Phase 2 step5_shared)
    uniform       - all 6 quantizable layers quantized        (from Phase 0)
"""

from __future__ import annotations

import json
import sys
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


def paired_stats(a: list[float], b: list[float], label_a: str, label_b: str):
    """Paired comparison of two per-clip metric lists: mean delta, bootstrap CI, Wilcoxon."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    mean, lo, hi = QL.bootstrap_ci(list(d))
    try:
        stat, p = wilcoxon(a, b)
    except Exception:
        stat, p = float("nan"), float("nan")
    return {
        "comparison": f"{label_a} - {label_b}",
        "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
        "wilcoxon_p": float(p),
        "significant": bool(lo is not None and (lo > 0 or hi < 0)),
    }


def main() -> None:
    torch.set_num_threads(2)
    p0 = HERE.parent / "onnx" / "phase0"
    p2 = HERE.parent / "onnx" / "phase2"
    out_dir = HERE.parent / "results" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tf, _ = QL.load_model(DEFAULT_CKPT)
    dev = QL.load_samples(HERE.parent / "splits" / "dev_manifest.csv", limit=200)

    configs = {
        "fp32": model,
        "stem_only": QL.make_session(p0 / "only_stem.onnx"),
        "blocks5": QL.make_session(p2 / "step5_shared.onnx"),
        "uniform": QL.make_session(p0 / "uniform_int8.onnx"),
    }

    from src.eval.metrics import lsd, si_sdr, stoi_metric

    per_clip: dict[str, dict[str, list[float]]] = {}
    for name, backend in configs.items():
        is_ort = hasattr(backend, "run")
        si, st, es, ld = [], [], [], []
        print(f"evaluating {name} ...", flush=True)
        with torch.no_grad():
            for s in dev:
                wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
                spec, feats = tf.analyze(wav)
                if is_ort:
                    gain = torch.from_numpy(backend.run(["gain"], {"feats": feats.numpy()})[0])
                else:
                    gain = backend(feats)
                enh = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()
                if s.clean is not None:
                    si.append(si_sdr(s.clean, enh))
                    st.append(stoi_metric(s.clean, enh, s.sr, extended=False))
                    es.append(stoi_metric(s.clean, enh, s.sr, extended=True))
                    ld.append(lsd(s.clean, enh))
        per_clip[name] = {"si_sdr": si, "stoi": st, "estoi": es, "lsd": ld}
        print(f"  {name:10s} mean SI-SDR={np.mean(si):.4f}  STOI={np.mean(st):.4f}", flush=True)

    # ------------------------------------------------------------------ headline comparisons
    print("\n" + "=" * 86)
    print("PAIRED TESTS (n=200 dev clips, 95% bootstrap CI + Wilcoxon signed-rank)")
    print("=" * 86)

    results = {}
    comparisons = [
        ("blocks5", "uniform", "keeping stem FP32 vs quantizing everything"),
        ("fp32", "uniform", "FP32 vs uniform INT8 (total quantization cost)"),
        ("fp32", "blocks5", "FP32 vs stem-FP32 mixed (residual cost)"),
        ("fp32", "stem_only", "FP32 vs stem-only quantized"),
    ]
    for a, b, desc in comparisons:
        print(f"\n{desc}")
        for metric in ("si_sdr", "stoi", "estoi"):
            r = paired_stats(per_clip[a][metric], per_clip[b][metric], a, b)
            results[f"{a}_vs_{b}__{metric}"] = r
            flag = "*" if r["significant"] else " "
            print(f"  {metric:6s} delta={r['mean_delta']:+.5f} "
                  f"[{r['ci_lo']:+.5f},{r['ci_hi']:+.5f}]{flag} wilcoxon_p={r['wilcoxon_p']:.2e}")

    # ------------------------------------------------------------------ interaction analysis
    print("\n" + "=" * 86)
    print("NON-ADDITIVITY: does quantization error compound through the network?")
    print("=" * 86)
    fp32_si = np.asarray(per_clip["fp32"]["si_sdr"])
    d_stem = np.asarray(per_clip["stem_only"]["si_sdr"]) - fp32_si
    d_blocks = np.asarray(per_clip["blocks5"]["si_sdr"]) - fp32_si
    d_uniform = np.asarray(per_clip["uniform"]["si_sdr"]) - fp32_si
    interaction = d_uniform - (d_stem + d_blocks)
    m_i, lo_i, hi_i = QL.bootstrap_ci(list(interaction))
    print(f"  stem-only degradation           : {d_stem.mean():+.5f} dB")
    print(f"  blocks-only (5 layers)          : {d_blocks.mean():+.5f} dB")
    print(f"  sum if independent              : {(d_stem+d_blocks).mean():+.5f} dB")
    print(f"  ACTUAL combined (uniform)       : {d_uniform.mean():+.5f} dB")
    print(f"  interaction (actual - sum)      : {m_i:+.5f} dB [{lo_i:+.5f},{hi_i:+.5f}]"
          f"{'*' if (lo_i>0 or hi_i<0) else ' '}")
    ratio = abs(d_uniform.mean() / (d_stem + d_blocks).mean()) if (d_stem+d_blocks).mean() else float('nan')
    print(f"  combined / sum-if-independent   : {ratio:.2f}x")
    results["interaction"] = {"mean": m_i, "ci_lo": lo_i, "ci_hi": hi_i,
                               "d_stem": float(d_stem.mean()), "d_blocks": float(d_blocks.mean()),
                               "d_uniform": float(d_uniform.mean()), "ratio": float(ratio)}

    # ------------------------------------------------------------------ size / benefit summary
    ckpt = torch.load(str(DEFAULT_CKPT), map_location="cpu")
    sd = ckpt["model"]
    total_b = sum(v.numel() * 4 for v in sd.values())
    stem_b = sum(v.numel() * 4 for k, v in sd.items() if k.startswith("stem.0."))
    size_blocks5 = total_b - (total_b - stem_b - sum(
        v.numel() * 4 for k, v in sd.items()
        if k.startswith(("stem.1.", "blocks.0.norm", "blocks.1.norm", "blocks.2.norm",
                         "blocks.3.norm", "band_gru.")))) * 0  # placeholder, use measured below
    print("\n" + "=" * 86)
    print("COST/BENEFIT OF EXEMPTING THE STEM")
    print("=" * 86)
    print(f"  stem is {stem_b} bytes = {100*stem_b/total_b:.3f}% of FP32 model weights")
    d_si = np.mean(np.asarray(per_clip['blocks5']['si_sdr']) - np.asarray(per_clip['uniform']['si_sdr']))
    print(f"  exempting it costs +{stem_b*0.75/1024:.1f} KB and recovers {d_si:+.4f} dB SI-SDR")
    results["stem_fraction_of_weights"] = float(stem_b / total_b)
    results["recovered_db"] = float(d_si)

    with open(out_dir / "phase2b_stem_verification.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out_dir / 'phase2b_stem_verification.json'}")


if __name__ == "__main__":
    main()
