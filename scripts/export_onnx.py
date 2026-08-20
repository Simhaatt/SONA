"""Export the locked GTCRN gain model to ONNX, then apply static INT8 (QDQ) quantization.

Only the neural net (per-band feature -> per-band gain) is exported. The STFT / Bark-band
feature extraction and mask-apply/ISTFT stay in Python/numpy: on real edge hardware that DSP glue
runs in optimized C, so exporting it to ONNX wouldn't tell us anything about the model's edge
cost, and would only complicate parity-checking. This mirrors how the model is actually deployed
in `web_demo` / `realtime/stream_processor.py` -- DSP wrapper + a swappable gain estimator.

Quantization: GTCRN is Conv2d-heavy, and this onnxruntime CPU build has no ConvInteger kernel for
this graph, which is what plain `quantize_dynamic` produces for Conv layers (and dynamic
quantization only touches the tiny Linear head anyway, so it barely shrinks the model). Static
QDQ quantization (QuantizeLinear/DequantizeLinear wrapping, calibrated on real held-out TRAIN
clips -- never the test set used for the paper's quality-parity check) is the standard,
broadly-supported path for CNNs and is what's used here instead.

Usage:
    python scripts/export_onnx.py --checkpoint ".../ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25/last.pt" \\
        --calib-manifest "E:\\Rohit\\FYP\\vbd_local\\train_manifest.csv" --out results/onnx/gtcrn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib-manifest", required=True, help="TRAIN manifest -- calibration data only")
    ap.add_argument("--calib-samples", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "results" / "onnx" / "gtcrn"))
    ap.add_argument("--dummy-frames", type=int, default=250, help="time steps for the export trace")
    ap.add_argument("--dynamo", action="store_true",
                    help="use torch's dynamo ONNX exporter. Off by default: it emits Split with a "
                         "num_outputs attribute that ONNX Runtime 1.20.1 cannot load during "
                         "static quantization.")
    args = ap.parse_args()

    import numpy as np
    import torch

    from src.data import PairDataset
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.config import Config

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config(checkpoint["cfg"])
    tf = build_frontend(cfg)
    model = build_model(cfg, tf)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    n_bands, n_feat = tf.n_bands, tf.n_feat_channels
    dummy = torch.randn(1, args.dummy_frames, n_bands, n_feat, dtype=torch.float32)

    fp32_path = out / "model_fp32.onnx"
    torch.onnx.export(
        model,
        (dummy,),
        str(fp32_path),
        input_names=["feats"],
        output_names=["gain"],
        dynamic_axes={"feats": {1: "time"}, "gain": {1: "time"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=args.dynamo,
    )
    print(f"FP32 ONNX graph written to {fp32_path} ({fp32_path.stat().st_size / 1024:.1f} KB)")

    # --------------------------------------------------------------- calibration data (train set)
    calib_samples = list(PairDataset(manifest=args.calib_manifest, limit=args.calib_samples))
    calib_feats: list[np.ndarray] = []
    with torch.no_grad():
        for s in calib_samples:
            wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
            _, feats = tf.analyze(wav)
            calib_feats.append(feats.numpy())

    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class _FeatsReader(CalibrationDataReader):
        def __init__(self, feats: list[np.ndarray]) -> None:
            self._it = iter({"feats": f} for f in feats)

        def get_next(self):
            return next(self._it, None)

    int8_path = out / "model_int8.onnx"
    quantize_static(
        str(fp32_path),
        str(int8_path),
        calibration_data_reader=_FeatsReader(calib_feats),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        extra_options={"CalibMovingAverage": True},
    )
    print(f"INT8 (QDQ, static) ONNX graph written to {int8_path} "
          f"({int8_path.stat().st_size / 1024:.1f} KB), calibrated on "
          f"{len(calib_feats)} TRAIN clips from {args.calib_manifest}")

    # ------------------------------------------------------------- sanity: PyTorch vs FP32-ONNX
    import onnxruntime as ort

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    onnx_out = sess.run(["gain"], {"feats": dummy.numpy()})[0]
    max_abs_diff = abs(torch_out - onnx_out).max()
    print(f"PyTorch vs FP32-ONNX max abs diff on dummy input: {max_abs_diff:.3e}")
    if max_abs_diff > 1e-3:
        print("WARNING: diff larger than expected for FP32 export -- inspect before trusting benchmarks.")


if __name__ == "__main__":
    main()
