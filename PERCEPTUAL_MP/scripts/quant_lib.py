"""Shared helpers for the perceptually-guided mixed-precision quantization study.

Wraps the pieces that every phase needs: load the frozen checkpoint, export an ONNX variant with
a chosen per-layer precision map, and score a variant over a manifest through the SAME DSP
front-end used everywhere else in the repo (so only the gain-estimation backend differs, exactly
as in the paper's fair-comparison design).

Reused by phase0_probe.py and (later) quant_sensitivity.py / quant_allocate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

# The 7 weight-bearing layers in the exported GTCRN graph, in forward order.
LAYERS = [
    "/stem/stem.0/Conv",
    "/blocks.0/conv/Conv",
    "/blocks.1/conv/Conv",
    "/blocks.2/conv/Conv",
    "/blocks.3/conv/Conv",
    "/band_gru/GRU",
    "/head/MatMul",
]

# ORT 1.20.1 does NOT quantize GRU: requesting it yields 0 QDQ nodes, no size change and no
# speedup (verified in phase0_probe 0b/0d). So the band GRU is permanently FP32 on this runtime
# and is excluded from the sensitivity study and the allocation search. This is itself a
# reportable finding, not a bug to work around.
UNQUANTIZABLE = ["/band_gru/GRU"]
QUANTIZABLE = [n for n in LAYERS if n not in UNQUANTIZABLE]

SHORT = {
    "/stem/stem.0/Conv": "stem",
    "/blocks.0/conv/Conv": "block0_d1",
    "/blocks.1/conv/Conv": "block1_d2",
    "/blocks.2/conv/Conv": "block2_d4",
    "/blocks.3/conv/Conv": "block3_d8",
    "/band_gru/GRU": "band_gru",
    "/head/MatMul": "head",
}


# ------------------------------------------------------------------------------- model loading
def load_model(ckpt_path: str | Path, device: str = "cpu"):
    """Return (model, torch_frontend, cfg) for the frozen checkpoint, in eval mode."""
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.config import Config

    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = Config(ckpt["cfg"])
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tf, cfg


def load_samples(manifest: str | Path, limit: int | None = None):
    from src.data import PairDataset

    return list(PairDataset(manifest=str(manifest), limit=limit))


# ------------------------------------------------------------------------------- ONNX export
def export_fp32(model, tf, out_path: str | Path, n_frames: int = 250) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, n_frames, tf.n_bands, tf.n_feat_channels, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["feats"],
        output_names=["gain"],
        dynamic_axes={"feats": {1: "time"}, "gain": {1: "time"}},
        opset_version=17,
        do_constant_folding=True,
    )
    return out_path


def calib_features(tf, samples) -> list[np.ndarray]:
    """Per-band feature tensors for calibration clips (calibration split ONLY -- never dev/test)."""
    feats = []
    with torch.no_grad():
        for s in samples:
            wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
            _, f = tf.analyze(wav)
            feats.append(f.numpy())
    return feats


def _reader(calib_feats: list[np.ndarray]):
    """Fresh CalibrationDataReader -- the ORT API consumes it once, so build one per call."""
    from onnxruntime.quantization import CalibrationDataReader

    class _FeatsReader(CalibrationDataReader):
        def __init__(self, feats):
            self._it = iter({"feats": f} for f in feats)

        def get_next(self):
            return next(self._it, None)

    return _FeatsReader(calib_feats)


def quantize_variant(
    fp32_path: str | Path,
    out_path: str | Path,
    calib_feats: list[np.ndarray],
    nodes_to_quantize: list[str] | None = None,
    weight_bits: int = 8,
    activation_bits: int = 8,
    per_channel: bool = False,
) -> Path:
    """Static QDQ quantization of a chosen subset of layers.

    `nodes_to_quantize=None` means "every quantizable layer" (uniform). Passing a list restricts
    quantization to exactly those nodes, which is what makes the per-layer sensitivity study and
    the mixed-precision allocation possible.
    """
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    def _qtype(bits: int):
        if bits == 8:
            return QuantType.QInt8
        if bits == 4:
            return QuantType.QInt4
        raise ValueError(f"unsupported bit-width {bits}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        calibration_data_reader=_reader(calib_feats),
        quant_format=QuantFormat.QDQ,
        weight_type=_qtype(weight_bits),
        activation_type=_qtype(activation_bits),
        per_channel=per_channel,
        extra_options={"CalibMovingAverage": True},
    )
    if nodes_to_quantize is not None:
        kwargs["nodes_to_quantize"] = list(nodes_to_quantize)
    quantize_static(str(fp32_path), str(out_path), **kwargs)
    return out_path


def make_session(path: str | Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if threads is not None:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


# ------------------------------------------------------------------------------- evaluation
def eval_variant(tf, samples, backend, label: str) -> list[dict]:
    """Score one gain-estimation backend over `samples`, returning per-clip metric rows.

    `backend` is either a torch model (FP32 reference) or an onnxruntime InferenceSession.
    The DSP path (STFT -> features -> mask apply -> ISTFT) is identical either way.
    Also returns the raw predicted mask so callers can compute mask-domain MSE without re-running.
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
            enh = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()

            row = {"method": label, "uid": s.uid, "_mask": gain_np}
            if s.clean is not None:
                row["si_sdr"] = si_sdr(s.clean, enh)
                row["stoi"] = stoi_metric(s.clean, enh, s.sr, extended=False)
                row["estoi"] = stoi_metric(s.clean, enh, s.sr, extended=True)
                row["lsd"] = lsd(s.clean, enh)
            rows.append(row)
    return rows


def paired_deltas(ref_rows: list[dict], var_rows: list[dict], keys=("si_sdr", "stoi", "estoi", "lsd")):
    """Per-clip (variant - reference) deltas, plus mask-domain MSE. Assumes same clip order."""
    ref_by = {r["uid"]: r for r in ref_rows}
    out = []
    for v in var_rows:
        r = ref_by.get(v["uid"])
        if r is None:
            continue
        d = {"uid": v["uid"]}
        for k in keys:
            if k in r and k in v:
                d[f"d_{k}"] = v[k] - r[k]
        m_ref, m_var = r.get("_mask"), v.get("_mask")
        if m_ref is not None and m_var is not None and m_ref.shape == m_var.shape:
            d["mask_mse"] = float(np.mean((m_ref - m_var) ** 2))
        out.append(d)
    return out


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 1234):
    """Mean and percentile bootstrap CI. Returns (mean, lo, hi); (None,)*3 if empty."""
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return None, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(vals.mean()), float(lo), float(hi)


def model_size_kb(path: str | Path) -> float:
    return Path(path).stat().st_size / 1024.0
