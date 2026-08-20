"""Edge-simulated benchmark: PyTorch vs FP32-ONNX vs INT8-ONNX gain-estimation backend.

This is *edge-representative*, not an on-device measurement: no Raspberry Pi / ARM hardware was
used (see NEXT_STEPS_PLAN.md -- explicitly cut this cycle). Instead it approximates a
constrained-core edge device by pinning ONNX Runtime's intra-op thread count to 1 and 2 on this
CPU. Report results as such, not as real Pi numbers.

Only the neural-net gain-estimation step is swapped between backends (PyTorch vs ONNX); the STFT /
Bark-band feature extraction and mask-apply/ISTFT are the identical `TorchFrontend` DSP path used
everywhere else in the harness, so RTF differences are attributable to the model backend alone,
and the quality-parity check isolates quantization error from DSP differences.

Usage:
    python scripts/bench_onnx_edge.py --checkpoint ".../last.pt" \\
        --onnx-dir results/onnx/gtcrn --variants fp32,int8,mixed \\
        --manifest vbd_local/test_manifest.csv --limit 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--onnx-dir", required=True, help="dir holding model_<variant>.onnx files")
    ap.add_argument("--variants", default="fp32,int8",
                    help="comma-separated variant names; each needs model_<name>.onnx in --onnx-dir")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--threads", default="1,2", help="ONNX intra-op thread counts to benchmark")
    ap.add_argument("--out", default=str(ROOT / "results" / "onnx" / "edge_bench"))
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    import torch

    from src.data import PairDataset
    from src.eval.metrics import si_sdr
    from src.eval.report import save_rows_csv
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.config import Config

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_dir = Path(args.onnx_dir)

    torch.set_num_threads(1)  # keep the PyTorch CPU reference comparable to the 1-thread ONNX run

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config(checkpoint["cfg"])
    tf = build_frontend(cfg)
    model = build_model(cfg, tf)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    samples = list(PairDataset(manifest=args.manifest, limit=(args.limit or None)))
    if not samples:
        raise SystemExit(f"No samples found in {args.manifest}")

    thread_counts = [int(x) for x in args.threads.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    missing = [v for v in variants if not (onnx_dir / f"model_{v}.onnx").exists()]
    if missing:
        raise SystemExit(
            "Missing ONNX file(s) for variant(s) " + ", ".join(missing) +
            f" -- expected model_<name>.onnx in {onnx_dir}"
        )
    sessions: dict[str, dict[int, ort.InferenceSession]] = {v: {} for v in variants}
    for kind in variants:
        path = onnx_dir / f"model_{kind}.onnx"
        for n in thread_counts:
            so = ort.SessionOptions()
            so.intra_op_num_threads = n
            sessions[kind][n] = ort.InferenceSession(
                str(path), sess_options=so, providers=["CPUExecutionProvider"]
            )
    model_size_kb = {
        kind: (onnx_dir / f"model_{kind}.onnx").stat().st_size / 1024 for kind in variants
    }

    def run_backend(feats_np: np.ndarray, backend: str, threads: int | None = None):
        if backend == "pytorch":
            with torch.no_grad():
                t0 = time.perf_counter()
                gain = model(torch.from_numpy(feats_np))
                elapsed = time.perf_counter() - t0
            return gain, elapsed
        sess = sessions[backend][threads]
        t0 = time.perf_counter()
        gain = sess.run(["gain"], {"feats": feats_np})[0]
        elapsed = time.perf_counter() - t0
        return torch.from_numpy(gain), elapsed

    rows: list[dict] = []
    for s in samples:
        wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            spec, feats = tf.analyze(wav)
        feats_np = feats.numpy()
        audio_dur = len(s.noisy) / s.sr

        backends = [("pytorch", None)] + [
            (kind, n) for kind in variants for n in thread_counts
        ]
        for backend, threads in backends:
            gain, elapsed = run_backend(feats_np, backend, threads)
            with torch.no_grad():
                enhanced = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()
            row = {
                "backend": backend,
                "threads": threads if threads is not None else "-",
                "uid": s.uid,
                "rtf": elapsed / audio_dur if audio_dur > 0 else float("nan"),
                "latency_ms": elapsed * 1000.0,
            }
            if s.clean is not None:
                row["si_sdr"] = si_sdr(s.clean, enhanced)
            rows.append(row)

    save_rows_csv(rows, out / "per_sample.csv")

    # --------------------------------------------------------------------------- summary
    import numpy as np

    def _mean(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    summary_rows = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["backend"], r["threads"])
        groups.setdefault(key, []).append(r)
    for (backend, threads), grp in groups.items():
        summary_rows.append(
            {
                "backend": backend,
                "threads": threads,
                "n": len(grp),
                "mean_rtf": _mean([g["rtf"] for g in grp]),
                "p95_latency_ms": float(np.percentile([g["latency_ms"] for g in grp], 95)),
                "mean_latency_ms": _mean([g["latency_ms"] for g in grp]),
                "mean_si_sdr": _mean([g.get("si_sdr") for g in grp]),
                "model_size_kb": model_size_kb.get(backend, None),
            }
        )
    save_rows_csv(summary_rows, out / "summary.csv")

    print("\n=== Edge-simulated benchmark summary ===\n")
    header = f"{'backend':10s} {'threads':7s} {'n':>5s} {'mean_rtf':>10s} {'p95_lat_ms':>11s} {'mean_si_sdr':>12s} {'size_kb':>9s}"
    print(header)
    for r in sorted(summary_rows, key=lambda r: (r["backend"], str(r["threads"]))):
        size = f"{r['model_size_kb']:.1f}" if r["model_size_kb"] else "-"
        sisdr = f"{r['mean_si_sdr']:.3f}" if r["mean_si_sdr"] is not None else "-"
        print(
            f"{r['backend']:10s} {str(r['threads']):7s} {r['n']:5d} {r['mean_rtf']:10.5f} "
            f"{r['p95_latency_ms']:11.3f} {sisdr:>12s} {size:>9s}"
        )
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
