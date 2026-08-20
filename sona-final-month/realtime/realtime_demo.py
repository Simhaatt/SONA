"""SONA realtime enhancement demo.

Three run modes, all driven by the same block-streaming core (stream_processor):

    # live microphone -> speaker (needs `pip install sounddevice`)
    python realtime/realtime_demo.py --mode live

    # offline: stream a wav through the realtime path, write enhanced wav
    python realtime/realtime_demo.py --mode wav --in noisy.wav --out enhanced.wav

    # self-test: synthetic tone through the pipeline, prints latency/RTF
    python realtime/realtime_demo.py --mode selftest

The enhancer is the locked GTCRN checkpoint loaded via the main repo's
`src.controller.load_neural_enhancer` (array-in / array-out). If torch or the
checkpoint is unavailable, it falls back to a light spectral noise-gate so the
demo path still runs end-to-end (clearly logged).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_processor import StreamProcessor  # noqa: E402

DEFAULT_SR = 16000
REPO = Path(__file__).resolve().parents[2] / "noise cancellation"
CHECKPOINT = (REPO / "results" / "kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07"
              / "extracted" / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25" / "last.pt")


# ---------------------------------------------------------------- enhancer ----
def spectral_gate(block: np.ndarray, sr: int = DEFAULT_SR) -> np.ndarray:
    """Fallback: soft over-subtraction noise gate on a single block."""
    n_fft = 512
    if block.shape[0] < n_fft:
        block = np.pad(block, (0, n_fft - block.shape[0]))
    win = np.hanning(len(block))
    spec = np.fft.rfft(block * win)
    mag = np.abs(spec)
    floor = np.percentile(mag, 20)
    gain = np.clip((mag - floor) / (mag + 1e-9), 0.05, 1.0)
    out = np.fft.irfft(spec * gain, n=len(block))
    return (out / (win + 1e-6)).astype(np.float64)[: block.shape[0]]


def make_enhancer(sr: int = DEFAULT_SR, force_fallback: bool = False):
    """Return (process_fn, label). process_fn: block(np) -> block(np)."""
    if not force_fallback:
        try:
            if str(REPO) not in sys.path:
                sys.path.insert(0, str(REPO))
            from src.controller import load_neural_enhancer  # type: ignore
            if not CHECKPOINT.exists():
                raise FileNotFoundError(f"checkpoint missing: {CHECKPOINT}")
            enh = load_neural_enhancer(CHECKPOINT, name="gtcrn_ch96_h96_b4_mw1p25_mr0p25")

            def _proc(block: np.ndarray) -> np.ndarray:
                y = enh.enhance(block.astype(np.float32), sr=sr)
                return np.asarray(y, dtype=np.float64)

            return _proc, f"GTCRN checkpoint ({enh.device})"
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[warn] GTCRN unavailable ({exc}); using spectral-gate fallback")
    return (lambda b: spectral_gate(b, sr)), "spectral-gate fallback"


# ------------------------------------------------------------------- modes ----
def run_wav(in_path: Path, out_path: Path, block: int, hop: int, sr: int,
            force_fallback: bool = False) -> dict:
    from scipy.io import wavfile
    rate, data = wavfile.read(in_path)
    x = data.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        x = x / np.iinfo(data.dtype).max
    process, label = make_enhancer(sr=rate, force_fallback=force_fallback)
    sp = StreamProcessor(block_size=block, hop_size=hop, process=process)
    out = []
    chunk = hop  # emulate a soundcard delivering hop-sized frames
    t0 = time.perf_counter()
    for i in range(0, len(x), chunk):
        for y in sp.push(x[i : i + chunk]):
            out.append(y)
    out.append(sp.flush())
    wall = time.perf_counter() - t0
    y = np.concatenate([np.asarray(o) for o in out]) if out else np.zeros(0)
    peak = np.max(np.abs(y)) or 1.0
    y = (y / peak * 0.98)
    wavfile.write(out_path, rate, (y * 32767).astype(np.int16))
    audio_s = len(x) / rate
    stats = {
        "enhancer": label, "in": str(in_path), "out": str(out_path),
        "audio_seconds": round(audio_s, 3), "wall_seconds": round(wall, 3),
        "realtime_factor": round(wall / max(audio_s, 1e-9), 4),
        "algorithmic_latency_ms": round(1000.0 * block / rate, 1),
        "blocks": sp.n_blocks,
    }
    return stats


def run_selftest(block: int, hop: int, sr: int) -> dict:
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.default_rng(0).standard_normal(len(t))
    process, label = make_enhancer(sr=sr, force_fallback=True)
    sp = StreamProcessor(block_size=block, hop_size=hop, process=process)
    t0 = time.perf_counter()
    n_out = 0
    for i in range(0, len(x), hop):
        for y in sp.push(x[i : i + hop]):
            n_out += len(y)
    n_out += len(sp.flush())
    wall = time.perf_counter() - t0
    audio_s = len(x) / sr
    return {
        "enhancer": label, "audio_seconds": round(audio_s, 3),
        "wall_seconds": round(wall, 3),
        "realtime_factor": round(wall / audio_s, 4),
        "algorithmic_latency_ms": round(1000.0 * block / sr, 1),
        "output_samples": n_out, "blocks": sp.n_blocks,
    }


def run_live(block: int, hop: int, sr: int, seconds: float, force_fallback: bool) -> dict:
    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "live mode needs sounddevice: pip install sounddevice "
            f"(import failed: {exc})")
    process, label = make_enhancer(sr=sr, force_fallback=force_fallback)
    sp = StreamProcessor(block_size=block, hop_size=hop, process=process)
    print(f"[live] enhancer={label} sr={sr} block={block} hop={hop} "
          f"latency~{1000.0*block/sr:.0f}ms — Ctrl+C to stop")

    def callback(indata, outdata, frames, time_info, status):  # pragma: no cover
        if status:
            print(status, file=sys.stderr)
        mono = indata[:, 0].astype(np.float64)
        produced = []
        for y in sp.push(mono):
            produced.append(y)
        buf = np.concatenate(produced) if produced else np.zeros(frames)
        if len(buf) < frames:
            buf = np.pad(buf, (0, frames - len(buf)))
        outdata[:, 0] = buf[:frames]

    with sd.Stream(samplerate=sr, blocksize=hop, channels=1, callback=callback):
        sd.sleep(int(seconds * 1000))
    return {"enhancer": label, "seconds": seconds, "realtime_factor": round(sp.realtime_factor, 4)}


def main():
    ap = argparse.ArgumentParser(description="SONA realtime enhancement demo")
    ap.add_argument("--mode", choices=["live", "wav", "selftest"], default="selftest")
    ap.add_argument("--in", dest="in_path", type=Path)
    ap.add_argument("--out", dest="out_path", type=Path, default=Path("enhanced.wav"))
    ap.add_argument("--block", type=int, default=1024, help="block size (samples)")
    ap.add_argument("--hop", type=int, default=256, help="hop size (samples)")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--seconds", type=float, default=15.0, help="live capture length")
    ap.add_argument("--fallback", action="store_true", help="force spectral-gate fallback")
    args = ap.parse_args()

    if args.mode == "wav":
        if not args.in_path:
            raise SystemExit("--in is required for --mode wav")
        stats = run_wav(args.in_path, args.out_path, args.block, args.hop, args.sr, args.fallback)
    elif args.mode == "selftest":
        stats = run_selftest(args.block, args.hop, args.sr)
    else:
        stats = run_live(args.block, args.hop, args.sr, args.seconds, args.fallback)

    print("\n== realtime demo stats ==")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
