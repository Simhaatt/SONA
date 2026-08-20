"""Phase 1, part B step 1: render enhanced wavs for every variant, for WSL-side PESQ scoring.

Writes PERCEPTUAL_MP/pesq_wavs/{clean,fp32,uniform_int8,only_<layer>}/<uid>.wav for the dev split,
mirroring the layout scripts/score_pesq_wsl.py already expects elsewhere in this project.
"""

from __future__ import annotations

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


def main() -> None:
    # This step only generates audio content -- it isn't a timing measurement (unlike Phase 0's
    # RTF benchmarks, which deliberately pin 1 thread for comparability) -- so use more threads
    # to finish faster under CPU contention from other processes on this machine.
    torch.set_num_threads(4)
    onnx_dir = HERE.parent / "onnx" / "phase0"  # reuse Phase 0's exported variants
    out_dir = HERE.parent / "pesq_wavs"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tf, cfg = QL.load_model(DEFAULT_CKPT)
    samples = QL.load_samples(HERE.parent / "splits" / "dev_manifest.csv", limit=200)
    print(f"rendering {len(samples)} dev clips across variants")

    variant_paths = {
        "uniform_int8": onnx_dir / "uniform_int8.onnx",
        "only_stem": onnx_dir / "only_stem.onnx",
        "only_block0_d1": onnx_dir / "only_block0_d1.onnx",
        "only_block1_d2": onnx_dir / "only_block1_d2.onnx",
        "only_block2_d4": onnx_dir / "only_block2_d4.onnx",
        "only_block3_d8": onnx_dir / "only_block3_d8.onnx",
        "only_head": onnx_dir / "only_head.onnx",
    }
    for name, p in variant_paths.items():
        if not p.exists():
            raise SystemExit(f"missing {p} -- run phase0_probe.py first")

    from src.utils.audio_io import save_audio

    # Resumable: skip any (uid, condition) wav that already exists on disk, so a killed/restarted
    # run picks up where it left off instead of redoing completed variants.
    def _pending(cond_dir: Path) -> list:
        cond_dir.mkdir(exist_ok=True)
        return [s for s in samples if not (cond_dir / f"{s.uid}.wav").exists()]

    # clean + fp32 reference (pytorch model, matches Phase 0's "fp32 reference")
    clean_dir, fp32_dir = out_dir / "clean", out_dir / "fp32"
    clean_dir.mkdir(exist_ok=True)
    pending_fp32 = _pending(fp32_dir)
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in pending_fp32:
            if s.clean is not None and not (clean_dir / f"{s.uid}.wav").exists():
                save_audio(clean_dir / f"{s.uid}.wav", s.clean, s.sr)
            wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
            spec, feats = tf.analyze(wav)
            gain = model(feats)
            enh = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()
            save_audio(fp32_dir / f"{s.uid}.wav", enh, s.sr)
    print(f"  clean + fp32: {len(pending_fp32)} pending, done ({time.perf_counter()-t0:.0f}s)")

    for name, onnx_path in variant_paths.items():
        cond_dir = out_dir / name
        pending = _pending(cond_dir)
        if not pending:
            print(f"  {name:16s} already complete, skipping")
            continue
        sess = QL.make_session(onnx_path, threads=4)
        t0 = time.perf_counter()
        with torch.no_grad():
            for s in pending:
                wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
                spec, feats = tf.analyze(wav)
                gain_np = sess.run(["gain"], {"feats": feats.numpy()})[0]
                gain = torch.from_numpy(gain_np)
                enh = tf.apply_band_gain(spec, gain, length=wav.shape[-1]).squeeze(0).numpy()
                save_audio(cond_dir / f"{s.uid}.wav", enh, s.sr)
        print(f"  {name:16s}: {len(pending)} pending, done ({time.perf_counter()-t0:.0f}s)")

    print(f"\nwrote wavs to {out_dir}")


if __name__ == "__main__":
    main()
