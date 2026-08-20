"""Phase D prep -- add the quantization comparison to the MOS listening test.

The listening test already covers magnitude vs complex mask, which is where SI-SDR and PESQ
contradict each other. It does not yet cover the other place a human verdict would settle
something: the claim that the stem-exempt mixed allocation is *indistinguishable from FP32*.

That claim currently rests on p = 0.25 on SI-SDR. A p-value above threshold is weak evidence of
sameness -- it is a failure to reject, not a demonstration. If listeners also cannot tell mixed
from FP32 while they *can* tell uniform INT8 apart, the claim becomes a positive result rather than
an absence of one, and that is the natural headline for the quantization study.

Renders the utterances already chosen for the test through three ONNX variants, level-matched the
same way, and appends them to the existing manifest so raters hear consistent material.

Requires onnxruntime, so on this machine run it under Python 3.12:

    py -3.12 scripts/add_mos_quant_conditions.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

VARIANTS = {
    "quant_fp32": "model_fp32.onnx",
    "quant_mixed": "model_mixed.onnx",
    "quant_uniform_int8": "model_int8.onnx",
}


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _match_level(wav: np.ndarray, target_rms: float) -> np.ndarray:
    cur = _rms(wav)
    if cur <= 0:
        return wav
    out = wav * (target_rms / cur)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default=str(ROOT / "results" / "onnx" / "gtcrn_variants"))
    ap.add_argument("--checkpoint", default=str(
        ROOT / "results" / "kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07" /
        "extracted" / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25" / "last.pt"))
    ap.add_argument("--manifest", default=str(ROOT / "vbd_local" / "test_manifest.csv"))
    ap.add_argument("--mos-dir", default=str(ROOT / "mos_test"))
    args = ap.parse_args()

    import onnxruntime as ort
    import torch

    from src.data import PairDataset
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.config import Config
    from src.utils.audio_io import save_audio

    mos = Path(args.mos_dir)
    audio_dir = mos / "audio"
    man_path = mos / "clip_manifest.csv"
    rows = list(csv.DictReader(man_path.open(newline="", encoding="utf-8")))
    existing = {r["condition"] for r in rows}
    if any(c in existing for c in VARIANTS):
        raise SystemExit("quantization conditions already present; nothing to do")

    # the utterances the test already uses, in order of first appearance
    wanted: list[str] = []
    for r in rows:
        if r["uid"] not in wanted and not int(r["is_attention_check"]):
            wanted.append(r["uid"])
    print(f"existing manifest: {len(rows)} clips over {len(wanted)} utterances")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(ckpt["cfg"])
    tf = build_frontend(cfg)
    model = build_model(cfg, tf)
    model.load_state_dict(ckpt["model"])
    model.eval()

    onnx_dir = Path(args.onnx_dir)
    sess = {}
    for name, fn in VARIANTS.items():
        p = onnx_dir / fn
        if not p.exists():
            raise SystemExit(f"missing {p}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess[name] = ort.InferenceSession(str(p), sess_options=so,
                                          providers=["CPUExecutionProvider"])
        print(f"  {name:20s} {p.stat().st_size/1024:8.1f} KB")

    by_uid = {s.uid: s for s in PairDataset(manifest=args.manifest, limit=None)
              if s.uid in set(wanted)}
    missing = [u for u in wanted if u not in by_uid]
    if missing:
        raise SystemExit(f"utterances not found in manifest: {missing}")

    counter = max(int(r["clip_id"][1:]) for r in rows)
    new_rows = []
    for uid in wanted:
        s = by_uid[uid]
        wav = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            spec, feats = tf.analyze(wav)
        feats_np = feats.numpy()
        target = _rms(s.noisy)
        for cond, _ in VARIANTS.items():
            gain = sess[cond].run(["gain"], {"feats": feats_np})[0]
            with torch.no_grad():
                enh = tf.apply_band_gain(spec, torch.from_numpy(gain),
                                         length=wav.shape[-1]).squeeze(0).numpy()
            counter += 1
            clip_id = f"c{counter:03d}"
            save_audio(audio_dir / f"{clip_id}.wav", _match_level(enh, target), s.sr)
            new_rows.append({"clip_id": clip_id, "uid": uid, "condition": cond,
                             "is_attention_check": 0})
        print(f"  rendered {uid}")

    all_rows = rows + new_rows
    with man_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "uid", "condition", "is_attention_check"])
        w.writeheader()
        w.writerows(all_rows)

    print(f"\nadded {len(new_rows)} clips -> {len(all_rows)} total")
    print("conditions now:", sorted({r['condition'] for r in all_rows}))
    print("\nThe comparison that matters: quant_fp32 vs quant_mixed should be "
          "indistinguishable;\nquant_uniform_int8 should be rated worse. If listeners separate "
          "mixed from FP32,\nthe indistinguishability claim is weaker than the p-value suggests "
          "and must be softened.")


if __name__ == "__main__":
    main()
