"""Run DeepFilterNet2 inference over a manifest, standalone in the py3.11 DeepFilterNet2 venv.

DeepFilterNet2 operates at 48 kHz internally; VoiceBank-DEMAND/MS-SNSD clips here are 16 kHz.
Resamples noisy -> 48k -> enhance -> resample back to 16k, so the output is directly comparable
to every other method's output in the shared eval harness (which all run at 16 kHz).

Deliberately minimal deps (torch, torchaudio, df, soundfile) -- this venv does NOT have the main
repo's src/ package. Run in the py3.11 venv:

    E:\\Rohit\\FYP\\noise cancellation\\.venv_dfn2\\Scripts\\python.exe \\
        scripts\\run_deepfilternet2_inference.py --manifest "E:\\...\\test_manifest.csv" \\
        --out results\\deepfilternet2_voicebank
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with clean,noisy columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 means full manifest")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    import torchaudio
    from df.enhance import enhance, init_df

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, df_state, _ = init_df(default_model="DeepFilterNet2")
    model_sr = df_state.sr()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]
    print(f"processing {len(rows)} clips, DeepFilterNet2 @ {model_sr} Hz")

    timing_rows = []
    for i, row in enumerate(rows):
        noisy_path = Path(row["noisy"])
        uid = noisy_path.stem
        out_path = out / f"{uid}.wav"
        if out_path.exists():
            continue

        wav, sr = torchaudio.load(str(noisy_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != model_sr:
            wav = torchaudio.functional.resample(wav, sr, model_sr)

        t0 = time.perf_counter()
        with torch.no_grad():
            enhanced = enhance(model, df_state, wav)
        elapsed = time.perf_counter() - t0

        enhanced_16k = torchaudio.functional.resample(enhanced, model_sr, 16000)
        sf.write(str(out_path), enhanced_16k.squeeze(0).numpy(), 16000)

        audio_dur_s = wav.shape[-1] / model_sr
        timing_rows.append({"uid": uid, "rtf": elapsed / audio_dur_s if audio_dur_s > 0 else None})

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}")

    if timing_rows:
        rtfs = [r["rtf"] for r in timing_rows if r["rtf"] is not None]
        mean_rtf = sum(rtfs) / len(rtfs) if rtfs else float("nan")
        print(f"processed {len(timing_rows)} new clips, mean RTF (CPU, this venv) = {mean_rtf:.4f}")
        with open(out / "timing.csv", "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["uid", "rtf"])
            if fh.tell() == 0:
                w.writeheader()
            w.writerows(timing_rows)
    else:
        print("all clips already present, nothing to do")

    print(f"wrote enhanced wavs to {out}")


if __name__ == "__main__":
    main()
