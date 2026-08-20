"""Build the embedding banks used by every downstream experiment.

Banks produced:
  database   MS-SNSD clean_train subset -- the retrieval index
  queries    MS-SNSD clean_test         -- held out from the database, used as ANN queries
  speakers   VoiceBank-DEMAND test      -- carries speaker IDs, for semantic retrieval
  noisy      MS-SNSD noisy mixtures     -- carries noise_type + SNR labels, for semantic retrieval

Database and queries come from disjoint MS-SNSD splits, so recall is never measured against
vectors the query itself contributed to.

Usage:
    python scripts/extract_embeddings.py --n-database 10000 --model wavlm_base_plus
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embeddings import embed_files, load_model, save_bank  # noqa: E402

from src.config import MS_SNSD, VBD_TEST_CLEAN  # noqa: E402
from src.config import NOISY_MANIFEST as MS_SNSD_NOISY_MANIFEST  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    ap.add_argument("--n-database", type=int, default=10000)
    ap.add_argument("--n-queries", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "embeddings"))
    ap.add_argument("--only", default="", help="comma-separated subset of banks to build")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.model
    want = {s.strip() for s in args.only.split(",") if s.strip()}

    model, device, model_sr = load_model(args.model)
    print(f"model={args.model} device={device} sr={model_sr}")

    def wanted(n: str) -> bool:
        return (not want) or (n in want)

    # ------------------------------------------------------------------ database (index)
    if wanted("database"):
        files = sorted(MS_SNSD.glob("clean_train/*.wav"))[: args.n_database]
        print(f"\n[database] MS-SNSD clean_train: {len(files)} clips")
        vecs = embed_files(files, model, device, model_sr)
        save_bank(out_dir, "database", vecs, [{"uid": f.stem, "source": "ms_snsd_clean_train"}
                                               for f in files])

    # ------------------------------------------------------------------ queries
    if wanted("queries"):
        files = sorted(MS_SNSD.glob("clean_test/*.wav"))[: args.n_queries]
        print(f"\n[queries] MS-SNSD clean_test: {len(files)} clips")
        vecs = embed_files(files, model, device, model_sr)
        save_bank(out_dir, "queries", vecs, [{"uid": f.stem, "source": "ms_snsd_clean_test"}
                                              for f in files])

    # ------------------------------------------------- speakers (VoiceBank, has speaker IDs)
    if wanted("speakers"):
        files = sorted(VBD_TEST_CLEAN.glob("*.wav"))
        print(f"\n[speakers] VoiceBank-DEMAND test clean: {len(files)} clips")
        vecs = embed_files(files, model, device, model_sr)
        meta = []
        for f in files:
            m = re.match(r"(p\d+)", f.stem)
            meta.append({"uid": f.stem, "speaker": m.group(1) if m else "unknown",
                          "source": "voicebank_test_clean"})
        save_bank(out_dir, "speakers", vecs, meta)

    # ------------------------------------------- noisy (MS-SNSD mixtures, noise_type + SNR)
    if wanted("noisy") and MS_SNSD_NOISY_MANIFEST.exists():
        with open(MS_SNSD_NOISY_MANIFEST, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if Path(r["noisy"]).exists()]
        print(f"\n[noisy] MS-SNSD noisy mixtures: {len(rows)} clips")
        files = [Path(r["noisy"]) for r in rows]
        vecs = embed_files(files, model, device, model_sr)
        meta = [{"uid": Path(r["noisy"]).stem, "noise_type": r.get("noise_type", "unknown"),
                  "snr_db": r.get("snr_db", ""), "source": "ms_snsd_noisy"} for r in rows]
        save_bank(out_dir, "noisy", vecs, meta)

    print(f"\nall banks written to {out_dir}")


if __name__ == "__main__":
    main()
