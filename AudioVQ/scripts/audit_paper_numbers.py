"""Audit: verify every quantitative claim in PAPER_DRAFT.md against its source result file.

Numbers get copied by hand into prose and tables, and a single transcription slip invalidates a
claim while looking perfectly plausible. This re-derives the headline figures straight from the
CSV/JSON artefacts and compares them to what the draft asserts, so any drift is caught before
submission rather than by a reviewer.

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, claimed: float, actual: float | None, tol: float = 0.005) -> None:
    global CHECKS
    CHECKS += 1
    if actual is None:
        FAILURES.append(f"{label}: no source value found (claimed {claimed})")
        print(f"  MISSING  {label:52s} claimed {claimed}")
        return
    ok = abs(claimed - actual) <= tol
    status = "ok" if ok else "MISMATCH"
    print(f"  {status:8s} {label:52s} claimed {claimed:<9.4f} source {actual:.4f}")
    if not ok:
        FAILURES.append(f"{label}: draft says {claimed}, source says {actual:.4f}")


def bench(model: str) -> list[dict]:
    p = RES / "retrieval" / f"benchmark_{model}.csv"
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("bits_per_vector", "recall@10", "build_search_s"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def rec(model: str, method: str) -> float | None:
    r = next((x for x in bench(model) if x["method"] == method), None)
    return r["recall@10"] if r else None


def sem(model: str, task: str, method: str) -> float | None:
    p = RES / "semantic" / f"semantic_{model}.csv"
    if not p.exists():
        return None
    for r in csv.DictReader(open(p)):
        if r["task"] == task and r["method"] == method:
            return float(r["precision_at_k"])
    return None


def ani(model: str, *path):
    p = RES / "anisotropy" / f"anisotropy_{model}_database.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    for k in path:
        d = d.get(k, {}) if isinstance(d, dict) else None
        if d is None:
            return None
    return d if isinstance(d, (int, float)) else None


def scale(model: str, n: int, col: str) -> float | None:
    p = RES / "scale" / f"scale_{model}.csv"
    if not p.exists():
        return None
    for r in csv.DictReader(open(p)):
        if int(r["n"]) == n:
            return float(r[col])
    return None


print("=" * 84)
print("AUDIT: paper claims vs source artefacts")
print("=" * 84)

print("\n[§5.1] anisotropy")
check("WavLM mean pairwise cosine", 0.830, ani("wavlm_base_plus", "raw", "mean_pairwise_cosine"))
check("wav2vec2 mean pairwise cosine", 0.929, ani("wav2vec2_base", "raw", "mean_pairwise_cosine"))
check("HuBERT mean pairwise cosine", 0.853, ani("hubert_base", "raw", "mean_pairwise_cosine"))
check("WavLM effective dim frac", 0.029, ani("wavlm_base_plus", "raw", "effective_dim_frac"), 0.002)
check("wav2vec2 effective dim frac", 0.014, ani("wav2vec2_base", "raw", "effective_dim_frac"), 0.002)
check("HuBERT effective dim frac", 0.021, ani("hubert_base", "raw", "effective_dim_frac"), 0.002)

print("\n[§5.2] within-vector crest (raw)")
check("WavLM crest raw", 10.22, ani("wavlm_base_plus", "within_vector", "raw", "crest_mean"), 0.05)
check("wav2vec2 crest raw", 3.62, ani("wav2vec2_base", "within_vector", "raw", "crest_mean"), 0.05)
check("HuBERT crest raw", 12.53, ani("hubert_base", "within_vector", "raw", "crest_mean"), 0.05)

print("\n[§5.4] matched-budget Recall@10, data-oblivious")
check("WavLM oblivious 1b", 0.4829, rec("wavlm_base_plus", "scalar-1b-rot-gaussian"))
check("WavLM oblivious 4b", 0.8791, rec("wavlm_base_plus", "scalar-4b-rot-gaussian"))
check("wav2vec2 oblivious 1b", 0.5299, rec("wav2vec2_base", "scalar-1b-rot-gaussian"))
check("wav2vec2 oblivious 4b", 0.8965, rec("wav2vec2_base", "scalar-4b-rot-gaussian"))
check("HuBERT oblivious 1b", 0.6982, rec("hubert_base", "scalar-1b-rot-gaussian"))
check("HuBERT oblivious 4b", 0.9212, rec("hubert_base", "scalar-4b-rot-gaussian"))

print("\n[§5.4] OPQ baselines")
check("WavLM OPQ m96", 0.7260, rec("wavlm_base_plus", "OPQ-m96-8b"))
check("WavLM OPQ m384", 0.8990, rec("wavlm_base_plus", "OPQ-m384-8b"))
check("wav2vec2 OPQ m384", 0.8472, rec("wav2vec2_base", "OPQ-m384-8b"))

print("\n[§5.3] semantic retention at 1 bit/coord")
for model, lbl in [("wavlm_base_plus", "WavLM"), ("wav2vec2_base", "wav2vec2"),
                   ("hubert_base", "HuBERT")]:
    for task in ("speaker", "noise_type"):
        f32 = sem(model, task, "float32")
        q = sem(model, task, "scalar-1b+rotation")
        if f32 and q:
            print(f"  derived  {lbl} {task} retention = {q/f32:.1%} "
                  f"(float32 {f32:.4f} -> {q:.4f})")

print("\n[§5.5] build cost")
r_opq = next((x for x in bench("wavlm_base_plus") if x["method"] == "OPQ-m384-8b"), None)
r_obl = next((x for x in bench("wavlm_base_plus") if x["method"] == "scalar-4b-rot-gaussian"), None)
if r_opq and r_obl:
    ratio = r_opq["build_search_s"] / r_obl["build_search_s"]
    check("OPQ/oblivious build-time ratio (claimed 434x)", 434.0, ratio, 40.0)

print("\n[§5.6] scale study gaps at N=1000 and N=10000")
for model, lbl in [("wavlm_base_plus", "WavLM"), ("wav2vec2_base", "wav2vec2"),
                   ("hubert_base", "HuBERT")]:
    for n in (1000, 10000):
        g = scale(model, n, "gap")
        if g is not None:
            print(f"  derived  {lbl} gap at N={n:<6d} = {g:+.4f}")

print("\n" + "=" * 84)
if FAILURES:
    print(f"AUDIT FAILED: {len(FAILURES)} of {CHECKS} checks disagree with source data")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print(f"AUDIT PASSED: all {CHECKS} checked claims match their source artefacts")
