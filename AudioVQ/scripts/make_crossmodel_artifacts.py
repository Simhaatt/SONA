"""Cross-model figures and consolidated LaTeX tables for the manuscript.

Fig 7  RQ1 curves for both embedding models side by side
Fig 8  the mechanism result: rotation's benefit vs raw within-vector crest factor
Table 1  structural properties of the two embedding spaces
Table 2  matched-budget retrieval, both models
Table 3  geometry vs semantics, both models
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [("wavlm_base_plus", "WavLM-Base+"), ("wav2vec2_base", "wav2vec2-base"),
          ("hubert_base", "HuBERT-base")]
C_OBL, C_PQ, C_OPQ, C_NOROT = "#c1121f", "#2a6f97", "#0b525b", "#adb5bd"


def load_bench(model):
    p = ROOT / "results" / "retrieval" / f"benchmark_{model}.csv"
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("bits_per_vector", "recall@1", "recall@10", "recall@100", "build_search_s"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def load_sem(model):
    p = ROOT / "results" / "semantic" / f"semantic_{model}.csv"
    if not p.exists():
        return []
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("bits", "precision_at_k", "retained_frac"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def load_ani(model):
    p = ROOT / "results" / "anisotropy" / f"anisotropy_{model}_database.json"
    return json.loads(p.read_text()) if p.exists() else {}


def series(rows, pred, key="recall@10"):
    s = sorted([r for r in rows if pred(r)], key=lambda r: r["bits_per_vector"])
    return [r["bits_per_vector"] for r in s], [r[key] for r in s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out = ROOT / "results" / "figures"
    tab = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    is_obl = lambda r: r["method"].startswith("scalar") and "no-rot" not in r["method"]
    is_norot = lambda r: r["method"].endswith("no-rot")
    is_pq = lambda r: r["method"].startswith("PQ-")
    is_opq = lambda r: r["method"].startswith("OPQ-")

    # ------------------------------------------------------------------ Fig 7: RQ1 all models
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.6 * len(MODELS), 4.3), sharey=True)
    for ax, (model, label) in zip(axes, MODELS):
        rows = load_bench(model)
        for pred, lbl, c, m in [
            (is_opq, "OPQ (trained)", C_OPQ, "s"),
            (is_pq, "PQ (trained)", C_PQ, "^"),
            (is_obl, "rotation+scalar (untrained)", C_OBL, "o"),
            (is_norot, "scalar, no rotation", C_NOROT, "v"),
        ]:
            x, y = series(rows, pred)
            if x:
                ax.plot(x, y, marker=m, color=c, lw=1.9, ms=6.5, label=lbl)
        ani = load_ani(model)
        crest = ani.get("within_vector", {}).get("raw", {}).get("crest_mean", float("nan"))
        pr = ani.get("raw", {}).get("effective_dim_frac", float("nan")) * 100
        ax.set_title(f"{label}\neff. dim {pr:.1f}% of d,  raw crest {crest:.2f}", fontsize=10)
        ax.set_xlabel("bits per vector")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Recall@10")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Data-oblivious vs data-dependent quantization, three embedding models",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig7_crossmodel_rq1.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------- Fig 8: rotation benefit vs raw crest (mechanism)
    crests, gains, names = [], [], []
    for model, label in MODELS:
        rows = load_bench(model)
        ani = load_ani(model)
        crest = ani.get("within_vector", {}).get("raw", {}).get("crest_mean")
        rel = []
        for b in (1, 2, 3, 4):
            a = next((r for r in rows if r["method"] == f"scalar-{b}b-rot-gaussian"), None)
            n = next((r for r in rows if r["method"] == f"scalar-{b}b-no-rot"), None)
            if a and n and n["recall@10"] > 0:
                rel.append((a["recall@10"] - n["recall@10"]) / n["recall@10"] * 100)
        if crest and rel:
            crests.append(crest); gains.append(float(np.mean(rel))); names.append(label)

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.scatter(crests, gains, s=180, color=C_OBL, zorder=3)
    for c, g, n in zip(crests, gains, names):
        ax.annotate(f"{n}\ncrest {c:.2f} → {g:+.1f}%", (c, g),
                    textcoords="offset points", xytext=(12, -6), fontsize=9)
    ax.axvline(3.37, ls="--", c="gray", lw=1.2)
    ax.text(3.45, max(gains) * 0.55, "isotropic ideal\n(crest ≈ 3.37)", fontsize=8, color="gray")
    ax.axhline(0, c="k", lw=1)
    ax.set_xlabel("raw within-vector crest factor  (spikiness before rotation)")
    ax.set_ylabel("mean relative Recall@10 gain from rotation (%)")
    ax.set_title("Rotation pays off where coordinates are spiky —\n"
                 "a threshold effect, not a proportional one", fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(2.5, 14.0)
    fig.tight_layout()
    fig.savefig(out / "fig8_mechanism_crossmodel.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------ Table 1: structure
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Structure of the two audio embedding spaces (10\,000 MS-SNSD "
             r"utterances, $d=768$). Both are far from the isotropic case that worst-case-optimal "
             r"data-oblivious quantization is designed for.}",
             r"\label{tab:structure}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Property & WavLM-Base+ & wav2vec2-base & isotropic ref. \\", r"\midrule"]
    a1, a2 = load_ani("wavlm_base_plus"), load_ani("wav2vec2_base")
    if a1 and a2:
        g = a1.get("gaussian_reference", {})
        rowspec = [
            ("mean pairwise cosine", "raw", "mean_pairwise_cosine", "{:+.3f}"),
            ("effective dim.\\ (\\% of $d$)", "raw", "effective_dim_frac", "{:.1%}"),
            ("variance in top-10 dirs", "raw", "var_top10", "{:.1%}"),
        ]
        for name, sec, key, fmt in rowspec:
            v1 = a1[sec][key]; v2 = a2[sec][key]; vg = g.get(key, float("nan"))
            lines.append(f"{name} & {fmt.format(v1)} & {fmt.format(v2)} & {fmt.format(vg)} " + r"\\")
        wv1, wv2 = a1["within_vector"], a2["within_vector"]
        lines.append(f"within-vector crest (raw) & {wv1['raw']['crest_mean']:.2f} & "
                     f"{wv2['raw']['crest_mean']:.2f} & {wv1['isotropic_ref']['crest_mean']:.2f} " + r"\\")
        lines.append(f"within-vector kurtosis (raw) & {wv1['raw']['within_kurtosis_mean']:+.2f} & "
                     f"{wv2['raw']['within_kurtosis_mean']:+.2f} & "
                     f"{wv1['isotropic_ref']['within_kurtosis_mean']:+.2f} " + r"\\")
        lines.append(r"\midrule")
        lines.append(f"within-vector crest (rotated) & {wv1['rotated']['crest_mean']:.2f} & "
                     f"{wv2['rotated']['crest_mean']:.2f} & "
                     f"{wv1['isotropic_ref']['crest_mean']:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tab / "table1_structure.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ Table 2: matched budget
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Recall@10 at matched bit budgets. The advantage of trained codebooks "
             r"grows as the budget shrinks, and reverses at the largest budget on wav2vec2.}",
             r"\label{tab:rq1}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & bits/vec & oblivious & PQ & OPQ \\", r"\midrule"]
    for model, label in MODELS:
        rows = load_bench(model)
        obl = sorted([r for r in rows if is_obl(r)], key=lambda r: r["bits_per_vector"])
        for i, o in enumerate(obl):
            b = o["bits_per_vector"]
            pq = min([r for r in rows if is_pq(r)],
                     key=lambda r: abs(r["bits_per_vector"] - b), default=None)
            opq = min([r for r in rows if is_opq(r)],
                      key=lambda r: abs(r["bits_per_vector"] - b), default=None)
            name = label if i == 0 else ""
            lines.append(f"{name} & {b:.0f} & {o['recall@10']:.4f} & "
                         f"{pq['recall@10']:.4f} & {opq['recall@10']:.4f} " + r"\\")
        lines.append(r"\midrule")
    lines = lines[:-1] + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tab / "table2_rq1.tex").write_text("\n".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ Table 3: geometry vs meaning
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Geometry versus meaning at 1 bit/coordinate ($\approx$30$\times$ "
             r"compression). Roughly half the true neighbours are lost, yet $\sim$96\,\% of "
             r"downstream retrieval performance is retained on both models.}",
             r"\label{tab:geo_vs_sem}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Model & Recall@10 kept & speaker P@10 kept & noise-type P@10 kept \\", r"\midrule"]
    for model, label in MODELS:
        rows = load_bench(model)
        sem = load_sem(model)
        o = next((r for r in rows if r["method"] == "scalar-1b-rot-gaussian"), None)
        cells = []
        for task in ("speaker", "noise_type"):
            f32 = next((r["precision_at_k"] for r in sem
                        if r["task"] == task and r["method"] == "float32"), None)
            q = next((r["precision_at_k"] for r in sem
                      if r["task"] == task and r["method"] == "scalar-1b+rotation"), None)
            cells.append(f"{q/f32:.1%}" if (f32 and q) else "--")
        if o:
            lines.append(f"{label} & {o['recall@10']:.1%} & {cells[0]} & {cells[1]} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tab / "table3_geometry_vs_meaning.tex").write_text("\n".join(lines), encoding="utf-8")

    print("figures:")
    for f in sorted(out.glob("fig[78]*.png")):
        print(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")
    print("tables:")
    for f in sorted(tab.glob("*.tex")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
