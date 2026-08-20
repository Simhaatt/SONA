"""Paper figures from the measured benchmark results.

Fig 1  recall vs bit budget -- the RQ1 operating curves
Fig 2  the RQ2 rotation ablation
Fig 3  recall vs build cost -- the practical trade that motivates data-oblivious VQ
Fig 4  within-vector coordinate distribution before/after rotation (the mechanism, F3)
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

C_OBL = "#c1121f"   # data-oblivious (method under test)
C_PQ = "#2a6f97"
C_OPQ = "#0b525b"
C_NOROT = "#adb5bd"


def load_rows(model: str) -> list[dict]:
    p = ROOT / "results" / "retrieval" / f"benchmark_{model}.csv"
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ("bits_per_vector", "recall@1", "recall@10", "recall@100",
                  "build_search_s", "compression_vs_fp32"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    return rows


def series(rows, pred, key="recall@10"):
    s = sorted([r for r in rows if pred(r)], key=lambda r: r["bits_per_vector"])
    return [r["bits_per_vector"] for r in s], [r[key] for r in s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    args = ap.parse_args()
    rows = load_rows(args.model)
    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    is_obl = lambda r: r["method"].startswith("scalar") and "no-rot" not in r["method"]
    is_norot = lambda r: r["method"].endswith("no-rot")
    is_pq = lambda r: r["method"].startswith("PQ-")
    is_opq = lambda r: r["method"].startswith("OPQ-")
    is_uni = lambda r: r["method"].startswith("uniform")

    # ------------------------------------------------------------------ Fig 1: recall vs budget
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, k in zip(axes, ("recall@10", "recall@100")):
        for pred, lbl, c, m in [
            (is_opq, "OPQ (data-dependent)", C_OPQ, "s"),
            (is_pq, "PQ (data-dependent)", C_PQ, "^"),
            (is_obl, "rotation+scalar (data-oblivious)", C_OBL, "o"),
            (is_norot, "scalar, no rotation", C_NOROT, "v"),
            (is_uni, "uniform scalar", "#dee2e6", "x"),
        ]:
            x, y = series(rows, pred, k)
            if x:
                ax.plot(x, y, marker=m, color=c, lw=1.8, ms=6, label=lbl)
        ax.axhline(1.0, ls="--", c="gray", lw=1, label="float32 exact")
        ax.set_xlabel("bits per vector")
        ax.set_ylabel(k.replace("recall@", "Recall@"))
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("Audio embedding retrieval: quality vs bit budget")
    axes[1].set_title("Same, Recall@100")
    fig.tight_layout()
    fig.savefig(out / "fig1_recall_vs_bits.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------ Fig 2: rotation ablation
    bits, with_rot, without = [], [], []
    for b in (1, 2, 3, 4):
        a = next((r for r in rows if r["method"] == f"scalar-{b}b-rot-gaussian"), None)
        n = next((r for r in rows if r["method"] == f"scalar-{b}b-no-rot"), None)
        if a and n:
            bits.append(b); with_rot.append(a["recall@10"]); without.append(n["recall@10"])
    x = np.arange(len(bits)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - w/2, with_rot, w, label="with random rotation", color=C_OBL)
    ax.bar(x + w/2, without, w, label="rotation disabled", color=C_NOROT)
    for i, (a, b_) in enumerate(zip(with_rot, without)):
        ax.text(i, max(a, b_) + 0.02, f"+{(a-b_)/b_*100:.0f}%", ha="center", fontsize=9,
                color=C_OBL, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{b} bit/coord" for b in bits])
    ax.set_ylabel("Recall@10"); ax.set_ylim(0, 1.0)
    ax.set_title("The random rotation is what makes scalar VQ work on audio")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_rotation_ablation.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------ Fig 3: recall vs build cost
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for pred, lbl, c, m in [
        (is_opq, "OPQ", C_OPQ, "s"), (is_pq, "PQ", C_PQ, "^"),
        (is_obl, "rotation+scalar (no training)", C_OBL, "o"),
    ]:
        sel = sorted([r for r in rows if pred(r)], key=lambda r: r["build_search_s"])
        if sel:
            ax.scatter([r["build_search_s"] for r in sel], [r["recall@10"] for r in sel],
                       label=lbl, color=c, marker=m, s=70, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("index build + search time (s, log scale)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Retrieval quality vs cost of building the index")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig3_recall_vs_buildcost.png", dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------ Fig 4: the mechanism
    ani_p = ROOT / "results" / "anisotropy" / f"anisotropy_{args.model}_database.json"
    if ani_p.exists():
        a = json.loads(ani_p.read_text())
        wv = a.get("within_vector", {})
        if wv:
            labels = ["raw\nembeddings", "after\nrotation", "isotropic\nideal"]
            crest = [wv["raw"]["crest_mean"], wv["rotated"]["crest_mean"],
                     wv["isotropic_ref"]["crest_mean"]]
            kurt = [wv["raw"]["within_kurtosis_mean"], wv["rotated"]["within_kurtosis_mean"],
                    wv["isotropic_ref"]["within_kurtosis_mean"]]
            fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
            for ax, vals, name in zip(axs, (crest, kurt),
                                      ("crest factor  max|c| / rms", "within-vector excess kurtosis")):
                cols = [C_NOROT, C_OBL, "#6c757d"]
                ax.bar(labels, vals, color=cols)
                for i, v in enumerate(vals):
                    ax.text(i, v, f"{v:.2f}", ha="center",
                            va="bottom" if v >= 0 else "top", fontsize=9)
                ax.set_title(name); ax.grid(axis="y", alpha=0.3)
            axs[0].set_ylabel("value")
            fig.suptitle("Mechanism: rotation flattens the within-vector coordinate profile "
                         "to the isotropic ideal", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "fig4_mechanism.png", dpi=180)
            plt.close(fig)

    print(f"wrote figures to {out}")
    for f in sorted(out.glob("*.png")):
        print(f"  {f.name}  ({f.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
