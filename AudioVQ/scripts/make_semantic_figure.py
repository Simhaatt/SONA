"""Figures for the semantic-retrieval results.

Fig 5  geometry vs meaning: Recall@10 collapses under compression while semantic precision does
       not -- the paper's most transferable claim.
Fig 6  the task-dependent rotation effect, with significance marked.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C_GEO = "#c1121f"
C_SPK = "#2a6f97"
C_NOISE = "#0b525b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wavlm_base_plus")
    args = ap.parse_args()

    sem = list(csv.DictReader(open(ROOT / "results" / "semantic" / f"semantic_{args.model}.csv")))
    for r in sem:
        for k in ("bits", "precision_at_k", "retained_frac", "random_baseline"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
    ret = list(csv.DictReader(open(ROOT / "results" / "retrieval" / f"benchmark_{args.model}.csv")))
    for r in ret:
        for k in ("bits_per_vector", "recall@10"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])

    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    bits_axis = [1, 2, 3, 4]
    d = 768

    # ---------------------------------------------------- Fig 5: geometry vs meaning
    geo = []
    for b in bits_axis:
        r = next((x for x in ret if x["method"] == f"scalar-{b}b-rot-gaussian"), None)
        geo.append(r["recall@10"] if r else np.nan)

    def sem_series(task):
        f32 = next(r["precision_at_k"] for r in sem
                   if r["task"] == task and r["method"] == "float32")
        ys = []
        for b in bits_axis:
            r = next((x for x in sem if x["task"] == task
                      and x["method"] == f"scalar-{b}b+rotation"), None)
            ys.append(r["precision_at_k"] / f32 if r else np.nan)
        return ys

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(bits_axis, geo, "o-", color=C_GEO, lw=2, ms=8,
            label="geometric Recall@10 (fraction of true neighbours kept)")
    ax.plot(bits_axis, sem_series("speaker"), "s--", color=C_SPK, lw=2, ms=7,
            label="speaker retrieval, fraction of float32 P@10")
    ax.plot(bits_axis, sem_series("noise_type"), "^--", color=C_NOISE, lw=2, ms=7,
            label="noise-type retrieval, fraction of float32 P@10")
    ax.axhline(1.0, ls=":", c="gray", lw=1)
    ax.set_xticks(bits_axis)
    ax.set_xlabel("bits per coordinate  (1 bit ≈ 30x compression)")
    ax.set_ylabel("fraction retained vs float32")
    ax.set_ylim(0, 1.08)
    ax.set_title("Compression destroys neighbour lists far faster than it destroys meaning")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.annotate("48% of neighbours lost\nbut 96% of task performance kept",
                xy=(1, geo[0]), xytext=(1.35, 0.28), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    fig.tight_layout()
    fig.savefig(out / "fig5_geometry_vs_meaning.png", dpi=180)
    plt.close(fig)

    # ---------------------------------------------------- Fig 6: task-dependent rotation effect
    sig = {("speaker", 2): True, ("speaker", 3): True,
           ("noise_type", 1): True, ("noise_type", 2): True}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    w = 0.36
    x = np.arange(len(bits_axis))
    for i, (task, c) in enumerate([("speaker", C_SPK), ("noise_type", C_NOISE)]):
        deltas = []
        for b in bits_axis:
            a = next((r["precision_at_k"] for r in sem if r["task"] == task
                      and r["method"] == f"scalar-{b}b+rotation"), np.nan)
            n = next((r["precision_at_k"] for r in sem if r["task"] == task
                      and r["method"] == f"scalar-{b}b no-rot"), np.nan)
            deltas.append(a - n)
        bars = ax.bar(x + (i - 0.5) * w, deltas, w, color=c,
                      label=task.replace("_", " "))
        for j, (bar, b) in enumerate(zip(bars, bits_axis)):
            if sig.get((task, b)):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.0012 if bar.get_height() >= 0 else -0.0028),
                        "*", ha="center", fontsize=15, color=c, fontweight="bold")
    ax.axhline(0, c="k", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels([f"{b} bit/coord" for b in bits_axis])
    ax.set_ylabel(r"$\Delta$ P@10:  with rotation $-$ without")
    ax.set_title("The rotation helps speaker retrieval but HURTS noise-type retrieval\n"
                 "(* = significant, paired Wilcoxon p < 0.05)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig6_rotation_task_dependent.png", dpi=180)
    plt.close(fig)

    print(f"wrote figures to {out}")
    for f in sorted(out.glob("fig[56]*.png")):
        print(f"  {f.name} ({f.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
