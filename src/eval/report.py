"""Turn metric rows into paper assets: CSV, Markdown/LaTeX tables, comparison figures."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Higher-is-better vs lower-is-better, for arrow annotations / bolding the best row.
LOWER_IS_BETTER = {"lsd", "wer", "rtf"}

# Preferred display order when present.
COL_ORDER = [
    "si_sdr_i", "si_sdr", "snr_i", "snr", "pesq", "stoi", "estoi",
    "dnsmos", "lsd", "wer", "rtf",
]


def save_rows_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    cols = list({k for r in rows for k in r})
    ordered = [c for c in ["method", "uid", "noise_type", "snr_db"] if c in cols]
    ordered += [c for c in COL_ORDER if c in cols]
    ordered += [c for c in cols if c not in ordered]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ordered)
        w.writeheader()
        w.writerows(rows)


def _fmt(v, col: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _metric_cols(summary: list[dict]) -> list[str]:
    present = {k for r in summary for k in r} - {"method", "n"}
    return [c for c in COL_ORDER if c in present] + sorted(present - set(COL_ORDER))


def markdown_table(summary: list[dict]) -> str:
    """Render an aggregated summary (from `aggregate`) as a Markdown table, best per column bold."""
    cols = _metric_cols(summary)
    best = _best_per_col(summary, cols)

    header = "| method | n | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 2)
    lines = [header, sep]
    for r in summary:
        cells = []
        for c in cols:
            txt = _fmt(r.get(c), c)
            if best.get(c) is not None and r.get(c) == best[c] and txt != "-":
                txt = f"**{txt}**"
            cells.append(txt)
        lines.append(f"| {r.get('method','')} | {r.get('n','')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def latex_table(summary: list[dict], caption: str = "Benchmark results.", label: str = "tab:bench") -> str:
    cols = _metric_cols(summary)
    best = _best_per_col(summary, cols)
    col_spec = "l" + "r" * (len(cols) + 1)
    head = " & ".join(["Method", "N"] + [c.replace("_", r"\_") for c in cols]) + r" \\"
    lines = [
        r"\begin{table}[t]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}", r"\toprule", head, r"\midrule",
    ]
    for r in summary:
        cells = []
        for c in cols:
            txt = _fmt(r.get(c), c)
            if best.get(c) is not None and r.get(c) == best[c] and txt != "-":
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join([str(r.get("method", "")), str(r.get("n", ""))] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _best_per_col(summary: list[dict], cols: list[str]) -> dict:
    best = {}
    for c in cols:
        vals = [r[c] for r in summary if isinstance(r.get(c), (int, float)) and r[c] is not None]
        if not vals:
            best[c] = None
        else:
            best[c] = min(vals) if c in LOWER_IS_BETTER else max(vals)
    return best


def plot_comparison(clean, noisy, enhanced, sr: int, path: str | Path, title: str = "") -> None:
    """Save a waveform + spectrogram before/after figure for the FYP report/demo."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..dsp import STFT

    st = STFT(n_fft=512, hop=128)
    sigs = [("clean", clean), ("noisy", noisy), ("enhanced", enhanced)]
    sigs = [(n, s) for n, s in sigs if s is not None]

    fig, axes = plt.subplots(2, len(sigs), figsize=(4 * len(sigs), 6))
    if len(sigs) == 1:
        axes = axes.reshape(2, 1)
    for j, (label, sig) in enumerate(sigs):
        t = np.arange(len(sig)) / sr
        axes[0, j].plot(t, sig, lw=0.5)
        axes[0, j].set_title(f"{label} waveform")
        axes[0, j].set_xlabel("s")
        mag = np.abs(st.stft(sig))
        axes[1, j].imshow(
            20 * np.log10(mag + 1e-6), origin="lower", aspect="auto", cmap="magma",
        )
        axes[1, j].set_title(f"{label} spectrogram (dB)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
