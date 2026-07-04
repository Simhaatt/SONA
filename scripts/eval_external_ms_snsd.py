"""Evaluate a saved VoiceBank model on the generated MS-SNSD external manifest.

This runner does not retrain. It loads a supervised checkpoint, evaluates the frozen enhancer, then
evaluates selected trust-region online-adaptation configs on the same manifest.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _add_repo_to_path(repo: str | None) -> Path:
    root = Path(repo).resolve() if repo else Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _csv_items(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _csv_ints(text: str) -> list[int]:
    return [int(x) for x in _csv_items(text)]


def _csv_floats(text: str) -> list[float]:
    return [float(x) for x in _csv_items(text)]


def _csv_optional_floats(text: str) -> list[float | None]:
    out: list[float | None] = []
    for x in _csv_items(text):
        out.append(None if x.lower() in ("none", "off", "unclipped") else float(x))
    return out


def _fmt_float(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _fmt_optional_float(x: float | None) -> str:
    return "none" if x is None else _fmt_float(x)


def _resolve_manifest_paths(manifest: Path, out: Path) -> Path:
    r"""Rewrite Windows absolute paths in a manifest when running from a Kaggle dataset.

    Locally the generated MS-SNSD manifest points at ``D:\noise cancellation\external_data\...``.
    After uploading the compact dataset zip to Kaggle, the same files live under the dataset root,
    for example ``/kaggle/input/.../MS-SNSD/...``. If the manifest paths already exist this leaves
    them untouched; otherwise it writes a fixed manifest beside the outputs.
    """

    with open(manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        return manifest

    dataset_root = manifest.parent.parent

    def fix_path(value: str) -> str:
        p = Path(value)
        if p.exists():
            return value

        normalized = value.replace("\\", "/")
        marker = "external_data/"
        if marker in normalized:
            rel = normalized.split(marker, 1)[1]
            candidate = dataset_root / rel
            if candidate.exists():
                return str(candidate)

        for marker in ("MS-SNSD/", "ms_snsd_sona/"):
            if marker in normalized:
                rel = normalized[normalized.index(marker):]
                candidate = dataset_root / rel
                if candidate.exists():
                    return str(candidate)

        return value

    changed = False
    for row in rows:
        for key in ("clean", "noisy"):
            if key in row and row[key]:
                fixed = fix_path(row[key])
                changed = changed or fixed != row[key]
                row[key] = fixed

    if not changed:
        return manifest

    fixed_manifest = out / "test_resolved_paths.csv"
    with open(fixed_manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return fixed_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None, help="repo root")
    ap.add_argument("--manifest", required=True, help="MS-SNSD SONA manifest CSV")
    ap.add_argument("--checkpoint", required=True, help="supervised model checkpoint .pt")
    ap.add_argument("--out", default="results_ms_snsd_external")
    ap.add_argument("--limit", type=int, default=0, help="0 means full manifest")
    ap.add_argument("--adapter", default="lowrank")
    ap.add_argument("--anchors", default="2.0")
    ap.add_argument("--steps", default="1")
    ap.add_argument("--max-gain-deltas", default="0.02")
    ap.add_argument("--smooth-time-weight", type=float, default=0.0)
    ap.add_argument("--smooth-band-weight", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--reward-kind", default="auto")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    _add_repo_to_path(args.repo)

    import torch

    from src.data import PairDataset
    from src.eval.evaluate import aggregate, evaluate_method
    from src.eval.metrics import available_metrics
    from src.eval.report import latex_table, markdown_table, save_rows_csv
    from src.eval.reward_squim import make_reward, squim_available
    from src.models import NeuralEnhancer, OnlineAdapter, make_adapter
    from src.train.train_supervised import build_frontend, build_model
    from src.utils.config import Config
    from src.utils.logging import get_logger
    from src.utils.seed import set_seed

    log = get_logger("external_ms_snsd")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(1234)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = Config(checkpoint["cfg"])
    tf = build_frontend(cfg).to(device)
    model = build_model(cfg, tf).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    manifest = _resolve_manifest_paths(Path(args.manifest), out)
    samples = list(PairDataset(manifest=manifest, limit=(args.limit or None)))
    if not samples:
        raise SystemExit(f"No samples found in {manifest}")

    metrics = ["si_sdr", "si_sdr_i", "snr", "snr_i", "pesq", "stoi", "estoi", "lsd"]
    reward = make_reward(args.reward_kind, device=device, sr=int(cfg.get_path("sr", 16000)))

    log.info("device=%s | metric backends=%s | SQUIM=%s", device, available_metrics(), squim_available())
    log.info("checkpoint=%s | manifest=%s | samples=%d", ckpt_path, manifest, len(samples))

    rows: list[dict] = []
    rows += evaluate_method(
        NeuralEnhancer(model, tf, device=device, name="neural_attention"),
        samples,
        include_metrics=metrics,
        method_name="neural_attention",
    )

    for adapter_kind in _csv_items(args.adapter):
        for steps in _csv_ints(args.steps):
            for anchor in _csv_floats(args.anchors):
                for max_delta in _csv_optional_floats(args.max_gain_deltas):
                    name = (
                        f"attention_online_{adapter_kind}"
                        f"_s{steps}_a{_fmt_float(anchor)}"
                        f"_d{_fmt_optional_float(max_delta)}"
                    )
                    log.info("eval %s", name)
                    rows += evaluate_method(
                        OnlineAdapter(
                            model,
                            tf,
                            device=device,
                            name=name,
                            steps=steps,
                            lr=args.lr,
                            anchor_weight=anchor,
                            max_gain_delta=max_delta,
                            smooth_time_weight=args.smooth_time_weight,
                            smooth_band_weight=args.smooth_band_weight,
                            reward_fn=reward,
                            adapter=make_adapter(adapter_kind, tf.n_bands),
                        ),
                        samples,
                        include_metrics=metrics,
                        method_name=name,
                    )

    summary = aggregate(rows, group_keys=("method",))
    save_rows_csv(rows, out / "per_sample.csv")
    save_rows_csv(summary, out / "summary.csv")
    (out / "summary.md").write_text(markdown_table(summary), encoding="utf-8")
    (out / "summary.tex").write_text(latex_table(summary), encoding="utf-8")

    print("\n=== MS-SNSD external validation summary ===\n")
    print(markdown_table(summary))

    by = {r["method"]: r for r in summary}
    base = by.get("neural_attention")
    if base:
        print("\nDelta vs frozen base:")
        for name, row in by.items():
            if not name.startswith("attention_online_"):
                continue
            parts = []
            for key in ("si_sdr_i", "pesq", "stoi", "estoi", "lsd"):
                b, v = base.get(key), row.get(key)
                if isinstance(b, (int, float)) and isinstance(v, (int, float)):
                    parts.append(f"{key} {v - b:+.3f}")
            print(f"  {name}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
