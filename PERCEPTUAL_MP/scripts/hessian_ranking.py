"""E2 -- Hessian-trace layer sensitivity (HAWQ-V2 proxy).

Why this exists. The study's negative result is that a *perceptual* sensitivity ranking (PESQ)
agrees with *output-error* rankings (SI-SDR, mask-MSE). But the field-standard way to rank layers
for mixed-precision allocation is neither of those: HAWQ (Dong et al., ICCV 2019) and HAWQ-V2
(NeurIPS 2020) rank by second-order curvature -- top Hessian eigenvalue or average Hessian trace.
Without that column the negative result compares a perceptual criterion against two output-error
criteria and never against the baseline a reviewer actually has in mind.

Method. HAWQ-V2 ranks a layer by its *average* Hessian trace, tr(H_l) / n_l. The trace is
estimated with Hutchinson's method, which needs no explicit Hessian:

    tr(H) = E_v[ v^T H v ],   v ~ Rademacher (+/-1)

and v^T H v is obtained from one double-backward: form g = grad(L), take the scalar g . v, and
differentiate that again.

The probe is drawn over the whole parameter set rather than one layer at a time, and each layer's
trace is read off from its own slice of the result. Cross-layer terms v_l^T H_lm v_m have zero
expectation under independent Rademacher entries, so every layer's estimate stays unbiased while
costing one double-backward per probe instead of one per (probe, layer) -- about 7x cheaper here.

Everything runs on the same speaker-disjoint dev split (p254/p256) used to pick the allocation, so
this column is directly comparable to the existing rankings and touches no test data.

Usage:
    python PERCEPTUAL_MP/scripts/hessian_ranking.py
    python PERCEPTUAL_MP/scripts/hessian_ranking.py --n-clips 50 --n-probes 32   # quick pass
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

# Torch module prefix for each quantizable layer, matching quant_lib's ONNX node names.
LAYER_PREFIX = {
    "stem": "stem.",
    "block0_d1": "blocks.0.conv.",
    "block1_d2": "blocks.1.conv.",
    "block2_d4": "blocks.2.conv.",
    "block3_d8": "blocks.3.conv.",
    "head": "head.",
}
# Reported for completeness -- it cannot be quantized on this runtime, so it is excluded from
# the ranking comparison exactly as it is excluded from the allocation search.
UNQUANTIZABLE = {"band_gru": "band_gru."}

SEED = 2026


def layer_params(model, prefix: str) -> list[torch.nn.Parameter]:
    return [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(
        REPO / "results" / "kaggle_voicebank_gtcrn_loss_focus_full824_rerun_2026-07-07" /
        "extracted" / "ckpt_gtcrn_ch96_h96_b4_mw1p25_mr0p25" / "last.pt"))
    ap.add_argument("--dev-manifest", default=str(HERE.parent / "splits" / "dev_manifest.csv"))
    ap.add_argument("--n-clips", type=int, default=100)
    ap.add_argument("--n-probes", type=int, default=64,
                    help="Hutchinson probe vectors per clip")
    ap.add_argument("--out-dir", default=str(HERE.parent / "results" / "hessian"))
    args = ap.parse_args()

    from src.data import PairDataset
    from src.train.train_supervised import _step, build_frontend, build_model
    from src.utils.config import Config

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = Config(ckpt["cfg"])
    tf = build_frontend(cfg)
    model = build_model(cfg, tf)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tr = cfg.get("train", {}) or {}
    step_kw = dict(
        mag_weight=tr.get("mag_weight", 0.0),
        n_fft=tr.get("n_fft", 512),
        hop=tr.get("hop", 128),
        mr_mag_weight=tr.get("mr_mag_weight", 0.0),
        snr_weight=tr.get("snr_weight", 0.0),
    )
    print(f"loss weights from checkpoint: {step_kw}")

    groups = {**LAYER_PREFIX, **UNQUANTIZABLE}
    params = {name: layer_params(model, pre) for name, pre in groups.items()}
    n_params = {name: int(sum(p.numel() for p in ps)) for name, ps in params.items()}
    for name, ps in params.items():
        if not ps:
            raise SystemExit(f"no parameters matched prefix for layer '{name}'")
    print("parameter counts:", n_params)

    # ---- INT8 perturbation per layer: || Q(w) - w ||^2, symmetric per-tensor, as deployed.
    # This is the second factor of the HAWQ-V2 criterion and the one that carries parameter
    # count and weight scale.
    def int8_perturbation(ps: list[torch.nn.Parameter]) -> float:
        total = 0.0
        for q in ps:
            w = q.detach()
            scale = w.abs().max() / 127.0
            if float(scale) == 0.0:
                continue
            qw = torch.clamp(torch.round(w / scale), -127, 127) * scale
            total += float(((qw - w) ** 2).sum())
        return total

    perturb = {name: int8_perturbation(ps) for name, ps in params.items()}
    print("INT8 perturbation ||Q(w)-w||^2:",
          {k: f"{v:.4e}" for k, v in perturb.items()})

    samples = list(PairDataset(manifest=args.dev_manifest, limit=args.n_clips))
    print(f"dev clips: {len(samples)}  probes/clip: {args.n_probes}")

    gen = torch.Generator().manual_seed(SEED)
    trace_sums = {name: 0.0 for name in groups}
    trace_sq = {name: 0.0 for name in groups}
    n_est = {name: 0 for name in groups}

    for i, s in enumerate(samples):
        noisy = torch.from_numpy(np.asarray(s.noisy, dtype=np.float32)).unsqueeze(0)
        clean = torch.from_numpy(np.asarray(s.clean, dtype=np.float32)).unsqueeze(0)

        model.zero_grad(set_to_none=True)
        loss = _step(model, tf, noisy, clean, **step_kw)

        flat = [p for name in groups for p in params[name]]
        grads = torch.autograd.grad(loss, flat, create_graph=True)
        gmap: dict[str, list[torch.Tensor]] = {}
        off = 0
        for name in groups:
            k = len(params[name])
            gmap[name] = list(grads[off:off + k])
            off += k

        # One probe over the whole parameter set estimates every layer's trace simultaneously:
        # cross-layer terms v_l^T H_lm v_m have zero expectation for m != l, so each layer's
        # component is still an unbiased estimate of tr(H_ll). Costs one double-backward per
        # probe instead of one per (probe, layer).
        for _ in range(args.n_probes):
            vmap = {name: [torch.randint(0, 2, q.shape, generator=gen, dtype=torch.float32) * 2 - 1
                           for q in params[name]] for name in groups}
            gv = sum((g * v).sum()
                     for name in groups
                     for g, v in zip(gmap[name], vmap[name]))
            hv = torch.autograd.grad(gv, flat, retain_graph=True)
            off2 = 0
            for name in groups:
                k2 = len(params[name])
                hv_l = hv[off2:off2 + k2]
                off2 += k2
                vhv = float(sum((h * v).sum() for h, v in zip(hv_l, vmap[name])))
                trace_sums[name] += vhv
                trace_sq[name] += vhv * vhv
                n_est[name] += 1

        del grads, gmap
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)} clips", flush=True)

    rows = []
    for name in groups:
        n = n_est[name]
        mean = trace_sums[name] / n
        var = max(trace_sq[name] / n - mean * mean, 0.0)
        se = float(np.sqrt(var / n))
        rows.append({
            "layer": name,
            "quantizable": name in LAYER_PREFIX,
            "n_params": n_params[name],
            "hessian_trace": mean,
            "hessian_trace_se": se,
            "avg_hessian_trace": mean / n_params[name],
            "int8_perturbation": perturb[name],
            # full HAWQ-V2 criterion: average curvature x actual quantization perturbation
            "hawq_v2_omega": (mean / n_params[name]) * perturb[name],
            "n_estimates": n,
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "hessian_ranking.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------ compare against the existing rankings
    prior_path = HERE.parent / "results" / "phase1" / "phase1a_local_rankings.json"
    correlations = {}
    q = [r for r in rows if r["quantizable"]]
    variants = {
        "avg_hessian_trace": {r["layer"]: r["avg_hessian_trace"] for r in q},
        "raw_hessian_trace": {r["layer"]: r["hessian_trace"] for r in q},
        "hawq_v2_omega": {r["layer"]: r["hawq_v2_omega"] for r in q},
    }

    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        sens = prior.get("sensitivity_mean_abs_delta", {})
        alias = {f"only_{k}": k for k in LAYER_PREFIX}
        for vname, order in variants.items():
            for metric in ("d_si_sdr", "mask_mse", "d_stoi", "d_estoi", "d_lsd"):
                pairs = []
                for key, vals in sens.items():
                    lay = alias.get(key)
                    if lay and lay in order and metric in vals:
                        pairs.append((order[lay], vals[metric]))
                if len(pairs) >= 3:
                    a, b = zip(*pairs)
                    rho, pv = stats.spearmanr(a, b)
                    correlations[f"{vname} vs {metric}"] = {
                        "spearman_rho": float(rho), "p": float(pv), "n_layers": len(pairs)}

    (out / "hessian_ranking.json").write_text(
        json.dumps({"rows": rows, "correlations": correlations}, indent=2), encoding="utf-8")

    order = sorted(q, key=lambda r: -r["hawq_v2_omega"])
    lines = ["# Hessian-trace layer sensitivity (HAWQ-V2 proxy)", "",
             f"Hutchinson estimator, {args.n_probes} Rademacher probes per clip over "
             f"{len(samples)} dev clips (speaker-disjoint split p254/p256; no test data touched). "
             "HAWQ-V2 ranks layers by **average** Hessian trace, tr(H)/n_params.", "",
             "Ranked by the full HAWQ-V2 criterion "
             "`Omega = (tr(H)/n) * ||Q(w)-w||^2` -- average curvature times the actual INT8 "
             "perturbation. The raw and averaged traces are shown alongside because they order "
             "the layers differently, which is itself the finding.", "",
             "| rank | layer | quantizable | params | tr(H) | avg tr(H)/n | ||Q(w)-w||^2 | Omega |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(order, 1):
        lines.append(f"| {i} | {r['layer']} | yes | {r['n_params']} | "
                     f"{r['hessian_trace']:.4e} | {r['avg_hessian_trace']:.4e} | "
                     f"{r['int8_perturbation']:.4e} | {r['hawq_v2_omega']:.4e} |")
    for r in rows:
        if not r["quantizable"]:
            lines.append(f"| -- | {r['layer']} | **no** (unquantizable) | {r['n_params']} | "
                         f"{r['hessian_trace']:.4e} | {r['avg_hessian_trace']:.4e} | "
                         f"{r['int8_perturbation']:.4e} | {r['hawq_v2_omega']:.4e} |")
    if correlations:
        lines += ["", "## Agreement with the existing sensitivity rankings", "",
                  "| comparison | Spearman rho | p | layers |", "|---|---|---|---|"]
        for k, v in correlations.items():
            lines.append(f"| {k} | {v['spearman_rho']:+.3f} | {v['p']:.4f} | {v['n_layers']} |")
    (out / "hessian_ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {out / 'hessian_ranking.md'}\n")
    print("HAWQ-V2 ranking by full Omega (most sensitive first, quantizable only):")
    for i, r in enumerate(order, 1):
        print(f"  {i}. {r['layer']:12s} Omega = {r['hawq_v2_omega']:.4e}  "
              f"(raw tr={r['hessian_trace']:.3e}, avg={r['avg_hessian_trace']:.3e})")
    print("\nranking by RAW trace:")
    for i, r in enumerate(sorted(q, key=lambda r: -r["hessian_trace"]), 1):
        print(f"  {i}. {r['layer']:12s} tr(H) = {r['hessian_trace']:.4e}")
    for k, v in correlations.items():
        print(f"  {k}: rho={v['spearman_rho']:+.3f} (p={v['p']:.4f})")


if __name__ == "__main__":
    main()
