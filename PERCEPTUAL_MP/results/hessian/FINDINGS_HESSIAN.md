# E2 — Hessian-trace sensitivity (HAWQ-V2): findings

Script: `PERCEPTUAL_MP/scripts/hessian_ranking.py`. Hutchinson estimator, 32 Rademacher probes per
clip over 50 dev clips (p254/p256, speaker-disjoint from TRAIN — no test data touched), 1 600
estimates per layer.

---

## 1. Why this experiment exists

Study 1's negative result is that a *perceptual* (PESQ) sensitivity ranking agrees with
*output-error* rankings (SI-SDR, mask-MSE). But the field-standard way to rank layers for
mixed-precision allocation is neither: HAWQ and HAWQ-V2 rank by second-order curvature. Without
that column, the negative result never compares against the baseline a reviewer actually has in
mind.

## 2. A correction made during this experiment

The first pass ranked layers by **average Hessian trace**, `tr(H)/n`, alone. That is *not* the
HAWQ-V2 criterion. HAWQ-V2 ranks by the expected loss perturbation from quantizing the layer:

```
Omega_l = (tr(H_l) / n_l) * || Q(w_l) - w_l ||^2
```

The second factor — the actual INT8 perturbation of that layer's weights — is what carries
parameter count and weight scale. Dropping it compares curvature *per weight* while ignoring how
many weights there are, which is precisely the axis on which this model is pathological: the head
has **97** parameters and each block has **166 080**.

Ranking on the incomplete criterion produced a striking but wrong conclusion (that HAWQ-V2
misranks). The corrected criterion reverses it. Reporting the incomplete version would have been a
strawman, and a reviewer would rightly have said the baseline was implemented incorrectly.

## 3. Result

| rank | layer | params | tr(H) | avg tr(H)/n | ‖Q(w)−w‖² | **Ω** |
|---|---|---:|---:|---:|---:|---:|
| 1 | **stem** | 1 152 | 4.79e+02 | 4.16e-01 | 2.66e-03 | **1.11e-03** |
| 2 | block3_d8 | 166 080 | 4.48e+02 | 2.70e-03 | 4.81e-02 | 1.30e-04 |
| 3 | block2_d4 | 166 080 | 4.58e+02 | 2.76e-03 | 4.53e-02 | 1.25e-04 |
| 4 | block0_d1 | 166 080 | 1.81e+02 | 1.09e-03 | 7.59e-02 | 8.25e-05 |
| 5 | block1_d2 | 166 080 | 1.25e+02 | 7.50e-04 | 6.56e-02 | 4.92e-05 |
| 6 | **head** | 97 | 8.62e+01 | 8.88e-01 | 3.18e-05 | **2.82e-05** |
| — | band_gru *(unquantizable)* | 55 872 | 5.49e+02 | 9.82e-03 | 6.68e-02 | 6.56e-04 |

**The full HAWQ-V2 criterion puts `stem` first and `head` last — exactly matching the empirical
INT8 damage ranking.**

### Rank agreement with the measured per-layer damage

| criterion | vs ΔSI-SDR | vs mask-MSE | vs ΔSTOI | vs ΔESTOI | vs ΔLSD |
|---|---:|---:|---:|---:|---:|
| **Ω (full HAWQ-V2)** | **+0.600** | +0.486 | +0.600 | +0.429 | +0.486 |
| raw tr(H) | **+0.714** | +0.543 | +0.657 | +0.486 | +0.429 |
| avg tr(H)/n *(incomplete)* | −0.143 | −0.314 | −0.200 | −0.371 | −0.429 |

## 4. What this means for the paper

**The negative result becomes comprehensive rather than partial.** Three independent families of
sensitivity criterion — perceptual (PESQ), output-error (SI-SDR, mask-MSE), and second-order
curvature (HAWQ-V2) — all identify the same layer as most sensitive and the same layer as least.
The criterion does not matter. That is a stronger version of Study 1's claim than "two output-error
metrics agree with a perceptual one", and it closes the obvious reviewer objection.

**A practical caveat worth one sentence in the paper.** *Average* Hessian trace is often used as
shorthand for HAWQ-V2 sensitivity. Here that shorthand **inverts the ranking**: it promotes the
97-parameter head from least to most sensitive, purely because dividing by 97 inflates it against
blocks divided by 166 080. When layer parameter counts span three orders of magnitude, the
perturbation term is not optional. Implement the full criterion.

## 5. Limitations — state these

- **n = 6 layers.** No correlation here reaches significance (p = 0.11–0.40). The claim rests on
  the agreement of the rank *orderings* — particularly that all three criteria agree on the top and
  bottom layer — and on the mechanism, not on a p-value. Say so explicitly rather than quoting ρ as
  though six points settled it.
- Hutchinson is a stochastic estimator; per-layer standard errors are recorded in
  `hessian_ranking.csv`.
- The band GRU is reported for completeness but excluded from the ranking comparison, exactly as it
  is excluded from the allocation search — it cannot be quantized on this runtime.
- Curvature is measured at the trained optimum under the training loss (SI-SDR + 1.25·log-mag +
  0.25·multi-resolution). A different loss would give different curvature.
