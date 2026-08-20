# Hessian-trace layer sensitivity (HAWQ-V2 proxy)

Hutchinson estimator, 32 Rademacher probes per clip over 50 dev clips (speaker-disjoint split p254/p256; no test data touched). HAWQ-V2 ranks layers by **average** Hessian trace, tr(H)/n_params.

Ranked by the full HAWQ-V2 criterion `Omega = (tr(H)/n) * ||Q(w)-w||^2` -- average curvature times the actual INT8 perturbation. The raw and averaged traces are shown alongside because they order the layers differently, which is itself the finding.

| rank | layer | quantizable | params | tr(H) | avg tr(H)/n | ||Q(w)-w||^2 | Omega |
|---|---|---|---|---|---|---|---|
| 1 | stem | yes | 1152 | 4.7929e+02 | 4.1605e-01 | 2.6591e-03 | 1.1063e-03 |
| 2 | block3_d8 | yes | 166080 | 4.4787e+02 | 2.6967e-03 | 4.8118e-02 | 1.2976e-04 |
| 3 | block2_d4 | yes | 166080 | 4.5767e+02 | 2.7557e-03 | 4.5295e-02 | 1.2482e-04 |
| 4 | block0_d1 | yes | 166080 | 1.8055e+02 | 1.0871e-03 | 7.5921e-02 | 8.2535e-05 |
| 5 | block1_d2 | yes | 166080 | 1.2455e+02 | 7.4992e-04 | 6.5584e-02 | 4.9183e-05 |
| 6 | head | yes | 97 | 8.6169e+01 | 8.8834e-01 | 3.1798e-05 | 2.8247e-05 |
| -- | band_gru | **no** (unquantizable) | 55872 | 5.4883e+02 | 9.8230e-03 | 6.6782e-02 | 6.5600e-04 |

## Agreement with the existing sensitivity rankings

| comparison | Spearman rho | p | layers |
|---|---|---|---|
| avg_hessian_trace vs d_si_sdr | -0.143 | 0.7872 | 6 |
| avg_hessian_trace vs mask_mse | -0.314 | 0.5441 | 6 |
| avg_hessian_trace vs d_stoi | -0.200 | 0.7040 | 6 |
| avg_hessian_trace vs d_estoi | -0.371 | 0.4685 | 6 |
| avg_hessian_trace vs d_lsd | -0.429 | 0.3965 | 6 |
| raw_hessian_trace vs d_si_sdr | +0.714 | 0.1108 | 6 |
| raw_hessian_trace vs mask_mse | +0.543 | 0.2657 | 6 |
| raw_hessian_trace vs d_stoi | +0.657 | 0.1562 | 6 |
| raw_hessian_trace vs d_estoi | +0.486 | 0.3287 | 6 |
| raw_hessian_trace vs d_lsd | +0.429 | 0.3965 | 6 |
| hawq_v2_omega vs d_si_sdr | +0.600 | 0.2080 | 6 |
| hawq_v2_omega vs mask_mse | +0.486 | 0.3287 | 6 |
| hawq_v2_omega vs d_stoi | +0.600 | 0.2080 | 6 |
| hawq_v2_omega vs d_estoi | +0.429 | 0.3965 | 6 |
| hawq_v2_omega vs d_lsd | +0.486 | 0.3287 | 6 |
