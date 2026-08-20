# Hessian-trace layer sensitivity (HAWQ-V2 proxy)

Hutchinson estimator, 8 Rademacher probes per clip over 4 dev clips (speaker-disjoint split p254/p256; no test data touched). HAWQ-V2 ranks layers by **average** Hessian trace, tr(H)/n_params.

| rank | layer | quantizable | params | tr(H) | avg tr(H)/n |
|---|---|---|---|---|---|
| 1 | stem | yes | 1152 | 8.2435e+02 | 7.1558e-01 |
| 2 | head | yes | 97 | 5.7137e+01 | 5.8905e-01 |
| 3 | block2_d4 | yes | 166080 | 4.6772e+02 | 2.8162e-03 |
| 4 | block3_d8 | yes | 166080 | 4.6670e+02 | 2.8101e-03 |
| 5 | block0_d1 | yes | 166080 | 2.9504e+02 | 1.7765e-03 |
| 6 | block1_d2 | yes | 166080 | 1.6026e+02 | 9.6495e-04 |
| -- | band_gru | **no** (unquantizable) | 55872 | 6.1434e+02 | 1.0995e-02 |

## Agreement with the existing sensitivity rankings

| comparison | Spearman rho | p | layers |
|---|---|---|---|
| avg_hessian_trace vs d_si_sdr | +0.143 | 0.7872 | 6 |
| avg_hessian_trace vs mask_mse | -0.029 | 0.9572 | 6 |
| avg_hessian_trace vs d_stoi | +0.086 | 0.8717 | 6 |
| avg_hessian_trace vs d_estoi | -0.086 | 0.8717 | 6 |
| avg_hessian_trace vs d_lsd | -0.143 | 0.7872 | 6 |
