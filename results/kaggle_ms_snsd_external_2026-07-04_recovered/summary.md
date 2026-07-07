| method | n | si_sdr_i | si_sdr | snr_i | snr | pesq | stoi | estoi | lsd | rtf |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neural_attention | 1100 | 2.586 | 5.086 | 3.582 | 6.564 | 1.323 | 0.804 | 0.664 | 11.907 | 0.001 |
| attention_online_lowrank_s1_a2_d0p01 | 1100 | 2.600 | 5.100 | 3.586 | 6.568 | 1.314 | 0.803 | 0.664 | 11.968 | 0.039 |
| attention_online_lowrank_s1_a2_d0p02 | 1100 | 2.609 | 5.109 | 3.584 | 6.565 | 1.307 | 0.803 | 0.663 | 12.017 | 0.038 |

Delta vs frozen `neural_attention`:

| method | delta_si_sdr_i | delta_pesq | delta_stoi | delta_estoi | delta_lsd |
|---|---:|---:|---:|---:|---:|
| attention_online_lowrank_s1_a2_d0p01 | +0.014 | -0.009 | -0.000364 | -0.000773 | +0.061 |
| attention_online_lowrank_s1_a2_d0p02 | +0.023 | -0.016 | -0.000726 | -0.001503 | +0.110 |

Recovered from the downloaded Kaggle notebook after the session output zip was lost. `per_sample.csv` was not recoverable from the notebook.
