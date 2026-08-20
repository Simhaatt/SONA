# MS-SNSD external validation (1100 clips, unseen dataset)

| label | method | si_sdr_i | pesq | stoi | estoi | lsd | rtf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Attention | neural_attention | 2.5861 | 1.3232 | 0.8035 | 0.6643 | 11.9075 | 0.0011 |
| GTCRN + MR loss | neural_gtcrn_ch96_h96_b4_mw1p25_mr0p25 | 3.1534 | 1.4027 | 0.8162 | 0.6904 | 11.6437 | 0.0058 |
