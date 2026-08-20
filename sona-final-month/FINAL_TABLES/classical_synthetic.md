# Classical DSP baselines (synthetic dev set, illustrative)

| method | si_sdr_i | pesq | stoi | estoi | lsd | rtf |
| --- | --- | --- | --- | --- | --- | --- |
| passthrough | -0.0000 | nan | 0.8356 | 0.4407 | 40.9441 | 0.0053 |
| hard_threshold | 1.1056 | nan | 0.6823 | 0.1345 | 9.3855 | 0.0071 |
| soft_threshold | -2.4492 | nan | 0.6481 | 0.0987 | 9.2816 | 0.0072 |
| spectral_subtraction | 4.1442 | nan | 0.8287 | 0.4635 | 29.8358 | 0.0074 |
| wiener | 3.5734 | nan | 0.8002 | 0.4178 | 19.7695 | 0.0082 |
