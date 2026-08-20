# Dither diagnostic

Post-training quantization approximates a source model, so **FP32 is a ceiling, not a competitor**. Any metric on which a quantized variant significantly beats FP32 is not measuring approximation quality on that corpus -- the extra quantization noise is acting as a dither. On such a metric, rank by **|deviation from FP32|** instead of by the absolute score.

Advantage is signed so that **positive means the quantized model beat FP32**. Bootstrap 95 % CI over 10000 resamples; `compromised` is flagged when the CI lies entirely above zero.

| corpus | method | metric | quantized | FP32 | advantage | 95% CI | compromised |
|---|---|---|---|---|---|---|---|
| voicebank_824 | blocks5_int8 | si_sdr | 18.3751 | 18.3765 | -0.0014 | [-0.0040, +0.0012] | no |
| voicebank_824 | blocks5_int8 | stoi | 0.9299 | 0.9299 | -0.0000 | [-0.0001, +0.0000] | no |
| voicebank_824 | blocks5_int8 | estoi | 0.8322 | 0.8322 | -0.0000 | [-0.0001, +0.0000] | no |
| voicebank_824 | blocks5_int8 | lsd | 6.4690 | 6.4694 | +0.0003 | [-0.0005, +0.0011] | no |
| voicebank_824 | mixed_stemfp32 | si_sdr | 18.3751 | 18.3765 | -0.0014 | [-0.0041, +0.0012] | no |
| voicebank_824 | mixed_stemfp32 | stoi | 0.9299 | 0.9299 | -0.0000 | [-0.0001, +0.0000] | no |
| voicebank_824 | mixed_stemfp32 | estoi | 0.8322 | 0.8322 | -0.0000 | [-0.0001, +0.0000] | no |
| voicebank_824 | mixed_stemfp32 | lsd | 6.4690 | 6.4694 | +0.0003 | [-0.0005, +0.0011] | no |
| voicebank_824 | stem_only_int8 | si_sdr | 18.3742 | 18.3765 | -0.0023 | [-0.0041, -0.0005] | no |
| voicebank_824 | stem_only_int8 | stoi | 0.9299 | 0.9299 | -0.0001 | [-0.0001, -0.0000] | no |
| voicebank_824 | stem_only_int8 | estoi | 0.8320 | 0.8322 | -0.0002 | [-0.0003, -0.0001] | no |
| voicebank_824 | stem_only_int8 | lsd | 6.5059 | 6.4694 | -0.0366 | [-0.0420, -0.0314] | no |
| voicebank_824 | uniform_int8 | si_sdr | 18.2671 | 18.3765 | -0.1094 | [-0.1269, -0.0925] | no |
| voicebank_824 | uniform_int8 | stoi | 0.9300 | 0.9299 | +0.0000 | [-0.0002, +0.0002] | no |
| voicebank_824 | uniform_int8 | estoi | 0.8295 | 0.8322 | -0.0027 | [-0.0032, -0.0022] | no |
| voicebank_824 | uniform_int8 | lsd | 6.5340 | 6.4694 | -0.0646 | [-0.0826, -0.0470] | no |
| ms_snsd_1100 | blocks5_int8 | si_sdr | 5.5642 | 5.5632 | +0.0010 | [-0.0044, +0.0057] | no |
| ms_snsd_1100 | blocks5_int8 | stoi | 0.8152 | 0.8154 | -0.0002 | [-0.0003, -0.0002] | no |
| ms_snsd_1100 | blocks5_int8 | estoi | 0.6896 | 0.6899 | -0.0003 | [-0.0004, -0.0002] | no |
| ms_snsd_1100 | blocks5_int8 | lsd | 11.6808 | 11.6689 | -0.0119 | [-0.0135, -0.0104] | no |
| ms_snsd_1100 | mixed_stemfp32 | si_sdr | 5.5642 | 5.5632 | +0.0010 | [-0.0043, +0.0059] | no |
| ms_snsd_1100 | mixed_stemfp32 | stoi | 0.8152 | 0.8154 | -0.0002 | [-0.0003, -0.0002] | no |
| ms_snsd_1100 | mixed_stemfp32 | estoi | 0.6896 | 0.6899 | -0.0003 | [-0.0004, -0.0002] | no |
| ms_snsd_1100 | mixed_stemfp32 | lsd | 11.6808 | 11.6689 | -0.0119 | [-0.0134, -0.0103] | no |
| ms_snsd_1100 | stem_only_int8 | si_sdr | 5.5892 | 5.5632 | +0.0260 | [+0.0086, +0.0476] | **YES** |
| ms_snsd_1100 | stem_only_int8 | stoi | 0.8154 | 0.8154 | -0.0000 | [-0.0001, +0.0000] | no |
| ms_snsd_1100 | stem_only_int8 | estoi | 0.6897 | 0.6899 | -0.0002 | [-0.0003, -0.0001] | no |
| ms_snsd_1100 | stem_only_int8 | lsd | 11.6955 | 11.6689 | -0.0267 | [-0.0325, -0.0211] | no |
| ms_snsd_1100 | uniform_int8 | si_sdr | 5.5255 | 5.5632 | -0.0377 | [-0.0669, -0.0070] | no |
| ms_snsd_1100 | uniform_int8 | stoi | 0.8212 | 0.8154 | +0.0058 | [+0.0051, +0.0065] | **YES** |
| ms_snsd_1100 | uniform_int8 | estoi | 0.6964 | 0.6899 | +0.0064 | [+0.0055, +0.0074] | **YES** |
| ms_snsd_1100 | uniform_int8 | lsd | 11.4474 | 11.6689 | +0.2215 | [+0.2019, +0.2417] | **YES** |

## Summary

Metrics on which at least one quantized variant significantly beat FP32:

- **ms_snsd_1100 / estoi** — 1 variant(s)
- **ms_snsd_1100 / lsd** — 1 variant(s)
- **ms_snsd_1100 / si_sdr** — 1 variant(s)
- **ms_snsd_1100 / stoi** — 1 variant(s)

On these cells, absolute scores cannot be used to rank allocations. This is the evidence for the paper's claim of *fidelity to FP32* rather than "better on every metric everywhere".
