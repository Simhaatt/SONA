# 1/Ratio@k vs Recall@k

Both metrics are purely geometric and need no labels. Recall@k asks *are these the same items*; 1/Ratio@k asks *how much worse in distance*. Dimitropoulos & Mamoulis (arXiv:2606.04522) propose the latter as a better proxy for downstream utility under search approximation; this table tests whether it also holds under quantization, which damages the vectors themselves.

| model | method | bits/vec | Recall@10 | 1/Ratio@10 | gap |
|---|---|---|---|---|---|
| wavlm_base_plus | float32-exact | 24576 | 1.0000 | 1.0000 | +0.0000 |
| wavlm_base_plus | scalar-1b-rot-gaussian | 800 | 0.4829 | 0.9703 | +0.4874 |
| wavlm_base_plus | scalar-1b-no-rot | 800 | 0.3437 | 0.9474 | +0.6037 |
| wavlm_base_plus | uniform-1b | 768 | 0.0443 | 0.8280 | +0.7837 |
| wavlm_base_plus | scalar-2b-rot-gaussian | 1568 | 0.6198 | 0.9848 | +0.3650 |
| wavlm_base_plus | scalar-2b-no-rot | 1568 | 0.4430 | 0.9620 | +0.5190 |
| wavlm_base_plus | uniform-2b | 1536 | 0.0726 | 0.8372 | +0.7646 |
| wavlm_base_plus | scalar-3b-rot-gaussian | 2336 | 0.7950 | 0.9961 | +0.2011 |
| wavlm_base_plus | scalar-3b-no-rot | 2336 | 0.5051 | 0.9720 | +0.4669 |
| wavlm_base_plus | uniform-3b | 2304 | 0.3434 | 0.9502 | +0.6068 |
| wavlm_base_plus | scalar-4b-rot-gaussian | 3104 | 0.8791 | 0.9988 | +0.1197 |
| wavlm_base_plus | scalar-4b-no-rot | 3104 | 0.5503 | 0.9770 | +0.4267 |
| wavlm_base_plus | uniform-4b | 3072 | 0.5182 | 0.9749 | +0.4567 |
