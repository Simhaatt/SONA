# Does a label-free geometric metric predict task retention?

Semantic retention is the measured ground truth: downstream retrieval precision after quantization divided by its float32 value, averaged over the speaker and noise-type tasks. Both candidate metrics are purely geometric and need no labels.

| model | method | semantic | Recall@10 | err | 1/Ratio@10 | err |
|---|---|---|---|---|---|---|
| hubert_base | scalar-1b-no-rot | 0.9918 | 0.4292 | -0.5626 | 0.9458 | -0.0461 |
| hubert_base | scalar-1b-rot-gaussian | 0.9792 | 0.6982 | -0.2810 | 0.9851 | +0.0059 |
| hubert_base | scalar-2b-no-rot | 0.9958 | 0.5877 | -0.4081 | 0.9723 | -0.0235 |
| hubert_base | scalar-2b-rot-gaussian | 0.9859 | 0.7771 | -0.2088 | 0.9925 | +0.0065 |
| hubert_base | scalar-3b-no-rot | 1.0016 | 0.6347 | -0.3669 | 0.9787 | -0.0229 |
| hubert_base | scalar-3b-rot-gaussian | 0.9962 | 0.8696 | -0.1266 | 0.9977 | +0.0015 |
| hubert_base | scalar-4b-no-rot | 1.0022 | 0.7141 | -0.2881 | 0.9861 | -0.0161 |
| hubert_base | scalar-4b-rot-gaussian | 1.0008 | 0.9212 | -0.0796 | 0.9992 | -0.0015 |
| hubert_base | uniform-1b | 0.9460 | 0.0058 | -0.9402 | 0.7768 | -0.1692 |
| hubert_base | uniform-2b | 0.9675 | 0.1681 | -0.7994 | 0.8520 | -0.1155 |
| hubert_base | uniform-3b | 0.9965 | 0.4850 | -0.5115 | 0.9601 | -0.0363 |
| hubert_base | uniform-4b | 1.0021 | 0.7266 | -0.2755 | 0.9886 | -0.0134 |
| wav2vec2_base | scalar-1b-no-rot | 0.9651 | 0.5245 | -0.4406 | 0.9603 | -0.0048 |
| wav2vec2_base | scalar-1b-rot-gaussian | 0.9631 | 0.5299 | -0.4332 | 0.9632 | +0.0000 |
| wav2vec2_base | scalar-2b-no-rot | 0.9826 | 0.6604 | -0.3222 | 0.9818 | -0.0008 |
| wav2vec2_base | scalar-2b-rot-gaussian | 0.9807 | 0.7085 | -0.2722 | 0.9870 | +0.0062 |
| wav2vec2_base | scalar-3b-no-rot | 0.9958 | 0.7925 | -0.2033 | 0.9936 | -0.0022 |
| wav2vec2_base | scalar-3b-rot-gaussian | 0.9924 | 0.7935 | -0.1989 | 0.9935 | +0.0012 |
| wav2vec2_base | scalar-4b-no-rot | 0.9994 | 0.8739 | -0.1255 | 0.9978 | -0.0017 |
| wav2vec2_base | scalar-4b-rot-gaussian | 0.9982 | 0.8965 | -0.1017 | 0.9985 | +0.0003 |
| wav2vec2_base | uniform-1b | 0.9534 | 0.0133 | -0.9401 | 0.7201 | -0.2333 |
| wav2vec2_base | uniform-2b | 0.9774 | 0.2258 | -0.7516 | 0.8634 | -0.1140 |
| wav2vec2_base | uniform-3b | 1.0003 | 0.5665 | -0.4338 | 0.9682 | -0.0321 |
| wav2vec2_base | uniform-4b | 1.0040 | 0.7470 | -0.2570 | 0.9908 | -0.0133 |
| wavlm_base_plus | scalar-1b-no-rot | 0.9711 | 0.3437 | -0.6274 | 0.9474 | -0.0236 |
| wavlm_base_plus | scalar-1b-rot-gaussian | 0.9624 | 0.4829 | -0.4795 | 0.9703 | +0.0079 |
| wavlm_base_plus | scalar-2b-no-rot | 0.9842 | 0.4430 | -0.5412 | 0.9620 | -0.0222 |
| wavlm_base_plus | scalar-2b-rot-gaussian | 0.9774 | 0.6198 | -0.3576 | 0.9848 | +0.0074 |
| wavlm_base_plus | scalar-3b-no-rot | 0.9949 | 0.5051 | -0.4898 | 0.9720 | -0.0230 |
| wavlm_base_plus | scalar-3b-rot-gaussian | 0.9946 | 0.7950 | -0.1996 | 0.9961 | +0.0015 |
| wavlm_base_plus | scalar-4b-no-rot | 0.9988 | 0.5503 | -0.4485 | 0.9770 | -0.0218 |
| wavlm_base_plus | scalar-4b-rot-gaussian | 0.9987 | 0.8791 | -0.1196 | 0.9988 | +0.0001 |
| wavlm_base_plus | uniform-1b | 0.8891 | 0.0443 | -0.8448 | 0.8280 | -0.0611 |
| wavlm_base_plus | uniform-2b | 0.9247 | 0.0726 | -0.8521 | 0.8372 | -0.0875 |
| wavlm_base_plus | uniform-3b | 0.9855 | 0.3434 | -0.6421 | 0.9502 | -0.0353 |
| wavlm_base_plus | uniform-4b | 0.9953 | 0.5182 | -0.4771 | 0.9749 | -0.0204 |

## Summary

| metric | mean abs error vs semantic | Pearson r | Spearman rho |
|---|---|---|---|
| Recall@10 | 0.4280 | +0.754 | +0.689 |
| **1/Ratio@10** | **0.0328** | +0.732 | +0.674 |

## Discrimination check

A metric pinned near 1.0 everywhere would be useless. Uniform 1-bit destroys retrieval and is the case that decides whether 1/Ratio@10 discriminates:

| model | Recall@10 | 1/Ratio@10 | semantic |
|---|---|---|---|
| wavlm_base_plus | 0.0443 | 0.8280 | 0.8891 |
| wav2vec2_base | 0.0133 | 0.7201 | 0.9534 |
| hubert_base | 0.0058 | 0.7768 | 0.9460 |
