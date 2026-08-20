# Does a label-free geometric metric predict task retention?

Semantic retention is the measured ground truth: downstream retrieval precision after quantization divided by its float32 value, averaged over the speaker and noise-type tasks. Both candidate metrics are purely geometric and need no labels.

| model | method | semantic | Recall@10 | err | 1/Ratio@10 | err |
|---|---|---|---|---|---|---|
| hubert_base | scalar-1b-rot-gaussian | 0.9792 | 0.6982 | -0.2810 | 0.9851 | +0.0059 |
| hubert_base | scalar-2b-rot-gaussian | 0.9859 | 0.7771 | -0.2088 | 0.9925 | +0.0065 |
| hubert_base | scalar-3b-rot-gaussian | 0.9962 | 0.8696 | -0.1266 | 0.9977 | +0.0015 |
| hubert_base | scalar-4b-rot-gaussian | 1.0008 | 0.9212 | -0.0796 | 0.9992 | -0.0015 |
| wav2vec2_base | scalar-1b-rot-gaussian | 0.9631 | 0.5299 | -0.4332 | 0.9632 | +0.0000 |
| wav2vec2_base | scalar-2b-rot-gaussian | 0.9807 | 0.7085 | -0.2722 | 0.9870 | +0.0062 |
| wav2vec2_base | scalar-3b-rot-gaussian | 0.9924 | 0.7935 | -0.1989 | 0.9935 | +0.0012 |
| wav2vec2_base | scalar-4b-rot-gaussian | 0.9982 | 0.8965 | -0.1017 | 0.9985 | +0.0003 |
| wavlm_base_plus | scalar-1b-rot-gaussian | 0.9624 | 0.4829 | -0.4795 | 0.9703 | +0.0079 |
| wavlm_base_plus | scalar-2b-rot-gaussian | 0.9774 | 0.6198 | -0.3576 | 0.9848 | +0.0074 |
| wavlm_base_plus | scalar-3b-rot-gaussian | 0.9946 | 0.7950 | -0.1996 | 0.9961 | +0.0015 |
| wavlm_base_plus | scalar-4b-rot-gaussian | 0.9987 | 0.8791 | -0.1196 | 0.9988 | +0.0001 |

## Summary

| metric | mean abs error vs semantic | Pearson r | Spearman rho |
|---|---|---|---|
| Recall@10 | 0.2382 | +0.984 | +0.993 |
| **1/Ratio@10** | **0.0033** | +0.972 | +0.993 |

## Discrimination check

A metric pinned near 1.0 everywhere would be useless. Uniform 1-bit destroys retrieval and is the case that decides whether 1/Ratio@10 discriminates:

| model | Recall@10 | 1/Ratio@10 | semantic |
|---|---|---|---|
