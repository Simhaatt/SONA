# E1 / E1b — does a label-free geometric metric predict task retention?

Scripts: `AudioVQ/scripts/ratio_at_k.py`, `AudioVQ/scripts/metric_predicts_task.py`.
3 embedding models × 19 methods; 36 points where a geometric metric and measured downstream
retention are both available.

---

## 1. The question

The paper argues Recall@k understates deployment quality under quantization. A practitioner's
immediate reply: *then what should I measure, short of running the downstream task — which needs
labels my deployment may not have?*

Two candidates, both purely geometric and label-free:

- **Recall@10** — the field default.
- **1/Ratio@10** — the inverse approximation ratio proposed by Dimitropoulos & Mamoulis
  (arXiv:2606.04522) for ANN *search*, computed here under *quantization*.

Ground truth is measured semantic retention: downstream speaker- and noise-type-retrieval precision
after quantization, divided by its float32 value.

## 2. Result

| metric | mean abs error vs semantic retention | Pearson r | Spearman ρ |
|---|---:|---:|---:|
| Recall@10 | 0.4280 | +0.754 | +0.689 |
| **1/Ratio@10** | **0.0328** | +0.732 | +0.674 |

**1/Ratio@10 is ~13× better calibrated than Recall@10** (0.033 vs 0.428 mean absolute error), while
the two rank methods almost identically.

The precise statement, which matters: **Recall@k preserves the *ordering* of quantizers but not the
*calibration*; 1/Ratio@k preserves both.** Recall will tell you which quantizer is better. It will
badly misinform you about how much you are losing.

At the paper's headline operating point (1 bit/coordinate, ~30× compression) on WavLM:

| | value |
|---|---|
| Recall@10 | 0.483 |
| 1/Ratio@10 | 0.970 |
| measured semantic retention | 0.965 |

The label-free geometric metric lands within half a point of the task measurement that recall
misses by 48 points.

## 3. The discrimination check — and why it was needed

An earlier version of this analysis covered only 12 points, all with semantic retention ≥ 0.96. A
high correlation over a narrow range is weak evidence however large the coefficient, and a metric
pinned near 1.0 everywhere would look calibrated on good quantizers while being useless in
practice.

Two fixes widened the range: an alias bug meant the `no-rot` variants never matched (the semantic
CSV uses a space, not a hyphen), and the uniform scalar quantizer was absent from the semantic
sweep entirely although the code already supported it. Uniform 1-bit destroys retrieval, so
including it is the case that decides whether the metric discriminates.

| model | Recall@10 | 1/Ratio@10 | semantic retention |
|---|---:|---:|---:|
| WavLM | 0.0443 | 0.828 | 0.889 |
| wav2vec2 | 0.0133 | 0.720 | 0.953 |
| HuBERT | 0.0058 | 0.777 | 0.946 |

**1/Ratio@10 does discriminate** — it drops to 0.72–0.83 on the broken quantizer rather than
sitting at 1.0. It is *conservative* there, under-predicting retention by 0.13–0.23, which is the
safe direction for a deployment metric.

## 4. An unplanned result that is stronger than the headline

Uniform 1-bit loses **95–99 % of true nearest neighbours** (Recall@10 = 0.006–0.044) while retaining
**89–95 % of downstream task accuracy**.

The geometry–meaning dissociation the paper is built on is therefore not a mild effect at moderate
compression. At aggressive compression it is close to total: the neighbour list is essentially
destroyed and the task barely notices. This is the most extreme demonstration in the project and it
came out of a check added to guard against a weak correlation, not from the planned sweep.

## 5. What to claim

1. Recall@k is well-ordered but badly calibrated under quantization; report 1/Ratio@k alongside it.
2. 1/Ratio@k is label-free, so it is available to deployments that cannot run a labelled downstream
   evaluation — which is the practical contribution.
3. The dissociation strengthens with compression, up to ~99 % neighbour loss at ~9 % task loss.

## 6. Limitations

- 36 points across 3 models and 4 bit-widths; not a general claim about all quantizers.
- Semantic retention here means speaker and noise-type retrieval on two corpora. Other downstream
  tasks may weight the neighbour list differently — a task that genuinely needs exact nearest
  neighbours would not show this.
- 1/Ratio@k under-predicts on badly broken quantizers, so it is a safety-biased estimate rather
  than an unbiased one.
- Both metrics are computed on 10 000-vector indices; §5.6 of the paper already shows behaviour
  shifts with index size.
