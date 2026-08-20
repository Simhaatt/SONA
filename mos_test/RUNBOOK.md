# Phase D — MOS listening test runbook

Everything is built and validated. What is missing is **human raters**. This is the only part of
the plan that cannot be done by running something.

---

## What you need

| | |
|---|---|
| **Raters** | **18–20 people.** See the power note below for why not 13. |
| Time per rater | **12–18 minutes** (50 clips, one rating each) |
| Equipment | Headphones, quiet room. Laptop speakers invalidate the quantization arm — the differences are small |
| Who | Normal-hearing adults. No audio expertise needed; ACR is designed for naive listeners |
| Network | Raters on the same LAN as the machine running the app, or a tunnel |

Recruiting 20 classmates for 15 minutes is realistic in a single afternoon. That is the whole
remaining cost of Phase D.

## Why 18–20 and not 13

The dry run (15 simulated raters) put 91 paired observations on the table against a requirement of
~88 to detect a 0.3 MOS difference at 80 % power — i.e. **exactly at the threshold**. Two of the 15
were dropped by attention-check filtering, which is a normal rate.

So budget for attrition: **20 recruited → ~17 kept → comfortable margin.** At 13 or fewer the
`analyze_mos.py` output will label the study a pilot and the equivalence result will be
inconclusive rather than negative, which is a much weaker thing to report.

## Running it

```bash
python mos_test/app.py
```

Serves on port **7861**. Raters open `http://<your-ip>:7861`, enter any rater id, and rate all 50
clips on the ACR 1–5 scale. Responses append to `mos_test/responses.csv`.

Verified working against the current 50-clip manifest: all clips serve, and **condition labels are
never sent to the browser** — the test is genuinely blind.

### Instructions to give raters

> Rate the **quality of the speech** on each clip from 1 (bad) to 5 (excellent). Some clips are
> noisy, some have been processed. Judge how pleasant and clear the speech sounds overall. Use
> headphones. Do not go back and change earlier ratings.

Do not tell them what the conditions are, that some clips are near-identical, or which comparison
matters. That is the point of blinding.

## What the test answers

50 clips = 7 utterances × 7 conditions + 1 attention check.

**Arm 1 — complex vs magnitude mask.** SI-SDR says the complex mask wins (+0.313 dB, p < 1e-6);
PESQ says it loses (−0.016, p = 0.005). The metrics contradict each other, so a human verdict
settles which one tracked perception.

**Arm 2 — FP32 vs mixed vs uniform INT8.** The paper claims the stem-exempt mixed model is
*indistinguishable* from FP32. That currently rests on p = 0.25, which is a **failure to reject**,
not a demonstration of sameness — an underpowered study produces the same p-value. Human ratings
plus an equivalence test convert this into a positive result.

The expected pattern, if the paper's claim holds:

| comparison | expected |
|---|---|
| `quant_fp32` vs `quant_mixed` | **equivalent** within ±0.3 MOS |
| `quant_fp32` vs `quant_uniform_int8` | uniform audibly **worse** |
| `quant_mixed` vs `quant_uniform_int8` | mixed **better** |

**If listeners do separate mixed from FP32, the claim must be softened.** Say so; do not re-run
until it comes out right.

## Analysis

```bash
python scripts/analyze_mos.py
```

Produces per-condition MOS with bootstrap CIs, paired comparisons, equivalence tests, and a
MOS-vs-STOI correlation.

Two things were added specifically for this test:

1. **Paired within (rater, utterance).** Every rater hears every utterance under every condition,
   so pairing removes rater severity and utterance difficulty from the error term. Far more
   sensitive than comparing marginal means at these sample sizes.
2. **TOST equivalence testing** against a **declared ±0.3 MOS margin** (override with
   `MOS_EQUIV_MARGIN`). Equivalence is concluded when the 90 % CI of the difference lies entirely
   inside the margin. The margin is stated up front rather than chosen after seeing the data — a
   reader can disagree with 0.3, but cannot accuse it of being fitted.

The script drops raters who fail the attention check, and prints a power note that flags the study
as under-powered if the paired sample is too small to support an equivalence conclusion.

## Validation already done

The full pipeline was dry-run on 15 simulated raters with a known ground truth, and recovered it:

| comparison | simulated truth | recovered |
|---|---|---|
| gtcrn vs complex_gtcrn | tie | no significant difference ✓ |
| fp32 vs mixed | equivalent | **EQUIVALENT**, TOST p = 0.0091 ✓ |
| fp32 vs uniform INT8 | uniform worse | fp32 better, p = 0.002 ✓ |
| noisy vs gtcrn | enhancement helps | gtcrn better, p = 1.8e-11 ✓ |

A deliberately careless simulated rater was caught by the attention check. So the analysis is known
to work *before* anyone spends time rating — if it had broken after collection, the data could not
have been regenerated.

## Reporting

- Under 10 raters: report as a **pilot listening study**, indicative only. The script says so.
- Report the equivalence margin in the paper text, not just the p-value.
- Report how many raters were excluded and why.
- If Arm 1 comes out a tie, that is a real finding: it means the SI-SDR/PESQ disagreement is below
  the perceptual threshold, and *neither* metric should have been trusted to call a winner.
