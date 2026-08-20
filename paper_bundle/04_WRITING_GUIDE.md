# How to write this paper — and how not to sound like a machine wrote it

Two halves. **Part A** is how research papers are actually written and what reviewers respond to.
**Part B** is the specific, evidence-backed problem of prose that reads as LLM-generated, which is
now a real risk to credibility in 2026 — editors and reviewers actively flag it, and the markers
are documented in the literature rather than folklore.

Read Part B properly. It is not about cheating detectors. It is that LLM-default prose is *bad
scientific writing* for identifiable reasons, and fixing those reasons makes the paper better
regardless of who or what drafted it.

---

# PART A — How research papers are actually written

## A1. The single most useful principle

George Gopen and Judith Swan, *The Science of Scientific Writing* (American Scientist 78(6), 1990)
— still the best thing written on this, and short. Its core claim:

> The meaning of prose is not what the writer intends, but what the reader interprets. Readers
> interpret more easily and uniformly when information sits where they expect it.

Concrete rules that follow, all of which you can apply mechanically:

1. **Put the subject and verb close together.** Readers hold everything between them in working
   memory and cannot start interpreting until the verb arrives.
2. **Old information first, new information last.** The start of a sentence links to what came
   before; the end is the stress position where emphasis lands. If your important point is in the
   middle of a sentence, readers will not weight it.
3. **One unit of discourse, one point.** If a paragraph has two points, it has none.
4. **The topic position of a paragraph sets what the paragraph is about.** Do not open a paragraph
   with a subordinate clause about something else.

Example from your own material, before and after:

> **Weak:** Because the stem convolution is only 3.8 KB and represents 0.133 % of the model's
> weights, which is a very small fraction, it was surprising that when it was exempted from
> quantization the resulting model was found to be statistically indistinguishable from FP32.

> **Better:** The stem convolution is 3.8 KB — 0.133 % of the model's weights. Exempting it from
> quantization yields a model statistically indistinguishable from FP32.

The second version puts the surprising fact in the stress position, splits one overloaded sentence
into two, and drops the throat-clearing ("it was surprising that", "which is a very small
fraction"). It is also 60 % shorter.

## A2. Structure

The advice that actually works: **find three recent papers in your target venue making a
contribution of similar *shape* to yours, and follow their structure.** Structure is
discipline-specific and sub-area-specific — a benchmark/analysis paper looks nothing like an
algorithms paper. Do not invent a structure.

For this paper (an empirical analysis paper with a methodological thesis), the shape is:

| section | job | length |
|---|---|---|
| Abstract | the whole paper in 200 words, with numbers | 1 para |
| Introduction | problem → gap → what we did → contributions list | 1–1.5 pp |
| Related work | position, do not survey | 0.75 pp |
| Framework | make the thesis falsifiable | 0.5 pp |
| Studies 1–3 | method + result per leg | 5–6 pp |
| Remedies | validated countermeasures | 1.5 pp |
| Human validation | MOS as arbiter | 0.75 pp |
| Reporting protocol | the generalizable contribution | 0.5 pp |
| Limitations | honest, specific, non-performative | 0.5 pp |
| Conclusion | what changed about what we know | 1 para |

## A3. The contributions list

Reviewers read the abstract, the contributions list, and the figures. Then they decide. Everything
else confirms or undermines that decision.

Rules for the list:
- **Each item is a claim you can point at a table for.** If you cannot name the table, it is not a
  contribution.
- **Quantify.** "We show quantization damage is concentrated" is weak. "0.133 % of weights carry
  98.7 % of INT8 loss" is a contribution.
- **State scope in the item itself.** Not "we show X"; "we show X on three self-supervised models
  and two corpora."
- **Four to six items.** Fewer looks thin; more looks padded.

## A4. Related work positions, it does not summarise

The failure mode is a list of paragraphs each summarising one paper. Nobody reads it and it wins
you nothing.

What works: state what is already known, cite it, and say explicitly what you are *not* claiming.
This paper is in unusually good shape here, because the citation pass found genuine prior work on
the central claim. Use that:

> That Recall@k is a pessimistic proxy is not our discovery. Dimitropoulos & Mamoulis show it for
> ANN search approximation and propose 1/Ratio@k. Their setting differs in the way that matters:
> they hold vectors exact and vary search effort, so a missed neighbour is merely a slightly more
> distant true vector; under quantization the vectors themselves are damaged.

Conceding the prior claim and then drawing a sharp distinction is *stronger* than not citing it.
Reviewers who know the field will find the paper anyway, and the ones who do not will trust you
more for volunteering it.

## A5. Reporting negative and corrected results

This paper is full of them, and that is an asset if framed right. The framing that works:

- **State the hypothesis you actually held, before the result.** "We set out to test whether X. It
  does not." Reviewers can tell the difference between a pre-registered prediction and a
  rationalised outcome.
- **Say what would have falsified it.** Your crest-factor prediction was recorded in advance with
  an explicit falsification condition. Say so — it is rare and it is worth points.
- **When you narrow a claim, show the test that could have preserved the stronger version.** You
  narrowed the crest factor from a quantitative predictor to a binary screen after four tests. That
  history is evidence of discipline; do not hide it in a footnote.
- **Corrections are strengths when you found them yourself.** The scene classifier's 0.844 → 0.748
  under grouped CV, and the WER `bus` effect vanishing under FDR, are both cases where you caught
  your own optimism. A reviewer catching them would be fatal; you catching them is credibility.

## A6. Things reviewers reliably punish

- Claims in the abstract that the tables do not support.
- Absolute timings without hardware and load stated. (You fixed this; keep it fixed.)
- Missing the obvious baseline. (Which is exactly why the Hessian column had to exist.)
- Significance without multiplicity correction when many tests are run.
- "Future work will..." where a remedy is claimed but not validated.
- Figures that require reading the caption to understand the axes.

---

# PART B — Not sounding like an LLM wrote it

## B1. Why this matters now, factually

This is measurable, not vibes. The reference study is Kobak et al., *Delving into ChatGPT usage in
academic writing through excess vocabulary* (arXiv:2406.07016), which analysed **14.2 million
PubMed abstracts (2010–2024)**, projecting expected 2024 word frequencies from 2021–22 and
measuring the excess.

Findings that matter to you:

- At least **10 % of 2024 abstracts** show LLM processing; **over 30 % in computational fields.**
- The 2024 vocabulary surge was **almost entirely stylistic verbs and adjectives**, unlike previous
  shifts (COVID drove *content* words like "coronavirus"). That signature — flowery style words
  with no new subject matter — is the tell.
- Frequency ratios: **"delves" ×25.2**, "showcasing" ×9.2, "underscores" ×9.1.

So the risk is not a detector tool. It is a human reviewer in a computational field, where the base
rate is above 30 %, recognising the register.

## B2. The word list — search and destroy

**Highest risk (near-diagnostic).** If any of these survive into your final draft, you have a
problem:

> delve / delves / delving · underscore / underscores · showcase / showcasing · intricate ·
> meticulous / meticulously · pivotal · realm · tapestry · commendable · indispensable ·
> interplay · multifaceted · testament

**High frequency, still flagged.** Not wrong words, but the density is the signal — an LLM uses
them two to three times as often as a person:

> comprehensive · crucial · notably · particularly · additionally · furthermore · moreover ·
> insights · potential · enhancing · leveraging · robust · nuanced · seamless · foster ·
> landscape · navigate · align / alignment · significant (when not statistical) · myriad

**Phrase-level tells:**

> "It is important to note that" · "It is worth noting that" · "plays a crucial role in" ·
> "sheds light on" · "paves the way for" · "a deeper understanding of" · "in the realm of" ·
> "stands as a testament to" · "not only … but also" (as a reflex) · "In conclusion," ·
> "Overall," opening a paragraph

Practical step: before submission, grep the manuscript for the first list and count the second.

```bash
grep -oiE "delv|underscore|showcas|intricate|meticulous|pivotal|realm|tapestry|commendable|multifacet|testament|interplay" main.tex | sort | uniq -c | sort -rn
```

## B3. The structural tells, which matter more than the words

Swapping words is cosmetic. These are the real markers:

**1. Relentless three-item lists.** LLMs default to triples everywhere — three contributions, three
reasons, three limitations, three adjectives. Real arguments have uneven shapes: sometimes two
reasons, sometimes five. Vary it deliberately.

**2. Every paragraph the same length.** Human technical writing has a four-line paragraph next to a
fourteen-line one, because the ideas are not the same size. Uniform paragraphs read as generated.

**3. Symmetrical hedging.** "While X has advantages, it also has limitations." "Although promising,
challenges remain." This balances every claim into saying nothing. Commit: say which way the
evidence points, then state the specific exception.

**4. Summary sentences that add nothing.** "In summary, these results demonstrate the importance of
careful evaluation." Delete every sentence that would survive unchanged in a different paper.

**5. No numbers where numbers belong.** LLM prose says "significantly improved" where a person
writes "+0.313 dB (p < 1e-6)". Your paper has the numbers — use them in the prose, not only in
tables.

**6. Uniform register.** Real papers shift: terse in Methods, discursive in Discussion, blunt in
Limitations. Generated text holds one temperature throughout.

**7. Absence of specificity and texture.** The most-cited human marker: generated text has no
particular fingerprints. Yours should be full of them — ORT 1.20.1 emitting 0 QDQ nodes for the
GRU, the machine sitting at 6 % load, a p-value of exactly 0.25. Concrete, checkable, slightly
awkward details are the strongest anti-signal there is, and they are also just better science.

## B4. What to do instead — a concrete rewrite pass

Take one paragraph at a time and apply these in order:

1. **Delete throat-clearing.** "It is important to note that the stem is small" → "The stem is
   small."
2. **Replace an adjective with a number.** "substantially smaller" → "3.8 KB, 0.133 % of weights."
3. **Break the symmetry.** If the paragraph balances a pro against a con, decide which one wins and
   say so.
4. **Vary sentence length deliberately.** Put a four-word sentence next to a thirty-word one. This
   is the single fastest way to make prose stop sounding generated.
5. **Add the specific detail you left out** because it felt too in-the-weeds. It is the good part.
6. **Read it aloud.** Anything you would not say to a colleague at a whiteboard gets cut.

## B5. A worked example, in your material

**LLM-default version:**

> It is important to note that quantization plays a crucial role in enabling the deployment of deep
> learning models on resource-constrained devices. Our comprehensive analysis delves into the
> intricate relationship between layer sensitivity and quantization error, providing valuable
> insights into how practitioners might approach mixed-precision allocation. While uniform
> quantization offers simplicity, our findings underscore the potential benefits of a more nuanced,
> targeted approach.

Every marker is present: "important to note", "plays a crucial role", "comprehensive", "delves",
"intricate", "valuable insights", "While X, Y" symmetry, "underscore", "potential", "nuanced" — and
not one number.

**Rewrite:**

> We expected quantization damage to be spread across layers. It is not. A single 3.8 KB stem
> convolution — 0.133 % of the weights — carries 98.7 % of the INT8 loss on held-out data.
> Exempting it gives a model statistically indistinguishable from FP32 (p = 0.25) at 3.2×
> compression, and the resulting file is *smaller* than uniform INT8, because quantizing everything
> makes ONNX Runtime insert 143 extra QDQ nodes.

Shorter, entirely concrete, has a surprise in it, and no reviewer will wonder who wrote it.

## B6. What not to over-correct

Do not manufacture errors, slang, or false personality to seem human. Reviewers punish that harder
than blandness. And do not avoid a precise word because it appears on a list — if a mechanism
genuinely *is* intricate, and you have measured it, say so. The lists above describe *unearned*
usage; earned usage is fine.

The target is not "sounds human." It is **specific, committed, and quantified.** Prose with those
three properties does not read as generated, because current models do not default to any of them.

---

## Checklist before submission

- [ ] Every abstract claim traced to a table or figure
- [ ] `audit_paper_numbers.py` passes for both studies
- [ ] Hardware and load stated for every absolute timing
- [ ] Multiplicity correction reported wherever many tests were run
- [ ] Every remedy validated, none merely proposed
- [ ] Grep for the B2 word list returns nothing from the first group
- [ ] Paragraph lengths vary visibly down each page
- [ ] No sentence would survive unchanged in a different paper
- [ ] Read aloud end to end
- [ ] Author names and affiliation replace the `SPARC FYP Team` placeholder

## Sources

- Gopen & Swan, *The Science of Scientific Writing*, American Scientist 78(6):550–558, 1990 —
  https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf
- Kobak et al., *Delving into ChatGPT usage in academic writing through excess vocabulary*,
  arXiv:2406.07016 — https://arxiv.org/html/2406.07016v1
- FSU research on why ChatGPT overuses particular words —
  https://news.fsu.edu/news/science-technology/2025/02/17/why-does-chatgpt-delve-so-much-fsu-researchers-begin-to-uncover-why-chatgpt-overuses-certain-words/
- Longitudinal PubMed word-overuse study (2020–2024) —
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12679996/
- On the unreliability of human and automated AI detection —
  https://arxiv.org/pdf/2312.05241
