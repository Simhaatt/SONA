# Venue — decision deferred, format decided

**Decision taken 2026-08-21:** write the paper in **IEEE format now**, choose the venue later.

That is a safe order, because in IEEEtran the conference/journal choice is a **single
documentclass option**. Nothing about the writing is locked by deferring the venue:

```latex
\documentclass[conference]{IEEEtran}   % what both existing drafts use
\documentclass[journal]{IEEEtran}      % switch for TASLP / Access — one line
```

Both existing manuscripts are already IEEEtran and compile:

- `AudioVQ/paper/recall_wrong_yardstick.pdf` — 5 pp, 16 refs
- `PERCEPTUAL_MP/paper/stem_exempt_quantization.pdf` — 4 pp, 6 refs

Write the merged paper by reusing their sections rather than starting from a blank file.

---

## The constraint: **no budget**

This rules out more than it first appears, and it inverts a common assumption.

| venue | cost to publish | notes |
|---|---:|---|
| **IEEE/ACM TASLP** | **$0** | subscription model, no APC. ⚠️ overlength page charges apply beyond the standard allowance |
| **arXiv** | **$0** | not peer-reviewed; permanently citable, timestamps the work |
| IEEE Access | ~$1,950 APC | broad scope, ~4–6 week decision |
| ICASSP OJSP track | $1,795 APC **+** author registration | worst of both |
| ICASSP 2026 | **$900–$1,240** author registration + travel | see the trap below |
| Interspeech | ~€600–800 + travel | same structure |

### The conference trap — verified against ICASSP 2026's fee table

| category | author rate (required to publish) |
|---|---:|
| IEEE SPS **Student** member | **$900** |
| IEEE **Student** member | $1,035 |
| Student, no IEEE membership | **$1,240** |

The cheap student rates ($250–$395) are **non-author only**. Every accepted paper must be covered
by at least one full **Author Registration**, and there is **no student discount on it** — a
student author pays the same as a professor. Plus flights, hotel and visa.

**So conferences are the most expensive option available, not the cheapest.** With no budget the
constraint pushes toward journals, which is counterintuitive and worth stating plainly.

---

## Recommendation when the time comes

**1. arXiv the day the draft is finished.** Free, immediate, zero risk, and TASLP permits
preprints. It also timestamps the work, which matters here: Dimitropoulos & Mamoulis published the
adjacent claim in June 2026, and the citation pass already had to narrow one contribution because
of it.

**2. IEEE/ACM TASLP** as the peer-reviewed target. Free to publish and the flagship venue for this
area.

Two consequences to plan for:

- **Study 3 (embedding retrieval) probably gets cut.** Page pressure plus TASLP's scope. That
  loses the `1/Ratio@k` result from *this* paper — but `recall_wrong_yardstick.pdf` already exists
  standalone, so it becomes its own submission rather than being wasted.
- **Review takes 6–12 months.** Reviewers will push on the two caveats already documented in
  `01_PAPER_DETAILS.md`: no listening test, and two architectures sharing a backbone. Both are
  written into the limitations, which is the right place. If a reviewer demands the listening test,
  it is built and validated — see `mos_test/RUNBOOK.md`.

## Before committing to TASLP, verify

- **The current overlength page allowance and per-page charge.** The paper is scoped at 12–16 pp;
  this should shape the outline rather than surprise you at submission. It is the one way a "free"
  venue can still generate an invoice.
- Whether your university holds an **IEEE open-access agreement** — some do, which would put IEEE
  Access back on the table at no cost to you.
- Whether the department has a **conference budget**, and whether **ISCA / IEEE SPS student grants
  or fee waivers** apply. Both bodies run programmes that can cover author registration for
  students without institutional funding. Worth asking your supervisor — it is a normal request.

## Still outstanding regardless of venue

**Author names.** `SPARC FYP Team` and the project email are placeholders in both manuscripts. Real
names and affiliation go in before submission anywhere.
