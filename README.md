# chartcheck

**Auditable, omission-aware evaluation of clinical AI text — scored against the source record, claim by claim.**

chartcheck scores LLM-generated clinical summaries and chart Q&A on the
[PDSQI-9](https://arxiv.org/abs/2501.08977) attributes, but it does the two hard
attributes — *Accurate* and *Thorough* — the right way: as **claim-level entailment
in both directions**, with a full audit trail. No black-box 1–5 from a judge that
read the whole chart and shrugged.

It runs out of the box with **no API key and no dependencies** (a heuristic backend,
for demos and CI). Point it at a Claude model for real evaluation.

---

## The problem

The PDSQI-9 is a validated instrument for scoring LLM clinical summaries. Its own
stated limitation, and the thing every health system bumps into when they try to
scale it: the *Accurate* and *Thorough* attributes *"require someone or something
to review the actual record."* You cannot grade faithfulness or completeness
without comparing the output **to the source**. So teams reach for LLM-as-judge,
and then can't tell whether to trust the judge.

chartcheck is a turnkey answer to that specific gap.

## The idea

The two record-dependent attributes are not "read the whole 200-page chart"
problems. They are claim-level entailment problems pointed in two directions:

| PDSQI-9 attribute | What it really is | chartcheck metric |
|---|---|---|
| **Accurate** | every claim in the output is entailed by the record | **precision** (1 − hallucination rate) |
| **Thorough** | every *salient* fact in the record appears in the output | **recall** (1 − omission rate) |

Decompose the output into atomic claims, check each against the record → precision.
Decompose the record into salient facts, check each against the output → recall.
This is the [FActScore](https://arxiv.org/abs/2305.14251) decompose-then-verify
idea, run **both ways** — because in clinical summaries the dangerous failure is
usually the thing that was *left out*, not the thing that was made up. (Tang et al.,
*npj Digital Medicine* 2023, found GPT-4 medical summaries omitted clinically
relevant information in ~47% of cases — as often as they hallucinated.) Precision-only
faithfulness metrics are blind to exactly the error that hurts patients most.

## Why this is different from "just run an LLM judge"

1. **Bidirectional + omission-first.** Most eval tooling scores faithfulness
   (precision) only. chartcheck makes *omission* a first-class, equally-weighted axis.
2. **Salience is explicit, not hand-waved.** Recall against *every* atomic fact in a
   chart punishes good, succinct summaries. So "what counts as salient" is a
   pluggable, task-specific object (`SalienceSpec`) — the honest version of
   "generate the rubric from the record." A discharge summary and a medication-rec
   note care about different facts; you say which.
3. **Glass-box, not a number.** Every sub-score traces to specific claims and quoted
   evidence spans. You can show a clinician *why* a summary scored 0.43.
4. **It knows how much to trust itself.** `calibrate()` measures the judge's
   agreement with human labels (quadratic weighted κ) and **refuses to call results
   trustworthy until agreement clears a threshold** — operationalizing the
   error-analysis-first discipline ([Hamel Husain](https://hamel.dev/blog/posts/evals-faq/)).
5. **The evaluator is itself tested.** `meta_eval()` injects *known* defects
   (omission, hallucination, value error, negation flip, wrong patient) into good
   summaries and reports how often chartcheck catches each — a sensitivity profile
   for your judge, not just for the model under test.
6. **A deterministic floor.** Unambiguous, never-acceptable failures (e.g. "no known
   drug allergies" when the record lists one) are caught by hard rules, not left to a
   probabilistic judge.

## Install

```bash
git clone https://github.com/sanjaybasu/chartcheck
cd chartcheck
pip install -e .            # core, zero dependencies
pip install -e ".[llm]"     # add the Anthropic backend for real evaluation
```

## Quickstart (no API key)

```python
from chartcheck import evaluate, Task

card = evaluate(record_text, summary_text, task=Task.pdsqi_summary())
print(card.summary())
print(card.omissions)        # the salient facts the summary dropped
print(card.hallucinations)   # claims unsupported by / contradicting the record
```

```bash
python examples/quickstart.py
```

scores a deliberately flawed summary of a synthetic record:

```
chartcheck scorecard  (pdsqi_summary)   [uncalibrated]
------------------------------------------------------------
  Accurate (precision) : 0.43   4/7 claims unsupported
  Thorough (recall)    : 0.42   11/19 salient facts omitted
  ...
  Unsupported / contradicted claims:
    [contradicted] Hemoglobin A1c is 7.2%        -> value in claim differs from record
    [contradicted] No known drug allergies       -> negation/polarity differs from record
  Omitted salient facts:
    (allergies and adverse reactions) Penicillin (anaphylaxis).
    (abnormal or actionable results) potassium 5.3 mmol/L (high)
    (abnormal or actionable results) eGFR 38 mL/min (low)
    ...
  Deterministic findings:
    [critical] Output states no allergies, but the record documents one.
```

The same flawed summary that a holistic judge might wave through with a "4/5 —
looks complete" gets the dropped penicillin allergy, the wrong A1c, and the missing
hyperkalemia surfaced as discrete, quotable findings.

## Real evaluation (Claude backend)

```python
from chartcheck import evaluate, AnthropicBackend, Task

card = evaluate(record, summary, task=Task.pdsqi_summary(),
                backend=AnthropicBackend(model="claude-opus-4-8"))
```

`AnthropicBackend` does claim decomposition, salient-fact extraction, bidirectional
entailment, and rubric judging with the model. (`export ANTHROPIC_API_KEY=...`.)

## Calibrate before you trust it

```python
from chartcheck import calibrate

report = calibrate(labeled_examples, threshold=0.6)   # [{record, output, human:{accurate:1-5,...}}]
print(report.summary())   # per-axis weighted κ; "TRUSTWORTHY" only if it clears threshold
```

## Stress-test the evaluator itself

```python
from chartcheck import meta_eval

report = meta_eval([{"record": record, "output": good_summary}])
print(report.summary())   # per-failure-mode detection sensitivity
```

## How this maps to the approaches people propose

The conversation that prompted this listed three:

- **Synthetic stub records targeting specific failure modes** → `meta_eval()` + `inject()`.
- **Dynamic, record-derived rubrics** → `SalienceSpec`: salience is derived per task
  rather than fixed, without pretending the hard part (what matters) is solved for free.
- **Reference datasets with task-specific rubrics** → bring your own records;
  `Task` carries the rubric and salience.

And it respects the two cautions experts always raise: *define the objective first*
(you choose the axes and salience) and *do error analysis before scaling* (calibration
is a required step, not an afterthought).

## Honest limitations

- The **offline backend is a lexical heuristic** for demos and CI — it is not
  accurate enough for real scoring. Use `AnthropicBackend` (or implement the
  five-method `Backend` interface for another model).
- **Salience is genuinely hard.** chartcheck makes it explicit and pluggable; it does
  not pretend to solve "what matters in this chart" universally.
- Claim decomposition quality bounds everything downstream (a known sensitivity of
  decompose-then-verify metrics). Calibrate on your own data and tasks.
- This is an evening project, not a certified medical device. It is decision support
  for *evaluation*, physician-facing, and should be validated before operational use.

## Related work

- **PDSQI-9** — Croxford et al., *JAMIA* 2025 ([10.1093/jamia/ocaf068](https://doi.org/10.1093/jamia/ocaf068); [arXiv:2501.08977](https://arxiv.org/abs/2501.08977)). The rubric chartcheck scores.
- **Epic's [evaluation-instruments](https://github.com/epic-open-source/evaluation-instruments)** — an open PDSQI-9 *LLM-as-judge* implementation. Complementary: chartcheck adds the auditable grounding layer for the two record-dependent axes.
- **FActScore** — Min et al., EMNLP 2023 ([arXiv:2305.14251](https://arxiv.org/abs/2305.14251)). Decompose-then-verify, precision-only; chartcheck runs it bidirectionally.
- **Error analysis for LLM evals** — [Hamel Husain](https://hamel.dev/blog/posts/evals-faq/). Why calibration comes first.
- **Omission in clinical summaries** — Tang et al., *npj Digital Medicine* 2023 ([s41746-023-00896-7](https://www.nature.com/articles/s41746-023-00896-7)).

## License

MIT
