# chartcheck

Reference-grounded evaluation of clinical AI text (note summaries, chart Q&A),
scored on the [PDSQI-9](https://arxiv.org/abs/2501.08977) attributes. The two
attributes that depend on the source record — *Accurate* and *Thorough* — are
computed as claim-level entailment in both directions, with a per-claim audit trail.

The library runs with no API key and no third-party dependencies (a heuristic
backend, for demonstration and CI). A Claude-based backend is provided for model-graded
evaluation.

---

## Background

The PDSQI-9 (Croxford et al., *JAMIA* 2025) is a validated, nine-attribute
instrument for scoring LLM-generated clinical summaries. Two of its attributes,
*Accurate* and *Thorough*, require comparison of the output against the source
record; they cannot be scored from the output alone. This is the step that does not
scale by human review, and the reason teams substitute an LLM judge.

Two observations from the evaluation literature motivate chartcheck's design:

1. *Faithfulness is measured by decomposition.* Reference-grounded factuality metrics
   decompose a generation into atomic claims and verify each against a source
   ([FActScore](https://arxiv.org/abs/2305.14251), Min et al., EMNLP 2023). This
   gives claim-level, attributable scores rather than a single holistic judgment.

2. *Omission and hallucination are distinct, and both are material.* In a 450-note
   evaluation, omissions occurred about twice as often as hallucinations (3.45% vs
   1.47% of sentences), while hallucinations were more often rated "major" (44% vs
   16.7%) (Asgari et al., *npj Digital Medicine* 2025;8:274). A precision-only
   faithfulness score does not observe omission at all.

chartcheck applies decomposition in both directions so that the two error classes
are measured separately:

| PDSQI-9 attribute | Operationalization | Metric |
|---|---|---|
| *Accurate* | each claim in the output is entailed by the record | precision (1 − unsupported-claim rate) |
| *Thorough* | each salient fact in the record appears in the output | recall (1 − omission rate) |

The remaining seven attributes do not require the record and are scored by a
structured rubric judge.

## Design

The library follows several established evaluation practices:

- *Bidirectional claim checking.* The output is decomposed into atomic claims and
  each is checked against the record (precision); the record is decomposed into
  salient facts and each is checked against the output (recall). Scores are reported
  per claim and per fact, with quoted evidence spans.

- *Salience is treated as a task-dependent input, not a constant.* Scoring recall
  against every atomic fact in a record penalizes appropriately concise summaries,
  and which facts are salient differs by task (a medication-reconciliation note and a
  discharge summary do not share the same critical set). chartcheck represents this as
  an explicit `SalienceSpec` per `Task`. This is the component most dependent on local
  definition and is documented as such.

- *Calibration precedes aggregate scoring.* Error analysis on real traces is the
  recommended first step before relying on an automated judge (Husain, *LLM Evals*).
  `calibrate()` compares the judge against human labels using a quadratic weighted
  Cohen's κ and reports per-attribute agreement; it does not return a "trustworthy"
  verdict until agreement clears a configurable threshold. This mirrors how the
  PDSQI-9 itself was validated (inter-rater ICC, Cronbach's α).

- *The evaluator is evaluated.* `meta_eval()` injects known defects (omission,
  hallucination, value error, negation flip, wrong patient) into reference summaries
  and reports detection sensitivity by defect class, so the evaluator's blind spots
  are measured rather than assumed.

- *Deterministic checks for unambiguous failures.* Some failures (e.g. an output
  asserting no allergies when the record documents one) are categorical and are
  handled by rules rather than a probabilistic judge.

## Install

```bash
git clone https://github.com/sanjaybasu/chartcheck
cd chartcheck
pip install -e .            # core, no dependencies
pip install -e ".[llm]"     # adds the Anthropic backend
```

## Usage

```python
from chartcheck import evaluate, Task

card = evaluate(record_text, summary_text, task=Task.pdsqi_summary())
print(card.summary())
card.omissions        # salient facts absent from the output
card.hallucinations   # claims unsupported by, or contradicting, the record
card.pdsqi9()         # all nine attributes on a 1-5 scale
```

For model-graded evaluation:

```python
from chartcheck import evaluate, AnthropicBackend
card = evaluate(record, summary, backend=AnthropicBackend(model="claude-opus-4-8"))
```

## Reproducible demonstration

`python examples/quickstart.py` scores a deliberately flawed summary of a synthetic
record. The flawed summary omits an allergy and several abnormal results, misstates
an A1c value, and asserts no allergies:

```
  Accurate (precision) : 0.43   4/7 claims unsupported
  Thorough (recall)    : 0.42   11/19 salient facts omitted
  Unsupported / contradicted claims:
    [contradicted] Hemoglobin A1c is 7.2%        -> value in claim differs from record
    [contradicted] No known drug allergies       -> negation/polarity differs from record
  Omitted salient facts:
    (allergies and adverse reactions) Penicillin (anaphylaxis).
    (abnormal or actionable results) potassium 5.3 mmol/L (high)
  Deterministic findings:
    [critical] Output states no allergies, but the record documents one.
```

A faithful summary of the same record scores 0.92 precision / 1.00 recall. The
findings are reported as discrete, quotable items rather than a single score.
`examples/meta_eval_demo.py` and `examples/calibrate_demo.py` demonstrate the
evaluator-evaluation and calibration steps.

## Scope and limitations

- The offline backend is a lexical heuristic for demonstration and CI; it is not
  intended for scoring real summaries. Use `AnthropicBackend`, or implement the
  five-method `Backend` interface for another model.
- Salience definition is task- and site-specific. chartcheck makes it explicit and
  configurable; it does not provide a universal definition of what is salient.
- Decompose-then-verify metrics are sensitive to decomposition quality (Wanner et
  al., 2024, arXiv:2403.11903). Calibrate on the target task and data before use.
- This is research tooling for evaluation, intended to support reviewers, not a
  certified medical device. Validate before any operational use.

## Related work

- PDSQI-9 — Croxford et al., *JAMIA* 2025 ([10.1093/jamia/ocaf068](https://doi.org/10.1093/jamia/ocaf068); [arXiv:2501.08977](https://arxiv.org/abs/2501.08977)).
- PDSQI-9 as an LLM judge — Epic's [evaluation-instruments](https://github.com/epic-open-source/evaluation-instruments). chartcheck is complementary, adding the reference-grounded claim layer for the two record-dependent attributes.
- FActScore — Min et al., EMNLP 2023 ([arXiv:2305.14251](https://arxiv.org/abs/2305.14251)).
- Hallucination and omission rates in clinical summaries — Asgari et al., *npj Digital Medicine* 2025;8:274 ([s41746-025-01670-7](https://www.nature.com/articles/s41746-025-01670-7)); Tang et al., *npj Digital Medicine* 2023 ([s41746-023-00896-7](https://www.nature.com/articles/s41746-023-00896-7)).
- Error analysis for LLM evaluation — Husain, [*LLM Evals*](https://hamel.dev/blog/posts/evals-faq/).

## License

MIT
