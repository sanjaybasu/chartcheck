"""PDSQI-9 rubric, task specifications, and salience definitions.

The PDSQI-9 (Provider Documentation Summarization Quality Instrument) defines nine
attributes for scoring LLM-generated clinical summaries. Croxford et al., JAMIA
2025; arXiv:2501.08977.

The instrument's own stated limitation is that two of its attributes -- *Accurate*
and *Thorough* -- "require someone or something to review the actual record."
chartcheck's premise is that those two attributes are not holistic "read the whole
chart" problems; they are claim-level entailment problems in two directions:

    Accurate  = every claim in the output is entailed by the record   (precision)
    Thorough  = every salient fact in the record appears in the output (recall)

The remaining attributes do not require the record (except Cited, which only needs
to know whether a statement is attributable, not whether it is correct), so they
are scored by a structured judge.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Axis:
    """One PDSQI-9 attribute."""

    key: str
    label: str
    description: str
    grounded: bool  # True if scoring requires comparison against the source record
    reverse: bool = False  # True if a high count is bad (e.g. stigmatizing language)


# The nine PDSQI-9 attributes. `accurate` and `thorough` are computed by
# bidirectional claim checking (see core.evaluate); the rest are judge-scored.
PDSQI9: List[Axis] = [
    Axis("cited", "Cited",
         "Statements are attributable to a location in the source record.", grounded=False),
    Axis("accurate", "Accurate",
         "No statement contradicts or is unsupported by the record (precision).", grounded=True),
    Axis("thorough", "Thorough",
         "No salient fact from the record is omitted (recall).", grounded=True),
    Axis("useful", "Useful",
         "Clinically useful for the stated task.", grounded=False),
    Axis("organized", "Organized",
         "Logical, navigable structure.", grounded=False),
    Axis("comprehensible", "Comprehensible",
         "Clear and unambiguous.", grounded=False),
    Axis("succinct", "Succinct",
         "Free of unnecessary or redundant content.", grounded=False),
    Axis("synthesized", "Synthesized",
         "Integrates information across the record rather than copying it.", grounded=False),
    Axis("stigmatizing", "Stigmatizing",
         "Free of stigmatizing or biased language.", grounded=False, reverse=True),
]


@dataclass(frozen=True)
class SalienceSpec:
    """Defines which facts in a record "count" for the Thorough (recall) axis.

    This is the genuinely hard, task-dependent part of clinical summary
    evaluation. Recall against *every* atomic fact in a 200-page chart punishes
    good, succinct summaries. So salience is a first-class, pluggable object:
    the categories of fact a given task cares about, a natural-language
    definition the LLM backend uses, and keyword cues the dependency-free
    offline backend uses.
    """

    instruction: str
    categories: List[str]
    keywords: Dict[str, List[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Task:
    """What the AI was asked to do, plus how to score it."""

    name: str
    instruction: str
    salience: SalienceSpec
    axes: List[Axis] = field(default_factory=lambda: list(PDSQI9))

    @staticmethod
    def pdsqi_summary() -> "Task":
        """Default task: multi-document clinical summary, scored on the full PDSQI-9."""
        salience = SalienceSpec(
            instruction=(
                "A fact is salient if a covering clinician reading only the summary "
                "would be expected to act on it or be harmed by not knowing it: active "
                "problems and diagnoses, current medications and dose changes, allergies "
                "and adverse reactions, abnormal or actionable results, and the plan / "
                "outstanding follow-up. Routine normal findings and administrative "
                "boilerplate are not salient."
            ),
            categories=[
                "active problems / diagnoses",
                "medications and dose changes",
                "allergies and adverse reactions",
                "abnormal or actionable results",
                "plan and outstanding follow-up",
            ],
            keywords={
                "active problems / diagnoses": [
                    "diagnos", "history of", "h/o", "hx", "disease", "disorder",
                    "failure", "cancer", "diabetes", "ckd", "copd", "chf", "depression",
                    "hypertension", "htn", "kidney", "renal", "problem",
                ],
                "medications and dose changes": [
                    "mg", "mcg", "started", "stopped", "increased", "decreased",
                    "discontinued", "titrat", "daily", "bid", "tid", "insulin", "tablet",
                ],
                "allergies and adverse reactions": [
                    "allerg", "anaphylaxis", "reaction", "intoleran", "rash", "adverse",
                ],
                "abnormal or actionable results": [
                    "elevated", "low", "high", "abnormal", "positive", "critical",
                    "a1c", "egfr", "creatinine", "potassium", "inr", "ejection fraction",
                    "bp ", "blood pressure", "%",
                ],
                "plan and outstanding follow-up": [
                    "plan", "follow up", "follow-up", "refer", "schedule", "recheck",
                    "monitor", "return", "pending", "await",
                ],
            },
        )
        return Task(name="pdsqi_summary",
                    instruction="Summarize this patient's record for a covering clinician.",
                    salience=salience)

    @staticmethod
    def qa(question: str) -> "Task":
        """Chart Q&A: only facts relevant to the question are salient."""
        salience = SalienceSpec(
            instruction=(
                f"A fact is salient only if it is needed to correctly and completely "
                f"answer the question: '{question}'. Facts unrelated to the question are "
                f"not salient even if clinically important."
            ),
            categories=["facts required to answer the question"],
            keywords={},  # offline salience for Q&A falls back to question-term overlap
        )
        # Q&A is not scored on summary-shaped axes like Succinct/Synthesized.
        axes = [a for a in PDSQI9 if a.key not in ("succinct", "synthesized", "organized")]
        return Task(name="qa", instruction=question, salience=salience, axes=axes)
