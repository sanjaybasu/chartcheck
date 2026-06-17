"""The evaluate() orchestrator and the Scorecard it returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backends import Backend, OfflineBackend
from .deterministic import Check, Finding, default_checks, run_checks
from .results import AxisScore, ClaimVerdict, Coverage, FactVerdict, Support
from .rubric import Task


def _likert(fraction: float) -> float:
    """Map a 0..1 precision/recall onto the PDSQI-9 1..5 Likert scale."""
    return round(1 + 4 * max(0.0, min(1.0, fraction)), 2)


@dataclass
class Scorecard:
    task: str
    precision: float                       # "Accurate": supported / all claims
    recall: float                          # "Thorough": covered / all salient facts
    claims: List[ClaimVerdict]
    facts: List[FactVerdict]
    axes: List[AxisScore]
    findings: List[Finding]
    trusted: Optional[bool] = None         # set by calibrate(); None = uncalibrated

    # -- convenience views ------------------------------------------------- #
    @property
    def accurate(self) -> float:
        return self.precision

    @property
    def thorough(self) -> float:
        return self.recall

    @property
    def hallucinations(self) -> List[ClaimVerdict]:
        return [c for c in self.claims if c.support != Support.SUPPORTED]

    @property
    def omissions(self) -> List[FactVerdict]:
        return [f for f in self.facts if f.coverage == Coverage.OMITTED]

    def pdsqi9(self) -> Dict[str, float]:
        """All nine attributes on a 1..5 scale (Accurate/Thorough mapped from p/r)."""
        scores = {a.key: a.score for a in self.axes}
        scores["accurate"] = _likert(self.precision)
        scores["thorough"] = _likert(self.recall)
        return scores

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "precision": self.precision,
            "recall": self.recall,
            "trusted": self.trusted,
            "pdsqi9": self.pdsqi9(),
            "claims": [vars(c) | {"support": c.support.value} for c in self.claims],
            "facts": [vars(f) | {"coverage": f.coverage.value} for f in self.facts],
            "findings": [vars(x) for x in self.findings],
        }

    def summary(self) -> str:
        lines = []
        trust = ("uncalibrated" if self.trusted is None
                 else "TRUSTED" if self.trusted else "NOT TRUSTED (judge under-agrees)")
        lines.append(f"chartcheck scorecard  ({self.task})   [{trust}]")
        lines.append("-" * 60)
        lines.append(f"  Accurate (precision) : {self.precision:.2f}   "
                     f"{len(self.hallucinations)}/{len(self.claims)} claims unsupported")
        lines.append(f"  Thorough (recall)    : {self.recall:.2f}   "
                     f"{len(self.omissions)}/{len(self.facts)} salient facts omitted")
        for a in self.axes:
            lines.append(f"  {a.label:<20} : {a.score:.1f}   {a.rationale}")
        if self.hallucinations:
            lines.append("\n  Unsupported / contradicted claims:")
            for c in self.hallucinations:
                lines.append(f"    [{c.support.value}] {c.claim}")
                if c.note:
                    lines.append(f"        -> {c.note}")
        if self.omissions:
            lines.append("\n  Omitted salient facts:")
            for f in self.omissions:
                lines.append(f"    ({f.category}) {f.fact}")
        if self.findings:
            lines.append("\n  Deterministic findings:")
            for x in self.findings:
                lines.append(f"    [{x.severity}] {x.message}  ({x.evidence})")
        return "\n".join(lines)


def evaluate(record: str, output: str, task: Optional[Task] = None,
             backend: Optional[Backend] = None,
             deterministic_checks: Optional[List[Check]] = None) -> Scorecard:
    """Score an AI-generated clinical output against its source record.

    Args:
        record:  the source clinical record (one document or many concatenated).
        output:  the AI-generated text to evaluate (summary, Q&A answer, ...).
        task:    a Task defining the rubric and salience. Defaults to PDSQI-9 summary.
        backend: an evaluation Backend. Defaults to the dependency-free OfflineBackend.
        deterministic_checks: hard rules to run. Defaults to chartcheck's built-ins.

    Returns:
        A Scorecard with claim-level and fact-level audit trails. Note that
        `trusted` is None until you run calibrate() against human labels.
    """
    task = task or Task.pdsqi_summary()
    backend = backend or OfflineBackend()
    checks = default_checks() if deterministic_checks is None else deterministic_checks

    # Accurate (precision): does every claim in the output hold up against the record?
    claims = backend.decompose_claims(output, task)
    claim_verdicts = [backend.check_support(c, record) for c in claims]
    supported = sum(1 for c in claim_verdicts if c.support == Support.SUPPORTED)
    precision = supported / len(claim_verdicts) if claim_verdicts else 1.0

    # Thorough (recall): does the output cover every salient fact in the record?
    facts = backend.extract_salient_facts(record, task)
    fact_verdicts = [backend.check_coverage(f, output) for f in facts]
    covered = sum(1 for f in fact_verdicts if f.coverage == Coverage.COVERED)
    recall = covered / len(fact_verdicts) if fact_verdicts else 1.0

    # Remaining PDSQI-9 attributes (do not require reading the record).
    axes = [backend.judge_axis(a, record, output, task)
            for a in task.axes if not a.grounded]

    findings = run_checks(record, output, checks)

    return Scorecard(task=task.name, precision=precision, recall=recall,
                     claims=claim_verdicts, facts=fact_verdicts, axes=axes,
                     findings=findings)
