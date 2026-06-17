"""Calibration: how much should you trust this evaluator?

An LLM judge you have not checked against human labels is not evidence. The
discipline (per Hamel Husain's error-analysis-first guidance) is: label a small
set by hand, measure agreement, and only then trust the judge at scale.

`calibrate` runs the evaluator over human-labeled examples, computes a quadratic
weighted Cohen's kappa per PDSQI-9 axis, and refuses to call the evaluator
"trusted" until agreement clears a threshold. Pass the resulting report to
`apply_trust` (or read it yourself) before believing aggregate scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .backends import Backend, OfflineBackend
from .core import Scorecard, evaluate
from .rubric import Task


def weighted_kappa(a: List[int], b: List[int], categories=(1, 2, 3, 4, 5)) -> float:
    """Quadratic weighted Cohen's kappa for two ordinal raters (pure Python)."""
    cats = list(categories)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    if n == 0:
        return float("nan")
    O = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[idx[x]][idx[y]] += 1
    W = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    row = [sum(O[i]) for i in range(k)]
    col = [sum(O[i][j] for i in range(k)) for j in range(k)]
    E = [[row[i] * col[j] / n for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k))
    return 1.0 if den == 0 else 1 - num / den


def _clamp_likert(x: float) -> int:
    return max(1, min(5, int(round(x))))


@dataclass
class CalibrationReport:
    n: int
    threshold: float
    per_axis: Dict[str, float]      # axis key -> weighted kappa
    overall: float                  # mean kappa across scored axes
    trustworthy: bool
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"chartcheck calibration  (n={self.n}, threshold kappa>={self.threshold})",
                 "-" * 56]
        for axis, kappa in sorted(self.per_axis.items()):
            mark = "ok " if kappa >= self.threshold else "LOW"
            lines.append(f"  [{mark}] {axis:<16} weighted kappa = {kappa:.2f}")
        lines.append("-" * 56)
        verdict = "TRUSTWORTHY" if self.trustworthy else "NOT TRUSTWORTHY -- do not scale yet"
        lines.append(f"  overall mean kappa = {self.overall:.2f}  ->  {verdict}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def calibrate(labeled: List[dict], task: Optional[Task] = None,
              backend: Optional[Backend] = None, threshold: float = 0.6,
              cache_path: Optional[str] = None) -> CalibrationReport:
    """Measure evaluator-vs-human agreement on a hand-labeled set.

    Args:
        labeled: list of {"record": str, "output": str, "human": {axis_key: 1..5}}.
        task:    the Task being evaluated (default PDSQI-9 summary).
        backend: evaluation backend (default OfflineBackend).
        threshold: minimum mean weighted kappa to be called trustworthy.
        cache_path: optional path to persist per-example machine scores (resumable).

    Returns:
        A CalibrationReport with per-axis kappa and a trustworthy verdict.
    """
    task = task or Task.pdsqi_summary()
    backend = backend or OfflineBackend()

    machine_by_axis: Dict[str, List[int]] = {}
    human_by_axis: Dict[str, List[int]] = {}

    def score_one(ex):
        card = evaluate(ex["record"], ex["output"], task=task, backend=backend)
        return card.pdsqi9()

    if cache_path:
        from .io_utils import cached_map
        machine_scores = cached_map(labeled, score_one, cache_path)
    else:
        machine_scores = [score_one(ex) for ex in labeled]

    for ex, m in zip(labeled, machine_scores):
        for axis, human_val in ex.get("human", {}).items():
            if axis not in m:
                continue
            machine_by_axis.setdefault(axis, []).append(_clamp_likert(m[axis]))
            human_by_axis.setdefault(axis, []).append(_clamp_likert(human_val))

    per_axis = {axis: weighted_kappa(human_by_axis[axis], machine_by_axis[axis])
                for axis in human_by_axis}
    scored = [k for k in per_axis.values() if k == k]  # drop NaN
    overall = sum(scored) / len(scored) if scored else float("nan")
    notes = []
    if not scored:
        notes.append("no overlapping human labels found; nothing to calibrate")
    return CalibrationReport(n=len(labeled), threshold=threshold, per_axis=per_axis,
                             overall=overall, trustworthy=bool(scored) and overall >= threshold,
                             notes=notes)


def apply_trust(card: Scorecard, report: CalibrationReport) -> Scorecard:
    """Stamp a scorecard with the trust verdict from a calibration report."""
    card.trusted = report.trustworthy
    return card
