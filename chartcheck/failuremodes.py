"""Synthetic failure injection and meta-evaluation.

An evaluator you have not stress-tested is itself untested. This module injects
*known* defects into good summaries -- omission, hallucination, value error,
negation flip, wrong patient -- and checks whether chartcheck catches them. The
result is a sensitivity profile for the evaluator: which error classes it reliably
flags, and which slip through. This operationalizes the "synthetic stub record"
idea as a QA suite for the judge, not just for the model under test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .backends import Backend, OfflineBackend, split_sentences
from .core import evaluate
from .rubric import Task

_HALLUCINATION = ("Additionally, the patient was started on warfarin 5 mg daily "
                  "for newly diagnosed atrial fibrillation.")

MODES = ["omission", "hallucination", "value_error", "negation_flip", "wrong_patient"]


@dataclass
class Defect:
    mode: str
    description: str
    span: str = ""


def _pick(seq, seed):
    return seq[seed % len(seq)] if seq else None


def inject(output: str, mode: str, seed: int = 0) -> Tuple[str, Defect]:
    """Return (corrupted_output, Defect). Deterministic given (output, mode, seed)."""
    sents = split_sentences(output)

    if mode == "omission":
        cands = [s for s in sents if re.search(r"\d", s)] or sents
        target = _pick(cands, seed)
        corrupted = output.replace(target, "", 1) if target else output
        return corrupted, Defect("omission", "removed a salient fact from the summary", target or "")

    if mode == "hallucination":
        return (output.rstrip() + " " + _HALLUCINATION,
                Defect("hallucination", "added a claim absent from the record", _HALLUCINATION))

    if mode == "value_error":
        m = re.search(r"\d+(?:\.\d+)?", output)
        if not m:
            return output, Defect("value_error", "no numeric value to perturb")
        old = m.group(0)
        new = f"{float(old) + 1.5:.1f}" if "." in old else str(int(old) + 7)
        return (output[:m.start()] + new + output[m.end():],
                Defect("value_error", f"changed a value {old} -> {new}", old))

    if mode == "negation_flip":
        target = _pick([s for s in sents
                        if not re.search(r"\b(no|not|without|denies|negative)\b", s.lower())], seed)
        if not target:
            return output, Defect("negation_flip", "no affirmative statement to flip")
        flipped = "No " + target[0].lower() + target[1:]
        return output.replace(target, flipped, 1), Defect("negation_flip",
                                                          "flipped the polarity of a statement", target)

    if mode == "wrong_patient":
        m = re.search(r"\b(\d{1,3})[ -]?(year[- ]?old|yo|y/o)", output, re.I)
        if m:
            old, new = m.group(1), str((int(m.group(1)) + 11) % 100)
            return (output[:m.start(1)] + new + output[m.end(1):],
                    Defect("wrong_patient", f"changed age {old} -> {new}", m.group(0)))
        return output, Defect("wrong_patient", "no patient identifier to perturb")

    raise ValueError(f"unknown failure mode: {mode}")


@dataclass
class MetaEvalReport:
    per_mode: dict           # mode -> {"n", "caught", "sensitivity"}
    overall_sensitivity: float
    clean_precision: float
    clean_recall: float
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["chartcheck meta-evaluation (does the evaluator catch known defects?)",
                 "-" * 64,
                 f"  clean baseline: precision={self.clean_precision:.2f} "
                 f"recall={self.clean_recall:.2f}"]
        for mode in MODES:
            if mode in self.per_mode:
                m = self.per_mode[mode]
                lines.append(f"  {mode:<14} caught {m['caught']}/{m['n']}  "
                             f"sensitivity={m['sensitivity']:.2f}")
        lines.append("-" * 64)
        lines.append(f"  overall sensitivity = {self.overall_sensitivity:.2f}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def meta_eval(examples: List[dict], modes: Optional[List[str]] = None,
              task: Optional[Task] = None, backend: Optional[Backend] = None,
              eps: float = 1e-6) -> MetaEvalReport:
    """Inject known defects into good summaries and measure detection.

    Args:
        examples: list of {"record": str, "output": str} where output is a GOOD summary.
        modes:    failure modes to test (default: all).
        task/backend: as in evaluate().

    Returns:
        A MetaEvalReport with per-mode sensitivity.
    """
    task = task or Task.pdsqi_summary()
    backend = backend or OfflineBackend()
    modes = modes or MODES

    clean = [evaluate(ex["record"], ex["output"], task=task, backend=backend)
             for ex in examples]
    n = len(clean) or 1
    clean_p = sum(c.precision for c in clean) / n
    clean_r = sum(c.recall for c in clean) / n

    per_mode = {}
    for mode in modes:
        tested = caught = 0
        for ex, base in zip(examples, clean):
            corrupted, _ = inject(ex["output"], mode)
            if corrupted == ex["output"]:
                continue  # injection was a no-op for this example
            tested += 1
            card = evaluate(ex["record"], corrupted, task=task, backend=backend)
            degraded = (card.precision < base.precision - eps
                        or card.recall < base.recall - eps)
            new_finding = len(card.findings) > len(base.findings)
            if degraded or new_finding:
                caught += 1
        per_mode[mode] = {"n": tested, "caught": caught,
                          "sensitivity": caught / tested if tested else float("nan")}

    sens = [m["sensitivity"] for m in per_mode.values() if m["n"]]
    overall = sum(sens) / len(sens) if sens else float("nan")
    return MetaEvalReport(per_mode, overall, clean_p, clean_r)
