"""chartcheck -- auditable, omission-aware evaluation of clinical AI text.

Quickstart:

    from chartcheck import evaluate, Task
    card = evaluate(record_text, summary_text, task=Task.pdsqi_summary())
    print(card.summary())

By default this uses the dependency-free OfflineBackend (heuristics, for demos).
For real evaluation, pass an AnthropicBackend:

    from chartcheck import evaluate, AnthropicBackend
    card = evaluate(record, summary, backend=AnthropicBackend(model="claude-opus-4-8"))
"""

from .backends import AnthropicBackend, Backend, OfflineBackend
from .calibrate import CalibrationReport, calibrate, weighted_kappa
from .core import Scorecard, evaluate
from .failuremodes import MetaEvalReport, inject, meta_eval
from .results import (AxisScore, ClaimVerdict, Coverage, FactVerdict, Support)
from .rubric import PDSQI9, Axis, SalienceSpec, Task

__version__ = "0.1.0"

__all__ = [
    "evaluate", "Scorecard", "Task", "SalienceSpec", "Axis", "PDSQI9",
    "Backend", "OfflineBackend", "AnthropicBackend",
    "calibrate", "CalibrationReport", "weighted_kappa",
    "inject", "meta_eval", "MetaEvalReport",
    "ClaimVerdict", "FactVerdict", "AxisScore", "Support", "Coverage",
    "__version__",
]
