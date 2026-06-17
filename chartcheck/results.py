"""Small result types shared across the package (kept dependency-free)."""

from dataclasses import dataclass
from enum import Enum


class Support(str, Enum):
    SUPPORTED = "supported"        # claim is entailed by the record
    UNSUPPORTED = "unsupported"    # claim has no basis in the record
    CONTRADICTED = "contradicted"  # claim conflicts with the record


class Coverage(str, Enum):
    COVERED = "covered"   # salient record fact appears in the output
    OMITTED = "omitted"   # salient record fact is missing from the output


@dataclass
class ClaimVerdict:
    """One claim from the AI output, checked against the record (precision side)."""

    claim: str
    support: Support
    evidence: str        # quoted span from the record, or "" if none found
    confidence: float    # 0..1
    note: str = ""


@dataclass
class FactVerdict:
    """One salient record fact, checked against the AI output (recall side)."""

    fact: str
    category: str
    coverage: Coverage
    evidence: str        # quoted span from the output, or "" if omitted
    confidence: float    # 0..1
    note: str = ""


@dataclass
class AxisScore:
    """A judge-scored PDSQI-9 attribute (1..5 Likert)."""

    key: str
    label: str
    score: float
    rationale: str
