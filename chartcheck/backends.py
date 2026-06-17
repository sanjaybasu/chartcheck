"""Evaluation backends.

A backend supplies the five primitive operations chartcheck needs:

    decompose_claims      output  -> list of atomic claims
    extract_salient_facts record  -> list of salient facts (task-dependent)
    check_support         claim   -> entailed by / unsupported by / contradicted by record
    check_coverage        fact    -> covered by / omitted from output
    judge_axis            axis    -> 1..5 Likert score for a non-grounded attribute

Two backends ship:

    OfflineBackend   Zero-dependency lexical heuristics. Not accurate enough for
                     real evaluation, but it runs instantly with no API key so you
                     can see the mechanism and audit trail end to end.
    AnthropicBackend Calls a Claude model for each primitive. This is the backend
                     you use for real evaluation.
"""

from __future__ import annotations

import json
import re
from typing import List

from .results import (AxisScore, ClaimVerdict, Coverage, FactVerdict, Support)
from .rubric import Axis, Task


# --------------------------------------------------------------------------- #
# Shared text utilities
# --------------------------------------------------------------------------- #

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "is", "are", "was", "were", "be", "been", "as", "that", "this", "it",
    "his", "her", "their", "patient", "pt", "has", "have", "had", "who", "which",
    "from", "but", "not", "no", "denies", "without", "negative", "none",
}
_NEGATION_CUES = {
    "no", "not", "without", "denies", "denied", "negative", "none", "absent",
    "unremarkable", "never", "ruled", "r/o", "neg", "non",
}
_WORD = re.compile(r"[a-z0-9.%]+")
# Standalone numbers only: do not pull "1" out of "A1c" or "3" out of "3b".
_NUM = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])")


def split_sentences(text: str) -> List[str]:
    """Split clinical text into sentence-ish units (newlines, bullets, ; and . )."""
    rough = re.split(r"[\n;]+|(?<=[a-z0-9%])\.\s+", text)
    out = []
    for s in rough:
        s = re.sub(r"^[\s\-\*••]+", "", s).strip()
        if len(s) >= 3:
            out.append(s)
    return out


def _content_tokens(text: str) -> List[str]:
    toks = _WORD.findall(text.lower())
    return [t.strip(".") for t in toks if t.strip(".") and t.strip(".") not in _STOPWORDS]


def _numbers(text: str) -> set:
    return set(_NUM.findall(text.lower()))


def _has_negation(text: str) -> bool:
    return bool(set(_WORD.findall(text.lower())) & _NEGATION_CUES)


def _token_match(a: str, b: str) -> bool:
    """Loose match tolerant of simple morphology (allergy/allergies, diabetes/diabetic)."""
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
        return True
    return False


def _overlap(claim_tokens: List[str], sent_tokens: List[str]) -> float:
    """Fraction of claim tokens that have a match in the sentence (recall-of-claim)."""
    if not claim_tokens:
        return 0.0
    hits = sum(1 for c in claim_tokens if any(_token_match(c, s) for s in sent_tokens))
    return hits / len(claim_tokens)


def _best_match(needle: str, haystack_sents: List[str]):
    """Return (best_overlap, best_sentence) for needle against a list of sentences."""
    nt = _content_tokens(needle)
    best_o, best_s = 0.0, ""
    for sent in haystack_sents:
        o = _overlap(nt, _content_tokens(sent))
        if o > best_o:
            best_o, best_s = o, sent
    return best_o, best_s


# --------------------------------------------------------------------------- #
# Backend interface
# --------------------------------------------------------------------------- #

class Backend:
    """Backend interface. Subclass and implement the five primitives."""

    def decompose_claims(self, output: str, task: Task) -> List[str]:
        raise NotImplementedError

    def extract_salient_facts(self, record: str, task: Task) -> List[FactVerdict]:
        """Return FactVerdicts with coverage left unset (filled by check_coverage)."""
        raise NotImplementedError

    def check_support(self, claim: str, record: str) -> ClaimVerdict:
        raise NotImplementedError

    def check_coverage(self, fact: FactVerdict, output: str) -> FactVerdict:
        raise NotImplementedError

    def judge_axis(self, axis: Axis, record: str, output: str, task: Task) -> AxisScore:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Offline heuristic backend (no API key, no dependencies)
# --------------------------------------------------------------------------- #

class OfflineBackend(Backend):
    """Lexical-entailment heuristics. For demos and CI, not for real scoring."""

    SUPPORT_T = 0.60   # overlap to count a claim as supported
    NEG_T = 0.30       # overlap above which a negation flip counts as a contradiction
    COVER_T = 0.50     # overlap to count a fact as covered

    def decompose_claims(self, output: str, task: Task) -> List[str]:
        claims = []
        for sent in split_sentences(output):
            for piece in re.split(r",\s+and\s+|\s+and\s+", sent) if len(sent) > 80 else [sent]:
                piece = piece.strip()
                if _content_tokens(piece):
                    claims.append(piece)
        return claims

    def extract_salient_facts(self, record: str, task: Task) -> List[FactVerdict]:
        facts: List[FactVerdict] = []
        seen = set()
        kw = task.salience.keywords
        q_tokens = _content_tokens(task.instruction) if not kw else []
        for sent in split_sentences(record):
            low = sent.lower()
            category = None
            if kw:
                for cat, cues in kw.items():
                    if any(c in low for c in cues):
                        category = cat
                        break
            else:  # Q&A: salient if it overlaps the question
                if _overlap(q_tokens, _content_tokens(sent)) >= 0.3:
                    category = "relevant to question"
            if category is None:
                continue
            # Strip a leading section label ("RESULTS:", "MEDICATIONS:") so it does
            # not dilute the coverage overlap denominator.
            fact_text = re.sub(r"^[A-Z][A-Z /]{1,20}:\s*", "", sent).strip()
            key = " ".join(sorted(set(_content_tokens(fact_text))))
            if not key or key in seen:
                continue
            seen.add(key)
            facts.append(FactVerdict(fact=fact_text, category=category,
                                     coverage=Coverage.OMITTED, evidence="", confidence=0.0))
        return facts

    def check_support(self, claim: str, record: str) -> ClaimVerdict:
        sents = split_sentences(record)
        best_o, best_s = _best_match(claim, sents)
        c_nums, s_nums = _numbers(claim), _numbers(best_s)

        if c_nums and s_nums and best_o >= 0.5 and not (c_nums & s_nums):
            return ClaimVerdict(claim, Support.CONTRADICTED, best_s, max(0.5, best_o),
                                note="value in claim differs from record")
        if best_o >= self.NEG_T and (_has_negation(claim) != _has_negation(best_s)):
            return ClaimVerdict(claim, Support.CONTRADICTED, best_s, max(0.5, best_o),
                                note="negation/polarity differs from record")
        if best_o >= self.SUPPORT_T:
            return ClaimVerdict(claim, Support.SUPPORTED, best_s, best_o)
        return ClaimVerdict(claim, Support.UNSUPPORTED, best_s, round(1 - best_o, 2),
                            note="no supporting span found in record")

    def check_coverage(self, fact: FactVerdict, output: str) -> FactVerdict:
        sents = split_sentences(output)
        best_o, best_s = _best_match(fact.fact, sents)
        f_nums, s_nums = _numbers(fact.fact), _numbers(best_s)
        value_lost = bool(f_nums) and best_o >= 0.5 and not (f_nums & s_nums)
        if best_o >= self.COVER_T and not value_lost:
            fact.coverage = Coverage.COVERED
            fact.evidence = best_s
            fact.confidence = best_o
        else:
            fact.coverage = Coverage.OMITTED
            fact.evidence = ""
            fact.confidence = round(1 - best_o, 2)
            if value_lost:
                fact.note = "fact mentioned but value differs"
        return fact

    def judge_axis(self, axis: Axis, record: str, output: str, task: Task) -> AxisScore:
        # Crude proxies so the offline demo populates every axis. Real judging
        # requires an LLM backend; these are deliberately simple.
        words = output.split()
        if axis.key == "succinct":
            n = len(words)
            score = 5.0 if n <= 200 else 4.0 if n <= 350 else 3.0 if n <= 600 else 2.0
            return AxisScore(axis.key, axis.label, score, f"{n} words")
        if axis.key == "stigmatizing":
            flags = [w for w in ("noncompliant", "non-compliant", "abuser", "frequent flyer",
                                 "drug-seeking", "difficult") if w in output.lower()]
            score = 5.0 if not flags else 2.0
            return AxisScore(axis.key, axis.label, score,
                             "no stigmatizing terms" if not flags else f"flagged: {flags}")
        if axis.key == "organized":
            has_structure = bool(re.search(r"\n\s*[\-\*•]|\n\w[\w ]{0,30}:", output))
            return AxisScore(axis.key, axis.label, 4.0 if has_structure else 3.0,
                             "structured" if has_structure else "free text")
        return AxisScore(axis.key, axis.label, 3.0, "offline backend: not assessed")


# --------------------------------------------------------------------------- #
# Anthropic (Claude) backend
# --------------------------------------------------------------------------- #

class AnthropicBackend(Backend):
    """Real evaluation backend backed by a Claude model.

    Requires the `anthropic` package and ANTHROPIC_API_KEY. Imported lazily so the
    rest of chartcheck works with no dependencies installed.
    """

    def __init__(self, model: str = "claude-opus-4-8", max_tokens: int = 1500,
                 client=None):
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic  # lazy
            self._client = anthropic.Anthropic()
        return self._client

    def _json(self, system: str, user: str) -> dict:
        msg = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        m = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        return json.loads(m.group(0) if m else text)

    def decompose_claims(self, output: str, task: Task) -> List[str]:
        sys = ("You decompose clinical text into atomic claims: each a single, "
               "independently checkable assertion. Return JSON {\"claims\": [str, ...]}.")
        data = self._json(sys, f"Task: {task.instruction}\n\nText:\n{output}")
        return list(data.get("claims", []))

    def extract_salient_facts(self, record: str, task: Task) -> List[FactVerdict]:
        sys = ("You extract the salient facts a clinician must not miss, per the "
               "given salience definition. Return JSON {\"facts\": "
               "[{\"fact\": str, \"category\": str}, ...]}.")
        user = (f"Salience definition:\n{task.salience.instruction}\n\n"
                f"Categories: {task.salience.categories}\n\nRecord:\n{record}")
        data = self._json(sys, user)
        return [FactVerdict(f.get("fact", ""), f.get("category", ""),
                            Coverage.OMITTED, "", 0.0) for f in data.get("facts", [])]

    def check_support(self, claim: str, record: str) -> ClaimVerdict:
        sys = ("Decide whether the CLAIM is supported, unsupported, or contradicted "
               "by the RECORD. Quote the exact supporting/contradicting span. Return "
               "JSON {\"support\": \"supported|unsupported|contradicted\", "
               "\"evidence\": str, \"confidence\": 0..1, \"note\": str}.")
        d = self._json(sys, f"CLAIM:\n{claim}\n\nRECORD:\n{record}")
        return ClaimVerdict(claim, Support(d.get("support", "unsupported")),
                            d.get("evidence", ""), float(d.get("confidence", 0.5)),
                            d.get("note", ""))

    def check_coverage(self, fact: FactVerdict, output: str) -> FactVerdict:
        sys = ("Decide whether the FACT is covered by the OUTPUT (same meaning and "
               "value) or omitted. Quote the covering span if present. Return JSON "
               "{\"coverage\": \"covered|omitted\", \"evidence\": str, "
               "\"confidence\": 0..1, \"note\": str}.")
        d = self._json(sys, f"FACT:\n{fact.fact}\n\nOUTPUT:\n{output}")
        fact.coverage = Coverage(d.get("coverage", "omitted"))
        fact.evidence = d.get("evidence", "")
        fact.confidence = float(d.get("confidence", 0.5))
        fact.note = d.get("note", "")
        return fact

    def judge_axis(self, axis: Axis, record: str, output: str, task: Task) -> AxisScore:
        sys = (f"Score the clinical text on the PDSQI-9 attribute '{axis.label}': "
               f"{axis.description} Use a 1-5 Likert scale (5 best"
               f"{', 5 = no stigmatizing language' if axis.reverse else ''}). "
               f"Return JSON {{\"score\": 1..5, \"rationale\": str}}.")
        d = self._json(sys, f"Task: {task.instruction}\n\nText:\n{output}")
        return AxisScore(axis.key, axis.label, float(d.get("score", 3)),
                         d.get("rationale", ""))
