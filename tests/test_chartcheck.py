"""Tests for chartcheck (offline backend, no API key needed)."""

from pathlib import Path

import pytest

from chartcheck import (Coverage, Support, Task, calibrate, evaluate, inject,
                        meta_eval, weighted_kappa)
from chartcheck.failuremodes import MODES

DATA = Path(__file__).parent.parent / "examples" / "data"
RECORD = (DATA / "record.txt").read_text()
GOOD = (DATA / "summary_good.txt").read_text()
FLAWED = (DATA / "summary_flawed.txt").read_text()


def test_good_summary_scores_well():
    card = evaluate(RECORD, GOOD)
    assert card.precision >= 0.8, card.summary()
    assert card.recall >= 0.7, card.summary()


def test_flawed_summary_is_penalized():
    card = evaluate(RECORD, FLAWED)
    assert card.recall < 0.8
    assert card.precision < 1.0


def test_penicillin_allergy_is_flagged_as_omission():
    card = evaluate(RECORD, FLAWED)
    omitted = " ".join(f.fact.lower() for f in card.omissions)
    assert "penicillin" in omitted, card.summary()


def test_deterministic_allergy_contradiction_fires():
    card = evaluate(RECORD, FLAWED)
    assert any(f.severity == "critical" and "allerg" in f.message.lower()
               for f in card.findings), card.summary()


def test_a1c_value_error_is_caught():
    card = evaluate(RECORD, FLAWED)
    # The A1c claim (7.2 vs record 9.2) must not be counted as supported.
    a1c = [c for c in card.claims if "a1c" in c.claim.lower()]
    assert a1c and all(c.support != Support.SUPPORTED for c in a1c), card.summary()


def test_no_known_allergies_contradicts_record():
    card = evaluate(RECORD, FLAWED)
    nkda = [c for c in card.claims if "allerg" in c.claim.lower()]
    assert nkda and any(c.support == Support.CONTRADICTED for c in nkda)


def test_weighted_kappa_bounds():
    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert weighted_kappa([1, 1, 5, 5], [5, 5, 1, 1]) < 0.0  # systematic disagreement


def test_calibrate_runs_and_reports():
    labeled = [
        {"record": RECORD, "output": GOOD, "human": {"accurate": 5, "thorough": 5}},
        {"record": RECORD, "output": FLAWED, "human": {"accurate": 2, "thorough": 2}},
    ]
    report = calibrate(labeled)
    assert "accurate" in report.per_axis
    assert report.n == 2


@pytest.mark.parametrize("mode", MODES)
def test_inject_changes_text(mode):
    corrupted, defect = inject(GOOD, mode)
    assert defect.mode == mode
    # The demo summary has demographics, numbers, allergies, and a plan, so every
    # failure mode has something to corrupt.
    assert corrupted != GOOD


def test_meta_eval_detects_defects():
    report = meta_eval([{"record": RECORD, "output": GOOD}])
    assert report.overall_sensitivity > 0.5, report.summary()


def test_qa_task_runs():
    card = evaluate(RECORD, "The patient is allergic to penicillin.",
                    task=Task.qa("What are the patient's allergies?"))
    assert card.precision >= 0.5
