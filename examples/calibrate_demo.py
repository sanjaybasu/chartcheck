"""Calibration: measure evaluator-vs-human agreement before trusting it at scale.

    python examples/calibrate_demo.py

Builds a tiny hand-labeled set (here, illustrative labels on the Accurate and
Thorough axes) and computes a quadratic weighted kappa per axis. In practice you
would label ~50-100 real examples following error-analysis-first discipline.
"""

from pathlib import Path

from chartcheck import calibrate

DATA = Path(__file__).parent / "data"
record = (DATA / "record.txt").read_text()
good = (DATA / "summary_good.txt").read_text()
flawed = (DATA / "summary_flawed.txt").read_text()

# Each example carries human Likert labels (1-5). Mix faithful and flawed cases.
labeled = [
    {"record": record, "output": good,   "human": {"accurate": 5, "thorough": 5}},
    {"record": record, "output": flawed, "human": {"accurate": 2, "thorough": 2}},
    {"record": record, "output": good,   "human": {"accurate": 5, "thorough": 4}},
    {"record": record, "output": flawed, "human": {"accurate": 2, "thorough": 1}},
]

report = calibrate(labeled, threshold=0.6)
print(report.summary())
print("\nUntil this reads TRUSTWORTHY, aggregate scores are not evidence.")
