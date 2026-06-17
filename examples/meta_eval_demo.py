"""Meta-evaluation: does chartcheck catch known, injected defects?

    python examples/meta_eval_demo.py

Takes a GOOD summary, injects each failure mode (omission, hallucination, value
error, negation flip, wrong patient), and reports how often the evaluator flags
the corruption. This is how you stress-test the evaluator itself.
"""

from pathlib import Path

from chartcheck import meta_eval

DATA = Path(__file__).parent / "data"
record = (DATA / "record.txt").read_text()
good = (DATA / "summary_good.txt").read_text()

report = meta_eval([{"record": record, "output": good}])
print(report.summary())
