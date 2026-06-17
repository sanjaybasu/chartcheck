"""Score a flawed clinical summary against its source record.

Run from the repo root with no API key:

    python examples/quickstart.py

This uses the dependency-free OfflineBackend. For real evaluation, swap in
AnthropicBackend (see the bottom of this file).
"""

from pathlib import Path

from chartcheck import Task, evaluate

DATA = Path(__file__).parent / "data"
record = (DATA / "record.txt").read_text()
flawed = (DATA / "summary_flawed.txt").read_text()

card = evaluate(record, flawed, task=Task.pdsqi_summary())
print(card.summary())

# The audit trail is structured, not a black-box number:
print("\nPDSQI-9 (1-5):", card.pdsqi9())

# For real evaluation:
#     from chartcheck import AnthropicBackend
#     card = evaluate(record, flawed, backend=AnthropicBackend(model="claude-opus-4-8"))
