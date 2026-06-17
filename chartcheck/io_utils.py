"""Crash-safe IO: atomic writes and a resume-on-restart map.

Batch LLM evaluation is slow and costs money, so an interrupted run must never
re-pay for work it already did. `cached_map` persists each result the moment it is
produced (atomically), and skips anything already on disk when re-run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, List


def atomic_write(path, obj) -> None:
    """Write JSON to `path` via a temp file + rename (no torn files on crash)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def cached_map(items: list, fn: Callable, cache_path) -> List:
    """Apply `fn` to each item, persisting results so a restart resumes cleanly.

    `fn` must return a JSON-serializable value. Results are keyed by index.
    """
    cache_path = Path(cache_path)
    done = {}
    if cache_path.exists():
        done = {int(k): v for k, v in json.loads(cache_path.read_text()).items()}
    results = []
    for i, item in enumerate(items):
        if i not in done:
            done[i] = fn(item)
            atomic_write(cache_path, {str(k): v for k, v in done.items()})
        results.append(done[i])
    return results
