"""tekken_coach — a read-only Tekken 8 coaching side-car.

The capture pipeline (reader -> segmenter -> xref) writes a labeled .jsonl event log; the
coaching layer reads it. See docs/ for the full technical design. This chunk (C0) ships the
data schemas and the session store; other subpackages are homes for later chunks.
"""

__version__ = "0.1.0"
