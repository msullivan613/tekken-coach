"""Segmenter: frame stream -> Interactions via a state machine (docs/04). Chunk C3.

C3a delivers the deterministic core — the NEUTRAL -> COMMIT -> CONTACT -> FOLLOWUP -> NEUTRAL
machine (docs/04 §2), the four derived outputs (docs/04 §3), the clean ``defender_reaction``
subset ``{blocked, hit, evaded, whiff_punished}``, single-hit punish detection, pure-whiff discard,
and round/match-boundary truncation (docs/04 §4.8). The docs/04 §4 edge-case catalogue is C3b.

Public surface:

* :class:`Segmenter` — the streaming consumer (``feed`` per frame, ``close`` at end of stream).
* :func:`segment_frames` — run a whole ``FrameRecord`` iterable through a fresh segmenter.
* :class:`SegmenterConfig` / :data:`DEFAULT_CONFIG` — the tunable thresholds and window sizes.
"""

from __future__ import annotations

from tekken_coach.segment.segmenter import (
    DEFAULT_CONFIG,
    Segmenter,
    SegmenterConfig,
    segment_frames,
)

__all__ = [
    "DEFAULT_CONFIG",
    "Segmenter",
    "SegmenterConfig",
    "segment_frames",
]
