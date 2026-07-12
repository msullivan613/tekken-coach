"""Coaching layer — render a session event log via a backend (Skill / API), docs/06. Chunk C5.

The **default** backend is a Claude Code Skill (``skill/`` at the repo root): zero marginal cost,
run by hand in Claude Code on the session ``.jsonl``. This package is the **optional** API backend
(``--coach api``) plus the shared prompt-assembly seam that keeps the two in lockstep:

* :func:`build_rubric_and_instructions` assembles the system prompt from the ``skill/`` sources —
  the single source of truth (docs/06 §3); the API backend is never a second copy of the rubric.
* :func:`coach_session` calls the Claude API and returns a :class:`CoachResult`, degrading to a
  Skill-path pointer when no credential is available.

C6 (CLI) wires ``--coach skill|api`` to these; this package exposes the backend, not the CLI.
"""

from __future__ import annotations

from tekken_coach.coach.api import (
    BACKEND_API,
    BACKEND_SKILL_FALLBACK,
    MODEL,
    CoachResult,
    coach_session,
)
from tekken_coach.coach.prompt import (
    ASSETS_DIR,
    SKILL_DIR,
    SKILL_SOURCES,
    build_rubric_and_instructions,
    read_event_log_text,
)

__all__ = [
    "ASSETS_DIR",
    "BACKEND_API",
    "BACKEND_SKILL_FALLBACK",
    "MODEL",
    "SKILL_DIR",
    "SKILL_SOURCES",
    "CoachResult",
    "build_rubric_and_instructions",
    "coach_session",
    "read_event_log_text",
]
