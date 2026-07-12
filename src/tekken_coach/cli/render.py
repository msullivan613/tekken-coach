"""Terminal rendering for the ``tekken-coach`` CLI (docs/07 §2, §3).

The pipeline emits *data* (the session ``.jsonl``); this module is the first renderer over it
(docs/07 §5 — v1 avoids baking presentation into the pipeline). Two things get rendered:

* the **capture hand-off** (``--coach skill``, the default): capture wrote a log; tell the user it
  is ready and how to coach it in Claude Code (docs/07 §2). The Skill path never calls the API.
* the **coaching report** (``--coach api`` or ``coach <log>``): the between-matches report text, or
  the graceful "no credential" note when the API backend fell back (docs/07 §3).

TTY-aware (docs/07 §3): light color/box-drawing for headers on a real terminal, degrading to plain
ASCII when stdout is piped so logs stay clean. All output goes through a :class:`Renderer` bound to
one stream, so tests capture it with a ``StringIO`` and no monkeypatching of ``print``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from tekken_coach.coach import BACKEND_SKILL_FALLBACK, CoachResult

# The Skill hand-off line (docs/07 §2). One place so the CLI and any test assert the same text.
SKILL_HANDOFF = "open this repo and run the tekken-coach skill on that file."

_GREEN = "\033[32m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


class Renderer:
    """Writes CLI output to one stream, using ANSI only when that stream is a TTY (docs/07 §3)."""

    def __init__(self, stream: TextIO | None = None, *, color: bool | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        # Auto-detect from the stream unless forced (tests force it; a pipe degrades to plain text).
        self._color = color if color is not None else _stream_is_tty(self._stream)

    def _c(self, text: str, *codes: str) -> str:
        """Wrap ``text`` in ANSI ``codes`` when color is on, else return it unchanged."""
        if not self._color or not codes:
            return text
        return "".join(codes) + text + _RESET

    def line(self, text: str = "") -> None:
        print(text, file=self._stream)

    def header(self, text: str) -> None:
        """A section header — bold + a rule on a TTY, a plain underline otherwise."""
        self.line(self._c(text, _BOLD))
        if not self._color:
            self.line("=" * len(text))

    # -- capture hand-off (--coach skill) ----------------------------------

    def capture_handoff(self, path: Path, matches: int, interactions: int) -> None:
        """The default skill-path hand-off after capture (docs/07 §2). Never calls the API."""
        check = self._c("✔", _GREEN)  # ✔
        arrow = self._c("→", _BOLD)  # →
        counts = f"({matches} {_plural(matches, 'match', 'matches')}, {interactions} interactions)"
        self.line(f"{check} Session recorded: {path}  {self._c(counts, _DIM)}")
        self.line(f"{arrow} Coach it in Claude Code:  {SKILL_HANDOFF}")

    def log_handoff(self, path: Path, matches: int, interactions: int) -> None:
        """The skill-path hand-off for an existing log (``coach <log> --coach skill``, docs/07)."""
        arrow = self._c("→", _BOLD)
        counts = f"({matches} {_plural(matches, 'match', 'matches')}, {interactions} interactions)"
        self.line(f"Session log: {path}  {self._c(counts, _DIM)}")
        self.line(f"{arrow} Coach it in Claude Code:  {SKILL_HANDOFF}")

    # -- coaching report (--coach api, or `coach <log>`) -------------------

    def coach_result(self, result: CoachResult) -> None:
        """Render a :class:`CoachResult`: the report text, or the graceful fallback note (§3)."""
        if result.backend == BACKEND_SKILL_FALLBACK or result.report is None:
            # No usable credential — C5 guarantees this never crashes; point at the Skill path.
            self.line(result.message)
            return
        self.header("Coaching report")
        self.line(result.report)

    def notice(self, text: str) -> None:
        """A non-fatal notice (data-freshness, online-refusal); dimmed on a TTY (docs/07 §4)."""
        self.line(self._c(f"note: {text}", _DIM))


def _stream_is_tty(stream: TextIO) -> bool:
    """Whether ``stream`` is an interactive terminal (a piped/redirected stream is not)."""
    isatty = getattr(stream, "isatty", None)
    return bool(isatty()) if callable(isatty) else False


def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural
