"""Capture-mode orchestration — the live/clean state machine (docs/01, docs/00 §4).

This is the one place a capture mode is visible. Everything below the *trigger* — the segmenter,
the xref, the session store, the coaching cadence — is identical for both modes (docs/01 §5:
*no ``if mode ==`` past the reader/trigger layer*). The two modes differ only in:

* **which signal opens/closes a recording unit** — a :class:`ModePolicy` decision, and
* **when coaching fires** — per unit (live: coach match *N* in the downtime before *N+1*) vs. once
  at end of the batch (clean).

The data flow per captured frame (docs/00 §4):

    Poll.frame → Segmenter.feed → label_interaction (xref) → SessionWriter.append

with a :meth:`SessionWriter.flush` at each round end so a crash loses at most one round. A recording
*unit* is one match (live) or one replay (clean); each gets its own :class:`Segmenter` and a
:class:`MatchSummary`.

**No mid-match output** is a hard invariant (docs/01 §3.2), enforced structurally: the orchestrator
never renders. It calls its ``reporter`` callback *only* from :meth:`_close_unit` (after leaving the
unit) and :meth:`finish` (end of stream) — never while :attr:`is_recording` is true.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tekken_coach.cli.source import Poll
from tekken_coach.reader.state import SignalKind, StateSignal
from tekken_coach.schemas import (
    CaptureMode,
    FrameRecord,
    Interaction,
    LabeledInteraction,
    MatchState,
    MatchSummary,
)
from tekken_coach.segment.segmenter import DEFAULT_CONFIG, Segmenter, SegmenterConfig
from tekken_coach.session.store import SessionWriter

# A move-id → interaction the xref labels. Kept as a callable so the orchestrator neither loads
# assets nor knows the xref's signature (docs/00 §3: xref is pure, wired at the edge).
Labeler = Callable[[Interaction], LabeledInteraction]
# char_id → display name (from the move maps), falling back to a stable ``char:<id>`` on a miss.
CharResolver = Callable[[int], str]
# Rendered between/after matches; takes no argument (it reads the log + live counts by closure).
Reporter = Callable[[], None]

# Match phases in which a live match is actively in progress (pre/in/round-over). ``match_over``
# is deliberately excluded so it closes the unit while the segmenter still sees the boundary frame.
_LIVE_ACTIVE = frozenset({MatchState.pre_round, MatchState.in_round, MatchState.round_over})
# Phases whose transition triggers a crash-safety flush (docs/00 §4 round-end flush).
_FLUSH_PHASES = frozenset({MatchState.round_over, MatchState.match_over})


class CaptureError(Exception):
    """A capture-time configuration/consistency error surfaced to the CLI (docs/07 §4)."""


class CharacterMismatchError(CaptureError):
    """The observed character on the user's side ≠ the configured ``--char`` (docs/01 §5).

    A hard error: getting the user's side wrong inverts all coaching, so capture refuses rather than
    recording an inverted session.
    """


@dataclass(frozen=True)
class ModePolicy:
    """What separates live from clean above the shared pipeline (docs/01 §5).

    ``is_capturing`` is the trigger; ``coach_per_unit`` is the cadence. Nothing else in the
    orchestrator branches on mode.
    """

    mode: CaptureMode
    coach_per_unit: bool  # coach per closed unit (live) vs. once at batch end (clean)
    refuses_online: bool  # count/refuse online frames (clean defense-in-depth, docs/01 §4.3)
    _capturing: Callable[[StateSignal], bool]

    def is_capturing(self, signal: StateSignal) -> bool:
        """Whether this frame belongs to an active recording unit."""
        return self._capturing(signal)


def _live_capturing(signal: StateSignal) -> bool:
    """Live: record while a live match is in an active phase (docs/01 §3.1)."""
    return signal.kind is SignalKind.live_match and signal.match_state in _LIVE_ACTIVE


def _clean_capturing(signal: StateSignal) -> bool:
    """Clean: record only during offline replay playback; the online-refusal is baked in here
    (``should_buffer_clean`` is false for an online session — docs/01 §4.3 defense-in-depth)."""
    return signal.should_buffer_clean


def live_policy() -> ModePolicy:
    """The live-capture policy: per-match coaching, arm on live-match-active (docs/01 §3.1)."""
    return ModePolicy(
        mode=CaptureMode.live,
        coach_per_unit=True,
        refuses_online=False,
        _capturing=_live_capturing,
    )


def clean_policy() -> ModePolicy:
    """The clean-capture policy: coach once at batch end; buffer only offline replay (docs/01)."""
    return ModePolicy(
        mode=CaptureMode.clean,
        coach_per_unit=False,
        refuses_online=True,
        _capturing=_clean_capturing,
    )


def policy_for(mode: CaptureMode) -> ModePolicy:
    """Resolve the :class:`ModePolicy` for a capture mode."""
    return live_policy() if mode is CaptureMode.live else clean_policy()


class CaptureOrchestrator:
    """Drives the shared pipeline from a :class:`Poll` stream under a :class:`ModePolicy`.

    Feed each poll to :meth:`process`; call :meth:`finish` once the stream ends. The orchestrator
    holds the current open unit's :class:`Segmenter`, the running interaction count, and the
    per-unit round set — nothing mode-specific beyond the injected policy.
    """

    def __init__(
        self,
        *,
        policy: ModePolicy,
        writer: SessionWriter,
        labeler: Labeler,
        char_resolver: CharResolver,
        user_player: int,
        user_char: str,
        reporter: Reporter,
        config: SegmenterConfig = DEFAULT_CONFIG,
    ) -> None:
        self._policy = policy
        self._writer = writer
        self._label = labeler
        self._resolve_char = char_resolver
        self._user_player = user_player
        self._user_char = user_char
        self._reporter = reporter
        self._cfg = config

        self._seg: Segmenter | None = None
        self._in_unit = False
        self._match_no = 0
        self._interaction_count = 0
        self._unit_match_id = ""
        self._unit_rounds: set[int] = set()
        self._last_frame: FrameRecord | None = None
        self._prev_phase: MatchState | None = None
        self._online_refused = 0

    # -- introspection (used by the no-mid-match invariant test) -----------

    @property
    def is_recording(self) -> bool:
        """True iff a recording unit is open — no output may be emitted while this holds."""
        return self._in_unit

    @property
    def interaction_count(self) -> int:
        return self._interaction_count

    @property
    def online_refused(self) -> int:
        """Frames refused in clean mode because the signal was an online match (docs/01 §4.3)."""
        return self._online_refused

    # -- the state machine -------------------------------------------------

    def process(self, poll: Poll) -> None:
        """Advance the machine by one poll (a :class:`~tekken_coach.cli.source.Poll`)."""
        signal = poll.signal
        frame = poll.frame

        # Defense-in-depth accounting: the clean policy refuses online frames (``is_capturing`` is
        # already false for them); note it so the end-of-session report can warn (docs/01 §4.3).
        # Driven by a policy flag, not an ``if mode ==`` — the pipeline below stays mode-agnostic.
        if self._policy.refuses_online and signal.online:
            self._online_refused += 1

        capturing_now = self._policy.is_capturing(signal)

        if capturing_now and not self._in_unit:
            self._open_unit(frame)
        if self._in_unit:
            self._feed(frame, signal)
        # Close on the boundary frame (which was still fed above, so the segmenter truncated it).
        if self._in_unit and not capturing_now:
            self._close_unit()

    def _open_unit(self, frame: FrameRecord) -> None:
        """Start a recording unit: validate the user's side, then open a fresh segmenter (§5)."""
        self._validate_user_char(frame)
        self._match_no += 1
        self._unit_match_id = f"{self._writer.header.created_at}#{self._match_no}"
        self._seg = Segmenter(self._unit_match_id, self._cfg)
        self._unit_rounds = set()
        self._prev_phase = None
        self._in_unit = True
        self._last_frame = frame

    def _validate_user_char(self, frame: FrameRecord) -> None:
        """Hard-error on a configured-vs-observed character mismatch on the user's side (§5).

        Accepts either the resolved name (``--char jin``) or the raw ``char:<id>`` form
        (``--char char:6``): once a memory char map resolves id 6 to ``jin`` (Part B), the stub form
        must keep validating too, so a config or muscle-memory ``char:6`` is not rejected for Jin.
        """
        observed_id = frame.players[self._user_player].char_id
        observed = self._resolve_char(observed_id)
        if _char_matches(observed, self._user_char) or _char_matches(
            f"char:{observed_id}", self._user_char
        ):
            return
        raise CharacterMismatchError(
            f"configured --char {self._user_char!r} but P{self._user_player + 1} is "
            f"{observed!r} (char_id {observed_id}). Getting the user's side wrong inverts all "
            f"coaching (docs/01 §5); fix --user/--char and retry."
        )

    def _feed(self, frame: FrameRecord, signal: StateSignal) -> None:
        """Segment one frame, label + append what closed, and flush at round end (docs/00 §4)."""
        assert self._seg is not None
        self._last_frame = frame
        self._unit_rounds.add(frame.round)
        for interaction in self._seg.feed(frame):
            self._emit(interaction)
        phase = signal.match_state
        if phase in _FLUSH_PHASES and phase is not self._prev_phase:
            self._writer.flush()  # round-end / match-end crash-safety flush
        self._prev_phase = phase

    def _emit(self, interaction: Interaction) -> None:
        self._writer.append(self._label(interaction))
        self._interaction_count += 1

    def _close_unit(self) -> None:
        """Finalize the open unit: flush the segmenter tail, record the summary, (live) coach."""
        assert self._seg is not None
        for interaction in self._seg.close():
            self._emit(interaction)
        self._writer.flush()
        self._record_match_summary()
        self._seg = None
        self._in_unit = False
        self._prev_phase = None
        if self._policy.coach_per_unit:
            self._reporter()  # post-match downtime — never mid-match (is_recording is now False)

    def _record_match_summary(self) -> None:
        """Append this unit's :class:`MatchSummary` to the (in-memory) header (docs/03 §5)."""
        frame = self._last_frame
        assert frame is not None
        opponent_idx = 1 - self._user_player
        opponent = self._resolve_char(frame.players[opponent_idx].char_id)
        user_hp = frame.players[self._user_player].health
        opp_hp = frame.players[opponent_idx].health
        result = "win" if user_hp > opp_hp else "loss" if user_hp < opp_hp else "draw"
        self._writer.header.matches.append(
            MatchSummary(
                match_id=self._unit_match_id,
                opponent_char=opponent,
                result=result,
                rounds=len(self._unit_rounds),
            )
        )

    def finish(self) -> None:
        """End of stream: close any open unit, then (clean) coach over the whole session log."""
        if self._in_unit:
            self._close_unit()  # reports here if coach_per_unit (last match's downtime)
        if not self._policy.coach_per_unit:
            self._reporter()  # clean: one report over the whole batch (docs/01 §4.1)


def _char_matches(observed: str, configured: str) -> bool:
    """Case/space-insensitive character-name comparison for the §5 validation."""
    return observed.strip().casefold() == configured.strip().casefold()
