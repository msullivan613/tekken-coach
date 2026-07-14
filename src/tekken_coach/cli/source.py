"""The capture-source seam — the single boundary between the reader and the state machine (docs/01).

The orchestrator ([orchestrate][tekken_coach.cli.orchestrate]) reads only :class:`Poll` records —
``(FrameRecord, StateSignal)`` pairs — never raw memory. That is what lets the identical state
machine run against the live game and against a scripted test stream (the plan's test strategy:
*"drive the pipeline on recorded fixtures with a fake reader replaying a FrameRecord stream"*).

Two implementations share the :class:`CaptureSource` protocol:

* :class:`ReaderCaptureSource` — the **real** path. Wraps :class:`WinMemorySource` + the reader's
  :func:`decode_frame` / :func:`read_state_signal`, resolving a version-matched offset table on
  ``attach``. Its ``polls`` loop is an endless live poll (the user Ctrl-Cs, or the process is lost).
* :class:`ScriptedCaptureSource` — the **fake** path. Replays a pre-built list of :class:`Poll`s
  with no game, no sleep, and deterministically. Tests build the live/clean lifecycles out of these.

Real-game phase sourcing (docs/02 §8, round-gating): the real T8 build (5.02.01) has no usable
global match-phase enum, so :class:`ReaderCaptureSource` *derives* the full ``menu``…``match_over``
phase per frame from the per-player ``frames_since_round_start`` counter plus the global
``match_flag`` word via a single :class:`~tekken_coach.reader.decode.MatchPhaseTracker`, building
the :class:`StateSignal` from the derived phase instead of the (raising) strict
:func:`read_state_signal`. A legacy table with real global phase codes keeps the
:func:`read_state_signal` path. The whole state machine is also
exercised through :class:`ScriptedCaptureSource`, and ``coach <log>`` works with no capture at all.

Read-only, by construction: this module reads frames and signals; it never writes memory or injects
input (the reader package has no such primitive — docs/02 §2).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

from tekken_coach.reader.decode import (
    FrameReader,
    MatchPhaseTracker,
    MemoryReadError,
    derive_match_phase,
    phase_signal,
    read_match_flag,
    read_state_signal,
    stamp_phase,
    table_derives_round_phase,
)
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import OffsetTable
from tekken_coach.reader.state import MODE_OFFLINE, StateSignal, classify_state
from tekken_coach.schemas import (
    ActionState,
    CounterState,
    FrameRecord,
    HeatState,
    MatchState,
    PlayerFrame,
)

# The live poll cadence. A real game only advances its frame counter every ~16.7 ms (60 fps); a
# poll faster than that re-reads the same frame (docs/02 §6 note on poll_frames' interval).
DEFAULT_POLL_INTERVAL = 0.05

# Consecutive player-decode faults a live poll tolerates before it treats the game as out of a
# match (main menu / matchmaking / character select / post-match results) and yields an idle poll
# (Part A, docs/01 §4.3). At the 0.05 s cadence this is ~0.2 s — long enough to ride out a transient
# mid-match read glitch (which must NOT close a recording unit, or one match fragments into two and
# coaches early), short enough to close promptly and coach in the downtime when a match really ends.
IDLE_FAULT_THRESHOLD = 4


@dataclass(frozen=True)
class Poll:
    """One capture poll: the decoded frame plus the side-signal that gates it (docs/01 §4.3).

    ``frame`` feeds the segmenter; ``signal`` drives the capture triggers. The signal's
    ``match_state`` is what C6 gates transitions on, *not* the persisted ``frame.match_state``
    (they agree on a calibrated build, but the gate is the strict read — docs/01 §4.3).
    """

    frame: FrameRecord
    signal: StateSignal


# The signal an out-of-match (menu / pre-match / post-match) poll carries: a menu phase classified
# to ``idle`` (Part A). It is not a capturing phase, so the orchestrator closes any open recording
# unit on it and (live) coaches — the players-gone "match over → coach in the downtime" trigger that
# complements the flag-churn ``match_over`` edge (docs/01 §3.1).
_IDLE_SIGNAL = classify_state(MatchState.menu, MODE_OFFLINE)


def _menu_player() -> PlayerFrame:
    """A neutral placeholder player for :data:`_MENU_FRAME` — never read by the orchestrator."""
    return PlayerFrame(
        char_id=0,
        move_id=0,
        move_frame=0,
        action_state=ActionState.neutral,
        health=0,
        pos=(0.0, 0.0, 0.0),
        facing=1,
        block_stun=False,
        hit_stun=False,
        counter_state=CounterState.none,
        throw_active=False,
        airborne=False,
        juggle=False,
        heat=HeatState(active=False, timer_ms=0, engager_used=False),
        rage=False,
    )


# A neutral placeholder frame for an idle poll emitted *before any match has loaded* (no last-good
# frame yet, e.g. ``live`` launched at the main menu). Inert by construction: the orchestrator reads
# only ``poll.signal`` while idle-and-not-in-a-unit, never ``poll.frame`` (docs/01 §4.3 / the
# no-mid-match invariant), so this frame is never fed to the segmenter. It exists only to satisfy
# the ``Poll(frame, signal)`` shape. Its ``match_state`` is ``menu`` so that if it ever *were* fed,
# segmenter would treat it as a boundary, not active play.
_MENU_FRAME = FrameRecord(
    frame=0,
    match_state=MatchState.menu,
    round=0,
    timer_ms=0,
    players=[_menu_player(), _menu_player()],
)


def _idle_poll(last_good: FrameRecord | None) -> Poll:
    """Build the idle poll for an out-of-match instant (Part A).

    After a match (``last_good`` set) the boundary is the last good frame with its ``match_state``
    stamped to ``menu`` — a non-active phase the segmenter closes an open interaction on, and an
    idempotent re-feed: the orchestrator re-feeds this boundary frame before closing the unit
    (``_feed`` runs before ``_close_unit``), and a ``menu`` frame is a clean segmenter boundary
    (``feed`` truncates any open interaction and returns), so feeding it once — or, on a repeated
    frame number, again — is a no-op past the first. Before any match (``last_good`` is None) the
    inert :data:`_MENU_FRAME` sentinel stands in (the orchestrator is not in a unit, so it is never
    fed).
    """
    frame = (
        last_good.model_copy(update={"match_state": MatchState.menu})
        if last_good is not None
        else _MENU_FRAME
    )
    return Poll(frame=frame, signal=_IDLE_SIGNAL)


class PollAction(Enum):
    """What the live poll loop should do with one poll's read outcomes (Part A)."""

    record = auto()  # a real decoded frame — yield it (feed/record)
    idle = auto()  # alive but out of a match — yield an idle boundary/menu poll
    skip = auto()  # a sub-threshold transient player fault — sleep and retry, yield nothing
    process_lost = auto()  # the liveness read failed → the process is gone — propagate the fault


@dataclass(frozen=True)
class PollStep:
    """The decision for one poll: an action and (for record/idle) the :class:`Poll` to yield."""

    action: PollAction
    poll: Poll | None = None


def decide_poll(
    *,
    global_ok: bool,
    poll: Poll | None,
    misses: int,
    last_good: FrameRecord | None,
    threshold: int = IDLE_FAULT_THRESHOLD,
) -> PollStep:
    """Decide one live poll's action from its read outcomes — the tested core of Part A (pure).

    Liveness is read first and for free (docs/01 §4.3): a **global** field (module-relative, ticks
    in every mode) is probed before the player frame, so the two faults separate cleanly:

    * ``global_ok`` False — the liveness read itself faulted → the process is gone →
      ``process_lost`` (the caller re-raises the read fault so C6 classifies it, surfacing after
      the match).
    * ``poll`` set — the player frame decoded → ``record`` it.
    * ``poll`` None but alive — the player holder slot was unreadable (menu / character select /
      post-match): out of a match. Only after ``threshold`` **consecutive** such faults do we go
      ``idle`` (closing an open unit); below it we ``skip`` (a lone mid-match glitch must not close
      a unit). ``misses`` is that running count, including this poll.
    """
    if not global_ok:
        return PollStep(PollAction.process_lost)
    if poll is not None:
        return PollStep(PollAction.record, poll)
    if misses >= threshold:
        return PollStep(PollAction.idle, _idle_poll(last_good))
    return PollStep(PollAction.skip)


class _PollSequencer:
    """Track the consecutive-fault count + last good frame across polls, then :func:`decide_poll`.

    The stateful half of the Part A decision, split from the I/O so it is unit-testable without a
    process: feed it one poll outcome — a real :class:`Poll` when the player decoded, else the
    liveness verdict — and it returns the :class:`PollStep` the live loop acts on. A success resets
    the fault run and records the last good frame (the idle boundary); a live-but-faulted poll
    increments the run. Pure given the outcomes.
    """

    def __init__(self, threshold: int = IDLE_FAULT_THRESHOLD) -> None:
        self._threshold = threshold
        self._misses = 0
        self._last_good: FrameRecord | None = None

    def step(self, *, global_ok: bool, poll: Poll | None) -> PollStep:
        if poll is not None:
            self._misses = 0
            self._last_good = poll.frame
        elif global_ok:
            self._misses += 1
        return decide_poll(
            global_ok=global_ok,
            poll=poll,
            misses=self._misses,
            last_good=self._last_good,
            threshold=self._threshold,
        )


def _read_poll(
    source: MemorySource,
    table: OffsetTable,
    reader: FrameReader,
    tracker: MatchPhaseTracker | None,
) -> tuple[bool, Poll | None, MemoryReadError | None]:
    """One live poll's reads: the liveness probe first, then the player frame (Part A).

    Returns ``(global_ok, poll, lost)``. A **global** read is the liveness probe — it succeeds
    whenever the process is alive. If it faults the process is gone: ``(False, None, <fault>)``, and
    the caller re-raises ``<fault>`` as ``process_lost``. If it succeeds but the per-player decode
    faults (a null holder slot at the menu / character select / post-match), the process is alive
    but out of a match: ``(True, None, None)`` → the caller idles. On a full decode: ``(True, poll,
    None)``. The derived path probes ``match_flag``; the legacy path probes the strict state signal
    (both read module-relative globals, off the same anchor as ``frame_counter``).
    """
    if tracker is not None:
        try:
            match_flag = read_match_flag(source, table)
        except MemoryReadError as exc:
            return False, None, exc
        try:
            frame = reader.read_frame(source, table).frame
        except MemoryReadError:
            return True, None, None
        phase = derive_match_phase(tracker, table, frame, match_flag)
        return True, Poll(frame=stamp_phase(frame, phase), signal=phase_signal(phase)), None
    try:
        signal = read_state_signal(source, table)
    except MemoryReadError as exc:
        return False, None, exc
    try:
        frame = reader.read_frame(source, table).frame
    except MemoryReadError:
        return True, None, None
    return True, Poll(frame=frame, signal=signal), None


class CaptureSource(Protocol):
    """The producer seam the orchestrator consumes (docs/00 §4 producer boundary)."""

    @property
    def game_version(self) -> str:
        """The offset-table version stamped on the session header (valid after :meth:`attach`)."""
        ...

    @property
    def char_names(self) -> dict[int, str]:
        """Memory char id -> name for the running build (valid after :meth:`attach`).

        The observation-grounded map that lets ``--char <name>`` validate against the reader's
        **memory** char ids — a different space from the movemap's framedata ids (Part B, project
        memory ``t8-reader-model-holder-aob``). Empty when the build's table declares none.
        """
        ...

    def attach(self) -> None:
        """Acquire the source (attach to the game / prepare the stream). Idempotent-safe."""
        ...

    def polls(self) -> Iterator[Poll]:
        """Yield :class:`Poll`s until the stream ends (fake) or forever (live, until Ctrl-C)."""
        ...

    def close(self) -> None:
        """Release the source. Safe to call more than once."""
        ...


class ScriptedCaptureSource:
    """A :class:`CaptureSource` that replays a fixed list of :class:`Poll`s (the fake reader).

    Deterministic and game-free: ``polls`` yields the scripted sequence once and stops, which is how
    a test drives a whole live or clean lifecycle (arm → record a match → downtime, or a batch of
    replays) without the process. ``attach``/``close`` are no-ops beyond bookkeeping.
    """

    def __init__(self, script: list[Poll], *, game_version: str = "2.01.01") -> None:
        self._script = list(script)
        self._game_version = game_version
        self.attached = False
        self.closed = False

    @property
    def game_version(self) -> str:
        return self._game_version

    @property
    def char_names(self) -> dict[int, str]:
        return {}  # the fake reader resolves char ids through the movemap (docs/05); no memory map

    def attach(self) -> None:
        self.attached = True

    def polls(self) -> Iterator[Poll]:
        yield from self._script

    def close(self) -> None:
        self.closed = True


class ReaderCaptureSource:
    """The real :class:`CaptureSource` over the live reader (docs/02, docs/01 §4.3).

    ``attach`` opens the process read-only, detects the running build, and selects the matching
    offset table (failing closed with the §4 runbook on an unknown version). ``polls`` decodes a
    full :func:`FrameRecord` each cadence tick and sources the gating :class:`StateSignal` two ways
    (docs/02 §8, Stage 1 round-gating): on the real T8 build it *derives* the round phase from the
    per-player counter through a :class:`RoundPhaseTracker` and stamps it over the frame; on a
    legacy table with real global phase codes it uses the strict :func:`read_state_signal`.
    """

    def __init__(
        self,
        process: str,
        offsets_dir: str,
        *,
        version_override: str | None = None,
        interval: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self._process = process
        self._offsets_dir = offsets_dir
        self._version_override = version_override
        self._interval = interval
        self._source: MemorySource | None = None
        self._table: OffsetTable | None = None
        self._version: str | None = None

    @property
    def game_version(self) -> str:
        """The detected/selected offset-table version (available after :meth:`attach`)."""
        if self._version is None:
            raise RuntimeError("ReaderCaptureSource.game_version read before attach()")
        return self._version

    @property
    def char_names(self) -> dict[int, str]:
        """The running build's memory char id -> name map (Part B), empty until :meth:`attach`."""
        if self._table is None:
            return {}
        return self._table.char_names_by_id()

    def attach(self) -> None:
        if self._source is not None:
            return  # already attached — idempotent so a caller may attach eagerly to read version
        # Imported here so this module (and the offline test suite) load without pymem: constructing
        # WinMemorySource is the only thing that needs the Windows extra (docs/02 §3 posture).
        from tekken_coach.reader.offsets import select_offset_table
        from tekken_coach.reader.version import detect_running_version
        from tekken_coach.reader.win_source import WinMemorySource

        source: MemorySource = WinMemorySource(self._process)
        version = self._version_override or detect_running_version(self._process)
        self._table = select_offset_table(version, self._offsets_dir)
        self._source = source
        self._version = version

    def polls(self) -> Iterator[Poll]:
        assert self._source is not None and self._table is not None, "attach() before polls()"
        source = self._source
        table = self._table
        reader = FrameReader()  # wraps decode_frame with dropped-frame accounting (docs/02 §7)
        # The match-phase deriver is the one stateful thing threaded across polls (docs/02 §8): the
        # real T8 build has no usable global phase enum, so we derive the full menu…match_over phase
        # from the player counter plus the global match_flag word.
        derives = table_derives_round_phase(table)
        tracker = MatchPhaseTracker(table.sanity.round_start_health) if derives else None
        # The out-of-match tolerance (Part A): the sequencer + decide_poll own the record/idle/skip/
        # lost decision; this loop only does the reads and acts on the verdict. Launching `live` at
        # the main menu (the player holder slot is null) now waits idle instead of crashing.
        return self._live_polls(source, table, reader, tracker, _PollSequencer())

    def _live_polls(
        self,
        source: MemorySource,
        table: OffsetTable,
        reader: FrameReader,
        tracker: MatchPhaseTracker | None,
        seq: _PollSequencer,
    ) -> Iterator[Poll]:  # pragma: no cover - endless live loop; the decision helpers are tested
        while True:
            global_ok, poll, lost = _read_poll(source, table, reader, tracker)
            step = seq.step(global_ok=global_ok, poll=poll)
            if step.action is PollAction.process_lost:
                assert lost is not None
                raise lost  # the liveness fault → C6 classifies it as process_lost (docs/02 §7)
            if step.action is PollAction.skip:
                time.sleep(self._interval)  # a sub-threshold glitch: retry without closing a unit
                continue
            assert step.poll is not None
            yield step.poll
            time.sleep(self._interval)

    def close(self) -> None:
        self._source = None
