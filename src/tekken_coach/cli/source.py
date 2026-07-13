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
from typing import Protocol

from tekken_coach.reader.decode import (
    FrameReader,
    MatchPhaseTracker,
    derive_match_phase,
    phase_signal,
    read_match_flag,
    read_state_signal,
    stamp_phase,
    table_derives_round_phase,
)
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import OffsetTable
from tekken_coach.reader.state import StateSignal
from tekken_coach.schemas import FrameRecord

# The live poll cadence. A real game only advances its frame counter every ~16.7 ms (60 fps); a
# poll faster than that re-reads the same frame (docs/02 §6 note on poll_frames' interval).
DEFAULT_POLL_INTERVAL = 0.05


@dataclass(frozen=True)
class Poll:
    """One capture poll: the decoded frame plus the side-signal that gates it (docs/01 §4.3).

    ``frame`` feeds the segmenter; ``signal`` drives the capture triggers. The signal's
    ``match_state`` is what C6 gates transitions on, *not* the persisted ``frame.match_state``
    (they agree on a calibrated build, but the gate is the strict read — docs/01 §4.3).
    """

    frame: FrameRecord
    signal: StateSignal


class CaptureSource(Protocol):
    """The producer seam the orchestrator consumes (docs/00 §4 producer boundary)."""

    @property
    def game_version(self) -> str:
        """The offset-table version stamped on the session header (valid after :meth:`attach`)."""
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
        while True:
            if tracker is not None:
                # Decode the frame, read the match_flag, derive the full match phase, and stamp it
                # over the bogus seeded global reads. The signal is built from the derived phase
                # (classify_state maps menu -> idle, match_over -> a live_match edge).
                frame = reader.read_frame(source, table).frame
                match_flag = read_match_flag(source, table)
                phase = derive_match_phase(tracker, table, frame, match_flag)
                signal = phase_signal(phase)
                frame = stamp_phase(frame, phase)
            else:
                # Legacy build with real global phase codes: strict signal first (it decides whether
                # to record), then the full frame the segmenter consumes.
                signal = read_state_signal(source, table)
                frame = reader.read_frame(source, table).frame
            yield Poll(frame=frame, signal=signal)
            time.sleep(self._interval)

    def close(self) -> None:
        self._source = None
