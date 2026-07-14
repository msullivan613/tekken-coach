"""Part A (live-at-menu tolerance) + Part B (memory char-id resolution) unit tests.

Part A: the live poll loop must tolerate the main menu — a null player holder slot faults the
per-player decode while the process is alive. The decision is factored into the pure
:func:`~tekken_coach.cli.source.decide_poll` + the tiny stateful :class:`_PollSequencer` so the
record/idle/skip/process_lost verdict is testable without a process (the live ``while True`` stays
``# pragma: no cover``). Each scripted outcome here models one poll's reads: a real :class:`Poll`
is a decoded frame, ``poll=None`` a player-decode ``MemoryReadError`` (out of a match), and
``global_ok=False`` a liveness-probe fault (the process gone).

Part B: ``--char <name>`` must resolve the reader's MEMORY char ids, a different space from the
movemap's framedata ids (project memory ``t8-reader-model-holder-aob``).
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.cli import capture as capture_mod
from tekken_coach.cli.orchestrate import _char_matches
from tekken_coach.cli.source import (
    _MENU_FRAME,
    IDLE_FAULT_THRESHOLD,
    Poll,
    PollAction,
    ReaderCaptureSource,
    _PollSequencer,
    decide_poll,
)
from tekken_coach.reader.offsets import load_offset_table
from tekken_coach.reader.state import SignalKind, StateSignal
from tekken_coach.schemas import MatchState
from tests.factories import make_frame_record

REPO_OFFSETS = Path("assets/offsets")


def _good_poll(frame_no: int) -> Poll:
    """A real live-match poll for a decoded frame numbered ``frame_no``."""
    frame = make_frame_record().model_copy(
        update={"frame": frame_no, "match_state": MatchState.in_round}
    )
    return Poll(frame=frame, signal=StateSignal(SignalKind.live_match, False, MatchState.in_round))


# ---------------------------------------------------------------------------
# Part A — the poll-decision logic
# ---------------------------------------------------------------------------


def test_sequencer_waits_idle_with_no_unit_at_session_start() -> None:
    # Launched at the main menu: every player decode faults while the process is alive. Below the
    # threshold each poll skips (no yield); the threshold-th goes idle with the inert menu sentinel
    # (no good frame has ever been seen), so the orchestrator waits, never opening a unit.
    seq = _PollSequencer()
    steps = [seq.step(global_ok=True, poll=None) for _ in range(IDLE_FAULT_THRESHOLD)]

    assert [s.action for s in steps[:-1]] == [PollAction.skip] * (IDLE_FAULT_THRESHOLD - 1)
    last = steps[-1]
    assert last.action is PollAction.idle
    assert last.poll is not None
    assert last.poll.frame is _MENU_FRAME  # inert sentinel — never fed (no unit is open)
    assert last.poll.signal.kind is SignalKind.idle


def test_sequencer_records_once_frames_resolve() -> None:
    # A match loads: the player decode succeeds → record the real poll and reset the fault run.
    seq = _PollSequencer()
    seq.step(global_ok=True, poll=None)  # one menu fault first
    step = seq.step(global_ok=True, poll=_good_poll(1000))

    assert step.action is PollAction.record
    assert step.poll is not None and step.poll.frame.frame == 1000


def test_sequencer_debounces_a_short_mid_match_glitch() -> None:
    # A transient mid-match read hiccup (fewer than the threshold) must NOT go idle — closing the
    # open unit would fragment one match into two and coach early. Below threshold every poll skips;
    # a subsequent success records, proving the unit was never closed.
    seq = _PollSequencer()
    seq.step(global_ok=True, poll=_good_poll(1000))  # in a match
    glitch = [seq.step(global_ok=True, poll=None) for _ in range(IDLE_FAULT_THRESHOLD - 1)]
    recovered = seq.step(global_ok=True, poll=_good_poll(1001))

    assert all(s.action is PollAction.skip for s in glitch)
    assert recovered.action is PollAction.record


def test_sequencer_closes_on_the_idle_boundary_after_a_match() -> None:
    # Match end → menu: after >= threshold consecutive player faults, yield an idle boundary whose
    # frame is the LAST GOOD frame stamped `menu`. That non-active phase closes the open unit (and,
    # live, coaches) — the players-gone match-over trigger. The frame NUMBER is preserved (an
    # idempotent re-feed) and the players carried, so the match summary reads the real last frame.
    seq = _PollSequencer()
    last_good = _good_poll(2048)
    seq.step(global_ok=True, poll=last_good)
    steps = [seq.step(global_ok=True, poll=None) for _ in range(IDLE_FAULT_THRESHOLD)]

    assert [s.action for s in steps[:-1]] == [PollAction.skip] * (IDLE_FAULT_THRESHOLD - 1)
    boundary = steps[-1]
    assert boundary.action is PollAction.idle
    assert boundary.poll is not None
    assert boundary.poll.frame.match_state is MatchState.menu  # a clean segmenter boundary
    assert boundary.poll.frame.frame == 2048  # same frame number → re-feed is a no-op
    assert boundary.poll.frame.players == last_good.frame.players  # real last-frame data preserved
    assert boundary.poll.signal.kind is SignalKind.idle


def test_sequencer_propagates_process_lost_when_the_liveness_read_fails() -> None:
    # The GLOBAL liveness read failing means the process is gone — propagate regardless of the fault
    # run (the caller re-raises so C6 classifies it as process_lost). A genuinely-closed game is not
    # mistaken for the menu.
    seq = _PollSequencer()
    seq.step(global_ok=True, poll=_good_poll(1000))
    step = seq.step(global_ok=False, poll=None)

    assert step.action is PollAction.process_lost
    assert step.poll is None


def test_decide_poll_is_pure_over_its_inputs() -> None:
    # The decision core, exercised directly: process_lost wins over everything; a decoded poll
    # records; a sub-threshold miss skips; an at-threshold miss idles.
    assert decide_poll(global_ok=False, poll=None, misses=0, last_good=None).action is (
        PollAction.process_lost
    )
    good = _good_poll(5)
    assert decide_poll(global_ok=True, poll=good, misses=0, last_good=None).action is (
        PollAction.record
    )
    assert decide_poll(global_ok=True, poll=None, misses=1, last_good=None, threshold=3).action is (
        PollAction.skip
    )
    assert decide_poll(global_ok=True, poll=None, misses=3, last_good=None, threshold=3).action is (
        PollAction.idle
    )


# ---------------------------------------------------------------------------
# Part B — memory char-id -> name resolution
# ---------------------------------------------------------------------------


def test_offset_table_exposes_the_memory_char_name_map() -> None:
    # The 5.02.01 table bakes the observed memory-id names; char_names_by_id gives the int-keyed
    # view the resolver wants (JSON object keys are strings).
    table = load_offset_table(REPO_OFFSETS / "5.02.01.json")
    assert table.char_names_by_id() == {
        0: "paul",
        6: "jin",
        7: "bryan",
        8: "kazuya",
        39: "armor_king",
    }


def test_char_resolver_prefers_the_memory_map_then_movemap_then_stub() -> None:
    assets = capture_mod.load_assets()
    resolve = assets.char_resolver(char_names={6: "jin", 12: "memory_wins"})

    assert resolve(6) == "jin"  # memory map — the reader's id space
    assert resolve(12) == "memory_wins"  # memory map wins over the movemap (Kazuya is 12 there)
    assert (
        resolve(999) == "char:999"
    )  # unobserved id → stable stub (so --char char:999 still works)

    # With no memory map the movemap is the fallback (legacy/fake builds, whose ids coincide).
    assert assets.char_resolver()(12) == "Kazuya"


def test_configured_char_validates_against_the_memory_id_space() -> None:
    # The live bug: P1 reads memory char_id 6, but --char jin was rejected because the resolver only
    # knew framedata ids. The orchestrator's validation predicate compares the RESOLVED name (now
    # the memory-map name) against --char case-insensitively, so `jin` validates and a wrong side
    # still hard-errors. --char char:N keeps working for an unobserved id via the stub.
    assets = capture_mod.load_assets()
    resolve = assets.char_resolver(char_names={6: "jin"})

    assert _char_matches(resolve(6), "Jin")  # case-insensitive → validates
    assert _char_matches(resolve(6), "jin")
    assert not _char_matches(resolve(6), "kazuya")  # a wrong side is rejected upstream
    assert _char_matches(resolve(999), "char:999")  # the stub still validates its own form


def test_reader_source_exposes_the_tables_memory_char_names() -> None:
    # White-box: the source surfaces the offset table's memory char map (post-attach) for the
    # resolver wiring, without needing a live process.
    src = ReaderCaptureSource("t8.exe", str(REPO_OFFSETS))
    assert src.char_names == {}  # before attach: empty
    src._table = load_offset_table(REPO_OFFSETS / "5.02.01.json")
    assert src.char_names[6] == "jin"
