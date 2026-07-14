"""The live match-end coaching trigger across **multiple matches with menu transitions**.

Live validation could never exercise the live *match-over → coach* transition end-to-end (every run
stopped at the win screen, so ``live``'s "1 match" came from the Ctrl-C shutdown-flush, not a real
match-end). These scripted tests prove — with no game — the whole live lifecycle across
``match₁ → menu → match₂``, the interaction the single-match tests never covered:

* a single session-long capture where the orchestrator's per-unit state (:class:`Segmenter`,
  ``_unit_rounds``, ``_prev_phase``) resets per match in ``_open_unit`` while the reader threads one
  match-phase tracker for the whole session (:class:`ReaderCaptureSource.polls`);
* **exactly one** coach call per match, each strictly in the post-match downtime (the no-mid-match
  invariant, docs/01 §3.2);
* per-match round-index reset, no cross-match segmenter/interaction bleed;
* both close paths — the tracker's ``match_over`` phase edge *and* the players-gone idle boundary
  (``_idle_poll``) — closing exactly once, with no double-close on the trailing menu polls.

The tracker's own per-match round-index restart is proven at the decode level (in
``test_reader_match_phase``, test ``two_matches_each_fire_match_over_and_restart_the_round_index``);
here we reproduce the tracker's *output* poll stream and prove the orchestrator consumes it right.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tekken_coach.cli import capture as capture_mod
from tekken_coach.cli.orchestrate import CaptureOrchestrator, live_policy
from tekken_coach.cli.source import Poll, _idle_poll
from tekken_coach.reader.decode import DerivedPhase, phase_signal
from tekken_coach.reader.state import SignalKind, StateSignal
from tekken_coach.schemas import FrameRecord, MatchState
from tekken_coach.session.store import SessionWriter, load_session
from tests.fixtures.coach import builder
from tests.fixtures.segment import streams

KAZUYA = "Kazuya"  # P1 in the streams (char_id 12, resolves via the committed movemap)


# ---------------------------------------------------------------------------
# Poll-stream builders (the fake reader's script)
# ---------------------------------------------------------------------------


def _live(frame: FrameRecord) -> Poll:
    """A live-match poll carrying the frame's own phase (docs/01 §3.1)."""
    return Poll(frame=frame, signal=StateSignal(SignalKind.live_match, False, frame.match_state))


def _match_over(frame: FrameRecord) -> Poll:
    """The tracker's ``match_over`` phase edge — the flag-churn close (docs/02 §8, Stage 2).

    This is the *other* close path from the idle boundary: a fully-decoded poll whose derived phase
    is ``match_over`` (kind ``live_match``, but ``match_over`` is not an active-capture phase), so
    the orchestrator closes the unit on it while the frame is still the boundary. Built exactly as
    the live derived path builds it (:func:`phase_signal`), so this is the real signal, not a stub.
    """
    over = frame.model_copy(update={"match_state": MatchState.match_over})
    return Poll(frame=over, signal=phase_signal(DerivedPhase(MatchState.match_over, frame.round)))


def _round_frames(round_no: int, frame_offset: int) -> list[FrameRecord]:
    """One round's frames (a −13 blocked / no-punish exchange → exactly one interaction), stamped
    to ``round_no`` with monotonic frame numbers and a leading ``pre_round`` approach."""
    out: list[FrameRecord] = []
    for i, f in enumerate(streams.blocked_no_punish()):
        update: dict[str, object] = {"round": round_no, "frame": f.frame + frame_offset}
        if i < 2:  # the idle approach frames open the round arc in pre_round
            update["match_state"] = MatchState.pre_round
        out.append(f.model_copy(update=update))
    return out


def _match(n_rounds: int, *, base_frame: int = 0) -> list[FrameRecord]:
    """A whole match: ``n_rounds`` rounds indexed 1..n, each a pre/in/round_over arc.

    Frame numbers are monotonic across the match (and, via ``base_frame``, across the session), so
    the segmenter never sees a spurious backwards jump; each round yields one interaction."""
    frames: list[FrameRecord] = []
    for i in range(n_rounds):
        frames += _round_frames(i + 1, base_frame + i * 1000)
    return frames


# ---------------------------------------------------------------------------
# The driver — one session-long orchestrator, a spy reporter asserting downtime
# ---------------------------------------------------------------------------


def _drive(out: Path, script: list[Poll]) -> tuple[CaptureOrchestrator, list[bool]]:
    """Drive one live :class:`CaptureOrchestrator` over ``script`` and return it plus, per reporter
    call, whether the orchestrator was *out* of a unit at that call (the no-mid-match invariant).

    A single orchestrator for the whole script mirrors a real ``live`` session: per-unit state
    resets in ``_open_unit``; nothing recreates the orchestrator between matches."""
    assets = capture_mod.load_assets()
    header = builder.build_header()
    header.matches.clear()
    header.capture_mode = live_policy().mode
    writer = SessionWriter(out, header)
    box: list[CaptureOrchestrator] = []
    downtime: list[bool] = []

    def reporter() -> None:
        downtime.append(not box[0].is_recording)

    orch = CaptureOrchestrator(
        policy=live_policy(),
        writer=writer,
        labeler=assets.labeler(),
        char_resolver=assets.char_resolver(),
        user_player=0,
        user_char=KAZUYA,
        reporter=reporter,
    )
    box.append(orch)
    for poll in script:
        orch.process(poll)
    orch.finish()
    writer.close()
    # Upgrade line 1 with the now-filled per-match summaries, exactly as run_capture does on close
    # (docs/03 §5); the writer stamped an empty-matches header on open for crash-safety.
    capture_mod._finalize_header(out, writer.header)
    return orch, downtime


def _by_match(out: Path) -> Counter[str]:
    """Interaction count keyed by ``match_id`` — proves interactions don't cross matches."""
    return Counter(i.match_id for i in load_session(out).interactions)


# ---------------------------------------------------------------------------
# The whole multi-match lifecycle (the audit's real risk)
# ---------------------------------------------------------------------------


def test_two_matches_across_a_menu_transition_coach_once_each_in_downtime(tmp_path: Path) -> None:
    # menu(idle) → match₁(3 rounds) → menu(idle) → match₂(2 rounds) → menu(idle). One orchestrator
    # for the whole session. Each match must be its own summary, coached exactly once in the
    # downtime; the round index must restart in match₂ (3 then 2, not 3 then 5); interactions must
    # not cross matches; the per-unit segmenter must not bleed.
    out = tmp_path / "live.jsonl"
    m1 = _match(3, base_frame=0)
    m2 = _match(2, base_frame=10_000)
    script = (
        [_idle_poll(None)]  # launched at the menu — no unit open, a clean no-op
        + [_live(f) for f in m1]
        + [_idle_poll(m1[-1])]  # match₁ over → the players-gone close, coach in the downtime
        + [_live(f) for f in m2]
        + [_idle_poll(m2[-1])]  # match₂ over → coach again
    )

    orch, downtime = _drive(out, script)

    session = load_session(out)
    assert len(session.header.matches) == 2  # two separate MatchSummaries
    s1, s2 = session.header.matches
    assert s1.match_id != s2.match_id
    assert (s1.rounds, s2.rounds) == (3, 2)  # round index restarts in match₂ (else it would read 3)
    assert downtime == [True, True]  # coached once per match, each strictly outside its unit
    # No interaction crosses a match: each match_id maps to its own match's interactions only.
    counts = _by_match(out)
    assert set(counts) == {s1.match_id, s2.match_id}
    assert counts[s1.match_id] == 3  # three rounds, one interaction each
    assert counts[s2.match_id] == 2
    assert orch.interaction_count == 5  # the running total the handoff line reports


def test_match_over_phase_edge_closes_and_coaches_exactly_once(tmp_path: Path) -> None:
    # The *other* close path: the tracker's ``match_over`` phase edge (flag-churn), not the idle
    # boundary. It closes the unit once; the trailing menu polls the tracker emits afterwards
    # (players still decodable, phase menu → idle) arrive with no unit open and must be a no-op —
    # no double-close, no second coach.
    out = tmp_path / "live.jsonl"
    m1 = _match(2, base_frame=0)
    script = (
        [_live(f) for f in m1]
        + [_match_over(m1[-1])]  # flag-churn match-over edge → close + coach
        + [_idle_poll(m1[-1])] * 3  # trailing menu: not in a unit → no-op
    )

    _, downtime = _drive(out, script)

    session = load_session(out)
    assert len(session.header.matches) == 1
    assert session.header.matches[0].rounds == 2
    assert downtime == [True]  # exactly one coach, in the downtime — no double-close
    assert len(session.interactions) == 2


def test_idle_boundary_close_does_not_double_close_on_trailing_menu_polls(tmp_path: Path) -> None:
    # The players-gone idle boundary close, then a run of further idle polls (the menu persists).
    # The first idle boundary closes + coaches once; every later idle poll is a no-op (no open
    # unit), so there is no missed close and no double close.
    out = tmp_path / "live.jsonl"
    m1 = _match(2, base_frame=0)
    script = [_live(f) for f in m1] + [_idle_poll(m1[-1])] * 5

    _, downtime = _drive(out, script)

    session = load_session(out)
    assert len(session.header.matches) == 1
    assert session.header.matches[0].rounds == 2
    assert downtime == [True]
    assert len(session.interactions) == 2


def test_launch_at_menu_then_a_full_match_records_exactly_one(tmp_path: Path) -> None:
    # ``live`` launched at the main menu: a run of leading idle polls (the inert menu sentinel, no
    # last-good frame yet) is a clean no-op while not in a unit, then a full match records and
    # coaches exactly once. The leading idle polls must not open a phantom unit or coach.
    out = tmp_path / "live.jsonl"
    m1 = _match(2, base_frame=0)
    script = [_idle_poll(None)] * 6 + [_live(f) for f in m1] + [_idle_poll(m1[-1])]

    _, downtime = _drive(out, script)

    session = load_session(out)
    assert len(session.header.matches) == 1
    assert session.header.matches[0].rounds == 2
    assert downtime == [True]
    assert len(session.interactions) == 2


def test_ctrl_c_mid_match_closes_and_coaches_that_match_once(tmp_path: Path) -> None:
    # Ctrl-C mid-match: the poll stream ends inside an open unit (no menu boundary ever arrives).
    # ``finish()`` must close the open unit and coach that match exactly once — the shutdown flush
    # that, before this brief, was the *only* way a live run recorded a match.
    out = tmp_path / "live.jsonl"
    m1 = _match(2, base_frame=0)
    script = [_live(f) for f in m1]  # stream just stops — no idle/match_over boundary

    orch, downtime = _drive(out, script)

    assert not orch.is_recording  # finish() left no unit open
    session = load_session(out)
    assert len(session.header.matches) == 1
    assert session.header.matches[0].rounds == 2
    assert downtime == [True]  # closed + coached once, in the downtime
    assert len(session.interactions) == 2
