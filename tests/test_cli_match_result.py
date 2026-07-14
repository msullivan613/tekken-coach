"""Match ``result`` = who won more rounds, not a terminal health snapshot (docs/01 §5).

Live validation showed ``matches[].result`` was always ``draw``: the old code derived the result
from a single frame's health, but at close the last frame is a results/menu frame where health is
reset/ambiguous, so both sides read equal. These tests drive scripted ``(FrameRecord, StateSignal)``
streams — KO rounds, a timeout decider, a trailing results screen, and both user sides — through a
raw :class:`CaptureOrchestrator` and assert the round-win tally decides the match.
"""

from __future__ import annotations

import io
from pathlib import Path

from tekken_coach.cli import capture as capture_mod
from tekken_coach.cli.config import resolve_settings
from tekken_coach.cli.render import Renderer
from tekken_coach.cli.source import Poll, ScriptedCaptureSource
from tekken_coach.reader.state import SignalKind, StateSignal
from tekken_coach.schemas import (
    ActionState,
    CounterState,
    FrameRecord,
    HeatState,
    MatchState,
    PlayerFrame,
)
from tekken_coach.session.store import load_session

KAZUYA_ID = 12  # resolves to "Kazuya" via the committed movemap (see test_cli.py)
FULL_HP = 150


def _pf(health: int) -> PlayerFrame:
    return PlayerFrame(
        char_id=KAZUYA_ID,
        move_id=0,
        move_frame=0,
        action_state=ActionState.neutral,
        health=health,
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


def _fr(frame_no: int, phase: MatchState, round_no: int, h0: int, h1: int) -> FrameRecord:
    return FrameRecord(
        frame=frame_no,
        match_state=phase,
        round=round_no,
        timer_ms=30000,
        players=[_pf(h0), _pf(h1)],
    )


def _live(frame: FrameRecord) -> Poll:
    return Poll(frame=frame, signal=StateSignal(SignalKind.live_match, False, frame.match_state))


def _idle(frame: FrameRecord) -> Poll:
    """The players-gone / results-menu boundary that closes the recording unit."""
    return Poll(frame=frame, signal=StateSignal(SignalKind.idle, False, MatchState.menu))


class _Round:
    """A scripted round: in-round frames ending at ``(h0, h1)``, optionally KO-capped."""

    def __init__(self, round_no: int, h0: int, h1: int, *, ko: bool) -> None:
        self.round_no = round_no
        self.h0 = h0
        self.h1 = h1
        self.ko = ko


def _script(rounds: list[_Round]) -> list[Poll]:
    """Build a live poll stream from a list of rounds, ending with the results-menu boundary."""
    polls: list[Poll] = []
    fno = 1000
    for rnd in rounds:
        # A pre-round full-health frame, an in-round exchange landing on the final health, then a
        # ``round_over`` KO cap when the round ends in a KO (a timeout round simply ends).
        polls.append(_live(_fr(fno, MatchState.pre_round, rnd.round_no, FULL_HP, FULL_HP)))
        polls.append(_live(_fr(fno + 1, MatchState.in_round, rnd.round_no, FULL_HP, FULL_HP)))
        polls.append(_live(_fr(fno + 2, MatchState.in_round, rnd.round_no, rnd.h0, rnd.h1)))
        if rnd.ko:
            polls.append(_live(_fr(fno + 3, MatchState.round_over, rnd.round_no, rnd.h0, rnd.h1)))
        fno += 100
    # The trailing results/menu boundary: last good frame, menu-stamped → closes the unit.
    last = polls[-1].frame
    polls.append(_idle(last))
    return polls


def _run(out: Path, script: list[Poll], *, user_player: int) -> None:
    """Drive the full live-capture path (so line 1's ``matches`` is finalized on close)."""
    user = "p1" if user_player == 0 else "p2"
    settings = resolve_settings(
        mode="live", coach="skill", user=user, char="Kazuya", out=str(out), config={}
    )
    capture_mod.run_capture(
        settings=settings,
        source=ScriptedCaptureSource(script),
        assets=capture_mod.load_assets(),
        renderer=Renderer(io.StringIO(), color=False),
    )


def _result(out: Path) -> str:
    session = load_session(out)
    assert len(session.header.matches) == 1
    return session.header.matches[0].result


def test_ko_majority_is_a_win(tmp_path: Path) -> None:
    """User (P1) KOs in rounds 1 and 3, loses round 2 → 2–1 → win."""
    out = tmp_path / "s.jsonl"
    script = _script(
        [
            _Round(1, FULL_HP, 0, ko=True),  # P1 wins
            _Round(2, 0, FULL_HP, ko=True),  # P2 wins
            _Round(3, 40, 0, ko=True),  # P1 wins
        ]
    )
    _run(out, script, user_player=0)
    assert _result(out) == "win"


def test_ko_minority_is_a_loss(tmp_path: Path) -> None:
    """User (P1) loses rounds 1 and 3 → 1–2 → loss."""
    out = tmp_path / "s.jsonl"
    script = _script(
        [
            _Round(1, 0, FULL_HP, ko=True),  # P2 wins
            _Round(2, FULL_HP, 0, ko=True),  # P1 wins
            _Round(3, 0, 60, ko=True),  # P2 wins
        ]
    )
    _run(out, script, user_player=0)
    assert _result(out) == "loss"


def test_one_one_with_a_decider(tmp_path: Path) -> None:
    """A 1–1 match settled by a third KO round resolves to the decider's winner."""
    out = tmp_path / "s.jsonl"
    script = _script(
        [
            _Round(1, FULL_HP, 0, ko=True),  # P1
            _Round(2, 0, FULL_HP, ko=True),  # P2
            _Round(3, 30, 0, ko=True),  # P1 → decider
        ]
    )
    _run(out, script, user_player=0)
    assert _result(out) == "win"


def test_timeout_decider_resolves_by_health(tmp_path: Path) -> None:
    """The final round ends without a KO; the higher-health side (the user) wins it → win."""
    out = tmp_path / "s.jsonl"
    script = _script(
        [
            _Round(1, FULL_HP, 0, ko=True),  # P1 KO
            _Round(2, 0, FULL_HP, ko=True),  # P2 KO
            _Round(3, 90, 45, ko=False),  # timeout: P1 higher health → P1 wins
        ]
    )
    _run(out, script, user_player=0)
    assert _result(out) == "win"


def test_trailing_results_frame_does_not_flip_the_tally(tmp_path: Path) -> None:
    """A results round appended after the decided rounds (full health both, no KO, no real play)
    must not register as a round → the win tally is unchanged."""
    out = tmp_path / "s.jsonl"
    script = _script(
        [
            _Round(1, FULL_HP, 0, ko=True),  # P1
            _Round(2, 20, 0, ko=True),  # P1 → 2–0 win already decided
        ]
    )
    # Splice a full-health results round (round 3) before the menu boundary — the game's post-match
    # counter-reset frame the live bug tripped on. It has no in-round play and no damage.
    boundary = script.pop()  # the _idle poll
    results = _fr(9000, MatchState.match_over, 3, FULL_HP, FULL_HP)
    script.append(_live(results))
    script.append(boundary)
    _run(out, script, user_player=0)
    assert _result(out) == "win"  # 2–0, not perturbed by the results frame


def test_result_is_from_the_users_point_of_view(tmp_path: Path) -> None:
    """The same stream is a win for P1 and a loss for P2 — result pivots on ``user_player``."""
    rounds = [
        _Round(1, FULL_HP, 0, ko=True),  # P1
        _Round(2, 0, FULL_HP, ko=True),  # P2
        _Round(3, 50, 0, ko=True),  # P1 → P1 wins the match
    ]
    out_p1 = tmp_path / "p1.jsonl"
    _run(out_p1, _script(rounds), user_player=0)
    assert _result(out_p1) == "win"

    out_p2 = tmp_path / "p2.jsonl"
    _run(out_p2, _script(rounds), user_player=1)
    assert _result(out_p2) == "loss"


def test_double_ko_style_even_round_does_not_tally_either_side(tmp_path: Path) -> None:
    """A round that ends level (both at equal health) tallies to neither side, so a lone even round
    leaves the match a draw rather than inventing a winner."""
    out = tmp_path / "s.jsonl"
    script = _script([_Round(1, 30, 30, ko=True)])  # equal health at the KO cap → no winner
    _run(out, script, user_player=0)
    assert _result(out) == "draw"
