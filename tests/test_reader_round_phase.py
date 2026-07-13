"""Stage 1 round-gating: deriving the match phase from the per-player frame counter (docs/02 §8).

The real Tekken 8 build exposes no usable global match-phase enum, so
:class:`~tekken_coach.reader.decode.RoundPhaseTracker` derives the phase + round index from
``frames_since_round_start`` (a per-round counter mirrored on both players) plus each player's
damage. These tests drive it over the committed calibration capture (offline Bryan P1 vs Paul P2,
3 rounds, build 5.02.01) and over two synthetic edge cases (a pause, a timeout round).
"""

from __future__ import annotations

import json
from pathlib import Path

from tekken_coach.reader.decode import RoundPhaseTracker
from tekken_coach.schemas import MatchState

FIXTURE = Path("tests/fixtures/reader/round-counter-bryan-paul.jsonl")
ROUND_START_HEALTH = 200  # assets/offsets/5.02.01.json sanity.round_start_health


def _load_rows() -> list[dict[str, int]]:
    """The calibration capture as ``{counter, p1_damage, p2_damage}`` rows (comments skipped)."""
    rows: list[dict[str, int]] = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _drive(rows: list[dict[str, int]]) -> list[tuple[MatchState, int]]:
    """Run the tracker over rows, returning the (match_state, round) derived for each."""
    tracker = RoundPhaseTracker(ROUND_START_HEALTH)
    return [
        (p.match_state, p.round)
        for p in (tracker.update(r["counter"], r["p1_damage"], r["p2_damage"]) for r in rows)
    ]


def _onsets(phases: list[tuple[MatchState, int]], state: MatchState) -> list[int]:
    """The round index at each *transition into* ``state`` (a change from the previous frame)."""
    out: list[int] = []
    prev: MatchState | None = None
    for match_state, round_no in phases:
        if match_state is state and match_state is not prev:
            out.append(round_no)
        prev = match_state
    return out


def test_three_rounds_each_end_on_the_losers_ko() -> None:
    rows = _load_rows()
    phases = _drive(rows)

    # Each round is decided when the loser's damage reaches round_start_health. Paul (P2) is KO'd in
    # all three rounds; the round index reaches 3, one KO per round, at the known KO values.
    assert _onsets(phases, MatchState.round_over) == [1, 2, 3]

    # The round winner (Bryan/P1) never takes lethal damage, so his side never trips round_over —
    # the threshold separates winner from loser cleanly (P1's damage tops out well under 200).
    assert max(r["p1_damage"] for r in rows) < ROUND_START_HEALTH


def test_each_real_round_starts_with_a_pre_round_transition() -> None:
    phases = _drive(_load_rows())
    starts = _onsets(phases, MatchState.pre_round)

    # The three real rounds each begin with a pre_round transition (round index 1, 2, 3). The
    # trailing results screen resets the counter and climbs again exactly like a new round, so Stage
    # 1 emits a spurious 4th start it cannot suppress — that is the documented Stage-2 gap (the
    # in-match flag), NOT a bug here. We pin both: the 3 real starts, and the results false start.
    assert starts[:3] == [1, 2, 3]
    assert starts == [1, 2, 3, 4]


def test_match_over_is_never_derived_in_stage_1() -> None:
    # match_over / menu / results detection is Stage 2; Stage 1 only ever derives the in-round arc.
    phases = _drive(_load_rows())
    assert all(state is not MatchState.match_over for state, _ in phases)
    assert {state for state, _ in phases} == {
        MatchState.pre_round,
        MatchState.in_round,
        MatchState.round_over,
    }


def test_a_pause_does_not_misfire_a_round_boundary() -> None:
    # A pause freezes the counter (it barely moves); a frozen counter is not a reset, so no spurious
    # round start and no phantom round increment — the round only advances on a real reset.
    tracker = RoundPhaseTracker(ROUND_START_HEALTH)
    counters = [300, 500, 700, 701, 701, 701, 702, 900]  # a freeze at 701 mid-round
    phases = [tracker.update(c, 0, 40) for c in counters]

    assert phases[0].match_state is MatchState.pre_round  # first frame opens round 1
    assert all(p.round == 1 for p in phases)  # never advances past round 1
    assert all(p.match_state is MatchState.in_round for p in phases[1:])  # no false boundary


def test_a_timeout_round_is_a_clean_boundary_with_no_ko() -> None:
    # A round that runs the clock out (neither player lethal) still resets the counter, so it is a
    # clean round boundary — pre_round + a round increment — with round_over never latching.
    tracker = RoundPhaseTracker(ROUND_START_HEALTH)
    round_one = [200, 800, 1400]  # climbs, no KO (both damages stay sub-lethal)
    round_two = [3, 600, 1200]  # counter reset -> a new round begins
    phases = [tracker.update(c, 80, 120) for c in round_one + round_two]

    assert _onsets([(p.match_state, p.round) for p in phases], MatchState.pre_round) == [1, 2]
    assert all(p.match_state is not MatchState.round_over for p in phases)  # no KO -> no round_over
    assert phases[-1].round == 2
