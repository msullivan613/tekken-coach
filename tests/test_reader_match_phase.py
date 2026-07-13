"""Stage 2 round-gating: deriving the full match phase from the counter + ``match_flag`` (§8).

:class:`~tekken_coach.reader.decode.MatchPhaseTracker` layers the menu/match-over edges on top of
Stage 1's round arc, keyed on the global ``match_flag`` word (``@0xd444``) that *holds* one value
while a stage is loaded and *churns* through low values in a menu.

No live fixture holds *both* the per-player counter and ``match_flag`` aligned (``phase.jsonl`` has
the flag but not the counter; ``round.jsonl`` the counter but not the flag — project memory
``capture-round-gating-deferred``), so the combined tracker is pinned with **synthetic** vectors
that reproduce each signal's observed shape: a menu churns the flag every poll on an idle counter; a
loaded stage holds one flag value while the counter resets/climbs per round. Stage 1's own
``RoundPhaseTracker`` fixture test stays untouched — the round arc is unchanged; this only adds the
match-over/menu edges on top.
"""

from __future__ import annotations

from tekken_coach.reader.decode import STAGE_HOLD_POLLS, DerivedPhase, MatchPhaseTracker
from tekken_coach.schemas import MatchState

HP = 200  # assets/offsets/5.02.01.json sanity.round_start_health
# Low UI values the menu flag was observed to cycle through (docs/02 §8 re-mine of phase.jsonl).
MENU_FLAGS = (16, 18, 40, 44, 56)
LOAD_HOLD = STAGE_HOLD_POLLS + 5  # a stage-load hold long enough for in_stage to confirm

Poll = tuple[int, int, int, int]  # (counter, p1_damage, p2_damage, match_flag)


# --------------------------------------------------------------------------- builders


def _menu(n: int, *, counter: int = 0, start: int = 0) -> list[Poll]:
    """``n`` menu polls: the flag churns (changes every poll), the counter idle, no damage."""
    return [(counter, 0, 0, MENU_FLAGS[(start + i) % len(MENU_FLAGS)]) for i in range(n)]


def _round(flag: int) -> list[Poll]:
    """One round: the counter climbs from a reset; P2 (the loser) takes lethal damage at the KO."""
    return [
        (5, 0, 0, flag),
        (300, 0, 40, flag),
        (700, 0, 120, flag),
        (1100, 0, 180, flag),
        (1400, 0, HP, flag),  # loser hits round_start_health -> round_over latches
    ]


def _stage(flag: int, n_rounds: int) -> list[Poll]:
    """A loaded stage: a flag-held load hold (counter idle), then ``n_rounds`` climbing rounds."""
    polls: list[Poll] = [(0, 0, 0, flag)] * LOAD_HOLD
    for _ in range(n_rounds):
        polls += _round(flag)
    return polls


def _results(flag: int) -> list[Poll]:
    """The post-match results screen: the flag still holds, the counter resets and climbs again."""
    return [(5, 0, 0, flag), (400, 0, 0, flag), (900, 0, 0, flag)]


def _arc(flag: int, n_rounds: int = 2) -> list[Poll]:
    """One whole match: stage -> results -> return-to-menu churn (frozen counter, no reset)."""
    return _stage(flag, n_rounds) + _results(flag) + _menu(10, counter=900)


def _drive(polls: list[Poll]) -> list[DerivedPhase]:
    tracker = MatchPhaseTracker(HP)
    return [tracker.update(*p) for p in polls]


def _transitions(phases: list[DerivedPhase]) -> list[tuple[MatchState, int]]:
    """Collapse consecutive-equal ``(match_state, round)`` into the change list."""
    out: list[tuple[MatchState, int]] = []
    prev: tuple[MatchState, int] | None = None
    for p in phases:
        key = (p.match_state, p.round)
        if key != prev:
            out.append(key)
            prev = key
    return out


def _count(phases: list[DerivedPhase], state: MatchState) -> int:
    return sum(1 for p in phases if p.match_state is state)


def _onsets(phases: list[DerivedPhase], state: MatchState) -> list[int]:
    """The round index at each *transition into* ``state``."""
    return [r for s, r in _transitions(phases) if s is state]


# --------------------------------------------------------------------------- full arc


def test_full_match_arc_menu_to_rounds_to_match_over_to_menu() -> None:
    phases = _drive(_arc(73, n_rounds=3))
    trans = _transitions(phases)
    states = [s for s, _ in trans]

    # Opens in menu (the load hold reads menu until a real round arms), runs the round arc, ends the
    # match on exactly one match_over at the stage-unload edge, then idles back to menu.
    assert states[0] is MatchState.menu
    assert states[-1] is MatchState.menu
    assert _count(phases, MatchState.match_over) == 1

    # The round arc surfaces all three round phases, and the round_over onsets climb 1..3.
    assert {MatchState.pre_round, MatchState.in_round, MatchState.round_over} <= set(states)
    assert _onsets(phases, MatchState.round_over) == [1, 2, 3]

    # The first real (armed) round is round 1, and match_over is the last active phase before menu.
    first_active = next(s for s in states if s is not MatchState.menu)
    assert first_active is MatchState.in_round
    assert states[states.index(MatchState.match_over) + 1] is MatchState.menu


def test_arc_is_value_agnostic_in_the_held_flag() -> None:
    # The gate keys on hold-vs-churn, never on the specific in-stage value: practice held 127, VS
    # held 73, and the held value even changes within a stage. The identical arc with 73 / 127 / an
    # arbitrary third value yields the identical phase sequence.
    baseline = _transitions(_drive(_arc(73)))
    assert _transitions(_drive(_arc(127))) == baseline
    assert _transitions(_drive(_arc(50_000))) == baseline


# --------------------------------------------------------------------------- false positive 1


def test_menu_hold_at_40_never_arms_a_match() -> None:
    # The ~37 s stable hold at 40 in the pre-match setup menus: the flag holds (in_stage is True),
    # but no round runs — the counter is idle, never advancing — so the tracker never arms. Result:
    # never a live round, never a match_over; it stays menu throughout. A stale non-zero idle
    # counter (not just 0) still never advances, so the value-agnostic advance guard holds.
    polls = _menu(8) + [(1234, 0, 0, 40)] * 40 + _menu(8, counter=1234)
    phases = _drive(polls)
    assert all(p.match_state is MatchState.menu for p in phases)
    assert _count(phases, MatchState.match_over) == 0


# --------------------------------------------------------------------------- false positive 2


def test_practice_substate_change_73_to_127_is_not_a_match_over() -> None:
    # Inside practice the flag jumped 73 -> 127 once, mid-round, then re-held. A single change is
    # below the churn debounce, so it must NOT flip the stage off / fire match_over: the round arc
    # continues uninterrupted with the counter still ticking.
    polls = _stage(73, 0)  # load + arm-ready hold
    polls += [(5, 0, 0, 73), (300, 0, 40, 73), (700, 0, 120, 73)]  # armed, mid-round on 73
    polls += [(900, 0, 140, 127)]  # the single 73 -> 127 substate change
    polls += [(1100, 0, 180, 127), (1400, 0, HP, 127)]  # re-holds 127, round runs to the KO
    phases = _drive(polls)

    assert _count(phases, MatchState.match_over) == 0  # the lone change is not a stage unload
    # The round still reaches its KO (round_over) — the arc was never interrupted by the blip.
    assert _onsets(phases, MatchState.round_over) == [1]
    # The poll of the 73 -> 127 change is still an armed in-round phase, not menu/match_over.
    assert phases[LOAD_HOLD + 3].match_state is MatchState.in_round


# --------------------------------------------------------------------------- two matches


def test_two_matches_each_fire_match_over_and_restart_the_round_index() -> None:
    # A session of two matches back-to-back: each ends with its own match_over, and the round index
    # restarts at 1 for the second match (a fresh RoundPhaseTracker per stage) rather than climbing
    # across the session.
    phases = _drive(_arc(73, n_rounds=2) + _arc(73, n_rounds=2))

    assert _count(phases, MatchState.match_over) == 2
    # Each match's two rounds are decided at index 1 then 2 — the second match restarts (not 3, 4).
    assert _onsets(phases, MatchState.round_over) == [1, 2, 1, 2]
