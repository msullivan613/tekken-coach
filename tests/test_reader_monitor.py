"""The pure core behind the ``monitor`` command (live state-map verification, docs/02 §8).

The live loop is a ``while True`` decode (``# pragma: no cover``); the tested parts are the
frame -> views reduction, the console formatting, and the emit-on-change stream.
"""

from __future__ import annotations

from tekken_coach.reader.decode import DerivedPhase
from tekken_coach.reader.monitor import (
    PlayerView,
    changed_views,
    format_phase,
    format_view,
    monitor_lines,
    view_of,
    views_of,
)
from tekken_coach.schemas import ActionState, FrameRecord, MatchState
from tests.factories import make_frame_record


def _frame_with(p1_state: ActionState, **p1_flags: bool) -> FrameRecord:
    """A frame with P1 in the given state (+flags) and P2 pinned to a stable neutral baseline."""
    fr = make_frame_record()
    p1 = fr.players[0].model_copy(update={"action_state": p1_state, **p1_flags})
    p2 = fr.players[1].model_copy(update={"action_state": ActionState.neutral})
    return fr.model_copy(update={"players": [p1, p2]})


def test_view_of_surfaces_action_state_and_true_flags() -> None:
    fr = _frame_with(ActionState.hitstun, hit_stun=True, juggle=True, airborne=True)
    v = view_of(0, fr.players[0])
    assert v.player == 1  # 1-based
    assert v.action_state == "hitstun"
    assert v.flags == ("hit_stun", "airborne", "juggle")  # fixed order, only the true ones
    assert v.char_id == fr.players[0].char_id
    assert v.key == ("hitstun", ("hit_stun", "airborne", "juggle"), fr.players[0].move_id)


def test_view_of_no_flags_reads_clean() -> None:
    fr = _frame_with(ActionState.neutral)
    v = view_of(1, fr.players[1])
    assert v.player == 2 and v.flags == ()
    assert "[-]" in format_view(v)  # empty flag set renders as a dash


def test_view_carries_the_round_counter_but_does_not_key_on_it() -> None:
    # The per-round counter rides the line for eyeballing (docs/02 §8) but is NOT in the change key,
    # so it does not flood a held state to one line per poll.
    fr = _frame_with(ActionState.neutral)
    pf = fr.players[0].model_copy(update={"frames_since_round_start": 742})
    v = view_of(0, pf)
    assert v.counter == 742
    assert "cnt=742" in format_view(v)
    assert 742 not in v.key  # counter is deliberately excluded from the change key


def test_format_view_optionally_appends_raw_words() -> None:
    fr = _frame_with(ActionState.blockstun, block_stun=True)
    pf = fr.players[0].model_copy(update={"raw_state": {"stun_type": 1, "simple_move_state": 2}})
    v = view_of(0, pf)
    plain = format_view(v)
    assert "blockstun" in plain and "block_stun" in plain and "raw(" not in plain
    withraw = format_view(v, show_raw=True)
    assert "raw(simple_move_state=2 stun_type=1)" in withraw  # sorted by name


def test_changed_views_emits_only_when_the_decoded_state_changes() -> None:
    n = views_of(_frame_with(ActionState.neutral))
    atk = views_of(_frame_with(ActionState.attack))
    blk = views_of(_frame_with(ActionState.blockstun, block_stun=True))
    stream = [
        (0.0, n),  # first sight -> both players emit
        (0.1, n),  # unchanged -> nothing
        (0.2, atk),  # P1 changed -> P1 only
        (0.3, blk),  # P1 changed -> P1 only
        (0.4, blk),  # unchanged
    ]
    emitted = [(round(t, 1), v.player, v.action_state) for t, v in changed_views(stream)]
    assert emitted == [
        (0.0, 1, "neutral"),
        (0.0, 2, "neutral"),
        (0.2, 1, "attack"),
        (0.3, 1, "blockstun"),
    ]


def test_changed_views_surfaces_each_move_in_a_string() -> None:
    # A string (1,2,1) stays action_state=attack throughout; keying on move_id too means each move
    # emits its own line instead of collapsing to the opening jab (the reported gap).
    def atk(move_id: int) -> list[PlayerView]:
        fr = _frame_with(ActionState.attack)
        return views_of(
            fr.model_copy(
                update={
                    "players": [
                        fr.players[0].model_copy(update={"move_id": move_id}),
                        fr.players[1],
                    ]
                }
            )
        )

    stream = [(0.0, atk(1586)), (0.1, atk(2)), (0.2, atk(1586)), (0.3, atk(1586))]
    p1 = [(round(t, 1), v.move_id) for t, v in changed_views(stream) if v.player == 1]
    assert p1 == [
        (0.0, 1586),
        (0.1, 2),
        (0.2, 1586),
    ]  # 1, 2, 1 — the held final frame does not repeat


def test_changed_views_tracks_players_independently() -> None:
    fr = make_frame_record()
    both_default = views_of(fr)
    # flip only P2 to hitstun
    p2_hit = fr.model_copy(
        update={
            "players": [
                fr.players[0],
                fr.players[1].model_copy(update={"action_state": ActionState.hitstun}),
            ]
        }
    )
    emitted = list(changed_views([(0.0, both_default), (0.1, views_of(p2_hit))]))
    # frame 0: both new; frame 1: only P2 changed
    assert [(v.player, v.action_state) for _, v in emitted][2:] == [(2, "hitstun")]


def test_format_phase_shows_state_round_counter_and_flag() -> None:
    line = format_phase("in_round", 2, 813, 73)
    assert "in_round" in line and "round=2" in line and "counter=813" in line and "flag=73" in line


def test_monitor_lines_emits_phase_on_change_and_views_on_change() -> None:
    # The live-loop core: a [match] line whenever the derived phase changes, per-player lines
    # whenever a player's decoded state changes — both on-change so a held situation is one line.
    def views_at(counter: int, p1_state: ActionState) -> list[PlayerView]:
        fr = _frame_with(p1_state)
        p1 = fr.players[0].model_copy(update={"frames_since_round_start": counter})
        p2 = fr.players[1].model_copy(update={"frames_since_round_start": counter})
        return views_of(fr.model_copy(update={"players": [p1, p2]}))

    stream = [
        (0.0, DerivedPhase(MatchState.pre_round, 1), 73, views_at(1, ActionState.neutral)),
        (0.1, DerivedPhase(MatchState.in_round, 1), 73, views_at(60, ActionState.neutral)),
        (0.2, DerivedPhase(MatchState.in_round, 1), 73, views_at(120, ActionState.attack)),
        (0.3, DerivedPhase(MatchState.round_over, 1), 73, views_at(180, ActionState.attack)),
    ]
    lines = list(monitor_lines(stream))
    match_lines = [ln for ln in lines if "[match]" in ln]
    # Three distinct phases -> three [match] lines (in_round repeats but does not re-emit).
    assert [ln.split("[match]")[1].split()[0] for ln in match_lines] == [
        "pre_round",
        "in_round",
        "round_over",
    ]
    # P1's state change (neutral -> attack) surfaces; the held neutral does not repeat every poll.
    assert sum("attack" in ln for ln in lines) == 1
    assert sum("P1" in ln and "neutral" in ln for ln in lines) == 1


def test_monitor_lines_flag_rides_line_but_does_not_drive_reemit() -> None:
    # The raw match_flag shows on the [match] line, but (like the counter) it is NOT in the change
    # key: a churning flag under a held phase must not flood one [match] line per poll.
    views = views_of(_frame_with(ActionState.neutral))
    stream = [
        (0.0, DerivedPhase(MatchState.menu, 0), 16, views),
        (0.1, DerivedPhase(MatchState.menu, 0), 40, views),  # flag churns, phase held
        (0.2, DerivedPhase(MatchState.menu, 0), 56, views),
    ]
    match_lines = [ln for ln in monitor_lines(stream) if "[match]" in ln]
    assert len(match_lines) == 1  # held menu phase -> one line despite the churning flag
    assert "flag=16" in match_lines[0]  # shows the flag at first sight
