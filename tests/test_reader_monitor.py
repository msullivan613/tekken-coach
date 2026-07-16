"""The pure core behind the ``monitor`` command (live state-map verification, docs/02 §8).

The live loop is a ``while True`` decode (``# pragma: no cover``); the tested parts are the
frame -> views reduction, the console formatting, and the emit-on-change stream.
"""

from __future__ import annotations

from tekken_coach.reader.decode import DerivedPhase
from tekken_coach.reader.monitor import (
    PlayerView,
    changed_views,
    format_input,
    format_phase,
    format_view,
    monitor_lines,
    view_of,
    views_of,
)
from tekken_coach.schemas import ActionState, FrameRecord, InputState, MatchState
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


# --- input-reconstruction probe (brief #9) -----------------------------------------------------


def _frame_with_input(dir_: int, buttons: list[str]) -> FrameRecord:
    """A frame whose P1 carries the given decoded input (P2 pinned to a stable neutral baseline)."""
    fr = _frame_with(ActionState.neutral)
    p1 = fr.players[0].model_copy(update={"input": InputState(dir=dir_, buttons=buttons)})
    return fr.model_copy(update={"players": [p1, fr.players[1]]})


def test_format_input_renders_dir_buttons_bare_dir_and_none() -> None:
    assert format_input((6, ("2",))) == "6:2"
    assert format_input((3, ("1", "2"))) == "3:1+2"
    assert format_input((5, ())) == "5:-"  # a bare direction, no button
    assert format_input(None) == "none"


def test_view_of_populates_input_from_the_frame() -> None:
    fr = _frame_with_input(3, ["1", "2"])
    v = view_of(0, fr.players[0])
    assert v.input == (3, ("1", "2"))
    assert v.input_key == (3, ("1", "2"))
    # Input is not in the default state key (so the normal monitor is unchanged)...
    assert v.key == ("neutral", (), fr.players[0].move_id)
    # ...and not shown unless asked for.
    assert "in=" not in format_view(v)
    assert "in=3:1+2" in format_view(v, show_input=True)


def test_view_of_none_input_is_none() -> None:
    fr = _frame_with(ActionState.neutral)
    p1 = fr.players[0].model_copy(update={"input": None})
    v = view_of(0, fr.model_copy(update={"players": [p1, fr.players[1]]}).players[0])
    assert v.input is None
    assert "in=none" in format_view(v, show_input=True)


def test_changed_views_with_input_surfaces_each_distinct_press_while_state_is_held() -> None:
    # The probe's whole point: standing still (action_state=neutral, move_id held) pressing 2 then 3
    # then holding df must emit three lines — the state key would collapse them to one.
    stream = [
        (0.0, views_of(_frame_with_input(5, []))),  # neutral, no buttons
        (0.1, views_of(_frame_with_input(5, ["2"]))),  # press 2
        (0.2, views_of(_frame_with_input(5, ["2"]))),  # hold 2 -> no re-emit
        (0.3, views_of(_frame_with_input(5, ["3"]))),  # press 3
        (0.4, views_of(_frame_with_input(3, []))),  # hold df, release buttons
    ]
    p1 = [
        (round(t, 1), v.input) for t, v in changed_views(stream, with_input=True) if v.player == 1
    ]
    assert p1 == [(0.0, (5, ())), (0.1, (5, ("2",))), (0.3, (5, ("3",))), (0.4, (3, ()))]


def test_monitor_lines_show_input_keys_and_renders_the_input() -> None:
    def at(dir_: int, buttons: list[str]) -> list[PlayerView]:
        return views_of(_frame_with_input(dir_, buttons))

    stream = [
        (0.0, DerivedPhase(MatchState.in_round, 1), 73, at(5, [])),
        (0.1, DerivedPhase(MatchState.in_round, 1), 73, at(3, ["2"])),  # df+2 while state held
    ]
    lines = list(monitor_lines(stream, show_input=True))
    p1_lines = [ln for ln in lines if "P1" in ln]
    assert sum("in=3:2" in ln for ln in p1_lines) == 1  # the df+2 press surfaced
    assert sum("in=5:-" in ln for ln in p1_lines) == 1  # the prior bare-neutral surfaced too


def test_monitor_lines_tolerates_a_menu_phase_with_no_player_views() -> None:
    # Part A: at the main menu the player decode faults, so the menu-tolerant stream yields a
    # menu phase with EMPTY views (no crash). monitor_lines must render the [match] menu line and
    # emit no per-player line — the contract the live loop depends on.
    stream: list[tuple[float, DerivedPhase, int, list[PlayerView]]] = [
        (0.0, DerivedPhase(MatchState.menu, 0), 40, [])
    ]
    lines = list(monitor_lines(stream))
    assert len(lines) == 1
    assert "[match]" in lines[0] and "menu" in lines[0] and "counter=0" in lines[0]
