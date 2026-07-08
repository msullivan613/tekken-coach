"""Fully-populated sample records for tests.

Every optional field is given a non-default value so round-trip tests actually exercise
serialization of the whole record, not just its required core.
"""

from __future__ import annotations

from tekken_coach.schemas import (
    ActionState,
    CaptureMode,
    CounterState,
    DefenderReaction,
    FollowUp,
    FollowUpResult,
    FrameRecord,
    HeatState,
    InputState,
    Interaction,
    InteractionContext,
    LabeledInteraction,
    Labels,
    MatchState,
    MatchSummary,
    MoveProperty,
    Outcome,
    PlayerFrame,
    SessionHeader,
    StringGap,
    Wall,
)


def make_player_frame(*, with_input: bool = True) -> PlayerFrame:
    return PlayerFrame(
        char_id=12,
        move_id=2145,
        move_frame=7,
        action_state=ActionState.attack,
        health=142,
        pos=(1.42, 0.0, -0.31),
        facing=1,
        block_stun=False,
        hit_stun=False,
        counter_state=CounterState.none,
        throw_active=False,
        airborne=False,
        juggle=False,
        heat=HeatState(active=True, timer_ms=3100, engager_used=True),
        rage=True,
        input=InputState(dir=6, buttons=["2"]) if with_input else None,
    )


def make_frame_record() -> FrameRecord:
    return FrameRecord(
        frame=128472,
        match_state=MatchState.in_round,
        round=2,
        timer_ms=41200,
        players=[make_player_frame(with_input=True), make_player_frame(with_input=False)],
    )


def make_interaction() -> Interaction:
    return Interaction(
        id="m3-r2-i017",
        match_id="2026-07-07T20:14:03Z#3",
        round=2,
        start_frame=128410,
        end_frame=128498,
        attacker=1,
        defender=0,
        attacker_move_id=2145,
        attacker_char_id=12,
        defender_char_id=7,
        context=InteractionContext(
            distance=1.6,
            attacker_heat=True,
            defender_heat=False,
            attacker_pressure=True,
            wall=Wall.none,
            defender_health_frac=0.71,
        ),
        defender_reaction=DefenderReaction.blocked,
        observed_advantage=-13,
        outcome=Outcome.no_punish,
        follow_up=FollowUp(move_id=0, result=FollowUpResult.none, reaction_frames=None),
        notes=["gap-tolerated:2 dropped frames"],
    )


def make_labels() -> Labels:
    return Labels(
        frame_data_matched=True,
        on_block=-13,
        was_punishable=True,
        punish_window=3,
        correct_punish="f,F+2 (i15)",
        user_punished_correctly=False,
        in_string=False,
        string_gap=StringGap.duckable,
        gap_size=4,
        duckable_high_hit=2,
        duck_punish="df+1 (i13)",
        move_property=MoveProperty.mid,
        is_knowledge_check=True,
        knowledge_check_ids=["punish_missed"],
    )


def make_labeled_interaction() -> LabeledInteraction:
    base = make_interaction()
    return LabeledInteraction(
        **base.model_dump(),
        attacker_move_name="df+2",
        attacker_char_name="Kazuya",
        defender_char_name="Jin",
        labels=make_labels(),
    )


def make_header(schema_version: str = "1.2.0") -> SessionHeader:
    return SessionHeader(
        schema_version=schema_version,
        created_at="2026-07-07T20:14:03Z",
        capture_mode=CaptureMode.clean,
        game_version="2.01.01",
        framedata_snapshot="2026-06-30",
        user_player=0,
        user_char="Jin",
        matches=[
            MatchSummary(
                match_id="2026-07-07T20:14:03Z#3",
                opponent_char="Kazuya",
                result="loss",
                rounds=3,
            )
        ],
    )
