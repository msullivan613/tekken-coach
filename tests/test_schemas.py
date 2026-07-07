"""Schema tests: field/enum coverage, lossless JSON round-trip, enum validation (03)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel, ValidationError

from tekken_coach.schemas import (
    FrameRecord,
    Interaction,
    LabeledInteraction,
    Labels,
    MatchState,
    PlayerFrame,
)
from tests.factories import (
    make_frame_record,
    make_interaction,
    make_labeled_interaction,
    make_player_frame,
)

# --- lossless round-trip: object -> JSON -> object (03 acceptance) -------------


@pytest.mark.parametrize(
    ("factory", "model"),
    [
        (make_frame_record, FrameRecord),
        (make_interaction, Interaction),
        (make_labeled_interaction, LabeledInteraction),
    ],
)
def test_json_round_trip_is_lossless(
    factory: Callable[[], BaseModel], model: type[BaseModel]
) -> None:
    original = factory()
    restored = model.model_validate_json(original.model_dump_json())
    assert restored == original
    # ...and stable across a second round-trip.
    assert restored.model_dump_json() == original.model_dump_json()


def test_labeled_interaction_carries_every_interaction_field() -> None:
    """LabeledInteraction is an Interaction superset (03 §3: '...allInteractionFields')."""
    labeled = make_labeled_interaction()
    interaction_fields = set(Interaction.model_fields)
    assert interaction_fields <= set(LabeledInteraction.model_fields)
    for name in interaction_fields:
        assert hasattr(labeled, name)


def test_pos_is_exactly_three_floats() -> None:
    dumped = make_player_frame().model_dump(mode="json")
    assert dumped["pos"] == [1.42, 0.0, -0.31]


def test_player_frame_input_may_be_null() -> None:
    pf = make_player_frame(with_input=False)
    restored = PlayerFrame.model_validate_json(pf.model_dump_json())
    assert restored.input is None


def test_frame_record_requires_exactly_two_players() -> None:
    one = make_player_frame()
    with pytest.raises(ValidationError):
        FrameRecord(frame=1, match_state=MatchState.in_round, round=1, timer_ms=1000, players=[one])
    with pytest.raises(ValidationError):
        FrameRecord(
            frame=1,
            match_state=MatchState.in_round,
            round=1,
            timer_ms=1000,
            players=[one, one, one],
        )


# --- string-related label fields present with spec names (03 §3) --------------


def test_string_label_fields_present() -> None:
    for name in ("in_string", "string_gap", "gap_size", "duckable_high_hit", "duck_punish"):
        assert name in Labels.model_fields


def test_unmatched_labels_allow_null_ground_truth() -> None:
    """The frame_data_matched:false path (05 §2.3) leaves ground-truth fields null."""
    labels = Labels(frame_data_matched=False, in_string=False, is_knowledge_check=False)
    assert labels.on_block is None
    assert labels.move_property is None
    assert labels.knowledge_check_ids == []


# --- invalid enum values are rejected (Pydantic validation) -------------------


def test_invalid_action_state_rejected() -> None:
    data = make_player_frame().model_dump()
    data["action_state"] = "flying"
    with pytest.raises(ValidationError):
        PlayerFrame(**data)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("defender_reaction", "vibed"),
        ("outcome", "won_hard"),
    ],
)
def test_invalid_interaction_enum_rejected(field: str, bad_value: str) -> None:
    data = make_interaction().model_dump()
    data[field] = bad_value
    with pytest.raises(ValidationError):
        Interaction(**data)


def test_invalid_match_state_rejected_via_json() -> None:
    payload = make_frame_record().model_dump(mode="json")
    payload["match_state"] = "loading"
    with pytest.raises(ValidationError):
        FrameRecord.model_validate(payload)


def test_invalid_string_gap_enum_rejected() -> None:
    data = make_labeled_interaction().model_dump()
    data["labels"]["string_gap"] = "sometimes"
    with pytest.raises(ValidationError):
        LabeledInteraction(**data)
