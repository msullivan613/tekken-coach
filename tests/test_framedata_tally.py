"""C2 tally tests (docs/03 §4, docs/06 §4.1).

The second phase of the two-phase rubric split: the xref sets a per-interaction trigger; the tally
applies the session-level recurrence rule (≥3× before it is a real knowledge check). Each of the six
starter patterns gets a fixture proving recurrence here (its trigger is proven in
``test_framedata_xref.py``), plus the grouping, example-id, and determinism guarantees.
"""

from __future__ import annotations

from tekken_coach.framedata.tally import build_tally, matchup_of
from tekken_coach.schemas import (
    Interaction,
    LabeledInteraction,
    Labels,
    StringGap,
)
from tests.factories import make_interaction


def _labeled(
    check_ids: list[str], *, move_id: int = 100, ex_id: str, **label_over: object
) -> LabeledInteraction:
    """A minimal LabeledInteraction carrying the given knowledge-check ids (grouping fields set)."""
    base: Interaction = make_interaction().model_copy(
        update={"id": ex_id, "attacker_move_id": move_id}
    )
    label_kwargs: dict[str, object] = {
        "frame_data_matched": True,
        "in_string": False,
        "is_knowledge_check": bool(check_ids),
        "knowledge_check_ids": check_ids,
        **label_over,
    }
    labels = Labels(**label_kwargs)  # type: ignore[arg-type]
    return LabeledInteraction(
        **base.model_dump(),
        attacker_move_name="X",
        attacker_char_name="Paul",
        defender_char_name="Kazuya",
        labels=labels,
    )


def _n_of(check_id: str, n: int, *, move_id: int = 100) -> list[LabeledInteraction]:
    return [_labeled([check_id], move_id=move_id, ex_id=f"m1-r1-i{i:03d}") for i in range(n)]


# ---------------------------------------------------------------------------
# recurrence rule (docs/06 §4.1): a pattern is a knowledge check only at >=3x
# ---------------------------------------------------------------------------


def test_below_threshold_is_not_recurring() -> None:
    tally = build_tally(_n_of("punish_missed", 2))
    entry = tally.entries[0]
    assert entry.count == 2
    assert entry.is_recurring is False
    assert tally.recurring() == []


def test_at_threshold_is_recurring() -> None:
    tally = build_tally(_n_of("punish_missed", 3))
    entry = tally.entries[0]
    assert entry.count == 3
    assert entry.is_recurring is True
    assert entry.knowledge_check_id == "punish_missed"
    assert entry.attacker_char == "Paul"
    assert entry.attacker_move_id == 100
    assert entry.matchup == "Paul vs Kazuya"
    assert entry.example_ids == ["m1-r1-i000", "m1-r1-i001", "m1-r1-i002"]
    assert tally.recurring() == [entry]


# Every one of the six starter patterns has a recurrence fixture (acceptance criterion).


def test_recurrence_all_six_patterns() -> None:
    for check_id in (
        "punish_missed",
        "respected_fake_gap",
        "challenged_true_string",
        "standing_duckable_high",
        "ate_low",  # the ate_low / ate_mid row
        "mashed_into_plus",
    ):
        tally = build_tally(_n_of(check_id, 3))
        recurring = tally.recurring()
        assert len(recurring) == 1, check_id
        assert recurring[0].knowledge_check_id == check_id
        assert recurring[0].count == 3


def test_ate_mid_half_of_the_row_also_recurs() -> None:
    tally = build_tally(_n_of("ate_mid", 3))
    assert tally.recurring()[0].knowledge_check_id == "ate_mid"


# ---------------------------------------------------------------------------
# grouping (docs/03 §4): by (knowledge_check_id, attacker_char, attacker_move_id, matchup)
# ---------------------------------------------------------------------------


def test_different_moves_group_separately() -> None:
    interactions = _n_of("punish_missed", 3, move_id=100) + _n_of("punish_missed", 2, move_id=200)
    tally = build_tally(interactions)
    by_move = {e.attacker_move_id: e for e in tally.entries}
    assert by_move[100].count == 3
    assert by_move[100].is_recurring is True
    assert by_move[200].count == 2
    assert by_move[200].is_recurring is False


def test_one_interaction_multiple_checks_counts_each() -> None:
    """An interaction tagged with two checks increments both groups."""
    both = _labeled(["punish_missed", "ate_low"], ex_id="m1-r1-i000")
    tally = build_tally([both])
    ids = {e.knowledge_check_id for e in tally.entries}
    assert ids == {"punish_missed", "ate_low"}
    assert all(e.count == 1 for e in tally.entries)


def test_untagged_interactions_do_not_appear() -> None:
    tally = build_tally([_labeled([], ex_id="m1-r1-i000")])
    assert tally.entries == []


def test_get_looks_up_a_group() -> None:
    tally = build_tally(_n_of("punish_missed", 3))
    entry = tally.get("punish_missed", "Paul", 100, "Paul vs Kazuya")
    assert entry is not None
    assert entry.count == 3
    assert tally.get("punish_missed", "Paul", 999, "Paul vs Kazuya") is None


def test_tally_is_deterministic_and_count_ordered() -> None:
    interactions = _n_of("ate_low", 2, move_id=200) + _n_of("punish_missed", 4, move_id=100)
    a = build_tally(interactions)
    b = build_tally(interactions)
    assert a == b
    # Higher count first (deterministic ordering).
    assert a.entries[0].knowledge_check_id == "punish_missed"
    assert a.entries[0].count == 4


def test_matchup_of_formats_attacker_vs_defender() -> None:
    labeled = _labeled(["punish_missed"], ex_id="m1-r1-i000")
    assert matchup_of(labeled) == "Paul vs Kazuya"


# A realistic label payload for each pattern round-trips through the tally unchanged; this guards
# that the grouping keys read the right fields (string/height/plus labels don't confuse grouping).


def test_recurrence_with_full_label_payloads() -> None:
    duck = [
        _labeled(
            ["standing_duckable_high"],
            ex_id=f"m1-r1-i{i:03d}",
            in_string=True,
            duckable_high_hit=2,
            duck_punish="df+1 (i13)",
            string_gap=None,
            move_property=None,
        )
        for i in range(3)
    ]
    tally = build_tally(duck)
    assert tally.recurring()[0].knowledge_check_id == "standing_duckable_high"

    true_string = [
        _labeled(
            ["challenged_true_string"],
            ex_id=f"m2-r1-i{i:03d}",
            in_string=True,
            string_gap=StringGap.true,
        )
        for i in range(3)
    ]
    assert build_tally(true_string).recurring()[0].count == 3
