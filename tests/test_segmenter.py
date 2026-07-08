"""C3a segmenter tests (docs/04 §2, §3, §6, §7).

Three layers:

* **Goldens** — each hand-authored stream (``tests/fixtures/segment/streams.py``) is run through
  the segmenter and compared frame-for-frame against a frozen ``Interaction`` list
  (``tests/fixtures/segment/goldens/*.json``). Regression = the segmenter reproduces the goldens.
* **Explicit per-scenario assertions** — the load-bearing derived values (reaction, observed
  advantage, outcome, follow-up) are asserted directly so the goldens are not self-fulfilling.
* **Invariants** — determinism (docs/04 §6) and the docs/04 §7 property tests hold across *every*
  fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tekken_coach.schemas import (
    DefenderReaction,
    FollowUpResult,
    Interaction,
    Outcome,
)
from tekken_coach.segment import Segmenter, SegmenterConfig, segment_frames
from tests.fixtures.segment import streams

MATCH_ID = "test-match#1"
GOLDENS_DIR = Path(__file__).parent / "fixtures" / "segment" / "goldens"


def _run(name: str) -> list[Interaction]:
    return segment_frames(streams.ALL_STREAMS[name](), match_id=MATCH_ID)


def _golden(name: str) -> list[dict[str, object]]:
    data = json.loads((GOLDENS_DIR / f"{name}.json").read_text())
    assert isinstance(data, list)
    return data


# ---------------------------------------------------------------------------
# Goldens: the segmenter reproduces the frozen interaction lists exactly (docs/04 §7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(streams.ALL_STREAMS))
def test_matches_golden(name: str) -> None:
    got = [it.model_dump(mode="json") for it in _run(name)]
    assert got == _golden(name)


# ---------------------------------------------------------------------------
# Per-scenario assertions on the derived outputs (docs/04 §3)
# ---------------------------------------------------------------------------


def test_neutral_dead_time_emits_nothing() -> None:
    assert _run("neutral_dead_time") == []


def test_spacing_whiff_out_of_range_emits_nothing() -> None:
    # Out of threat range → never commits → not an interaction (docs/04 §4.4).
    assert _run("spacing_whiff") == []


def test_in_range_whiff_discarded() -> None:
    # In range but the defender is uninvolved and it misses → COMMIT -> NEUTRAL discard (§2).
    assert _run("in_range_whiff_discard") == []


def test_blocked_no_punish() -> None:
    (it,) = _run("blocked_no_punish")
    assert it.attacker == 0 and it.defender == 1
    assert it.attacker_move_id == 800
    assert it.attacker_char_id == streams.KAZUYA and it.defender_char_id == streams.DEFENDER
    assert it.defender_reaction is DefenderReaction.blocked
    assert it.observed_advantage == -13  # defender free (120) − attacker free (133)
    assert it.outcome is Outcome.no_punish
    assert it.follow_up.result is FollowUpResult.none
    assert it.follow_up.move_id is None and it.follow_up.reaction_frames is None


def test_blocked_punished() -> None:
    (it,) = _run("blocked_punished")
    assert it.defender_reaction is DefenderReaction.blocked
    assert it.observed_advantage == -13
    assert it.outcome is Outcome.punished
    assert it.follow_up.result is FollowUpResult.hit
    assert it.follow_up.move_id == 900
    assert it.follow_up.reaction_frames == 3  # acted 3 frames after becoming actionable


def test_clean_hit() -> None:
    (it,) = _run("clean_hit")
    assert it.defender_reaction is DefenderReaction.hit
    assert it.observed_advantage == 8  # attacker plus on hit
    # A clean hit's ate_mid/ate_low needs the move's height (xref); the structural guess is neutral.
    assert it.outcome is Outcome.neutral
    assert it.follow_up.result is FollowUpResult.none


def test_sidestep_whiff_punish() -> None:
    (it,) = _run("sidestep_whiff_punish")
    # Evaded, then the whiff punish landed → the reaction is upgraded (docs/04 §4.4).
    assert it.defender_reaction is DefenderReaction.whiff_punished
    assert it.observed_advantage == -17  # deep whiff-recovery disadvantage
    assert it.outcome is Outcome.punished
    assert it.follow_up.result is FollowUpResult.hit
    assert it.follow_up.move_id == 930
    assert it.follow_up.reaction_frames == 9


def test_round_boundary_truncation() -> None:
    # A round ends mid-blockstun: the open interaction is emitted, truncated, not lost (§4.8).
    (it,) = _run("round_boundary_truncation")
    assert it.defender_reaction is DefenderReaction.blocked
    assert it.observed_advantage is None  # neither player became actionable before the boundary
    assert it.notes == ["truncated:round-boundary"]
    assert it.end_frame == 113


# ---------------------------------------------------------------------------
# C3b edge cases (docs/04 §4.1–§4.8) — independent per-field assertions
# ---------------------------------------------------------------------------


def test_stagger_is_its_own_reaction() -> None:
    # §4.1: a forced stagger is distinct from block/hit; the disadvantage folds into observed_adv.
    (it,) = _run("stagger_on_block")
    assert it.defender_reaction is DefenderReaction.stagger
    assert it.observed_advantage == -6  # defender free (115, stagger end) − attacker free (121)
    assert it.string_hits == []


def test_string_blocked_standing_records_per_hit() -> None:
    # §4.2 golden: a mid→high→mid string jailed and blocked standing → three per-hit records.
    (it,) = _run("string_blocked_standing")
    assert it.defender_reaction is DefenderReaction.blocked
    assert it.attacker_move_id == 100  # stays the string's ENTRY across all hits
    assert [(h.hit_index, h.defender_reaction, h.defender_crouching) for h in it.string_hits] == [
        (1, DefenderReaction.blocked, False),
        (2, DefenderReaction.blocked, False),  # the high, blocked STANDING → duck-punishable
        (3, DefenderReaction.blocked, False),
    ]


def test_string_ducked_high_records_evaded_hit() -> None:
    # §4.2: the defender ducks the high (crouch-blocks hit 1, hit 2 whiffs) → hit 2 is `evaded`,
    # crouching, which is the signal that this was correct play (no duck flag downstream).
    (it,) = _run("string_ducked_high")
    assert it.defender_reaction is DefenderReaction.blocked  # hit 1 (the mid) was blocked
    assert [(h.hit_index, h.defender_reaction, h.defender_crouching) for h in it.string_hits] == [
        (1, DefenderReaction.blocked, True),
        (2, DefenderReaction.evaded, True),  # ducked the high; it whiffed and broke the string
    ]
    # The "hit index at which the defender's state changed" (04 §4.2) is derivable: first hit whose
    # reaction/posture differs from hit 1 → hit 2.
    changed = next(
        h.hit_index
        for h in it.string_hits
        if (h.defender_reaction, h.defender_crouching)
        != (it.string_hits[0].defender_reaction, it.string_hits[0].defender_crouching)
    )
    assert changed == 2


def test_string_interrupted_closes_on_actionable_gap() -> None:
    # §4.2: a two-hit string the defender interrupts in the gap after hit 2 → string closes when
    # the defender became actionable and acted; two per-hit records, follow-up landed.
    (it,) = _run("string_interrupted")
    assert [(h.hit_index, h.defender_reaction) for h in it.string_hits] == [
        (1, DefenderReaction.blocked),
        (2, DefenderReaction.blocked),
    ]
    assert it.follow_up.result is FollowUpResult.hit
    assert it.follow_up.move_id == 962


def test_throw_broke() -> None:
    # §4.3: the defender breaks the throw in the tech window.
    (it,) = _run("throw_broke")
    assert it.defender_reaction is DefenderReaction.throw_broke
    assert it.string_hits == []


def test_thrown() -> None:
    # §4.3: the defender fails to break → thrown.
    (it,) = _run("thrown")
    assert it.defender_reaction is DefenderReaction.thrown


def test_knockdown_followup_extends_to_wakeup() -> None:
    # §4.3: on knockdown the follow-up window extends to the wakeup-actionable frame, so the
    # observed advantage spans the whole oki window (defender free at 133, attacker free at 113).
    (it,) = _run("knockdown_wakeup")
    assert it.defender_reaction is DefenderReaction.hit
    assert it.observed_advantage == 20


def test_counter_hit_reaction() -> None:
    # §4.5: the defender pressed and was counter-hit by the attack.
    (it,) = _run("counter_hit")
    assert it.defender_reaction is DefenderReaction.counter_hit


def test_mashed_into_counter_followup_signal() -> None:
    # §4.5: a plus move blocked, the defender mashes, their follow-up gets counter-hit. The raw
    # signal `mashed_into_plus` keys on is follow_up.result == got_counter_hit.
    (it,) = _run("mashed_into_counter")
    assert it.defender_reaction is DefenderReaction.blocked
    assert it.observed_advantage == 1  # the blocked move was plus
    assert it.follow_up.result is FollowUpResult.got_counter_hit
    assert it.follow_up.move_id == 965
    # The segmenter's outcome stays a conservative guess; xref finalizes mashed_into_ch.
    assert it.outcome is Outcome.neutral


def test_heat_activation_is_noted() -> None:
    # §4.6: a Heat activation within the interaction is noted (it shifts advantage mid-exchange).
    (it,) = _run("heat_activation")
    assert it.notes == ["heat-activated:107"]


def test_dropped_frames_tolerated() -> None:
    # §4.7: a small poll gap is bridged and noted; observed advantage is still measured.
    (it,) = _run("dropped_frames_tolerated")
    assert it.notes == ["gap-tolerated:2"]
    assert it.observed_advantage == -13  # counted across the tolerated gap


def test_dropped_frames_unreliable_nulls_advantage() -> None:
    # §4.7: a gap beyond the threshold still emits, but observed advantage is null (unreliable).
    (it,) = _run("dropped_frames_unreliable")
    assert it.notes == ["gap-tolerated:6"]
    assert it.observed_advantage is None


def test_attacker_pressure_carries_between_interactions() -> None:
    # Item 9b: the first exchange leaves the attacker plus, so the second opens with pressure.
    first, second = _run("attacker_pressure_carry")
    assert first.context.attacker_pressure is False
    assert first.observed_advantage == 8  # attacker plus on hit → carries pressure
    assert second.context.attacker_pressure is True
    assert second.notes == []  # resolves cleanly, no truncation


# ---------------------------------------------------------------------------
# Determinism (docs/04 §6): identical stream → identical interactions, always
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(streams.ALL_STREAMS))
def test_deterministic_rerun(name: str) -> None:
    first = [it.model_dump() for it in _run(name)]
    second = [it.model_dump() for it in _run(name)]
    assert first == second


def test_feed_and_close_equals_segment_frames() -> None:
    # The streaming API and the convenience wrapper agree (no hidden state in the wrapper).
    frames = streams.blocked_punished()
    seg = Segmenter(MATCH_ID)
    streamed: list[Interaction] = []
    for fr in frames:
        streamed.extend(seg.feed(fr))
    streamed.extend(seg.close())
    assert [it.model_dump() for it in streamed] == [
        it.model_dump() for it in segment_frames(frames, match_id=MATCH_ID)
    ]


def test_threat_range_is_the_commit_gate() -> None:
    # The blocked exchange happens at distance 1.5. Shrink threat range below it and the attack no
    # longer commits — proving the threshold is the gate and is genuinely configurable.
    assert _run("blocked_no_punish") != []  # default range 2.5 > 1.5
    narrow = SegmenterConfig(threat_range=1.0)
    assert segment_frames(streams.blocked_no_punish(), match_id=MATCH_ID, config=narrow) == []


# ---------------------------------------------------------------------------
# Property tests across every fixture (docs/04 §7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(streams.ALL_STREAMS))
def test_properties_hold(name: str) -> None:
    interactions = _run(name)
    for it in interactions:
        # attacker ≠ defender, both valid player indices.
        assert it.attacker != it.defender
        assert {it.attacker, it.defender} == {0, 1}
        # start_frame < end_frame.
        assert it.start_frame < it.end_frame
        # every interaction lies within a single round.
        assert it.round >= 1
        # docs/04 §4.2 string invariants: per-hit indices are contiguous 1..N, and the whole
        # string lies within one round (it is one interaction, which never spans a round, so this
        # is guaranteed structurally — asserted here as the load-bearing property).
        if it.string_hits:
            assert len(it.string_hits) >= 2  # single-hit interactions carry [] (04 §4.2)
            assert [h.hit_index for h in it.string_hits] == list(range(1, len(it.string_hits) + 1))
    # interactions never overlap and are ordered.
    for earlier, later in zip(interactions, interactions[1:], strict=False):
        assert earlier.end_frame <= later.start_frame


def test_string_lies_within_one_round() -> None:
    # docs/04 §4.2: a multi-hit string is one interaction, so it cannot span a round boundary.
    interactions = _run("string_blocked_standing")
    (it,) = interactions
    assert it.string_hits  # it is a string
    assert it.round == 1
    for fr in streams.string_blocked_standing():
        # every frame the interaction covers is in the interaction's round.
        if it.start_frame <= fr.frame <= it.end_frame:
            assert fr.round == it.round


def test_all_required_scenarios_present() -> None:
    # The docs/04 §7 scenario set (C3a + C3b §4.1–§4.8) is all wired (guards against silently
    # dropping a fixture).
    assert set(streams.ALL_STREAMS) == {
        # C3a clean paths.
        "neutral_dead_time",
        "blocked_no_punish",
        "blocked_punished",
        "clean_hit",
        "spacing_whiff",
        "in_range_whiff_discard",
        "sidestep_whiff_punish",
        "round_boundary_truncation",
        # C3b edge cases (docs/04 §4).
        "stagger_on_block",
        "string_blocked_standing",
        "string_ducked_high",
        "string_interrupted",
        "throw_broke",
        "thrown",
        "knockdown_wakeup",
        "counter_hit",
        "mashed_into_counter",
        "heat_activation",
        "dropped_frames_tolerated",
        "dropped_frames_unreliable",
        "attacker_pressure_carry",
    }
