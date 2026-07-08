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

from tekken_coach.schemas import DefenderReaction, FollowUpResult, Interaction, Outcome
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
    # interactions never overlap and are ordered.
    for earlier, later in zip(interactions, interactions[1:], strict=False):
        assert earlier.end_frame <= later.start_frame


def test_all_required_scenarios_present() -> None:
    # The docs/04 §7 3a scenario set is all wired (guards against silently dropping a fixture).
    assert set(streams.ALL_STREAMS) == {
        "neutral_dead_time",
        "blocked_no_punish",
        "blocked_punished",
        "clean_hit",
        "spacing_whiff",
        "in_range_whiff_discard",
        "sidestep_whiff_punish",
        "round_boundary_truncation",
    }
