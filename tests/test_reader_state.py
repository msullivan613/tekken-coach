"""Match/replay-state classification (docs/01 §4.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.decode import read_state_signal
from tekken_coach.reader.memory_source import FakeMemorySource
from tekken_coach.reader.offsets import OffsetTable, select_offset_table
from tekken_coach.reader.state import SignalKind, classify_state
from tekken_coach.schemas import MatchState
from tests.factories import make_frame_record
from tests.fixtures.reader.encode import advance_on_for, encode_frame, module_base_for

REPO_OFFSETS = Path("assets/offsets")
MODULE_BASE = 0x140000000


@pytest.fixture
def table() -> OffsetTable:
    return select_offset_table("2.01.01", REPO_OFFSETS)


def test_offline_match_in_round_is_live_match_not_online() -> None:
    sig = classify_state(MatchState.in_round, "offline_match")
    assert sig.kind is SignalKind.live_match
    assert sig.online is False
    assert sig.should_buffer_clean is False  # clean mode only buffers replay


def test_online_match_is_flagged_online_and_refused_by_clean() -> None:
    sig = classify_state(MatchState.in_round, "online_match")
    assert sig.kind is SignalKind.live_match
    assert sig.online is True
    # Defense-in-depth: clean capture must never buffer an online-match state (docs/01 §4.3).
    assert sig.should_buffer_clean is False


def test_replay_playback_is_bufferable_by_clean() -> None:
    sig = classify_state(MatchState.replay, "replay")
    assert sig.kind is SignalKind.replay_playback
    assert sig.online is False
    assert sig.should_buffer_clean is True


def test_menu_or_idle_is_idle() -> None:
    assert classify_state(MatchState.menu, "idle").kind is SignalKind.idle
    # An "active" phase but no match mode is still idle (nothing to capture).
    assert classify_state(MatchState.in_round, "idle").kind is SignalKind.idle


def test_practice_mode_active_is_live_match() -> None:
    # The doctor attaches during practice; practice is a live (offline) match state.
    sig = classify_state(MatchState.in_round, "practice")
    assert sig.kind is SignalKind.live_match
    assert sig.online is False


def test_read_state_signal_from_source(table: OffsetTable) -> None:
    fr = make_frame_record().model_copy(update={"match_state": MatchState.replay})
    image = encode_frame(fr, table, module_base=MODULE_BASE, game_mode="replay")
    source = FakeMemorySource(
        [image],
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )
    sig = read_state_signal(source, table)
    assert sig.kind is SignalKind.replay_playback
    assert sig.match_state is MatchState.replay
