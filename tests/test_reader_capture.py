"""Offline tests for capture serialize->reload (docs/02, docs/01 §4) via a FakeMemorySource.

Proves the acceptance criterion that ``capture`` produces a schema-valid ``FrameRecord`` JSON
fixture — without the game — by running the full poll->serialize->reload path against a scripted
fake source and asserting the reloaded frames validate against the C0 ``FrameRecord`` schema and
round-trip byte-for-byte.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.capture import (
    CAPTURE_FORMAT_VERSION,
    CaptureFile,
    load_capture,
    run_capture,
    write_capture,
)
from tekken_coach.reader.memory_source import FakeMemorySource, MemoryImage
from tekken_coach.reader.offsets import OffsetTable, select_offset_table
from tekken_coach.schemas import (
    ActionState,
    CounterState,
    FrameRecord,
    HeatState,
    MatchState,
    PlayerFrame,
)
from tests.fixtures.reader.encode import advance_on_for, encode_frame, module_base_for

REPO_OFFSETS = Path("assets/offsets")
MODULE_BASE = 0x140000000


@pytest.fixture
def table() -> OffsetTable:
    return select_offset_table("2.01.01", REPO_OFFSETS)


def _player(char_id: int, move_id: int, x: float) -> PlayerFrame:
    return PlayerFrame(
        char_id=char_id,
        move_id=move_id,
        move_frame=3,
        action_state=ActionState.attack,
        health=150,
        pos=(x, 0.0, 0.5),
        facing=1,
        block_stun=False,
        hit_stun=False,
        counter_state=CounterState.none,
        throw_active=False,
        airborne=False,
        juggle=False,
        heat=HeatState(active=True, timer_ms=8000, engager_used=False),
        rage=False,
        input=None,
    )


def _frames(n: int) -> list[FrameRecord]:
    out = []
    for i in range(n):
        out.append(
            FrameRecord(
                frame=1000 + i,
                match_state=MatchState.in_round,
                round=1,
                timer_ms=42000 - i,
                players=[_player(12, 2145, 1.0), _player(7, 800, -1.0 - 0.1 * i)],
            )
        )
    return out


def _source(frames: list[FrameRecord], table: OffsetTable) -> FakeMemorySource:
    images: list[MemoryImage] = [
        encode_frame(fr, table, module_base=MODULE_BASE, game_mode="practice") for fr in frames
    ]
    return FakeMemorySource(
        images,
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )


def test_capture_roundtrips_through_a_file(tmp_path: Path, table: OffsetTable) -> None:
    frames = _frames(5)
    capture = run_capture(_source(frames, table), table, 5, game_version="2.01.01")

    out = tmp_path / "capture.json"
    write_capture(out, capture)
    reloaded = load_capture(out)

    assert reloaded.meta.game_version == "2.01.01"
    assert reloaded.meta.format_version == CAPTURE_FORMAT_VERSION
    assert reloaded.meta.frame_count == 5
    # Byte-for-byte round-trip through the file.
    assert reloaded == capture
    # Every reloaded frame is a valid FrameRecord (schema-validated on load).
    assert all(isinstance(fr, FrameRecord) for fr in reloaded.frames)
    assert [fr.frame for fr in reloaded.frames] == [1000, 1001, 1002, 1003, 1004]


def test_capture_records_gap_markers(table: OffsetTable) -> None:
    # A scripted jump in the frame counter surfaces as a per-frame gap (docs/02 §7, docs/04 §4.7).
    frames = _frames(3)
    frames[2].frame = 1010  # skip from 1001 -> 1010: 8 frames missed
    capture = run_capture(_source(frames, table), table, 3, game_version="2.01.01")
    assert capture.meta.gaps == [0, 0, 8]  # frame - prev - 1 = 1010 - 1001 - 1


def test_reloaded_frames_match_the_c0_schema_exactly(tmp_path: Path, table: OffsetTable) -> None:
    frames = _frames(2)
    capture = run_capture(_source(frames, table), table, 2, game_version="2.01.01")
    out = tmp_path / "c.json"
    write_capture(out, capture)

    # Re-parse independently through FrameRecord to prove each line is schema-valid on its own.
    raw = load_capture(out).model_dump()
    for frame_dict in raw["frames"]:
        FrameRecord.model_validate(frame_dict)


def test_capture_file_rejects_a_malformed_frame() -> None:
    # Defensive: the FrameRecord constraint (exactly 2 players) is enforced on load.
    with pytest.raises(ValueError):
        CaptureFile.model_validate(
            {
                "meta": {
                    "game_version": "2.01.01",
                    "captured_at": "2026-07-08T00:00:00Z",
                    "frame_count": 1,
                    "gaps": [0],
                },
                "frames": [
                    {
                        "frame": 1,
                        "match_state": "in_round",
                        "round": 1,
                        "timer_ms": 1,
                        "players": [],  # invalid: FrameRecord needs exactly 2
                    }
                ],
            }
        )
