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
from tekken_coach.reader.faults import DecodeError
from tekken_coach.reader.memory_source import FakeMemorySource, MemoryImage
from tekken_coach.reader.offsets import FieldSpec, OffsetTable, select_offset_table
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


def test_capture_refuses_a_build_whose_match_phase_is_uncalibrated(table: OffsetTable) -> None:
    # The diagnostic/capture boundary (docs/02 §6). `decode_frame` decodes an unrecognized phase to
    # MatchState.unknown so the doctor can still validate the anchors on an uncalibrated build — but
    # capture WRITES FRAMES, and a capture that cannot tell an online ranked match from a practice
    # round must not run (docs/01 §4.3). It refuses before polling a single frame.
    frames = _frames(3)
    source = _source(frames, table)
    raw_in_round = next(k for k, v in table.state_codes.match_phase.items() if v == "in_round")
    del table.state_codes.match_phase[raw_in_round]  # the phase codes were never calibrated

    with pytest.raises(DecodeError, match=f"unknown match_phase code {raw_in_round}"):
        run_capture(source, table, 3, game_version="2.01.01")


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


def _derive_table(table: OffsetTable) -> OffsetTable:
    """A copy of the table exposing the counter + the global ``match_flag`` — i.e. it derives phase.

    ``frames_since_round_start`` at a free player-struct offset (128, past the last legacy field at
    64), and the Stage 2 ``match_flag`` global at a free global offset (128) — both so the encoder
    round-trips them. This turns on the round-gating (derived) path in ``run_capture`` (docs/02 §8).
    """
    derived = table.model_copy(deep=True)
    derived.players.fields["frames_since_round_start"] = FieldSpec(offset=128, kind="u32")
    derived.global_struct.fields["match_flag"] = FieldSpec(offset=128, kind="u32")
    return derived


def _arc_frame(frame: int, counter: int, p2_health: int) -> FrameRecord:
    """One frame of an in-match arc: P1 healthy, P2 at ``p2_health``, both carrying ``counter``."""

    def player(char_id: int, health: int) -> PlayerFrame:
        return _player(char_id, 800, 1.0).model_copy(
            update={"health": health, "frames_since_round_start": counter}
        )

    # A constant global match_state (``menu``) across every frame: the derived output varies anyway,
    # which is the point — it comes from the counter + flag, not from the (bogus) global phase read.
    return FrameRecord(
        frame=frame,
        match_state=MatchState.menu,  # overwritten by the derived phase
        round=0,  # overwritten by the derived round index
        timer_ms=42000,
        players=[player(12, 200), player(7, p2_health)],
    )


def _held_source(frames: list[FrameRecord], table: OffsetTable, flag: int) -> FakeMemorySource:
    """A source encoding every frame with the same held ``match_flag`` (a loaded-stage hold)."""
    images: list[MemoryImage] = [
        encode_frame(fr, table, module_base=MODULE_BASE, game_mode="practice", match_flag=flag)
        for fr in frames
    ]
    return FakeMemorySource(
        images,
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )


def test_capture_derives_the_match_phase_from_the_counter_and_flag_not_the_global_read() -> None:
    # The real-game scenario (docs/02 §8, round-gating): a build whose global match_phase is useless
    # but which exposes the per-player counter + the global match_flag. run_capture DERIVES phase
    # via MatchPhaseTracker (never consulting the strict gate, so it cannot raise) and stamps a
    # correct match_state + round over the bogus seeded global reads. Every frame encodes the SAME
    # global phase and the SAME held match_flag (a loaded stage), yet the derived arc still moves
    # through the round phases — which could ONLY come from the counter + damage.
    table = _derive_table(select_offset_table("2.01.01", REPO_OFFSETS))

    # A stage-load hold (counter idle at 0 -> in_stage confirms, no arm), then a 2-round arc
    # (counter climbs then resets; P2 KO'd to end each round). The held flag (73) keeps us in-stage.
    hold = [_arc_frame(1000 + i, 0, 200) for i in range(20)]  # >= STAGE_HOLD_POLLS
    arc = [
        _arc_frame(1020, 300, 200),  # counter climbs -> arms -> in_round, round 1
        _arc_frame(1021, 1000, 0),  # P2 KO'd          -> round_over, round 1
        _arc_frame(1022, 5, 200),  # counter reset      -> pre_round, round 2
        _arc_frame(1023, 500, 200),  # climbing         -> in_round,  round 2
        _arc_frame(1024, 1000, 0),  # P2 KO'd           -> round_over, round 2
    ]
    frames = hold + arc
    capture = run_capture(
        _held_source(frames, table, 73), table, len(frames), game_version="2.01.01"
    )

    states = [fr.match_state for fr in capture.frames]
    # The load hold reads menu until a real round arms; then the derived round arc plays out.
    assert set(states[:20]) == {MatchState.menu}
    assert states[20:] == [
        MatchState.in_round,
        MatchState.round_over,
        MatchState.pre_round,
        MatchState.in_round,
        MatchState.round_over,
    ]
    assert [fr.round for fr in capture.frames][20:] == [1, 1, 2, 2, 2]


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
