"""Golden raw-bytes -> FrameRecord decode, plus action_state / input-null / gap behavior.

The golden test packs bytes with an **independent** kind->format map (not the shipped decoder's
``_FORMATS`` nor the test encoder), keyed off the offset table's field offsets — the offset table
is the *data* contract; pack and unpack are independent *code*. So a real byte-layout bug is
caught, not masked by shared code.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tekken_coach.reader.decode import FrameReader, decode_frame, poll_frames
from tekken_coach.reader.faults import DecodeError
from tekken_coach.reader.memory_source import FakeMemorySource
from tekken_coach.reader.offsets import FieldSpec, OffsetTable, ScalarKind, select_offset_table
from tekken_coach.schemas import ActionState, CounterState, FrameRecord, MatchState
from tests.factories import make_frame_record
from tests.fixtures.reader.encode import advance_on_for, encode_frame, module_base_for

REPO_OFFSETS = Path("assets/offsets")
MODULE_BASE = 0x140000000

# Independent kind -> little-endian struct format (deliberately not imported from the decoder).
_PACK: dict[ScalarKind, str] = {
    "u8": "<B",
    "u16": "<H",
    "u32": "<I",
    "i32": "<i",
    "i64": "<q",
    "f32": "<f",
    "bool8": "<B",
    "ptr": "<Q",
}


@pytest.fixture
def table() -> OffsetTable:
    return select_offset_table("2.01.01", REPO_OFFSETS)


def _pack(kind: ScalarKind, value: float | int | bool) -> bytes:
    if kind == "f32":
        return struct.pack(_PACK[kind], float(value))
    return struct.pack(_PACK[kind], int(value))


def test_golden_bytes_decode_to_expected_framerecord(table: OffsetTable) -> None:
    image: dict[int, bytes] = {}
    g = table.global_struct
    gbase = MODULE_BASE + g.anchor.base_offset

    def putg(name: str, kind: ScalarKind, value: int) -> None:
        image[gbase + g.fields[name].offset] = _pack(kind, value)

    # Global/match state: frame 128472, in_round, round 2, timer 41200ms.
    putg("frame_counter", "u32", 128472)
    putg("match_phase", "u32", 2)  # in_round
    putg("game_mode", "u32", 4)  # practice
    putg("round", "u32", 2)
    putg("timer_ms", "u32", 41200)

    pf = table.players.fields
    pbase = MODULE_BASE + table.players.anchor.base_offset

    def putp(idx: int, name: str, kind: ScalarKind, value: float | int | bool) -> None:
        image[pbase + idx * table.players.stride + pf[name].offset] = _pack(kind, value)

    # Every player field defaults to 0/False unless set below, so start by zeroing all of them.
    for idx in (0, 1):
        for name, spec in pf.items():
            putp(idx, name, spec.kind, 0)

    # P1 (Kazuya, mid-attack, jab held, input resolvable).
    putp(0, "char_id", "u32", 12)
    putp(0, "move_id", "u32", 2145)
    putp(0, "move_frame", "u32", 7)
    putp(0, "health", "i32", 142)
    putp(0, "pos_x", "f32", 1.5)
    putp(0, "pos_y", "f32", 0.0)
    putp(0, "pos_z", "f32", -0.31)
    putp(0, "facing", "i32", 1)
    putp(0, "simple_state", "u32", 1)  # attack
    putp(0, "heat_active", "bool8", 1)
    putp(0, "heat_timer_ms", "u32", 3100)
    putp(0, "heat_engager_used", "bool8", 1)
    putp(0, "rage", "bool8", 1)
    putp(0, "input_valid", "bool8", 1)
    putp(0, "input_dir", "u8", 6)
    putp(0, "input_buttons", "u16", 0b0010)  # button "2"

    # P2 (Jin, in blockstun on a counter-hit posture, inputs NOT resolvable -> input == null).
    putp(1, "char_id", "u32", 7)
    putp(1, "move_id", "u32", 800)
    putp(1, "move_frame", "u32", 0)
    putp(1, "health", "i32", 150)
    putp(1, "pos_x", "f32", -1.5)
    putp(1, "pos_y", "f32", 0.0)
    putp(1, "pos_z", "f32", 0.2)
    putp(1, "facing", "i32", -1)
    putp(1, "simple_state", "u32", 0)
    putp(1, "block_stun", "bool8", 1)
    putp(1, "counter_state", "u32", 1)  # counter_hit
    putp(1, "input_valid", "bool8", 0)  # inputs unresolvable this frame

    source = FakeMemorySource(
        [image], module_bases={g.anchor.module: MODULE_BASE}, advance_on=gbase
    )
    fr = decode_frame(source, table)

    # Global.
    assert fr.frame == 128472
    assert fr.match_state is MatchState.in_round
    assert fr.round == 2
    assert fr.timer_ms == 41200
    assert len(fr.players) == 2

    # P1 — every §1 field, correct types, raw flags present, derived action_state.
    p1 = fr.players[0]
    assert (p1.char_id, p1.move_id, p1.move_frame, p1.health) == (12, 2145, 7, 142)
    assert p1.pos == pytest.approx((1.5, 0.0, -0.31))
    assert p1.facing == 1
    assert p1.action_state is ActionState.attack  # thin normalization from simple_state
    assert p1.counter_state is CounterState.none
    assert p1.block_stun is False and p1.hit_stun is False
    assert p1.heat.active is True and p1.heat.timer_ms == 3100 and p1.heat.engager_used is True
    assert p1.rage is True
    assert p1.input is not None
    assert p1.input.dir == 6 and p1.input.buttons == ["2"]

    # P2 — block_stun raw flag overrides simple_state into blockstun; counter_hit; input == null.
    p2 = fr.players[1]
    assert p2.char_id == 7
    assert p2.facing == -1
    assert p2.block_stun is True
    assert p2.action_state is ActionState.blockstun  # raw flag wins over the simple state
    assert p2.counter_state is CounterState.counter_hit
    assert p2.input is None  # docs/03 §1: input may be null; decoder tolerates it

    # The whole record validates against the C0 schema (it is a FrameRecord instance already).
    assert fr.model_dump()["players"][1]["input"] is None


def test_encoder_decoder_round_trip(table: OffsetTable) -> None:
    # The test encoder and the shipped decoder agree on a fully-populated record.
    fr = make_frame_record()
    image = encode_frame(fr, table, module_base=MODULE_BASE, game_mode="practice")
    source = FakeMemorySource(
        [image],
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )
    decoded = decode_frame(source, table)
    assert decoded.frame == fr.frame
    assert decoded.match_state is fr.match_state
    for got, want in zip(decoded.players, fr.players, strict=True):
        assert got.char_id == want.char_id
        assert got.move_id == want.move_id
        assert got.health == want.health
        assert got.pos == pytest.approx(want.pos)
        assert got.counter_state is want.counter_state
        assert got.heat.timer_ms == want.heat.timer_ms
        assert (got.input is None) == (want.input is None)


def test_health_computed_from_damage_taken(table: OffsetTable) -> None:
    # Tekken 8's struct has no direct HP field: with max_health set, the decoder reports
    # health = max_health - damage_taken, clamped to [0, max_health] (docs/02 §3, the fork's model).
    t = table.model_copy(deep=True)
    t.players.max_health = 200
    t.players.fields["damage_taken"] = FieldSpec(offset=128, kind="i32")  # within the 1024 stride

    g = t.global_struct
    gbase = MODULE_BASE + g.anchor.base_offset
    pf = t.players.fields
    pbase = MODULE_BASE + t.players.anchor.base_offset
    image: dict[int, bytes] = {}
    for spec in g.fields.values():
        image[gbase + spec.offset] = _pack(spec.kind, 0)
    for idx in (0, 1):
        for spec in pf.values():
            image[pbase + idx * t.players.stride + spec.offset] = _pack(spec.kind, 0)
    # P1 took 55 damage -> health 145; P2 took 250 (more than max) -> clamped to 0.
    image[pbase + 0 * t.players.stride + 128] = _pack("i32", 55)
    image[pbase + 1 * t.players.stride + 128] = _pack("i32", 250)

    source = FakeMemorySource(
        [image], module_bases={g.anchor.module: MODULE_BASE}, advance_on=gbase
    )
    fr = decode_frame(source, t)
    assert fr.players[0].health == 145
    assert fr.players[1].health == 0  # clamped, never negative


def test_input_null_when_group_absent(table: OffsetTable) -> None:
    # If the offset table has no input group at all, input decodes to None (not an error).
    stripped = table.model_copy(deep=True)
    for key in ("input_valid", "input_dir", "input_buttons"):
        stripped.players.fields.pop(key)
    fr = make_frame_record()
    image = encode_frame(fr, table, module_base=MODULE_BASE)  # encode with full table
    source = FakeMemorySource(
        [image],
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )
    decoded = decode_frame(source, stripped)  # decode with the input group removed
    assert all(p.input is None for p in decoded.players)


def test_missing_required_field_raises_decode_error(table: OffsetTable) -> None:
    broken = table.model_copy(deep=True)
    broken.players.fields.pop("char_id")
    fr = make_frame_record()
    image = encode_frame(fr, table, module_base=MODULE_BASE)
    source = FakeMemorySource(
        [image],
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )
    with pytest.raises(DecodeError):
        decode_frame(source, broken)


def _source_for(frames: list[FrameRecord], table: OffsetTable) -> FakeMemorySource:
    images = [encode_frame(fr, table, module_base=MODULE_BASE) for fr in frames]
    return FakeMemorySource(
        images,
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )


def test_frame_counter_gap_produces_gap_tolerated_marker(table: OffsetTable) -> None:
    # Frames at 100, 101, 104, 105 -> a 2-frame gap between 101 and 104. The marker must be
    # exactly "gap-tolerated:2" (== segmenter's missed = frame - prev - 1, docs/04 §4.7).
    counters = [100, 101, 104, 105]
    frames = []
    for c in counters:
        fr = make_frame_record()
        fr = fr.model_copy(update={"frame": c})
        frames.append(fr)
    source = _source_for(frames, table)

    reader = FrameReader()
    reads = [reader.read_frame(source, table) for _ in counters]

    assert [r.frame.frame for r in reads] == counters
    assert reads[0].gap == 0 and reads[0].gap_note is None  # first frame: no prior
    assert reads[1].gap == 0 and reads[1].gap_note is None  # 101 follows 100
    assert reads[2].gap == 2 and reads[2].gap_note == "gap-tolerated:2"  # 104 follows 101
    assert reads[3].gap == 0 and reads[3].gap_note is None  # 105 follows 104


def test_poll_frames_no_gap_on_contiguous_stream(table: OffsetTable) -> None:
    frames = [make_frame_record().model_copy(update={"frame": 500 + i}) for i in range(5)]
    source = _source_for(frames, table)
    reads = poll_frames(source, table, 5)
    assert [r.frame.frame for r in reads] == [500, 501, 502, 503, 504]
    assert all(r.gap == 0 and r.gap_note is None for r in reads)
