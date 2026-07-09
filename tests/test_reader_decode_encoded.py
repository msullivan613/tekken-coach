"""Decoding Tekken 8's real per-player layout: encoded state words + a transform component.

The C4a placeholder table describes a struct that does not exist — one ``bool8`` per flag, position
inline. The live build carries a few **encoded state words** whose integer values denote whole
situations, and keeps position in a separate transform component behind the entity's own pointer.
``decode`` handles both, chosen by what the *table* declares; these tests pin the encoded half.

Bytes are packed here with an **independent** format map (not the decoder's ``_FORMATS``, not the
test encoder), so a real byte-layout bug is caught rather than masked by shared code.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from tekken_coach.reader.decode import decode_frame
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    Anchor,
    ComponentAnchor,
    EncodedStateSpec,
    FieldSpec,
    OffsetTable,
    PlayerStruct,
    load_offset_table,
)
from tekken_coach.schemas import ActionState, CounterState
from tests.fixtures.reader.flat_source import FlatMemorySource
from tests.fixtures.reader.state_map import calibrated_state_map

MODULE = "Polaris-Win64-Shipping.exe"
MODULE_BASE = 0x140000000
_STRIDE = 0x4000
_P1_COMPONENT = 0x200020000
_P2_COMPONENT = 0x200021000
_HEAP_BASE = 0x200000000
_HEAP_SPAN = 0x30000

# The real within-struct offsets (facts/data, docs/02 §5) the base scan seeds into the table.
_MOVE_FRAME = 0x370
_COUNTER_STATE = 0x5F0
_RECOVERY_STATE = 0x5B4
_SIMPLE_MOVE_STATE = 0x640
_STUN_TYPE = 0x644
_THROW_TECH_STATE = 0x668
_COMPLEX_MOVE_STATE = 0x68C
_TRIPLE = 0x20
_SLOT = 0x100


def _encoded_table() -> OffsetTable:
    """The shape ``update-offsets --base-scan`` writes on the live build."""
    seed = load_offset_table(Path("assets/offsets/2.01.01.json"))
    fields = {
        "char_id": FieldSpec(offset=0x168, kind="u32"),
        "move_id": FieldSpec(offset=0x528, kind="u32"),
        "damage_taken": FieldSpec(offset=0x1260, kind="i32"),
        "move_frame": FieldSpec(offset=_MOVE_FRAME, kind="u32"),
        "counter_state": FieldSpec(offset=_COUNTER_STATE, kind="u32"),
        "recovery_state": FieldSpec(offset=_RECOVERY_STATE, kind="u32"),
        "simple_move_state": FieldSpec(offset=_SIMPLE_MOVE_STATE, kind="u32"),
        "stun_type": FieldSpec(offset=_STUN_TYPE, kind="u32"),
        "throw_tech_state": FieldSpec(offset=_THROW_TECH_STATE, kind="u32"),
        "complex_move_state": FieldSpec(offset=_COMPLEX_MOVE_STATE, kind="u32"),
        # Not yet calibrated on the real build; carried from the seed at their placeholder offsets.
        "facing": seed.players.fields["facing"],
        "heat_active": seed.players.fields["heat_active"],
        "heat_engager_used": seed.players.fields["heat_engager_used"],
        "heat_timer_ms": seed.players.fields["heat_timer_ms"],
        "rage": seed.players.fields["rage"],
    }
    players = PlayerStruct(
        anchor=Anchor(module=MODULE, base_offset=0x1000, pointer_path=[]),
        stride=_STRIDE,
        fields=fields,
        max_health=200,
        components={
            POSITION_COMPONENT: ComponentAnchor(
                slot_offset=_SLOT,
                pointer_path=[],
                fields={
                    "pos_x": FieldSpec(offset=_TRIPLE, kind="f32"),
                    "pos_y": FieldSpec(offset=_TRIPLE + 4, kind="f32"),
                    "pos_z": FieldSpec(offset=_TRIPLE + 8, kind="f32"),
                },
            )
        },
    )
    state_codes = seed.state_codes.model_copy(update={"encoded_state": calibrated_state_map()})
    return seed.model_copy(update={"players": players, "state_codes": state_codes})


def _blit(buf: bytearray, off: int, data: bytes) -> None:
    buf[off : off + len(data)] = data


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _f32(v: float) -> bytes:
    return struct.pack("<f", v)


def _source(
    table: OffsetTable,
    *,
    p1_state: dict[str, int],
    p2_state: dict[str, int] | None = None,
    p1_pos: tuple[float, float, float] = (1.5, 0.0, -0.31),
) -> FlatMemorySource:
    """A flat world holding a global struct, two entity structs, and two transform components."""
    module = bytearray(0x8000)
    heap = bytearray(_HEAP_SPAN)

    g = table.global_struct
    for name, value in (
        ("frame_counter", 128472),
        ("match_phase", 2),  # in_round
        ("game_mode", 4),  # practice
        ("round", 2),
        ("timer_ms", 41200),
    ):
        _blit(module, g.anchor.base_offset + g.fields[name].offset, _u32(value))

    # The players sit at the static anchor module+0x1000; this test is about the *player* layout,
    # so the anchor is a plain static offset rather than the live build's pointer chain.
    for index, (state, component, pos) in enumerate(
        (
            (p1_state, _P1_COMPONENT, p1_pos),
            (p2_state or {}, _P2_COMPONENT, (-1.0, 0.0, 0.5)),
        )
    ):
        off = 0x1000 + index * _STRIDE
        _blit(module, off + 0x168, _u32(12))  # char_id
        _blit(module, off + 0x528, _u32(2145))  # move_id
        _blit(module, off + 0x1260, struct.pack("<i", 58))  # damage_taken -> health 142
        _blit(module, off + _MOVE_FRAME, _u32(7))
        for name, raw in state.items():
            _blit(module, off + table.players.fields[name].offset, _u32(raw))
        _blit(module, off + _SLOT, struct.pack("<Q", component))
        for k, v in enumerate(pos):
            _blit(heap, (component - _HEAP_BASE) + _TRIPLE + 4 * k, _f32(v))

    return FlatMemorySource(
        [(MODULE_BASE, bytes(module)), (_HEAP_BASE, bytes(heap))],
        module_bases={MODULE: MODULE_BASE},
    )


@pytest.fixture
def table() -> OffsetTable:
    return _encoded_table()


def test_encoded_state_words_fold_into_the_player_flags(table: OffsetTable) -> None:
    # stun_type=2 is hit_stun; complex_move_state=2 is airborne+juggle. The decoder unions flags
    # across fields, so overlapping axes compose without the map enumerating their product.
    source = _source(table, p1_state={"stun_type": 2, "complex_move_state": 2})
    p1 = decode_frame(source, table).players[0]
    assert p1.hit_stun is True
    assert p1.block_stun is False
    assert p1.airborne is True
    assert p1.juggle is True
    # action_state is the thin fold: hitstun outranks airborne (docs/03 §1 Notes).
    assert p1.action_state is ActionState.hitstun


def test_simple_move_state_drives_action_state_when_nothing_overrides(table: OffsetTable) -> None:
    source = _source(table, p1_state={"simple_move_state": 1})
    assert decode_frame(source, table).players[0].action_state is ActionState.attack
    source = _source(table, p1_state={"simple_move_state": 3})
    assert decode_frame(source, table).players[0].action_state is ActionState.crouch


def test_blockstun_outranks_the_simple_posture(table: OffsetTable) -> None:
    # Mid-move blockstun: the game still reports an attack posture, but the defender is stuck. The
    # segmenter keys on block_stun (docs/04 §4.1), and action_state must not say "attack".
    source = _source(table, p1_state={"simple_move_state": 1, "stun_type": 1})
    p1 = decode_frame(source, table).players[0]
    assert p1.action_state is ActionState.blockstun
    assert p1.block_stun is True


def test_raw_state_carries_the_unmapped_words_for_calibration(table: OffsetTable) -> None:
    # An unmapped value contributes no flags — and would be invisible without raw_state. This is the
    # debuggability contract that makes the docs/02 §8 protocol possible from a captured fixture.
    source = _source(table, p1_state={"stun_type": 99})
    p1 = decode_frame(source, table).players[0]
    assert p1.hit_stun is False and p1.block_stun is False
    assert p1.raw_state is not None
    assert p1.raw_state["stun_type"] == 99
    assert set(p1.raw_state) == set(table.state_codes.encoded_state.flags)  # type: ignore[union-attr]


def test_an_uncalibrated_map_decodes_to_neutral_rather_than_lying(table: OffsetTable) -> None:
    # The shipped skeleton maps nothing. Everything reads neutral/false: structurally valid,
    # semantically empty — which the tooling reports loudly rather than passing off as working.
    empty = table.state_codes.model_copy(
        update={"encoded_state": EncodedStateSpec(flags={"stun_type": {}})}
    )
    blank = table.model_copy(update={"state_codes": empty})
    source = _source(blank, p1_state={"stun_type": 2})
    p1 = decode_frame(source, blank).players[0]
    assert p1.action_state is ActionState.neutral
    assert p1.hit_stun is False
    assert p1.raw_state == {"stun_type": 2}


def test_legacy_tables_carry_no_raw_state() -> None:
    # The C4c/legacy boolean layout has no encoded words to report.
    from tekken_coach.reader.memory_source import FakeMemorySource
    from tests.factories import make_frame_record
    from tests.fixtures.reader.encode import advance_on_for, encode_frame, module_base_for

    seed = load_offset_table(Path("assets/offsets/2.01.01.json"))
    fr = make_frame_record()
    source = FakeMemorySource(
        [encode_frame(fr, seed)],
        module_bases=module_base_for(seed),
        advance_on=advance_on_for(seed),
    )
    assert decode_frame(source, seed).players[0].raw_state is None


def test_position_is_read_through_the_transform_component(table: OffsetTable) -> None:
    source = _source(table, p1_state={}, p1_pos=(3.25, 0.0, -1.5))
    frame = decode_frame(source, table)
    assert frame.players[0].pos == pytest.approx((3.25, 0.0, -1.5))
    assert frame.players[1].pos == pytest.approx((-1.0, 0.0, 0.5))


def test_a_dead_component_pointer_is_a_classified_fault(table: OffsetTable) -> None:
    # Better a classified fault than a silently zeroed position, which the segmenter would read as
    # "the players are on top of each other" (docs/02 §7 — a wrong offset is worse than no run).
    source = _source(table, p1_state={})
    broken = table.model_copy(
        update={
            "players": table.players.model_copy(
                update={
                    "components": {
                        POSITION_COMPONENT: ComponentAnchor(
                            slot_offset=0x900,  # an all-zero slot: dereferences to nothing mapped
                            fields=table.players.components[POSITION_COMPONENT].fields,
                        )
                    }
                }
            )
        }
    )
    with pytest.raises(MemoryReadError):
        decode_frame(source, broken)


def test_counter_state_reads_the_hit_outcome_word(table: OffsetTable) -> None:
    source = _source(table, p1_state={"counter_state": 1})
    assert decode_frame(source, table).players[0].counter_state is CounterState.counter_hit
