"""C4i holder derivation: the RIP-relative code-sig find, the holder oracle, and the schema (§3).

The planted world (``tests/fixtures/reader/planted_holder.py``) hides the player holder behind an
AoB *code* signature in ``.text`` whose RIP-relative ``disp32`` points at a ``.data`` slot, and the
two players behind ``holder+0x30`` / ``holder+0x38`` as **separate** allocations. Nothing tells the
scan the holder slot RVA, so a passing derivation proves the RIP decode + code-sig find works, and
the schema tests prove the per-player-anchor model (no stride) is what the table carries.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from pydantic import ValidationError

from tekken_coach.reader.discovery.holderscan import (
    confirm_holder,
    decode_rip_relative,
    derive_holder_layout,
    find_holder_slot,
    resolve_holder,
)
from tekken_coach.reader.discovery.manifest import ProbeManifest, load_probe_manifest
from tekken_coach.reader.discovery.pe import parse_module_image
from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.offsets import (
    Anchor,
    ComponentAnchor,
    FieldSpec,
    OffsetTable,
    PlayerStruct,
    load_offset_table,
)
from tests.fixtures.reader.planted_chain import KAZUYA, MODULE, MODULE_BASE, P1_MOVE_ID
from tests.fixtures.reader.planted_holder import (
    AOB_PATTERN,
    DISP32_POS,
    HOLDER_SLOT_RVA,
    JIN,
    P1_BASE,
    P2_BASE,
    no_holder_source,
    planted_holder,
    planted_holder_idle,
)
from tests.fixtures.reader.state_map import calibrated_state_map

MANIFEST_PATH = Path("assets/offsets/probe-manifest.json")


def _manifest() -> ProbeManifest:
    return load_probe_manifest(MANIFEST_PATH)


def _seed() -> OffsetTable:
    return load_offset_table(Path("assets/offsets/2.01.01.json"))


# --- Phase 2: the RIP-relative displacement decode -----------------------------------------------


def test_rip_decode_computes_the_next_instruction_relative_slot() -> None:
    # An instruction at RVA 0x1500 whose disp32 (at byte 3) is D references the slot at
    # 0x1500 + 3 + 4 + D — RIP being the address of the *next* instruction on x64.
    disp = 0x2000
    site = 0x1500
    instr = b"\x4c\x89\x35" + struct.pack("<i", disp) + b"\x41\x88\x5e\x28"
    region = Region(base=0x1000, data=bytes(0x500) + instr + bytes(0x100))
    assert decode_rip_relative(region, site, DISP32_POS) == site + DISP32_POS + 4 + disp


def test_rip_decode_handles_a_negative_displacement() -> None:
    disp = -0x40
    site = 0x2000
    instr = b"\x4c\x89\x35" + struct.pack("<i", disp) + b"\x41\x88\x5e\x28"
    region = Region(base=0x1000, data=bytes(0x1000) + instr + bytes(0x100))
    assert decode_rip_relative(region, site, DISP32_POS) == site + DISP32_POS + 4 + disp


def test_rip_decode_returns_none_when_the_match_is_truncated() -> None:
    region = Region(base=0x1000, data=bytes(5))  # too short to hold disp32_pos + 4 bytes
    assert decode_rip_relative(region, 0x1002, DISP32_POS) is None


def test_find_holder_slot_locates_the_unique_rip_referenced_slot() -> None:
    source = planted_holder().before
    image = parse_module_image(lambda rva, n: source.read(MODULE_BASE + rva, n))
    slot = find_holder_slot(source, MODULE_BASE, image, pattern=AOB_PATTERN, disp32_pos=DISP32_POS)
    assert slot == HOLDER_SLOT_RVA


def test_find_holder_slot_returns_none_when_the_signature_is_absent() -> None:
    source = no_holder_source()
    image = parse_module_image(lambda rva, n: source.read(MODULE_BASE + rva, n))
    assert (
        find_holder_slot(source, MODULE_BASE, image, pattern=AOB_PATTERN, disp32_pos=DISP32_POS)
        is None
    )


# --- Phase 2: the holder oracle (resolve + structural, then behavioral) --------------------------


def test_resolve_holder_reads_both_players_and_the_char_id_pair() -> None:
    source = planted_holder().before
    m = _manifest()
    assert m.holder_scan is not None
    match = resolve_holder(source, MODULE_BASE, HOLDER_SLOT_RVA, m.holder_scan, m)
    assert match is not None
    assert match.player_bases == (P1_BASE, P2_BASE)
    assert set(match.char_ids) == {JIN, KAZUYA}
    assert match.move_ids[0] == P1_MOVE_ID


def test_confirm_holder_accepts_when_the_acting_move_id_changed() -> None:
    world = planted_holder()
    m = _manifest()
    assert m.holder_scan is not None
    match = resolve_holder(world.before, MODULE_BASE, HOLDER_SLOT_RVA, m.holder_scan, m)
    assert match is not None
    behavior = confirm_holder(
        world.during, match, module_base=MODULE_BASE, spec=m.holder_scan, manifest=m
    )
    assert behavior is not None and behavior.accepted
    assert behavior.opponent_damaged  # the jab connected mid-window


def test_confirm_holder_rejects_a_frozen_move_id() -> None:
    world = planted_holder_idle()
    m = _manifest()
    assert m.holder_scan is not None
    match = resolve_holder(world.before, MODULE_BASE, HOLDER_SLOT_RVA, m.holder_scan, m)
    assert match is not None
    behavior = confirm_holder(
        world.during, match, module_base=MODULE_BASE, spec=m.holder_scan, manifest=m
    )
    assert behavior is not None and not behavior.accepted


def test_derive_fails_closed_when_nothing_acted() -> None:
    world = planted_holder_idle()
    result = derive_holder_layout(
        world.before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=world.after,
        during=world.during,
        state_map=calibrated_state_map(),
    )
    assert result.player_anchor is None
    assert "char_id" in result.unresolved
    assert any("did NOT change" in n for n in result.notes)


def test_derive_still_finds_the_global_anchor_when_the_holder_is_absent() -> None:
    # The global oracle is independent of the holder — a missing AoB must not lose the match struct.
    from tests.fixtures.reader.planted_chain import GLOBAL_BASE_OFFSET

    before, after = no_holder_source(step=0), no_holder_source(step=1)
    result = derive_holder_layout(
        before,
        module=MODULE,
        module_base=MODULE_BASE,
        manifest=_manifest(),
        seed=_seed(),
        source_after=after,
        during=[after],
        state_map=calibrated_state_map(),
    )
    assert result.player_anchor is None
    assert result.global_anchor is not None
    assert result.global_anchor.base_offset == GLOBAL_BASE_OFFSET  # the derived one, not the seed
    assert any("AoB signature did not match" in n for n in result.notes)


# --- Phase 1: the schema enforces exactly one addressing model -----------------------------------


def _fields() -> dict[str, FieldSpec]:
    return {"char_id": FieldSpec(offset=0x168, kind="u32")}


def test_player_struct_accepts_the_holder_model() -> None:
    ps = PlayerStruct(
        anchor=Anchor(module=MODULE, base_offset=0x3400, pointer_path=[0]),
        player_slots=[ComponentAnchor(slot_offset=0x30), ComponentAnchor(slot_offset=0x38)],
        fields=_fields(),
    )
    assert ps.stride is None
    assert [s.slot_offset for s in ps.player_slots] == [0x30, 0x38]


def test_player_struct_accepts_the_legacy_stride_model() -> None:
    ps = PlayerStruct(
        anchor=Anchor(module=MODULE, base_offset=0x1000), stride=0x800, fields=_fields()
    )
    assert ps.stride == 0x800 and ps.player_slots == []


def test_player_struct_rejects_neither_addressing_model() -> None:
    with pytest.raises(ValidationError, match="neither stride nor player_slots"):
        PlayerStruct(anchor=Anchor(module=MODULE, base_offset=0x1000), fields=_fields())


def test_player_struct_rejects_both_addressing_models() -> None:
    with pytest.raises(ValidationError, match="both stride and player_slots"):
        PlayerStruct(
            anchor=Anchor(module=MODULE, base_offset=0x1000),
            stride=0x800,
            player_slots=[ComponentAnchor(slot_offset=0x30)],
            fields=_fields(),
        )
