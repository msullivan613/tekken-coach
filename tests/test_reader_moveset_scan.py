"""Tests for the brief #19 heap shape+gate moveset scan and the durable reference-path derivation.

Drives the pure discovery functions over a planted world (see
``tests.fixtures.reader.planted_moveset.planted_moveset_scan_source``) where the Bryan header is
reachable only through a pointer path (player -> object -> header), NOT a direct player slot — the
exact shape the 2026-07-19 live run proved. A gate-failing decoy rides along to prove the decoder
gate, not the cheap count/pointer filter, is what accepts a header.
"""

from __future__ import annotations

from collections.abc import Sequence

from tekken_coach.reader.discovery.heapscan import _region_buffers
from tekken_coach.reader.discovery.moveset_scan import (
    derive_reference_path,
    gate_survivors,
    scan_moveset,
    shape_survivors,
)
from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.moveset import BRYAN_GATE_PAIRS, gate_pairs_for, read_moveset_header
from tekken_coach.reader.slots import RegionIndex
from tests.fixtures.reader.planted_moveset import (
    EXPECTED_MOVESET_ANCHOR,
    GATE_DECOY_BASE,
    MICRO_DECOY_BASE,
    MOVESET_BASE,
    NEAR_EQUAL_DECOY_BASE,
    PLAYER_BASE,
    PLAYER_STRUCT_SPAN,
    REAL_CANCELS_COUNT,
    planted_moveset_scan_source,
)

# The old single shared count ceiling (brief #19) that filtered the true Bryan header out live.
_OLD_COUNT_MAX = 20000


def _buffers_and_index(source: MemorySource) -> tuple[Sequence[Region], RegionIndex]:
    buffers = _region_buffers(source, source.regions(), progress=None)
    return buffers, RegionIndex(source.regions())


def test_shape_survivors_include_both_headers() -> None:
    """The cheap filter keeps every header-shaped candidate — the real one AND the gate decoy."""
    source = planted_moveset_scan_source()
    buffers, index = _buffers_and_index(source)
    survivors = shape_survivors(buffers, index)
    assert MOVESET_BASE in survivors
    assert GATE_DECOY_BASE in survivors


def test_shape_filter_rejects_the_near_equal_and_micro_decoys() -> None:
    """Brief #20: the new ``cancels > moves`` check + raised floor cut the junk BEFORE the gate.

    The near-equal decoy (cancels < moves, like the 8,703 live false positives) and the micro decoy
    (counts below the floor) are both rejected at the cheap shape filter — never reaching the gate.
    """
    source = planted_moveset_scan_source()
    buffers, index = _buffers_and_index(source)
    survivors = shape_survivors(buffers, index)
    assert NEAR_EQUAL_DECOY_BASE not in survivors
    assert MICRO_DECOY_BASE not in survivors


def test_shape_survivors_is_a_small_set() -> None:
    """The tuned filter collapses the planted world to the two shape-valid headers (brief #20)."""
    source = planted_moveset_scan_source()
    buffers, index = _buffers_and_index(source)
    survivors = shape_survivors(buffers, index)
    assert set(survivors) == {MOVESET_BASE, GATE_DECOY_BASE}


def test_raised_ceiling_admits_the_real_header() -> None:
    """The real header's cancels count is ABOVE the old 20000 cap yet is now admitted (brief #20).

    This is the exact failure the live run exposed — the old shared ceiling rejected the true header
    before the gate. The raised, split ceiling keeps it, and it has the ``cancels > moves`` shape.
    """
    source = planted_moveset_scan_source()
    header = read_moveset_header(source, MOVESET_BASE)
    assert header.cancels_count == REAL_CANCELS_COUNT
    assert header.cancels_count > _OLD_COUNT_MAX
    assert header.cancels_count > header.moves_count
    # and it still survives the tuned shape filter end to end
    buffers, index = _buffers_and_index(source)
    assert MOVESET_BASE in shape_survivors(buffers, index)


def test_gate_is_the_decisive_discriminator() -> None:
    """Both headers survive the shape filter; only the real one reproduces Bryan's anchors."""
    source = planted_moveset_scan_source()
    buffers, index = _buffers_and_index(source)
    candidates = gate_survivors(source, shape_survivors(buffers, index), BRYAN_GATE_PAIRS)
    passed = {c.header_addr for c in candidates if c.gate_passed}
    assert passed == {MOVESET_BASE}
    # the decoy was considered (survived the shape filter) but rejected at the gate
    assert GATE_DECOY_BASE in {c.header_addr for c in candidates}


def test_scan_moveset_finds_the_single_real_header() -> None:
    """End to end: the scan locates exactly the real header by shape + gate, no direct slot."""
    source = planted_moveset_scan_source()
    scan = scan_moveset(source, pairs=BRYAN_GATE_PAIRS)
    assert [m.header_addr for m in scan.matches] == [MOVESET_BASE]
    assert scan.winner is not None
    assert scan.winner.header_addr == MOVESET_BASE


def test_derive_reference_path_recovers_the_one_hop_anchor() -> None:
    """Part C: the reverse scan derives the player -> object -> header ComponentAnchor."""
    source = planted_moveset_scan_source()
    scan = scan_moveset(source, pairs=BRYAN_GATE_PAIRS)
    assert scan.winner is not None
    anchor = derive_reference_path(
        source,
        scan.buffers,
        header_addr=scan.winner.header_addr,
        player_base=PLAYER_BASE,
        player_struct_span=PLAYER_STRUCT_SPAN,
    )
    assert anchor == EXPECTED_MOVESET_ANCHOR


def test_derive_reference_path_resolves_to_the_header() -> None:
    """The derived anchor actually resolves from the player base to the confirmed header address."""
    from tekken_coach.reader.decode import resolve_component

    source = planted_moveset_scan_source()
    scan = scan_moveset(source, pairs=BRYAN_GATE_PAIRS)
    assert scan.winner is not None
    anchor = derive_reference_path(
        source,
        scan.buffers,
        header_addr=scan.winner.header_addr,
        player_base=PLAYER_BASE,
        player_struct_span=PLAYER_STRUCT_SPAN,
    )
    assert anchor is not None
    assert resolve_component(source, PLAYER_BASE, anchor) == MOVESET_BASE


def test_derive_reference_path_returns_none_when_no_path_exists() -> None:
    """With a player base that reaches nothing, the derivation declines (caller re-scans)."""
    source = planted_moveset_scan_source()
    scan = scan_moveset(source, pairs=BRYAN_GATE_PAIRS)
    assert scan.winner is not None
    # OBJECT_BASE holds the header ref but is not a player struct pointing at itself -> no path.
    anchor = derive_reference_path(
        source,
        scan.buffers,
        header_addr=scan.winner.header_addr,
        player_base=PLAYER_BASE,
        player_struct_span=PLAYER_STRUCT_SPAN,
        max_hop=0x8,  # too small to bridge object base -> the ref slot (0x18)
    )
    assert anchor is None


def test_gate_pairs_for_known_and_unknown_char() -> None:
    """The char-keyed gate registry returns Bryan's anchors and None for an unmapped character."""
    assert gate_pairs_for("bryan") == BRYAN_GATE_PAIRS
    assert gate_pairs_for("BRYAN") == BRYAN_GATE_PAIRS
    assert gate_pairs_for("kazuya") is None
