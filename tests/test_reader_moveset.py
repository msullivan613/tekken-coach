"""Tests for the read-only tk_moveset reader — Phase 1 discovery/validation + Phase 2 build (#18).

Drives the production reader over a planted :class:`FlatMemorySource` shaped exactly like a live T8
moveset (see ``tests.fixtures.reader.planted_moveset``): Phase 1 validates the moveset pointer by
header shape + the decoder gate, and Phase 2 reads -> attributes owners -> joins into the ground
truth. The live run only fills in the discovered pointer; this proves everything else offline.
"""

from __future__ import annotations

from tekken_coach.reader.moveset import (
    BRYAN_GATE_PAIRS,
    build_notation_map,
    gate_pairs,
    read_attributed_cancels,
    read_cancels,
    read_moveset_header,
    self_check,
    validate_slot,
)
from tests.fixtures.reader.planted_moveset import (
    DECOY_BASE,
    EXPECTED_COLLISIONS,
    EXPECTED_NOTATION,
    EXPECTED_UNRESOLVED,
    MOVE_LAYOUT,
    NEUTRAL_MOVE_ID,
    PLANTED_CANCELS,
    planted_moveset_source,
)

# The live move_id a player would be sitting in — a real Bryan id, inside moves_count.
_LIVE_MOVE_ID = 1695


def test_read_moveset_header_reads_the_pointer_pairs() -> None:
    """The header read recovers the cancels/moves pointer + count pairs at their T8 offsets."""
    source, moveset_ptr = planted_moveset_source()
    header = read_moveset_header(source, moveset_ptr)
    assert header.cancels_count == len(PLANTED_CANCELS)
    assert header.moves_count > _LIVE_MOVE_ID
    assert header.cancels_ptr != 0 and header.moves_ptr != 0


def test_read_cancels_yields_every_row() -> None:
    """The global cancels array reads back one (command, dest) per planted row."""
    source, moveset_ptr = planted_moveset_source()
    header = read_moveset_header(source, moveset_ptr)
    cancels = read_cancels(source, header)
    assert len(cancels) == len(PLANTED_CANCELS)
    assert {c.dest_move_id for c in cancels} == {c.dest for c in PLANTED_CANCELS}


def test_phase1_gate_reproduces_bryans_anchors() -> None:
    """The decisive Phase-1 gate: Bryan's known from-neutral pairs all decode from the cancels."""
    source, moveset_ptr = planted_moveset_source()
    header = read_moveset_header(source, moveset_ptr)
    rows = gate_pairs(read_cancels(source, header), BRYAN_GATE_PAIRS)
    assert rows and all(row.found for row in rows), rows


def test_validate_slot_accepts_the_moveset() -> None:
    """A slot pointing at the real moveset passes every check (shape, readable, range, gate)."""
    source, moveset_ptr = planted_moveset_source()
    verdict = validate_slot(source, moveset_ptr, live_move_id=_LIVE_MOVE_ID)
    assert verdict.counts_plausible
    assert verdict.pointers_readable
    assert verdict.move_id_in_range
    assert verdict.gate_passed
    assert verdict.is_moveset


def test_validate_slot_passes_while_idle_out_of_range_move_id() -> None:
    """Brief #19 Part A: the idle move_id (0x8001) is out of range, yet the header still validates.

    32769 is the neutral/idle alias, not an index into the moves array, so ``move_id_in_range`` is
    False — but the decoder gate reads the static cancels and does not need a live move_id, so
    ``is_moveset`` no longer ANDs the range check and the true header passes while the player stands
    still (exactly the state moveset-probe asks for).
    """
    source, moveset_ptr = planted_moveset_source()
    verdict = validate_slot(source, moveset_ptr, live_move_id=32769)
    assert not verdict.move_id_in_range  # idle alias — informational only now
    assert verdict.gate_passed
    assert verdict.is_moveset


def test_validate_slot_rejects_a_decoy() -> None:
    """A readable but non-moveset object fails validation — no crash, just a negative verdict."""
    source, _ = planted_moveset_source()
    verdict = validate_slot(source, DECOY_BASE, live_move_id=_LIVE_MOVE_ID)
    assert not verdict.is_moveset


def test_validate_slot_rejects_an_unreadable_target() -> None:
    """A garbage slot value (nothing mapped there) is rejected without raising."""
    source, _ = planted_moveset_source()
    verdict = validate_slot(source, 0xDEADBEEF000, live_move_id=_LIVE_MOVE_ID)
    assert verdict.header is None
    assert not verdict.is_moveset


def test_read_attributed_cancels_assigns_owner_moves() -> None:
    """Owner attribution assigns each cancel to the move whose list holds it (from moves array)."""
    source, moveset_ptr = planted_moveset_source()
    header = read_moveset_header(source, moveset_ptr)
    cancels = read_attributed_cancels(source, header, MOVE_LAYOUT)
    by_owner = {(c.source_move_id, c.dest_move_id) for c in cancels}
    assert (NEUTRAL_MOVE_ID, 1695) in by_owner  # jab is from-neutral
    assert (1705, 1695) in by_owner  # jab is ALSO a b+1 follow-up
    assert (1574, 1582) in by_owner  # 4 -> 4,3


def test_phase2_build_reproduces_ground_truth() -> None:
    """The full read->attribute->join reconstructs every clean move, and degrades the rest."""
    source, moveset_ptr = planted_moveset_source()
    result = build_notation_map(source, moveset_ptr, MOVE_LAYOUT, neutral_move_id=NEUTRAL_MOVE_ID)
    for move_id, notation in EXPECTED_NOTATION.items():
        assert result.notation.get(move_id) == notation, move_id
    for move_id, competing in EXPECTED_COLLISIONS.items():
        assert result.collisions.get(move_id) == competing
        assert move_id not in result.notation
    for move_id in EXPECTED_UNRESOLVED:
        assert move_id in result.unresolved
        assert move_id not in result.notation


def test_self_check_classifies_hit_miss_missing() -> None:
    """The Phase-2 self-check labels a reproduced id HIT, a wrong one MISS, an unbuilt MISSING."""
    rebuilt = {1695: "1", 1628: "df+2"}
    ground_truth = {1695: "1", 1628: "b+2", 1765: "qcb+3"}
    rows = {r.move_id: r.status for r in self_check(rebuilt, ground_truth)}
    assert rows[1695] == "HIT"
    assert rows[1628] == "MISS"  # rebuilt a different notation — the failure that matters
    assert rows[1765] == "MISSING"  # out of v1 scope (a motion), not a wrong mapping


def test_phase2_self_check_against_planted_bryan() -> None:
    """End to end: the rebuilt map HITs Bryan's clean committed ids (no MISS on the gate subset)."""
    source, moveset_ptr = planted_moveset_source()
    result = build_notation_map(source, moveset_ptr, MOVE_LAYOUT, neutral_move_id=NEUTRAL_MOVE_ID)
    ground_truth = {p.move_id: p.notation for p in BRYAN_GATE_PAIRS}
    rows = self_check(result.notation, ground_truth)
    assert all(r.status == "HIT" for r in rows), rows
