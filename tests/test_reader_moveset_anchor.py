"""Tests for the brief #21 moveset-anchor grounding: solve the moves array + the grounding dumps.

The doc-derived ``tk_moveset`` offsets are wrong for the live 5.02.01 build (the 2026-07-19 dumps
proved it), so #21 grounds the layout on the trusted live ``move_id``. Two offline-testable pieces:

* :func:`~tekken_coach.reader.discovery.moveset_anchor.solve_moves_array` — the pure solver, driven
  here over planted ``(move_id, {slot: pointer})`` samples where exactly one slot's value tracks
  ``move_id`` linearly and the rest are constant / noise / a two-point coincidence.
* the grounding dumps — :func:`dump_move`, :func:`locate_moves_base_holder`,
  :func:`find_cancels_ptr_offset` — driven over the planted moveset world (``MOVES_BASE`` moves
  array, header at ``MOVESET_BASE`` with ``moves_ptr @ 0x230`` and ``cancels_ptr @ 0x1d0``).
"""

from __future__ import annotations

from tekken_coach.reader.discovery.heapscan import _region_buffers
from tekken_coach.reader.discovery.moveset_anchor import (
    MoveSample,
    MovesArray,
    dump_move,
    find_cancels_ptr_offset,
    locate_moves_base_holder,
    solve_moves_array,
)
from tekken_coach.reader.moveset import (
    BRYAN_GATE_PAIRS,
    MOVESET_CANCELS_PTR_OFFSET,
    KnownPair,
)
from tekken_coach.reader.slots import RegionIndex
from tests.fixtures.reader.planted_moveset import (
    CANCELS_BASE,
    GATE_DECOY_BASE,
    MOVE_SIZE,
    MOVES_BASE,
    MOVESET_BASE,
    planted_moveset_scan_source,
)

# The synthetic ground truth the solver must recover: one slot holds moves_base + id*stride.
_SLOT = 0x40
_MOVES_BASE = 0x302000000
_STRIDE = 0x1B0  # a realistic tk_move stride (bytes)


def _slots(move_id: int, *, tracked: int, constant: int, noise: int) -> dict[int, int]:
    """A player's pointer slots at one instant: the tracked slot plus a constant and noise slot."""
    return {
        _SLOT: _MOVES_BASE + move_id * _STRIDE,  # the current-move pointer
        0x10: constant,  # a slot that never moves (a shared descriptor)
        0x20: tracked,  # a slot that moves but NOT as a linear function of move_id
        0x80: noise,  # present only on some samples (dropped by the shared-offset intersection)
    }


def test_solver_recovers_exact_slot_base_and_stride() -> None:
    """The solver picks the linear slot and recovers the exact offset, base and stride."""
    samples = [
        MoveSample(1695, _slots(1695, tracked=0x900, constant=0x555, noise=0x1)),
        MoveSample(1725, _slots(1725, tracked=0x901, constant=0x555, noise=0x2)),
        MoveSample(1779, _slots(1779, tracked=0x902, constant=0x555, noise=0x3)),
    ]
    result = solve_moves_array(samples)
    assert result == MovesArray(slot_offset=_SLOT, moves_base=_MOVES_BASE, move_stride=_STRIDE)
    assert result is not None
    assert result.move_addr(1695) == _MOVES_BASE + 1695 * _STRIDE


def test_solver_rejects_a_two_point_coincidence() -> None:
    """A slot fitting a line through only 2 of 3 ids is rejected — the 3rd id kills the false line.

    The decoy slot ``0x20`` is collinear with the true current-move slot for the first two ids but
    breaks on the third; only the genuine slot survives the all-ids consistency check.
    """
    ids = (1695, 1725, 1779)
    decoy_base, decoy_stride = 0x700000, 0x100
    samples = []
    for i, move_id in enumerate(ids):
        slots = _slots(move_id, tracked=0, constant=0x555, noise=0)
        # Slot 0x20 lies on decoy_base + id*decoy_stride for the first two ids, then jumps off it.
        on_line = decoy_base + move_id * decoy_stride
        slots[0x20] = on_line if i < 2 else on_line + 0x999
        samples.append(MoveSample(move_id, slots))
    result = solve_moves_array(samples)
    assert result is not None
    assert result.slot_offset == _SLOT  # the real slot, not the 2-point decoy at 0x20


def test_solver_returns_none_when_no_slot_correlates() -> None:
    """With every slot constant or randomly churning (no linear slot), the solver declines."""
    samples = [
        MoveSample(1695, {0x10: 0x555, 0x20: 0x111}),
        MoveSample(1725, {0x10: 0x555, 0x20: 0x777}),
        MoveSample(1779, {0x10: 0x555, 0x20: 0x333}),
    ]
    assert solve_moves_array(samples) is None


def test_solver_needs_three_distinct_real_ids() -> None:
    """Two distinct ids are not enough — any pair defines a line, so the solver refuses to guess."""
    samples = [
        MoveSample(1695, _slots(1695, tracked=0, constant=0x555, noise=0)),
        MoveSample(1725, _slots(1725, tracked=0, constant=0x555, noise=0)),
    ]
    assert solve_moves_array(samples) is None


def test_solver_ignores_the_neutral_alias_sample() -> None:
    """The idle/neutral alias (0x8001) is not a moves-array index and is dropped before solving.

    Three real ids plus a neutral-alias sample whose tracked value is off the line: if the alias
    were fitted the line would break, so a clean solve proves it was filtered.
    """
    samples = [
        MoveSample(1695, _slots(1695, tracked=0, constant=0x555, noise=0)),
        MoveSample(1725, _slots(1725, tracked=0, constant=0x555, noise=0)),
        MoveSample(1779, _slots(1779, tracked=0, constant=0x555, noise=0)),
        # neutral alias: its tracked slot value is NOT moves_base + 0x8001*stride
        MoveSample(0x8001, {_SLOT: 0xDEADBEEF, 0x10: 0x555, 0x20: 0, 0x80: 0}),
    ]
    result = solve_moves_array(samples)
    assert result == MovesArray(slot_offset=_SLOT, moves_base=_MOVES_BASE, move_stride=_STRIDE)


def test_solver_deduplicates_repeated_move_ids() -> None:
    """Repeated samples of the same id collapse to one — a grind that re-catches a move is fine."""
    samples = [
        MoveSample(1695, _slots(1695, tracked=0, constant=0x555, noise=0)),
        MoveSample(1695, _slots(1695, tracked=0, constant=0x555, noise=0)),
        MoveSample(1725, _slots(1725, tracked=0, constant=0x555, noise=0)),
    ]
    # Only two DISTINCT real ids after dedup -> not enough to solve.
    assert solve_moves_array(samples) is None


# ---------------------------------------------------------------------------
# Phase 2 — grounding dumps over the planted moveset world
# ---------------------------------------------------------------------------


def test_dump_move_formats_the_known_tk_move_words() -> None:
    """``dump_move`` reads the right ``tk_move`` and shows its (planted) cancel pointer/count words.

    Move 1705 (``b+1``) owns a cancel list in the planted world, so its ``tk_move`` (0x10 stride)
    holds ``cancel_start_ptr`` at +0x000 and its count at +0x008 — both must appear in the dump.
    """
    source = planted_moveset_scan_source()
    moves = MovesArray(slot_offset=0x0, moves_base=MOVES_BASE, move_stride=MOVE_SIZE)
    text = dump_move(source, moves, 1705)
    assert f"move_id 1705 @ 0x{MOVES_BASE + 1705 * MOVE_SIZE:x}" in text
    # move 1705 owns one cancel (b -> b,1); its cancel_start points into the cancels array.
    assert "+0x000: 0x" in text and "+0x008: 0x0000000000000001" in text


def test_dump_move_is_total_over_unreadable_words() -> None:
    """A move whose window runs past mapped memory yields ``<unreadable>``, never an exception."""
    source = planted_moveset_scan_source()
    moves = MovesArray(slot_offset=0x0, moves_base=0x999000000, move_stride=MOVE_SIZE)
    text = dump_move(source, moves, 0)
    assert "<unreadable>" in text


def test_locate_moves_base_holder_finds_the_header() -> None:
    """The reverse-scan finds the header word holding ``moves_base`` and dumps the window around it.

    In the planted world the header at ``MOVESET_BASE`` stores ``MOVES_BASE`` at ``moves_ptr``
    (+0x230), so the located word is ``MOVESET_BASE + 0x230`` and the window reaches back to the
    ``cancels_ptr`` (+0x1d0, i.e. -0x60 from the located word), which holds ``CANCELS_BASE``.
    """
    source = planted_moveset_scan_source()
    buffers = _region_buffers(source, source.regions(), progress=None)
    text = locate_moves_base_holder(source, buffers, MOVES_BASE)
    located = MOVESET_BASE + 0x230
    assert f"around 0x{located:x}" in text
    assert "<- moves_ptr (moves_base)" in text
    # the neighbouring cancels_ptr word (at -0x60) holds CANCELS_BASE
    assert f"0x{CANCELS_BASE:016x}" in text


def test_locate_moves_base_holder_reports_when_absent() -> None:
    """A moves_base no object stores yields a clear 'no object' line, not an empty/false dump."""
    source = planted_moveset_scan_source()
    buffers = _region_buffers(source, source.regions(), progress=None)
    text = locate_moves_base_holder(source, buffers, 0xABCDEF000)
    assert "no heap object stores moves_base" in text


def test_find_cancels_ptr_offset_recovers_the_real_offset() -> None:
    """Brute-force identifies the real ``cancels_ptr`` offset by reproducing Bryan's anchors."""
    source = planted_moveset_scan_source()
    region_index = RegionIndex(source.regions())
    off = find_cancels_ptr_offset(source, MOVESET_BASE, BRYAN_GATE_PAIRS, region_index=region_index)
    assert off == MOVESET_CANCELS_PTR_OFFSET


def test_find_cancels_ptr_offset_declines_a_gate_failing_header() -> None:
    """A header whose cancels reproduce no anchor (the gate decoy) yields ``None``, not a guess."""
    source = planted_moveset_scan_source()
    region_index = RegionIndex(source.regions())
    off = find_cancels_ptr_offset(
        source, GATE_DECOY_BASE, BRYAN_GATE_PAIRS, region_index=region_index
    )
    assert off is None


def test_find_cancels_ptr_offset_flags_a_broken_decode() -> None:
    """If the decode/layout no longer holds, the gate reproduces nothing and brute-force is None.

    Modelled by gating a real header on anchors it does not contain (unknown move ids): the array is
    read fine, but no offset reproduces the anchors, so the tool honestly returns ``None`` rather
    than mislabelling a word as ``cancels_ptr``.
    """
    source = planted_moveset_scan_source()
    region_index = RegionIndex(source.regions())
    bogus = (KnownPair(4242, "1"), KnownPair(4243, "2"), KnownPair(4244, "3"))
    off = find_cancels_ptr_offset(source, MOVESET_BASE, bogus, region_index=region_index)
    assert off is None
