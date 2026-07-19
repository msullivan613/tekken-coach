"""Tests for the #21/#22 moveset-anchor grounding: solve the moves array + the grounding dumps.

The doc-derived ``tk_moveset`` offsets are wrong for the live 5.02.01 build (the 2026-07-19 dumps
proved it), so #21 grounds the layout on the trusted live ``move_id`` and #22 widens the search one
hop out. Offline-testable pieces:

* :func:`~tekken_coach.reader.discovery.moveset_anchor.solve_moves_array` — the pure solver, driven
  here over planted ``(move_id, {slot_key: pointer})`` samples where exactly one slot's value tracks
  ``move_id`` linearly and the rest are constant / noise / a two-point coincidence. Slot keys are
  composite (brief #22): ``(offset,)`` direct, ``(parent, sub)`` one hop out.
* :func:`sample_player_slots` — the one-hop sampler, driven over the planted moveset world
  (``PLAYER_BASE -> OBJECT_BASE -> header``).
* :func:`describe_slots` — the failed-solve diagnostic (constant / non-linear / linear-K-of-N).
* the grounding dumps — :func:`dump_move`, :func:`locate_moves_base_holder`,
  :func:`find_cancels_ptr_offset` — driven over the planted moveset world (``MOVES_BASE`` moves
  array, header at ``MOVESET_BASE`` with ``moves_ptr @ 0x230`` and ``cancels_ptr @ 0x1d0``).
"""

from __future__ import annotations

from collections.abc import Sequence

from tekken_coach.reader.discovery.heapscan import _region_buffers
from tekken_coach.reader.discovery.moveset_anchor import (
    MoveSample,
    MovesArray,
    SlotKey,
    describe_slots,
    dump_move,
    find_cancels_ptr_offset,
    format_slot_key,
    locate_moves_base_holder,
    sample_player_slots,
    solve_moves_array,
)
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemoryRegion
from tekken_coach.reader.moveset import (
    BRYAN_GATE_PAIRS,
    MOVESET_CANCELS_PTR_OFFSET,
    KnownPair,
)
from tekken_coach.reader.slots import RegionIndex
from tests.fixtures.reader.flat_source import FlatMemorySource
from tests.fixtures.reader.planted_moveset import (
    CANCELS_BASE,
    DECOY_BASE,
    GATE_DECOY_BASE,
    MOVE_SIZE,
    MOVES_BASE,
    MOVESET_BASE,
    MOVESET_REF_OFFSET,
    OBJECT_BASE,
    PLAYER_BASE,
    PLAYER_DISTRACTOR_SLOT,
    PLAYER_MOVESET_SLOT,
    planted_moveset_scan_source,
)

# The synthetic ground truth the solver must recover: one slot holds moves_base + id*stride.
_SLOT: SlotKey = (0x40,)
_MOVES_BASE = 0x302000000
_STRIDE = 0x1B0  # a realistic tk_move stride (bytes)


def _slots(move_id: int, *, tracked: int, constant: int, noise: int) -> dict[SlotKey, int]:
    """A player's pointer slots at one instant: the tracked slot plus a constant and noise slot."""
    return {
        _SLOT: _MOVES_BASE + move_id * _STRIDE,  # the current-move pointer
        (0x10,): constant,  # a slot that never moves (a shared descriptor)
        (0x20,): tracked,  # a slot that moves but NOT as a linear function of move_id
        (0x80,): noise,  # present only on some samples (dropped by the shared-key intersection)
    }


def test_solver_recovers_exact_slot_base_and_stride() -> None:
    """The solver picks the linear slot and recovers the exact key, base and stride."""
    samples = [
        MoveSample(1695, _slots(1695, tracked=0x900, constant=0x555, noise=0x1)),
        MoveSample(1725, _slots(1725, tracked=0x901, constant=0x555, noise=0x2)),
        MoveSample(1779, _slots(1779, tracked=0x902, constant=0x555, noise=0x3)),
    ]
    result = solve_moves_array(samples)
    assert result == MovesArray(slot_key=_SLOT, moves_base=_MOVES_BASE, move_stride=_STRIDE)
    assert result is not None
    assert result.move_addr(1695) == _MOVES_BASE + 1695 * _STRIDE


def test_solver_recovers_a_one_hop_composite_slot() -> None:
    """The tracking pointer lives one hop out — the solver names the composite ``(parent,sub)`` key.

    No *direct* slot tracks move_id (that is exactly the live #21 finding); the current-move pointer
    sits inside a sub-object, keyed ``(0x30, 0x18)``. Direct slots here are constant / a two-point
    coincidence, so only the one-hop key survives the all-ids fit.
    """
    hop_key: SlotKey = (0x30, 0x18)
    ids = (1695, 1725, 1779)
    samples = []
    for i, move_id in enumerate(ids):
        slots: dict[SlotKey, int] = {
            (0x30,): 0x999000,  # the parent pointer itself (constant)
            (0x40,): 0x555,  # a constant direct slot
            hop_key: _MOVES_BASE + move_id * _STRIDE,  # the tracking pointer, one hop out
        }
        # a direct slot that is collinear for the first two ids only (a 2-point decoy)
        slots[(0x20,)] = 0x700000 + move_id * 0x100 + (0 if i < 2 else 0x999)
        samples.append(MoveSample(move_id, slots))
    result = solve_moves_array(samples)
    assert result == MovesArray(slot_key=hop_key, moves_base=_MOVES_BASE, move_stride=_STRIDE)


def test_solver_tolerates_one_outlier_sample() -> None:
    """A slot linear over 4 of 5 ids (one stray) is still solved — the stray no longer sinks it."""
    ids = (1695, 1725, 1779, 1801, 1850)
    samples = []
    for i, move_id in enumerate(ids):
        slots = _slots(move_id, tracked=0, constant=0x555, noise=0)
        if i == 2:  # one stray sample: the tracking slot is off its line for this id only
            slots[_SLOT] = 0xDEADBEEF
        samples.append(MoveSample(move_id, slots))
    result = solve_moves_array(samples)
    assert result == MovesArray(slot_key=_SLOT, moves_base=_MOVES_BASE, move_stride=_STRIDE)


def test_solver_rejects_a_slot_linear_over_only_two_of_five() -> None:
    """A slot on a line for only 2 of 5 ids is rejected — below the 3-on-one-line floor."""
    ids = (1695, 1725, 1779, 1801, 1850)
    decoy_base, decoy_stride = 0x700000, 0x100
    samples = []
    for i, move_id in enumerate(ids):
        # No genuine slot: the one structured slot is on a line for the first two ids, then churns.
        on_line = decoy_base + move_id * decoy_stride
        slots: dict[SlotKey, int] = {
            (0x10,): 0x555,  # constant
            (0x20,): on_line if i < 2 else on_line + 0x1000 * (i + 1),  # 2 of 5, then scattered
        }
        samples.append(MoveSample(move_id, slots))
    assert solve_moves_array(samples) is None


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
        slots[(0x20,)] = on_line if i < 2 else on_line + 0x999
        samples.append(MoveSample(move_id, slots))
    result = solve_moves_array(samples)
    assert result is not None
    assert result.slot_key == _SLOT  # the real slot, not the 2-point decoy at 0x20


def test_solver_returns_none_when_no_slot_correlates() -> None:
    """With every slot constant or randomly churning (no linear slot), the solver declines."""
    samples = [
        MoveSample(1695, {(0x10,): 0x555, (0x20,): 0x111}),
        MoveSample(1725, {(0x10,): 0x555, (0x20,): 0x777}),
        MoveSample(1779, {(0x10,): 0x555, (0x20,): 0x333}),
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
        MoveSample(0x8001, {_SLOT: 0xDEADBEEF, (0x10,): 0x555, (0x20,): 0, (0x80,): 0}),
    ]
    result = solve_moves_array(samples)
    assert result == MovesArray(slot_key=_SLOT, moves_base=_MOVES_BASE, move_stride=_STRIDE)


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
# Phase 1 — the one-hop sampler + the failed-solve diagnostic (brief #22)
# ---------------------------------------------------------------------------


def test_sample_player_slots_captures_direct_and_one_hop_pointers() -> None:
    """The sampler records direct pointer slots AND the plausible pointers one hop out.

    In the planted world the player holds a pointer at ``0x30`` -> the intermediate object, whose
    ``0x18`` slot holds the header address; and a distractor pointer at ``0x80`` -> a decoy object.
    So the sampler must yield the direct keys ``(0x30,)`` / ``(0x80,)`` and the one-hop composite
    key ``(0x30, 0x18)`` holding the header address.
    """
    source = planted_moveset_scan_source()
    regions = RegionIndex(source.regions())
    slots, _ = sample_player_slots(source, PLAYER_BASE, regions, direct_end=0x200)
    assert slots[(PLAYER_MOVESET_SLOT,)] == OBJECT_BASE
    assert slots[(PLAYER_DISTRACTOR_SLOT,)] == DECOY_BASE
    # the current-move-style pointer, reached one hop out through the 0x30 slot
    assert slots[(PLAYER_MOVESET_SLOT, MOVESET_REF_OFFSET)] == MOVESET_BASE


def test_sample_player_slots_yields_no_hop_keys_from_a_pointerless_object() -> None:
    """A direct pointer to an object holding no plausible pointers contributes no one-hop keys.

    The distractor at ``0x80`` points at ``DECOY_BASE`` (0xff-filled — nothing 8-aligned/mapped in
    it), so it appears as a direct key but contributes no one-hop keys, and reading it never raises.
    """
    source = planted_moveset_scan_source()
    regions = RegionIndex(source.regions())
    slots, _ = sample_player_slots(source, PLAYER_BASE, regions, direct_end=0x200)
    assert (PLAYER_DISTRACTOR_SLOT,) in slots
    assert not any(key[:1] == (PLAYER_DISTRACTOR_SLOT,) and len(key) == 2 for key in slots)


# ---------------------------------------------------------------------------
# The sample census + the guarded, stepped-down direct read (brief #23)
# ---------------------------------------------------------------------------


class _WidthCappedSource:
    """A read-only source whose reads fault above a width cap — models an overrun mapping.

    The live failure this stands in for: a widened window that runs past the end of the struct's
    committed region, so the read faults at 0x4000 but succeeds narrower. ``only_at`` restricts the
    cap to one address, to fail a single sub-object deref while the rest of the sweep works.
    """

    def __init__(self, inner: FlatMemorySource, *, cap: int, only_at: int | None = None) -> None:
        self._inner = inner
        self._cap = cap
        self._only_at = only_at

    def read(self, address: int, size: int) -> bytes:
        if (self._only_at is None or address == self._only_at) and size > self._cap:
            raise MemoryReadError(f"capped: [0x{address:x}, +{size}) exceeds 0x{self._cap:x}")
        return self._inner.read(address, size)

    def module_base(self, module: str) -> int:
        return self._inner.module_base(module)

    def regions(self) -> Sequence[MemoryRegion]:
        return self._inner.regions()


def test_census_counts_are_exact_over_the_planted_world() -> None:
    """The census reports exactly what the sweep did over the planted player struct.

    Ground truth: two direct pointers (the moveset slot -> the intermediate object, and the
    distractor -> ``DECOY_BASE``); the object yields one plausible one-hop pointer (the header
    address at ``0x18``); the 0xff-filled decoy holds nothing plausible but IS readable, so both
    sub-objects are read and none skipped.
    """
    source = planted_moveset_scan_source()
    regions = RegionIndex(source.regions())
    slots, census = sample_player_slots(source, PLAYER_BASE, regions, direct_end=0x200)
    assert census.direct_bytes_read == 0x200
    assert census.direct_slots_scanned == 0x200 // 8
    assert census.direct_pointers == 2
    assert census.sub_objects_read == 2
    assert census.sub_objects_skipped == 0
    assert census.hop_pointers == 1
    # the invariant the census exists to make auditable
    assert census.total_keys == census.direct_pointers + census.hop_pointers
    assert census.total_keys == len(slots)


def test_census_counts_a_skipped_sub_object() -> None:
    """A direct pointer whose target cannot be read counts as a skipped sub-object, not a read one.

    ``DECOY_BASE`` is declared in the region map but its bytes are withheld from the source, so the
    deref fails: it still yields its direct key, and the census records the skip.
    """
    inner = planted_moveset_scan_source()
    source = _WidthCappedSource(inner, cap=0x200, only_at=DECOY_BASE)
    regions = RegionIndex(inner.regions())
    slots, census = sample_player_slots(
        source, PLAYER_BASE, regions, direct_end=0x200, hop_end=0x400
    )
    assert (PLAYER_DISTRACTOR_SLOT,) in slots
    assert census.sub_objects_skipped == 1
    assert census.sub_objects_read == 1


def test_sampler_returns_empty_when_every_direct_read_fails() -> None:
    """When the wide read AND every fallback width fail, the sampler yields nothing, never raising.

    This is the bug brief #23 fixes: the old room-bounded retry sat outside any ``try``, so a second
    fault propagated out of a function documented "never raises" and the capture was silently lost —
    a broken run masquerading as a quiet one.
    """
    source = _WidthCappedSource(planted_moveset_scan_source(), cap=0)
    regions = RegionIndex([MemoryRegion(base=PLAYER_BASE, size=0x100000)])
    slots, census = sample_player_slots(source, PLAYER_BASE, regions, direct_end=0x4000)
    assert slots == {}
    assert census.direct_bytes_read == 0
    assert census.total_keys == 0


def test_sampler_steps_down_to_a_narrower_width_and_still_finds_pointers() -> None:
    """A window overrunning the struct steps down, reports the narrower width, and still sweeps.

    The region map claims 0x100000 of room (so the room-bounded retry is no help — it re-tries the
    full 0x4000), but reads wider than 0x2000 fault. The sampler must halve down to 0x2000, record
    that as ``direct_bytes_read``, and still find the pointer planted at 0x30.
    """
    player = bytearray(0x2000)
    player[0x30:0x38] = OBJECT_BASE.to_bytes(8, "little")
    inner = FlatMemorySource(
        [
            (PLAYER_BASE, bytes(player)),
            (OBJECT_BASE, MOVESET_BASE.to_bytes(8, "little") * 8),
            (MOVESET_BASE, b"\x00" * 0x40),
        ],
        module_bases={},
        regions=[
            MemoryRegion(base=PLAYER_BASE, size=0x100000),
            MemoryRegion(base=OBJECT_BASE, size=0x40),
            MemoryRegion(base=MOVESET_BASE, size=0x40),
        ],
    )
    source = _WidthCappedSource(inner, cap=0x2000)
    regions = RegionIndex(inner.regions())
    slots, census = sample_player_slots(source, PLAYER_BASE, regions, direct_end=0x4000)
    assert census.direct_bytes_read == 0x2000
    assert slots[(0x30,)] == OBJECT_BASE
    assert census.direct_pointers == 1


def test_describe_slots_over_zero_sampled_slots_returns_nothing() -> None:
    """Samples carrying NO slots describe nothing — the exact ambiguity the census resolves.

    Three clean ids, zero swept pointers: ``describe_slots`` is empty, which is byte-for-byte the
    output of three ids whose every slot was constant. Only the census tells the two apart, so the
    CLI must branch on it rather than on this list.
    """
    samples = [MoveSample(move_id, {}) for move_id in (1695, 1725, 1779)]
    assert describe_slots(samples) == []


def test_describe_slots_classifies_constant_nonlinear_and_linear() -> None:
    """The diagnostic labels each slot key constant / non-linear / linear-K-of-N over the ids."""
    ids = (1695, 1725, 1779)
    samples = []
    for i, move_id in enumerate(ids):
        slots: dict[SlotKey, int] = {
            (0x10,): 0x555,  # constant
            (0x20,): _MOVES_BASE + move_id * _STRIDE,  # perfectly linear
            (0x30, 0x8): move_id * 3 if i != 2 else 0xDEAD,  # linear over 2 of 3 -> non-linear
        }
        samples.append(MoveSample(move_id, slots))
    by_key = {d.slot_key: d for d in describe_slots(samples)}
    assert by_key[(0x10,)].kind == "constant"
    assert by_key[(0x20,)].kind == "linear"
    assert by_key[(0x20,)].n_on_line == 3 and by_key[(0x20,)].n_ids == 3
    assert by_key[(0x30, 0x8)].kind == "nonlinear"
    # varying slots are ranked ahead of the constant one
    assert describe_slots(samples)[-1].slot_key == (0x10,)


def test_describe_slots_reports_no_variation_when_all_constant() -> None:
    """When every slot is constant, no description is 'linear'/'nonlinear' — the pivot signal."""
    samples = [
        MoveSample(1695, {(0x10,): 0x555, (0x20,): 0x777}),
        MoveSample(1725, {(0x10,): 0x555, (0x20,): 0x777}),
        MoveSample(1779, {(0x10,): 0x555, (0x20,): 0x777}),
    ]
    descriptions = describe_slots(samples)
    assert descriptions and all(d.kind == "constant" for d in descriptions)


def test_describe_slots_describe_line_is_human_readable() -> None:
    """A varying slot's ``describe()`` names the path and shows the ``(id -> value)`` points."""
    samples = [
        MoveSample(1695, {(0x30, 0x18): _MOVES_BASE + 1695 * _STRIDE}),
        MoveSample(1725, {(0x30, 0x18): _MOVES_BASE + 1725 * _STRIDE}),
        MoveSample(1779, {(0x30, 0x18): _MOVES_BASE + 1779 * _STRIDE}),
    ]
    text = describe_slots(samples)[0].describe()
    assert "player+0x30 -> object+0x18" in text
    assert "1695->0x" in text


def test_format_slot_key_direct_and_hop_and_given() -> None:
    """The slot-key formatter renders direct, one-hop, and the supplied-not-solved cases."""
    assert format_slot_key((0x40,)) == "player+0x40"
    assert format_slot_key((0x30, 0x18)) == "player+0x30 -> object+0x18"
    assert "not solved" in format_slot_key(())


# ---------------------------------------------------------------------------
# Phase 2 — grounding dumps over the planted moveset world
# ---------------------------------------------------------------------------


def test_dump_move_formats_the_known_tk_move_words() -> None:
    """``dump_move`` reads the right ``tk_move`` and shows its (planted) cancel pointer/count words.

    Move 1705 (``b+1``) owns a cancel list in the planted world, so its ``tk_move`` (0x10 stride)
    holds ``cancel_start_ptr`` at +0x000 and its count at +0x008 — both must appear in the dump.
    """
    source = planted_moveset_scan_source()
    moves = MovesArray(slot_key=(0x0,), moves_base=MOVES_BASE, move_stride=MOVE_SIZE)
    text = dump_move(source, moves, 1705)
    assert f"move_id 1705 @ 0x{MOVES_BASE + 1705 * MOVE_SIZE:x}" in text
    # move 1705 owns one cancel (b -> b,1); its cancel_start points into the cancels array.
    assert "+0x000: 0x" in text and "+0x008: 0x0000000000000001" in text


def test_dump_move_is_total_over_unreadable_words() -> None:
    """A move whose window runs past mapped memory yields ``<unreadable>``, never an exception."""
    source = planted_moveset_scan_source()
    moves = MovesArray(slot_key=(0x0,), moves_base=0x999000000, move_stride=MOVE_SIZE)
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
