"""Read-only heap region enumeration (C4h Phase 1, docs/02 §3).

The C4h layout derivation sweeps the *heap* for the entity struct, so the ``MemorySource`` seam
grows a ``regions()`` query. These tests pin two things: the ``VirtualQueryEx`` walk classifies and
bounds regions correctly (:func:`enumerate_committed_regions`, pure over a fake process map), and
enumeration is **read-only** — it returns spans without reading their bytes.
"""

from __future__ import annotations

from collections.abc import Callable

from tekken_coach.reader.memory_source import FakeMemorySource, MemoryRegion, MemorySource
from tekken_coach.reader.slots import RegionIndex, is_plausible_pointer
from tekken_coach.reader.win_source import (
    _BasicRegion,
    enumerate_committed_regions,
)

# Windows constants mirrored from win_source, spelled out here so the test states its own facts.
_COMMIT = 0x1000
_RESERVE = 0x2000
_FREE = 0x10000
_IMAGE = 0x1000000
_PRIVATE = 0x20000
_PAGE_RW = 0x04
_PAGE_NOACCESS = 0x01
_PAGE_GUARD = 0x100


def _map(regions: list[_BasicRegion]) -> Callable[[int], _BasicRegion | None]:
    """A fake ``VirtualQueryEx``: answer the region covering ``address``, else end the walk."""
    ordered = sorted(regions, key=lambda r: r.base)

    def query(address: int) -> _BasicRegion | None:
        for region in ordered:
            if region.base <= address < region.base + region.size:
                return region
            if address < region.base:  # a gap: VirtualQueryEx reports the free span up to the next
                return _BasicRegion(
                    base=address, size=region.base - address, state=_FREE, protect=0, type=0
                )
        return None

    return query


def test_enumerates_committed_readable_private_regions() -> None:
    heap = _BasicRegion(base=0x200000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE)
    got = enumerate_committed_regions(_map([heap]))
    assert got == [MemoryRegion(base=0x200000, size=0x1000)]


def test_skips_reserved_free_guard_and_noaccess() -> None:
    regions = [
        _BasicRegion(base=0x100000, size=0x1000, state=_RESERVE, protect=_PAGE_RW, type=_PRIVATE),
        _BasicRegion(
            base=0x101000, size=0x1000, state=_COMMIT, protect=_PAGE_NOACCESS, type=_PRIVATE
        ),
        _BasicRegion(
            base=0x102000, size=0x1000, state=_COMMIT, protect=_PAGE_RW | _PAGE_GUARD, type=_PRIVATE
        ),
        _BasicRegion(base=0x103000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
    ]
    got = enumerate_committed_regions(_map(regions))
    assert got == [MemoryRegion(base=0x103000, size=0x1000)]


def test_skips_module_image_regions_by_default() -> None:
    regions = [
        _BasicRegion(base=0x140000000, size=0x2000, state=_COMMIT, protect=_PAGE_RW, type=_IMAGE),
        _BasicRegion(base=0x200000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
    ]
    got = enumerate_committed_regions(_map(regions))
    assert got == [MemoryRegion(base=0x200000, size=0x1000)]
    # The image region *is* returned when the caller asks for it (the switch works, though unused).
    both = enumerate_committed_regions(_map(regions), skip_image=False)
    assert MemoryRegion(base=0x140000000, size=0x2000) in both


def test_a_giant_reservation_is_skipped_not_swept() -> None:
    regions = [
        _BasicRegion(base=0x200000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
        _BasicRegion(base=0x400000, size=1 << 40, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
    ]
    got = enumerate_committed_regions(_map(regions), max_region_bytes=0x100000)
    assert got == [MemoryRegion(base=0x200000, size=0x1000)]


def test_total_byte_cap_bounds_the_sweep() -> None:
    regions = [
        _BasicRegion(base=0x200000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
        _BasicRegion(base=0x300000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
        _BasicRegion(base=0x400000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
    ]
    got = enumerate_committed_regions(_map(regions), max_total_bytes=0x1000)
    assert got == [MemoryRegion(base=0x200000, size=0x1000)]  # stops once the cap is reached


def test_a_nonadvancing_query_terminates() -> None:
    # A query that never moves the cursor past `address` must not loop forever.
    def stuck(address: int) -> _BasicRegion | None:
        return _BasicRegion(base=address, size=0, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE)

    assert enumerate_committed_regions(stuck) == []


# ---------------------------------------------------------------------------
# The complete map for pointer VALIDATION (brief #24)
# ---------------------------------------------------------------------------
#
# The sweep caps above are correct for bounding a sweep and wrong as a validity oracle. Reusing them
# as one starved the moveset-anchor sampler to 13 plausible pointers out of 2048 slots: the game's
# arenas exceed the per-region cap, its committed total exceeds the running cap, and vtables — a C++
# object's most common pointer — live in module images. These pin that lifting the caps recovers
# them, and that the capped defaults are unchanged.


def _oversized_world() -> list[_BasicRegion]:
    """A map with everything the capped walk drops: a huge arena, a >4 GiB total, and an image."""
    return [
        _BasicRegion(base=0x200000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
        # A 1 GiB arena: past the 512 MiB per-region ceiling, so the capped walk skips it entirely.
        _BasicRegion(base=0x10000000, size=1 << 30, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
        # Two more 2 GiB arenas: together they push the running total past the 4 GiB cutoff.
        _BasicRegion(
            base=0x100000000, size=1 << 31, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE
        ),
        _BasicRegion(
            base=0x200000000, size=1 << 31, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE
        ),
        # The module image, where .rdata vtable targets live.
        # (Placed clear of the arenas above: the fake map serves the first covering region, so an
        # overlapping image would simply be swallowed by an arena rather than tested.)
        _BasicRegion(base=0x290000000, size=0x2000, state=_COMMIT, protect=_PAGE_RW, type=_IMAGE),
        # Past the 4 GiB running total, so the capped walk has already broken before reaching it.
        _BasicRegion(base=0x300000000, size=0x1000, state=_COMMIT, protect=_PAGE_RW, type=_PRIVATE),
    ]


def test_uncapped_enumeration_returns_the_whole_map() -> None:
    world = _oversized_world()
    # Capped (today's defaults): every arena and the image are gone, leaving only the two small
    # private regions. (Skipping the arenas means the running total never accumulates, so the walk
    # reaches the far region rather than breaking — the per-region ceiling did all the work here.)
    capped = enumerate_committed_regions(_map(world))
    assert capped == [
        MemoryRegion(base=0x200000, size=0x1000),
        MemoryRegion(base=0x300000000, size=0x1000),
    ]
    # Uncapped: every committed readable region, images included.
    complete = enumerate_committed_regions(
        _map(world), max_region_bytes=None, max_total_bytes=None, skip_image=False
    )
    assert complete == [
        MemoryRegion(base=0x200000, size=0x1000),
        MemoryRegion(base=0x10000000, size=1 << 30),
        MemoryRegion(base=0x100000000, size=1 << 31),
        MemoryRegion(base=0x200000000, size=1 << 31),
        MemoryRegion(base=0x290000000, size=0x2000),
        MemoryRegion(base=0x300000000, size=0x1000),
    ]


def test_lifting_one_cap_at_a_time_is_independent() -> None:
    world = _oversized_world()
    # Only the per-region ceiling lifted: the 4 GiB running total still truncates the walk, and the
    # image is still excluded. This pins that `None` means "no ceiling", not "no bounds at all".
    got = enumerate_committed_regions(_map(world), max_region_bytes=None)
    assert MemoryRegion(base=0x10000000, size=1 << 30) in got  # the ceiling is genuinely lifted
    assert MemoryRegion(base=0x290000000, size=0x2000) not in got  # skip_image still on
    # The walk breaks once the running total is reached (having appended the region that crossed
    # it), so the region *after* the cutoff is the one that proves the total cap still bites.
    assert MemoryRegion(base=0x300000000, size=0x1000) not in got


def test_complete_map_validates_pointers_the_capped_map_rejects() -> None:
    """The live failure, pinned: a pointer into a big arena is real but reads as junk when capped.

    This is brief #24's root cause in miniature. ``is_plausible_pointer`` is not at fault — its
    three mechanical tests are right — so the same value is fed to two :class:`RegionIndex` oracles
    that differ only in the map they were built from.
    """
    world = _oversized_world()
    into_arena = 0x10000000 + 0x5000  # squarely inside the 1 GiB region
    into_image = 0x290000000 + 0x100  # a vtable-shaped target in .rdata

    capped = RegionIndex(enumerate_committed_regions(_map(world)))
    complete = RegionIndex(
        enumerate_committed_regions(
            _map(world), max_region_bytes=None, max_total_bytes=None, skip_image=False
        )
    )

    assert not is_plausible_pointer(into_arena, capped)
    assert not is_plausible_pointer(into_image, capped)
    assert is_plausible_pointer(into_arena, complete)
    assert is_plausible_pointer(into_image, complete)


def test_region_index_describes_the_map_it_validates_against() -> None:
    # The census line: without it, a starved sweep is invisible the next time a cap bites.
    index = RegionIndex([MemoryRegion(base=0x200000, size=1 << 30)])
    description = index.describe()
    assert description.startswith("1 (total 1.00 GiB")
    assert "largest 1.00 GiB" in description
    assert RegionIndex([]).describe().startswith("0 (total 0.00 GiB")


def test_fake_source_serves_the_same_map_for_both_questions() -> None:
    planted = [MemoryRegion(base=0x5000, size=0x40)]
    source = FakeMemorySource(
        [{0x1000: b"\x00\x00\x00\x00"}],
        module_bases={"m": 0x1000},
        advance_on=0x1000,
        regions=planted,
    )
    # A fake has no caps to lift, so the budgeted and complete views coincide — but both must exist,
    # since the seam now asks two different questions.
    assert list(source.regions()) == planted
    assert list(source.mapped_regions()) == planted


def test_fake_source_regions_are_read_only_and_scripted() -> None:
    planted = [MemoryRegion(base=0x5000, size=0x40)]
    source = FakeMemorySource(
        [{0x1000: b"\x00\x00\x00\x00"}],
        module_bases={"m": 0x1000},
        advance_on=0x1000,
        regions=planted,
    )
    assert isinstance(source, MemorySource)  # the seam is satisfied
    assert list(source.regions()) == planted
    # Enumeration reads no bytes: it works even though the region's address is not a scripted read.
    assert source.regions()[0].base == 0x5000
