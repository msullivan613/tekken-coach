"""Read-only heap region enumeration (C4h Phase 1, docs/02 §3).

The C4h layout derivation sweeps the *heap* for the entity struct, so the ``MemorySource`` seam
grows a ``regions()`` query. These tests pin two things: the ``VirtualQueryEx`` walk classifies and
bounds regions correctly (:func:`enumerate_committed_regions`, pure over a fake process map), and
enumeration is **read-only** — it returns spans without reading their bytes.
"""

from __future__ import annotations

from collections.abc import Callable

from tekken_coach.reader.memory_source import FakeMemorySource, MemoryRegion, MemorySource
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
