"""A flat, multi-segment read-only ``MemorySource`` for the C4d base-scan tests.

The code-signature scan needs *random-access* reads: it parses the PE header at the module base,
sweeps a data section, then follows a pointer chain into the heap and reads struct fields at
arbitrary addresses. The C4c :class:`~tekken_coach.reader.memory_source.FakeMemorySource` serves
only whole regions keyed by an exact start address, so it cannot answer a sub-range read at
``module_base + 0x3c``. This fixture fills that gap: a set of ``(base, bytes)`` segments serving
any sub-range lying wholly inside one segment, raising
:class:`~tekken_coach.reader.faults.MemoryReadError` otherwise — modelling mapped regions with
unmapped gaps between them (which is what makes the bounded reads in ``basescan`` matter).

It is read-only — no write/inject method — mirroring the real seam (docs/02 §2). It is test-support
only (like ``encode.py``); the shipped reader never writes memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemoryRegion


class FlatMemorySource:
    """Serve arbitrary sub-range reads over a list of non-overlapping ``(base, bytes)`` segments."""

    def __init__(
        self,
        segments: Sequence[tuple[int, bytes]],
        *,
        module_bases: Mapping[str, int],
        regions: Sequence[MemoryRegion] | None = None,
    ) -> None:
        # Sort by base so lookups are deterministic; segments are expected non-overlapping.
        self._segments = sorted(segments, key=lambda s: s[0])
        self._module_bases = dict(module_bases)
        # Region enumeration (C4h Phase 1). Default: every non-module segment is one committed
        # readable heap region — the common case a heap sweep wants. A test may override to model a
        # specific map (guard pages, an image region to skip, a byte cap).
        if regions is None:
            module_addrs = set(self._module_bases.values())
            regions = [
                MemoryRegion(base=base, size=len(data))
                for base, data in self._segments
                if base not in module_addrs
            ]
        self._regions = list(regions)

    def read(self, address: int, size: int) -> bytes:
        for base, data in self._segments:
            if base <= address and address + size <= base + len(data):
                start = address - base
                return data[start : start + size]
        raise MemoryReadError(f"no mapped segment covers [0x{address:x}, +{size})")

    def module_base(self, module: str) -> int:
        base = self._module_bases.get(module)
        if base is None:
            raise MemoryReadError(f"module not loaded: {module!r}")
        return base

    def regions(self) -> Sequence[MemoryRegion]:
        """The planted committed-region map — read-only, mirroring the live seam (C4h Phase 1)."""
        return list(self._regions)

    def mapped_regions(self) -> Sequence[MemoryRegion]:
        """The same planted map: a fixture has no sweep caps to lift (brief #24)."""
        return list(self._regions)
