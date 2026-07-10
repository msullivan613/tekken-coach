"""A read-only source that replays a sequence of flat snapshots as successive game frames.

:class:`~tests.fixtures.reader.flat_source.FlatMemorySource` serves one static instant, which is all
the *derivation* tests need. The **doctor** (docs/02 §6) instead asserts things that only exist over
time — the frame counter increases, the inter-player distance changes — so it needs a source that
advances. This wraps N flat snapshots and steps to the next one whenever the global frame-counter
address is read, exactly as
:class:`~tekken_coach.reader.memory_source.FakeMemorySource` does for the C4a tables. The difference
is that this one serves arbitrary sub-range reads, so it works behind a pointer chain.

Read-only — no write method — mirroring the real seam (docs/02 §2).
"""

from __future__ import annotations

from collections.abc import Sequence

from tekken_coach.reader.memory_source import MemoryRegion
from tests.fixtures.reader.flat_source import FlatMemorySource


class TickingFlatSource:
    """Serve ``snapshots[i]``, advancing ``i`` each time ``advance_on`` is read."""

    def __init__(self, snapshots: Sequence[FlatMemorySource], *, advance_on: int) -> None:
        if not snapshots:
            raise ValueError("TickingFlatSource needs at least one snapshot")
        self._snapshots = list(snapshots)
        self._advance_on = advance_on
        self._cursor = -1  # the first frame-counter read advances to snapshot 0

    def read(self, address: int, size: int) -> bytes:
        if address == self._advance_on:
            self._cursor = min(self._cursor + 1, len(self._snapshots) - 1)
        return self._snapshots[max(self._cursor, 0)].read(address, size)

    def module_base(self, module: str) -> int:
        return self._snapshots[max(self._cursor, 0)].module_base(module)

    def regions(self) -> Sequence[MemoryRegion]:
        return self._snapshots[max(self._cursor, 0)].regions()
