"""The ``MemorySource`` seam — the read-only boundary between the decoder and the OS.

The reader never touches process memory directly; every read goes through a
:class:`MemorySource`. The interface is **read-only by construction**: it exposes exactly
two capabilities — read a byte range, and resolve a loaded module's base address — and *no*
mutation primitive. There is deliberately no ``write``/``inject`` method anywhere in this
Protocol, so no call site downstream can even name one (docs/02 §2, §5). The read-only
invariant is thus enforced by the type of the seam, not by discipline; a grep test
(``tests/test_reader_readonly.py``) additionally asserts the whole ``reader`` package contains
no write/inject token.

Two implementations:

* :class:`FakeMemorySource` (here) — serves scripted, in-memory byte buffers. Used by every
  offline test. It models a *live, ticking* process: reading the global frame-counter address
  advances it to the next scripted snapshot, so a sequence of snapshots replays as successive
  frames (the same way the real game advances between polls).
* The concrete Windows ``ReadProcessMemory``/pymem source — **C4b**, not here. It implements the
  same Protocol behind the seam and, like everything in this package, reads only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

# A scripted memory image: absolute address -> the bytes stored there. Regions must not overlap
# in a way that a single read straddles two entries (reads are served from one covering region).
MemoryImage = Mapping[int, bytes]


@runtime_checkable
class MemorySource(Protocol):
    """Read-only access to a target process's memory.

    Implementations expose only these two methods. There is no write/inject method, by design
    (docs/02 §2): the decoder resolves addresses and reads bytes, and can do nothing else.
    """

    def read(self, address: int, size: int) -> bytes:
        """Return exactly ``size`` bytes starting at ``address``.

        Raises :class:`tekken_coach.reader.faults.MemoryReadError` if the range is not
        readable (unmapped, process gone, access denied).
        """
        ...

    def module_base(self, module: str) -> int:
        """Return the load base address of ``module`` (e.g. the game executable).

        This is the anchor for module-base + static-offset addressing (docs/02 §3). Raises
        :class:`tekken_coach.reader.faults.MemoryReadError` if the module is not loaded.
        """
        ...


class FakeMemorySource:
    """A scripted, read-only :class:`MemorySource` for offline tests.

    Serves byte buffers from a list of *snapshots*, each a ``{address: bytes}`` image. It models
    a ticking process: whenever ``advance_on`` (the global frame-counter address) is read, the
    cursor steps to the next snapshot *before* serving, so the first frame reads snapshot 0, the
    second reads snapshot 1, and so on (clamped at the last). A single snapshot therefore replays
    as a static instant; a list replays as successive frames — which is how the doctor's
    monotonic-frame and moving-position checks (docs/02 §6) get distinct data offline.

    It has no write method — mirroring the real seam, it can only be read.
    """

    def __init__(
        self,
        snapshots: Sequence[MemoryImage],
        *,
        module_bases: Mapping[str, int],
        advance_on: int,
    ) -> None:
        if not snapshots:
            raise ValueError("FakeMemorySource needs at least one snapshot")
        self._snapshots: list[MemoryImage] = list(snapshots)
        self._module_bases = dict(module_bases)
        self._advance_on = advance_on
        # Starts at -1 so the first frame-counter read advances to snapshot 0.
        self._cursor = -1

    def read(self, address: int, size: int) -> bytes:
        if address == self._advance_on:
            self._cursor = min(self._cursor + 1, len(self._snapshots) - 1)
        image = self._snapshots[max(self._cursor, 0)]
        region = image.get(address)
        if region is None or len(region) < size:
            from tekken_coach.reader.faults import MemoryReadError

            raise MemoryReadError(
                f"no scripted bytes at 0x{address:x} (+{size}) in snapshot {max(self._cursor, 0)}"
            )
        return region[:size]

    def module_base(self, module: str) -> int:
        base = self._module_bases.get(module)
        if base is None:
            from tekken_coach.reader.faults import MemoryReadError

            raise MemoryReadError(f"module not loaded: {module!r}")
        return base
