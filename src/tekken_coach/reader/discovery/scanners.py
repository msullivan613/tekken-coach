"""Pure memory-scan primitives for offset re-discovery (docs/02 §3/§4).

These are the search primitives ``update-offsets`` runs over a captured memory image to re-derive
where each field lives after a game patch. They are **pure functions over raw bytes** — a
:class:`Region` is just ``(base_address, bytes)`` — so the whole search is offline-testable against
a synthetic image with a known planted layout, with no game and no ``pymem`` present. The live tool
reads a :class:`Region` off the process through the read-only
:class:`~tekken_coach.reader.memory_source.MemorySource` (see :func:`snapshot_region`) and then runs
exactly these functions.

Three primitives, matching the re-discovery technique (docs/02 §4):

* :func:`value_scan`  — addresses holding a given typed value (an *anchor*: full-health max, the
  known Kazuya char id, …).
* :func:`change_scan` — addresses whose typed value changed (or stayed equal) between two snapshots
  (move id when a button is pressed, position floats when the character moves, the global frame
  counter as it ticks).
* :func:`aob_scan`    — an array-of-bytes / wildcard pattern scan (``"48 8B ?? ?? 89"``), the
  signature form docs/02 §3 prefers because it survives minor relocation.

Nothing here reads process memory directly or writes anything — it slices ``bytes`` (docs/02 §2).
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

from tekken_coach.reader.decode import _FORMATS
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import ScalarKind


# A scanned span of a process's address space: the absolute ``base`` address the blob starts at and
# the raw ``data``. All scan results are returned as absolute addresses (``base + offset``).
@dataclass(frozen=True)
class Region:
    """A contiguous captured memory image: bytes ``data`` starting at absolute ``base``."""

    base: int
    data: bytes

    @property
    def end(self) -> int:
        """One past the last covered address."""
        return self.base + len(self.data)

    def covers(self, address: int, size: int) -> bool:
        """Whether ``[address, address+size)`` lies wholly within this region."""
        return self.base <= address and address + size <= self.end

    def read(self, address: int, size: int) -> bytes:
        """Slice ``size`` bytes at absolute ``address``; raises ``IndexError`` if out of range."""
        if not self.covers(address, size):
            raise IndexError(
                f"0x{address:x} (+{size}) outside region [0x{self.base:x}, 0x{self.end:x})"
            )
        start = address - self.base
        return self.data[start : start + size]

    def read_scalar(self, address: int, kind: ScalarKind) -> int | float | None:
        """Decode a scalar at ``address``, or ``None`` if the range is not covered.

        Returns a plain number (``bool8``/``ptr`` come back as ints) — the scanners compare numbers,
        not the semantic types the decoder produces.
        """
        fmt, size = _FORMATS[kind]
        if not self.covers(address, size):
            return None
        (value,) = struct.unpack(fmt, self.data[address - self.base : address - self.base + size])
        return value  # type: ignore[no-any-return]


def snapshot_region(
    source: MemorySource, module: str, base_offset: int, size: int, *, absolute: bool = False
) -> Region:
    """Read a :class:`Region` off ``source`` (read-only) for the scanners to work over.

    ``base_offset`` is module-relative by default (``module_base(module) + base_offset``) — the
    module-anchored addressing docs/02 §3 prefers — or absolute when ``absolute`` is set. This is
    the only bridge from a live :class:`MemorySource` into the pure scan world; it performs a single
    read and never mutates. It is source-agnostic, so the offline suite exercises it against a
    :class:`~tekken_coach.reader.memory_source.FakeMemorySource` carrying one big region.
    """
    base = base_offset if absolute else source.module_base(module) + base_offset
    return Region(base=base, data=source.read(base, size))


def value_scan(
    region: Region, value: int | float, kind: ScalarKind, *, align: int = 4
) -> list[int]:
    """Return every ``align``-stepped address in ``region`` whose ``kind`` value equals ``value``.

    The anchor primitive of docs/02 §4: locate a *known* value (full-health max, the Kazuya char
    id 12). ``align`` steps the scan (game structs are 4-/8-aligned; ``align=1`` scans unaligned).
    Float matching is exact on the packed bytes — pass a value that is representable, or prefer
    :func:`change_scan` for floats that only move.
    """
    fmt, size = _FORMATS[kind]
    needle = struct.pack(fmt, value)
    data = region.data
    hits: list[int] = []
    for offset in range(0, len(data) - size + 1, align):
        if data[offset : offset + size] == needle:
            hits.append(region.base + offset)
    return hits


def change_scan(
    before: Region,
    after: Region,
    kind: ScalarKind,
    *,
    align: int = 4,
    changed: bool = True,
    predicate: Callable[[float, float], bool] | None = None,
) -> list[int]:
    """Return addresses whose ``kind`` value changed (or stayed equal) between two snapshots.

    The behavioral primitive of docs/02 §4: a move id changes when a button is pressed; position
    floats change when the character moves; the global frame counter increments. ``before`` and
    ``after`` must describe the **same** address span (same ``base`` and length) — they are two
    reads of one region at different instants. With ``changed=False`` it returns addresses that held
    *steady*. An optional ``predicate(old, new)`` further filters (e.g. ``new > old`` for a
    monotonic counter); it is only consulted on addresses that changed.
    """
    if before.base != after.base or len(before.data) != len(after.data):
        raise ValueError("change_scan needs two snapshots of the same address span")
    fmt, size = _FORMATS[kind]
    a, b = before.data, after.data
    hits: list[int] = []
    for offset in range(0, len(a) - size + 1, align):
        (old,) = struct.unpack(fmt, a[offset : offset + size])
        (new,) = struct.unpack(fmt, b[offset : offset + size])
        differs = old != new
        if differs != changed:
            continue
        if changed and predicate is not None and not predicate(float(old), float(new)):
            continue
        hits.append(before.base + offset)
    return hits


@dataclass(frozen=True)
class AobPattern:
    """A parsed array-of-bytes pattern: fixed ``needle`` bytes plus a wildcard ``mask``.

    ``mask[i]`` is ``True`` where byte ``i`` must match ``needle[i]`` and ``False`` for a wildcard.
    """

    needle: bytes
    mask: tuple[bool, ...]

    def __len__(self) -> int:
        return len(self.needle)


def parse_aob(pattern: str) -> AobPattern:
    """Parse a wildcard byte pattern like ``"48 8B ?? ?? 89"`` into an :class:`AobPattern`.

    Tokens are whitespace-separated. A hex byte (``48``) is a fixed match; ``??`` or ``?`` or ``*``
    is a wildcard. Raises ``ValueError`` on an empty or malformed pattern.
    """
    tokens = pattern.split()
    if not tokens:
        raise ValueError("empty AOB pattern")
    needle = bytearray()
    mask: list[bool] = []
    for token in tokens:
        if token in ("??", "?", "*"):
            needle.append(0)
            mask.append(False)
        else:
            try:
                needle.append(int(token, 16))
            except ValueError as exc:
                raise ValueError(f"bad AOB token {token!r} (want a hex byte or wildcard)") from exc
            mask.append(True)
    return AobPattern(needle=bytes(needle), mask=tuple(mask))


def aob_scan(region: Region, pattern: str | AobPattern) -> list[int]:
    """Return every address in ``region`` where ``pattern`` matches, respecting wildcards.

    The signature primitive of docs/02 §3: AOB/pattern signatures survive minor relocation better
    than absolute addresses, so ``update-offsets`` prefers them for anchoring where the surrounding
    bytes are stable. Matching is byte-exact at every non-wildcard position.
    """
    pat = parse_aob(pattern) if isinstance(pattern, str) else pattern
    size = len(pat)
    if size == 0:
        raise ValueError("empty AOB pattern")
    data = region.data
    needle, mask = pat.needle, pat.mask
    hits: list[int] = []
    for offset in range(0, len(data) - size + 1):
        if all(not mask[i] or data[offset + i] == needle[i] for i in range(size)):
            hits.append(region.base + offset)
    return hits
