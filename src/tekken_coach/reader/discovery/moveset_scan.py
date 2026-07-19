"""Find a character's ``tk_moveset`` by heap **shape + gate** scan, not a direct player slot (#19).

Brief #18 assumed the player struct holds a single direct pointer to its ``tk_moveset`` header, and
discovered it by sweeping the player's own pointer slots. The 2026-07-19 live run disproved that:
none of the three plausible direct slots landed on a header shape. The header is reachable, but only
through a *pointer path* (player -> some object -> header), not one slot. So discovery cannot assume
a direct slot — it must find the header **by its shape**, anywhere on the heap, then confirm it with
the decisive decoder gate, and only *then* derive a durable reference path back to the player base.

Three pure, offline-testable pieces (the live shell in ``commands.py`` only times + prints them):

* :func:`shape_survivors` — the cheap pre-filter. Reads each enumerated region into a buffer
  **once** and sweeps 8-aligned candidate header addresses entirely in-process (no per-candidate
  syscall, which keeps this off the C4h full-heap perf wall). A candidate survives only if its four
  header words — the cancels/moves counts and pointers — are shaped like a real moveset. The
  overwhelming majority are rejected on the first count word.
* :func:`gate_survivors` — the decisive check. For each shape-survivor it reads the cancels array
  and runs :func:`~tekken_coach.reader.moveset.gate_pairs`: the header is the candidate whose
  cancels reproduce *all* of the character's known ``move_id -> notation`` anchors under the T8
  decode. This reads the **static** cancels array, so it works while the player is idle.
* :func:`derive_reference_path` — durability. A raw heap address moves every session, so this
  reverse-scans from the confirmed header back to a stable path from the player base, expressed
  as a :class:`~tekken_coach.reader.offsets.ComponentAnchor` (``slot_offset`` + ``pointer_path``),
  which ``moveset-build`` resolves with no manual address. When no player-relative path is found the
  caller falls back to re-running this shape+gate scan at startup (slower, but self-healing).

Read-only throughout (docs/02 §2): it enumerates regions, reads bytes, follows pointers, and writes
nothing. Perf: the naive full-heap byte sweep is what stalled C4h ``--derive``; the cheap four-word
count/pointer pre-filter *before* any deref or gate is what keeps this tractable. If even the
buffered sweep is too slow live, the numpy vectorization deferred for ``--derive`` applies to the
pre-filter — but only the handful of shape-survivors ever reach the gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tekken_coach.reader.decode import resolve_component
from tekken_coach.reader.discovery.basescan import Progress, _emit
from tekken_coach.reader.discovery.heapscan import _buffer_covering, _region_buffers
from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.moveset import (
    CANCEL_SIZE,
    CANCELS_COUNT_MAX,
    CANCELS_COUNT_MIN,
    MOVES_COUNT_MAX,
    MOVES_COUNT_MIN,
    MOVESET_CANCELS_COUNT_OFFSET,
    MOVESET_CANCELS_PTR_OFFSET,
    MOVESET_MOVES_COUNT_OFFSET,
    MOVESET_MOVES_PTR_OFFSET,
    GateRow,
    KnownPair,
    MovesetHeader,
    gate_pairs,
    read_cancels,
    read_moveset_header,
)
from tekken_coach.reader.offsets import ComponentAnchor
from tekken_coach.reader.slots import MIN_POINTER_VALUE, POINTER_SIZE, RegionIndex

# The four header words the cheap filter reads, as u64 indices into an 8-aligned candidate. All four
# documented offsets are multiples of 8 (the header is a pointer/count table), so a candidate at a
# word index ``i`` reads them at fixed word offsets — letting the sweep run over a memoryview cast
# without per-candidate struct unpacking.
_CANCELS_PTR_WORD = MOVESET_CANCELS_PTR_OFFSET // POINTER_SIZE
_CANCELS_COUNT_WORD = MOVESET_CANCELS_COUNT_OFFSET // POINTER_SIZE
_MOVES_PTR_WORD = MOVESET_MOVES_PTR_OFFSET // POINTER_SIZE
_MOVES_COUNT_WORD = MOVESET_MOVES_COUNT_OFFSET // POINTER_SIZE

# The furthest word the cheap filter touches; a candidate must have this many words after it.
_HEADER_LAST_WORD = max(_CANCELS_PTR_WORD, _CANCELS_COUNT_WORD, _MOVES_PTR_WORD, _MOVES_COUNT_WORD)

# Default max forward offset for a reverse-scan hop (player object -> the slot holding the header).
# An intermediate game object is at most a few KiB; bounding the hop keeps false pairings out.
DEFAULT_MAX_HOP = 0x2000


# ---------------------------------------------------------------------------
# Part B — the shape pre-filter (buffer-local, cheap-filtered)
# ---------------------------------------------------------------------------


def shape_survivors(
    buffers: Sequence[Region],
    region_index: RegionIndex,
    *,
    progress: Progress | None = None,
) -> list[int]:
    """Sweep the heap buffers for 8-aligned addresses shaped like a ``tk_moveset`` header.

    Buffer-local and cheap-filtered: for every 8-aligned candidate address it reads the two count
    words (each rejected against its own realistic range), requires the **moveset shape**
    ``cancels > moves``, then checks the two array pointers land in an enumerated region. Nothing is
    dereferenced and no cancel is read here — that is :func:`gate_survivors`' job on the few that
    survive. Uses a :class:`memoryview` cast to ``u64`` per buffer so the hot loop is index
    arithmetic, not per-candidate ``struct.unpack``.

    The ``cancels > moves`` requirement is what the 2026-07-19 live run proved essential (#20):
    without it, 8,703 of 9,026 survivors were contiguous near-equal-count arrays (two adjacent
    small words) that flooded the gate. A real moveset has far more cancels than moves, so this one
    cheap comparison — applied before the region lookups — collapses the survivors to a handful.
    """
    survivors: list[int] = []
    swept = 0
    for buffer in buffers:
        base = buffer.base
        data = buffer.data
        pad = (-base) % POINTER_SIZE  # regions are page-aligned, so this is 0 in practice
        if pad:
            base += pad
            data = data[pad:]
        usable = len(data) - (len(data) % POINTER_SIZE)
        words = memoryview(data)[:usable].cast("Q")
        n = len(words)
        if n <= _HEADER_LAST_WORD:
            continue
        swept += n - _HEADER_LAST_WORD
        for i in range(n - _HEADER_LAST_WORD):
            cancels_count = words[i + _CANCELS_COUNT_WORD]
            if not (CANCELS_COUNT_MIN <= cancels_count <= CANCELS_COUNT_MAX):
                continue
            moves_count = words[i + _MOVES_COUNT_WORD]
            if not (MOVES_COUNT_MIN <= moves_count <= MOVES_COUNT_MAX):
                continue
            if cancels_count <= moves_count:
                continue  # the defining moveset shape — kills the near-equal-count junk (brief #20)
            if not region_index.contains(words[i + _CANCELS_PTR_WORD], CANCEL_SIZE):
                continue
            if not region_index.contains(words[i + _MOVES_PTR_WORD], POINTER_SIZE):
                continue
            survivors.append(base + i * POINTER_SIZE)
    _emit(
        progress,
        f"  shape sweep: {len(survivors)} survivor(s) from {swept} candidate word(s) "
        f"in {len(buffers)} region(s)",
    )
    return survivors


# ---------------------------------------------------------------------------
# Part B — the decisive gate on the shape survivors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovesetCandidate:
    """A shape-survivor header address with the decoder gate applied (brief #19)."""

    header_addr: int
    header: MovesetHeader
    gate: list[GateRow]

    @property
    def gate_passed(self) -> bool:
        """Whether every known anchor was reproduced from this candidate's cancels."""
        return bool(self.gate) and all(row.found for row in self.gate)


def gate_survivors(
    source: MemorySource,
    survivors: Sequence[int],
    pairs: tuple[KnownPair, ...],
    *,
    progress: Progress | None = None,
) -> list[MovesetCandidate]:
    """Read each shape-survivor's cancels and gate it on the character's known anchors.

    The decisive step: the true header is the survivor whose cancels reproduce every anchor. A
    survivor whose header pointers turn out unreadable is dropped (never a crash). Returns *all*
    survivors with their gate result so the probe can show the near-misses, not only the winner.
    """
    candidates: list[MovesetCandidate] = []
    for addr in survivors:
        try:
            header = read_moveset_header(source, addr)
            cancels = read_cancels(source, header)
        except MemoryReadError:
            continue
        candidates.append(MovesetCandidate(addr, header, gate_pairs(cancels, pairs)))
    passers = sum(c.gate_passed for c in candidates)
    _emit(
        progress, f"  gate: {passers} of {len(candidates)} shape-survivor(s) reproduced the anchors"
    )
    return candidates


@dataclass(frozen=True)
class MovesetScan:
    """The full shape+gate scan outcome (brief #19 Part B): buffers reused, survivors, matches."""

    buffers: Sequence[Region]
    survivors: list[int]
    candidates: list[MovesetCandidate]

    @property
    def matches(self) -> list[MovesetCandidate]:
        """The gate-passers — one per loaded character with these anchors (ideally exactly one)."""
        return [c for c in self.candidates if c.gate_passed]

    @property
    def winner(self) -> MovesetCandidate | None:
        """The first gate-passing header, or ``None`` if none passed.

        Ideally exactly one candidate passes (one loaded character per anchor set); if several do
        (e.g. both players are the same character) they are all in :attr:`matches` and the caller
        reports the ambiguity — this just returns the first so the common single-match path is easy.
        """
        matches = self.matches
        return matches[0] if matches else None


def scan_moveset(
    source: MemorySource,
    *,
    pairs: tuple[KnownPair, ...],
    buffers: Sequence[Region] | None = None,
    region_index: RegionIndex | None = None,
    progress: Progress | None = None,
) -> MovesetScan:
    """Compose Part B: read the regions once, shape-filter, then gate the survivors.

    Reuses the C4h buffer infra so all candidate-scanning is in-process. The buffers are carried on
    the result so :func:`derive_reference_path` (Part C) can reverse-scan them without re-reading.
    """
    if buffers is None:
        buffers = _region_buffers(source, source.regions(), progress=progress)
    if region_index is None:
        region_index = RegionIndex(source.regions())
    survivors = shape_survivors(buffers, region_index, progress=progress)
    candidates = gate_survivors(source, survivors, pairs, progress=progress)
    return MovesetScan(buffers=buffers, survivors=survivors, candidates=candidates)


# ---------------------------------------------------------------------------
# Part C — derive a durable, player-relative reference path to the header
# ---------------------------------------------------------------------------


def _find_value_locations(buffers: Sequence[Region], value: int) -> list[int]:
    """Every 8-aligned location in the buffers that stores ``value`` as a little-endian u64.

    Uses ``bytes.find`` (C-level) so scanning the whole heap for one 8-byte needle is fast — this is
    how the reverse scan locates the slot pointing *at* the header without indexing every pointer.
    """
    import struct  # noqa: PLC0415

    needle = struct.pack("<Q", value)
    out: list[int] = []
    for buffer in buffers:
        data = buffer.data
        start = 0
        while True:
            pos = data.find(needle, start)
            if pos == -1:
                break
            if (buffer.base + pos) % POINTER_SIZE == 0:
                out.append(buffer.base + pos)
            start = pos + 1
    return out


def _player_pointer_slots(buffer: Region, player_base: int, span: int) -> list[tuple[int, int]]:
    """``(slot_offset, value)`` for each 8-aligned plausible-pointer slot in the player struct."""
    slots: list[tuple[int, int]] = []
    limit = min(span, buffer.end - player_base)
    for off in range(0, limit - POINTER_SIZE + 1, POINTER_SIZE):
        value = buffer.read_scalar(player_base + off, "ptr")
        if isinstance(value, int) and value >= MIN_POINTER_VALUE:
            slots.append((off, value))
    return slots


def _resolves(
    source: MemorySource, player_base: int, anchor: ComponentAnchor, expected: int
) -> bool:
    """Whether ``anchor`` from ``player_base`` actually lands on ``expected`` (read-only)."""
    try:
        return resolve_component(source, player_base, anchor) == expected
    except MemoryReadError:
        return False


def derive_reference_path(
    source: MemorySource,
    buffers: Sequence[Region],
    *,
    header_addr: int,
    player_base: int,
    player_struct_span: int,
    max_hop: int = DEFAULT_MAX_HOP,
    progress: Progress | None = None,
) -> ComponentAnchor | None:
    """Reverse-scan from ``header_addr`` back to a stable path from the player base (Part C).

    Two realistic shapes, in preference order (the live run showed the direct slot does not exist,
    so the second is the expected outcome):

    #. **Direct slot** (``pointer_path == []``): a pointer slot inside the player struct holds the
       header address outright.
    #. **One-hop** (``pointer_path == [off]``): a player slot points at an intermediate object, and
       a location ``obj + off`` inside that object holds the header address. Found by locating every
       slot that stores the header, then pairing it with a player slot whose value sits just below
       it (``obj <= L <= obj + max_hop``).

    Every proposal is confirmed by actually resolving it through the read-only source, so a
    coincidental pairing is rejected, never recorded. Returns ``None`` when no player-relative path
    within these bounds resolves — the caller then falls back to a startup shape+gate re-scan.
    """
    buffer = _buffer_covering(buffers, player_base, POINTER_SIZE)
    if buffer is None:
        _emit(progress, "  reverse scan: player base is not in an enumerated region")
        return None
    slots = _player_pointer_slots(buffer, player_base, player_struct_span)

    for off, value in slots:  # depth 1: a direct player slot holding the header address
        if value == header_addr:
            anchor = ComponentAnchor(slot_offset=off, pointer_path=[])
            if _resolves(source, player_base, anchor, header_addr):
                _emit(progress, f"  reverse scan: direct slot +0x{off:x} holds the header")
                return anchor

    locations = _find_value_locations(buffers, header_addr)  # depth 2: player -> object -> header
    for location in locations:
        for off, obj in slots:
            if obj <= location <= obj + max_hop:
                hop = location - obj
                anchor = ComponentAnchor(slot_offset=off, pointer_path=[hop])
                if _resolves(source, player_base, anchor, header_addr):
                    _emit(
                        progress,
                        f"  reverse scan: player +0x{off:x} -> object +0x{hop:x} -> header",
                    )
                    return anchor
    _emit(progress, "  reverse scan: no player-relative path found within bounds")
    return None
