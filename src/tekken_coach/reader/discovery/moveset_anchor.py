"""Ground the ``tk_moveset`` layout on the trusted live ``move_id`` (brief #21).

The 2026-07-19 dumps proved the doc-derived ``tk_moveset`` offsets are wrong for the live 5.02.01
build: the blind shape scan (#18-#20) was systematically matching shader/mesh objects because the
header offsets it read were for a different build (``cancels_ptr @ 0x1d0`` landed in DXBC/DXIL
data). Guessing new offsets from those false positives is unsound — they are not movesets.

So this module stops trusting the docs and grounds the layout on the one fact we fully trust: the
live ``move_id``. The reader already reads ``move_id @ 0x550`` and it is verified against
``bryan.json``. ``move_id`` is an **index into the moveset's ``moves`` array**, so the live value is
our foothold.

Two grounded steps, both offline-testable here (the live sampling shell in ``commands.py`` only
gathers the samples and prints the results):

* **Phase 1 — :func:`solve_moves_array`.** The game holds a pointer to the *current* move's
  ``tk_move`` so it can animate it; that pointer equals ``moves_base + move_id * move_stride``.
  Given samples of ``(move_id, {slot_offset: pointer_value})`` taken while the user performs several
  distinct known moves, this pure solver finds the slot whose value tracks ``move_id`` linearly and
  recovers ``moves_base`` and ``move_stride``. A ``>= 3``-id consistency check is what kills a
  two-point coincidental line, so the result is grounded entirely in observed ``move_id`` values
  with no doc offset involved.

* **Phase 2 — grounding dumps.** With ``moves_base + move_stride`` the ``tk_move`` for any known id
  is ``moves_base + id * move_stride``. :func:`dump_move` prints its raw words (its cancel-list
  reference points at the real cancels array); :func:`locate_moves_base_holder` reverse-scans the
  heap for the object storing ``moves_base`` — the real ``tk_moveset`` header — and dumps its words,
  so the real ``moves_ptr``/``cancels_ptr``/count offsets can be read off directly; and
  :func:`find_cancels_ptr_offset` self-validates a header candidate by brute-forcing which header
  word, treated as a cancels array, reproduces the character's anchors under the existing decode
  gate — simultaneously confirming (or flagging) that the ``tk_cancel`` layout + command decode
  still hold on this build.

Read-only throughout (docs/02 §2): it resolves addresses, reads bytes, and follows pointers — the
grounding uses the game's own observable ``move_id`` as truth, and the dumps print only the target's
own bytes for our inspection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tekken_coach.reader.decode import read_scalar
from tekken_coach.reader.discovery.moveset_scan import _find_value_locations
from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.moveset import (
    CANCEL_SIZE,
    CANCELS_COUNT_MAX,
    CANCELS_COUNT_MIN,
    KnownPair,
    MovesetHeader,
    gate_pairs,
    read_cancels,
)
from tekken_coach.reader.slots import POINTER_SIZE, RegionIndex

# A real move index is ``< 0x8000``; ``0x8001`` (32769) is the idle/neutral **alias**, which is NOT
# an index into the moves array (brief #19). A sample carrying it is dropped before the solve so the
# neutral alias never poisons the linear fit.
REAL_MOVE_ID_MAX = 0x8000

# The solver needs at least this many distinct real ids: two to solve ``base`` + ``stride`` and one
# or more to reject a slot that fits only a coincidental two-point line.
MIN_SAMPLES = 3


def _read_u64(source: MemorySource, address: int) -> int:
    """Read an unsigned 64-bit value (a pointer or count word) at ``address`` (read-only)."""
    return int(read_scalar(source, address, "ptr"))


# ---------------------------------------------------------------------------
# Phase 1 — solve the moves array from move_id correlation (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveSample:
    """One live observation: the current ``move_id`` and every player pointer-slot's value then."""

    move_id: int
    slots: Mapping[int, int]


@dataclass(frozen=True)
class MovesArray:
    """The solved moves-array foothold: the tracking slot and the array's ``base`` + ``stride``."""

    slot_offset: int
    moves_base: int
    move_stride: int

    def move_addr(self, move_id: int) -> int:
        """The address of ``move_id``'s ``tk_move`` (``moves_base + move_id * move_stride``)."""
        return self.moves_base + move_id * self.move_stride


def _real_dedup(samples: Sequence[MoveSample]) -> list[MoveSample]:
    """One sample per distinct **real** ``move_id`` (drop the neutral alias); first-seen wins."""
    seen: dict[int, MoveSample] = {}
    for sample in samples:
        if sample.move_id >= REAL_MOVE_ID_MAX:
            continue  # the idle/neutral alias is not a moves-array index — never fit against it
        seen.setdefault(sample.move_id, sample)
    return list(seen.values())


def _fits_linear(points: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    """``(moves_base, stride)`` if the ``(id, value)`` points form one positive line, else ``None``.

    Derives the stride from the two lowest ids (an integer, positive slope) and then requires the
    line ``value == moves_base + id * stride`` to hold for **all** points. A single false two-point
    line cannot survive the third id — which is why the solver demands ``>= 3`` distinct ids.
    """
    ordered = sorted(points)
    (id0, v0), (id1, v1) = ordered[0], ordered[1]
    span_id = id1 - id0
    span_v = v1 - v0
    if span_id == 0 or span_v % span_id != 0:
        return None
    stride = span_v // span_id
    if stride <= 0:
        return None
    moves_base = v0 - id0 * stride
    for move_id, value in ordered[2:]:
        if value != moves_base + move_id * stride:
            return None
    return moves_base, stride


def solve_moves_array(samples: Sequence[MoveSample]) -> MovesArray | None:
    """Find the current-move pointer slot and solve ``moves_base`` + ``move_stride`` (Phase 1).

    For each slot offset present in **every** (real, deduped) sample, the slot is the current-move
    pointer iff its value ``v(id)`` satisfies ``v(id) == moves_base + id * move_stride`` for every
    sampled id, with ``move_stride > 0``. Returns the winning slot with its base and stride, or
    ``None`` when fewer than :data:`MIN_SAMPLES` distinct real ids were seen or no slot correlates.

    Deterministic: candidate slot offsets are tried in ascending order, so the lowest-offset slot
    wins if (implausibly) two slots both fit the same several-point line.
    """
    kept = _real_dedup(samples)
    if len(kept) < MIN_SAMPLES:
        return None

    shared_offsets = set(kept[0].slots)
    for sample in kept[1:]:
        shared_offsets &= set(sample.slots)

    for offset in sorted(shared_offsets):
        points = [(sample.move_id, sample.slots[offset]) for sample in kept]
        fit = _fits_linear(points)
        if fit is not None:
            moves_base, stride = fit
            return MovesArray(slot_offset=offset, moves_base=moves_base, move_stride=stride)
    return None


# ---------------------------------------------------------------------------
# Phase 2 — grounding dumps (read the real layout from a known tk_move / header)
# ---------------------------------------------------------------------------

# How many u64 words :func:`dump_move` prints from a ``tk_move`` (the real stride is unknown, so a
# generous window shows the cancel pointer/count pair wherever it sits).
DEFAULT_MOVE_WORDS = 24

# The window :func:`locate_moves_base_holder` dumps around the word that stores ``moves_base`` —
# enough either side that the neighbouring ``cancels_ptr``/count pairs of the real header show up.
DEFAULT_HOLDER_BACK = 0x80
DEFAULT_HOLDER_FWD = 0x40


def dump_move(
    source: MemorySource,
    moves: MovesArray,
    move_id: int,
    *,
    n_words: int = DEFAULT_MOVE_WORDS,
) -> str:
    """Dump the raw words of ``move_id``'s ``tk_move`` at ``moves.move_addr(move_id)`` (read-only).

    Grounds the ``tk_move`` layout (#18's unconfirmed owner-attribution offsets): somewhere in these
    words is this move's cancel-list reference (a pointer into the cancels array + a count), which
    is the path to the real cancels. Total: an unreadable word is shown as ``<unreadable>``, never
    raising, so even a slightly-off stride still yields a usable dump.
    """
    move_addr = moves.move_addr(move_id)
    lines = [
        f"tk_move dump for move_id {move_id} @ 0x{move_addr:x} "
        f"(moves_base=0x{moves.moves_base:x}, stride=0x{moves.move_stride:x}):"
    ]
    for i in range(n_words):
        off = i * POINTER_SIZE
        try:
            value = _read_u64(source, move_addr + off)
            lines.append(f"    +0x{off:03x}: 0x{value:016x}")
        except MemoryReadError:
            lines.append(f"    +0x{off:03x}: <unreadable>")
    return "\n".join(lines)


def locate_moves_base_holder(
    source: MemorySource,
    buffers: Sequence[Region],
    moves_base: int,
    *,
    back: int = DEFAULT_HOLDER_BACK,
    fwd: int = DEFAULT_HOLDER_FWD,
) -> str:
    """Reverse-scan the heap for the object storing ``moves_base`` and dump its words (read-only).

    Reuses #19's value-locate: every 8-aligned location that holds ``moves_base`` as a little-endian
    u64 is a candidate ``tk_moveset`` header (the header's ``moves_ptr`` field). For each, dumps a
    window ``[location - back, location + fwd)`` — the located word (offset ``0``) is the real
    ``moves_ptr``; the real ``cancels_ptr``/count pairs sit at nearby negative offsets. Total: an
    unreadable word inside the window is shown as ``<unreadable>``.
    """
    locations = _find_value_locations(buffers, moves_base)
    if not locations:
        return f"no heap object stores moves_base 0x{moves_base:x} (widen the scan or re-solve)."

    lines = [
        f"{len(locations)} object(s) store moves_base 0x{moves_base:x} "
        "(each a tk_moveset header candidate; offset 0 == the real moves_ptr):"
    ]
    for location in locations:
        lines.append(f"  header candidate around 0x{location:x}:")
        for off in range(-back, fwd, POINTER_SIZE):
            try:
                value = _read_u64(source, location + off)
                marker = "  <- moves_ptr (moves_base)" if off == 0 else ""
                lines.append(f"    {off:+#06x}: 0x{value:016x}{marker}")
            except MemoryReadError:
                lines.append(f"    {off:+#06x}: <unreadable>")
    return "\n".join(lines)


def find_cancels_ptr_offset(
    source: MemorySource,
    header_addr: int,
    pairs: Sequence[KnownPair],
    *,
    word_start: int = 0,
    word_end: int = 0x400,
    region_index: RegionIndex | None = None,
) -> int | None:
    """Brute-force which header word is the real ``cancels_ptr`` via the existing gate (Phase 2).

    Walks each 8-aligned offset in ``[word_start, word_end)`` of the header at ``header_addr``,
    treating that word as a ``tk_cancel*`` and the **next** word as its count (the documented
    ptr-then-count adjacency). A candidate is only read when its count is in the plausible cancels
    range (and, if a :class:`RegionIndex` is given, the pointer lands in a mapped region), then its
    cancels are gated on the character's anchors. The offset whose array reproduces **all** anchors
    *is* ``cancels_ptr`` — which simultaneously confirms the ``tk_cancel`` layout + command decode
    still hold on this build. Returns the first such offset, or ``None`` if none reproduce them.

    Read-only and total: an unreadable word is skipped, never raised, so a mis-sized window is safe.
    """
    tuple_pairs = tuple(pairs)
    for off in range(word_start, word_end, POINTER_SIZE):
        try:
            cancels_ptr = _read_u64(source, header_addr + off)
            count = _read_u64(source, header_addr + off + POINTER_SIZE)
        except MemoryReadError:
            continue
        if not (CANCELS_COUNT_MIN <= count <= CANCELS_COUNT_MAX):
            continue
        if region_index is not None and not region_index.contains(cancels_ptr, CANCEL_SIZE):
            continue
        header = MovesetHeader(
            cancels_ptr=cancels_ptr,
            cancels_count=count,
            moves_ptr=0,
            moves_count=0,
            input_sequences_ptr=0,
            input_sequences_count=0,
        )
        try:
            cancels = read_cancels(source, header)
        except MemoryReadError:
            continue
        if all(row.found for row in gate_pairs(cancels, tuple_pairs)):
            return off
    return None
