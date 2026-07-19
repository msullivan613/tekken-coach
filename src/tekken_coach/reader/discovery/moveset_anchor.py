"""Ground the ``tk_moveset`` layout on the trusted live ``move_id`` (briefs #21, #22).

The 2026-07-19 dumps proved the doc-derived ``tk_moveset`` offsets are wrong for the live 5.02.01
build: the blind shape scan (#18-#20) was systematically matching shader/mesh objects because the
header offsets it read were for a different build (``cancels_ptr @ 0x1d0`` landed in DXBC/DXIL
data). Guessing new offsets from those false positives is unsound — they are not movesets.

So this module stops trusting the docs and grounds the layout on the one fact we fully trust: the
live ``move_id``. The reader already reads ``move_id @ 0x550`` and it is verified against
``bryan.json``. ``move_id`` is an **index into the moveset's ``moves`` array**, so the live value is
our foothold.

Brief #22 widens the *sampling surface*. Two clean #21 live runs (jab/d+4/b+3, then the
no-direction committed set) captured ``move_id`` fine but found **no direct player-struct slot** in
``0x0-0x1600`` tracking it linearly. The transitional-noise explanation was ruled out (the second
run used only committed moves), so the current-move pointer is not a direct slot in that window — it
almost certainly lives **one hop out**, in the animation/move sub-object the player struct points
at (or past the window). So the solver's key type is generalised from a bare direct offset to a
composite :data:`SlotKey`: ``(offset,)`` for a direct slot, ``(parent_offset, sub_offset)`` for a
slot one hop out. The linear-fit math is unchanged — it works on ``(id, value)`` points regardless
of where the value came from — but it is now outlier-tolerant (a single stray sample no longer sinks
a genuine slot), and when nothing solves it emits a diagnostic (:func:`describe_slots`) reporting
what *did* move, so a failed run still tells us whether anything tracks ``move_id`` at all.

Grounded steps, all offline-testable here (the live sampling shell in ``commands.py`` only gathers
the samples and prints the results):

* **Phase 1 — :func:`sample_player_slots` + :func:`solve_moves_array`.** The game holds a pointer to
  the *current* move's ``tk_move`` so it can animate it; that pointer equals
  ``moves_base + move_id * move_stride``. :func:`sample_player_slots` reads every plausible direct
  pointer slot in the (widened) player struct **and** dereferences each one hop to sample the
  plausible pointers inside that sub-object, keyed by composite :data:`SlotKey`. Given samples of
  ``(move_id, {slot_key: pointer_value})`` taken while the user performs several distinct known
  moves, :func:`solve_moves_array` finds the slot key whose value tracks ``move_id`` linearly and
  recovers ``moves_base`` and ``move_stride``. A ``>= 3``-id-on-one-line floor kills a two-point
  coincidental line; a strong-majority test lets one stray point be tolerated. When no slot key
  solves, :func:`describe_slots` classifies every sampled slot (constant / non-linear / linear over
  K-of-N ids) so the failure is signal, not just a dead end. Brief #23 makes that failure *honest*:
  the sampler also returns a :class:`SampleCensus` of what it actually swept, because a sweep that
  read nothing and a sweep that found everything constant otherwise produce the same empty
  diagnostic — and only the second is a reason to pivot.

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
from tekken_coach.reader.slots import (
    DEFAULT_SLOT_START,
    POINTER_SIZE,
    RegionIndex,
    is_plausible_pointer,
    pointer_candidates,
)

# A real move index is ``< 0x8000``; ``0x8001`` (32769) is the idle/neutral **alias**, which is NOT
# an index into the moves array (brief #19). A sample carrying it is dropped before the solve so the
# neutral alias never poisons the linear fit.
REAL_MOVE_ID_MAX = 0x8000

# The solver needs at least this many distinct real ids **on one line**: two to solve ``base`` +
# ``stride`` and one or more to reject a slot that fits only a coincidental two-point line. A floor,
# not a total — with outlier tolerance (brief #22) the line may skip a stray sample, but it
# must still cover at least this many ids to be believed.
MIN_SAMPLES = 3

# A composite key naming *where* a sampled pointer came from (brief #22). ``(offset,)`` is a direct
# player-struct slot; ``(parent_offset, sub_offset)`` is a slot one hop out — inside the sub-object
# the player slot ``parent_offset`` points at. Uniform, hashable, sortable; the empty tuple ``()``
# marks a ``MovesArray`` whose base/stride were supplied, not solved.
SlotKey = tuple[int, ...]

# How far to sweep the player struct for direct pointer slots. Widened past the old 0x1600 (the
# struct extends beyond the known fields — damage is at 0x1578) because #21 proved the current-move
# pointer is not a direct slot in 0x0-0x1600 (brief #22). Configurable via the CLI ``--window``.
DEFAULT_DIRECT_END = 0x4000

# The bounded sub-window swept inside each dereferenced sub-object for the one-hop pass. Kept small:
# the tracking pointer, if cached one hop out, sits near the head of the animation/move object.
DEFAULT_HOP_END = 0x800

# The narrowest direct window the sweep will step down to before giving up (brief #23). 0x1600 is
# the #21 window we know reads cleanly on the live struct, so a widened sweep that overruns the
# struct's region still returns the slots the old window would have — degraded, never empty.
MIN_DIRECT_WIDTH = 0x1600


def _read_u64(source: MemorySource, address: int) -> int:
    """Read an unsigned 64-bit value (a pointer or count word) at ``address`` (read-only)."""
    return int(read_scalar(source, address, "ptr"))


# ---------------------------------------------------------------------------
# Phase 1 — solve the moves array from move_id correlation (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveSample:
    """One live observation: the current ``move_id`` and every sampled pointer's value then.

    ``slots`` is keyed by :data:`SlotKey`, so a direct player slot ``(offset,)`` and a one-hop slot
    ``(parent_offset, sub_offset)`` sit in the same map — the solver treats them uniformly.
    """

    move_id: int
    slots: Mapping[SlotKey, int]


@dataclass(frozen=True)
class MovesArray:
    """The solved moves-array foothold: the tracking slot and the array's ``base`` + ``stride``."""

    slot_key: SlotKey
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


def _best_line(points: Sequence[tuple[int, int]]) -> tuple[int, int, int] | None:
    """The integer line best fitting the ``(id, value)`` points: ``(base, stride, n_on_line)``.

    Robust to outliers (brief #22): rather than demand *all* points lie on one line, this tries the
    line through every pair of distinct-id points — each pair defines a candidate integer
    ``stride > 0`` and ``moves_base`` — and returns the line ``value == moves_base + id*stride``
    holds for on the **most** ids. So one stray sample (e.g. a transient movement-animation id) no
    longer sinks a genuine slot: the true line still collects every other id. Returns ``None`` when
    no pair yields a positive-integer-stride line (all values equal, or none collinear).

    Deterministic: ties in the inlier count are broken by the lowest ``(base, stride)``.
    """
    ordered = sorted(points)
    best: tuple[int, int, int] | None = None
    for i in range(len(ordered)):
        id_i, v_i = ordered[i]
        for j in range(i + 1, len(ordered)):
            id_j, v_j = ordered[j]
            span_id = id_j - id_i
            span_v = v_j - v_i
            if span_id == 0 or span_v % span_id != 0:
                continue
            stride = span_v // span_id
            if stride <= 0:
                continue
            base = v_i - id_i * stride
            on_line = sum(1 for move_id, value in ordered if value == base + move_id * stride)
            candidate = (base, stride, on_line)
            if (
                best is None
                or on_line > best[2]
                or (on_line == best[2] and (base, stride) < (best[0], best[1]))
            ):
                best = candidate
    return best


def _is_strong_majority(on_line: int, total: int) -> bool:
    """Whether ``on_line`` of ``total`` distinct ids is a strong-enough majority to believe the fit.

    A strict majority (``> half``). Combined with the :data:`MIN_SAMPLES` floor this tolerates one
    stray point (4 of 5) while still rejecting a slot that fits only a coincidental minority.
    """
    return on_line * 2 > total


def solve_moves_array(samples: Sequence[MoveSample]) -> MovesArray | None:
    """Find the current-move pointer slot and solve ``moves_base`` + ``move_stride`` (Phase 1).

    For each slot key present in **every** (real, deduped) sample, the slot is the current-move
    pointer iff its value ``v(id)`` satisfies ``v(id) == moves_base + id * move_stride`` for a line
    covering at least :data:`MIN_SAMPLES` ids *and* a strong majority of them (with
    ``move_stride > 0``). Outlier-tolerant (brief #22): a single stray sample off the line no longer
    disqualifies an otherwise-linear slot. Returns the winning slot key with its base and stride, or
    ``None`` when fewer than :data:`MIN_SAMPLES` distinct real ids were seen or no slot correlates.

    Deterministic: the slot with the **most** ids on its line wins; ties break to the lowest slot
    key, so the result never depends on dict iteration order.
    """
    kept = _real_dedup(samples)
    if len(kept) < MIN_SAMPLES:
        return None

    shared_keys = set(kept[0].slots)
    for sample in kept[1:]:
        shared_keys &= set(sample.slots)

    total_ids = len(kept)
    best: tuple[int, SlotKey, int, int] | None = None  # (n_on_line, key, base, stride)
    for key in sorted(shared_keys):
        points = [(sample.move_id, sample.slots[key]) for sample in kept]
        line = _best_line(points)
        if line is None:
            continue
        base, stride, on_line = line
        if on_line < MIN_SAMPLES or not _is_strong_majority(on_line, total_ids):
            continue
        if best is None or on_line > best[0] or (on_line == best[0] and key < best[1]):
            best = (on_line, key, base, stride)

    if best is None:
        return None
    _, key, base, stride = best
    return MovesArray(slot_key=key, moves_base=base, move_stride=stride)


# ---------------------------------------------------------------------------
# Phase 1 — sample the player's pointers (direct + one hop out) and name where each came from
# ---------------------------------------------------------------------------


def format_slot_key(key: SlotKey) -> str:
    """Human-readable path for a :data:`SlotKey` (``player+0x30 -> object+0x18``)."""
    if not key:
        return "given (base/stride supplied, not solved)"
    if len(key) == 1:
        return f"player+0x{key[0]:x}"
    if len(key) == 2:
        return f"player+0x{key[0]:x} -> object+0x{key[1]:x}"
    return "player+" + " -> ".join(f"0x{part:x}" for part in key)


def _read_sub_object(
    source: MemorySource,
    regions: RegionIndex,
    base: int,
    hop_start: int,
    hop_end: int,
) -> bytes | None:
    """Read a bounded window inside the object at ``base`` for the one-hop pass, or ``None``.

    Bounded so a bad or short pointer can't spin: the read is clamped to the room left in the
    landing's mapped region (:meth:`RegionIndex.room_at`) and rounded down to pointer alignment, and
    an unreadable window yields ``None`` rather than raising. Read-only.
    """
    start = base + hop_start
    room = regions.room_at(start)
    if room < POINTER_SIZE:
        return None
    width = min(hop_end - hop_start, room)
    width -= width % POINTER_SIZE
    if width < POINTER_SIZE:
        return None
    try:
        return source.read(start, width)
    except MemoryReadError:
        return None


@dataclass(frozen=True)
class SampleCensus:
    """What one :func:`sample_player_slots` sweep actually did — the honest-failure record (#23).

    Without this, an empty sample is ambiguous: a sweep that read **nothing** (the widened window
    overran the struct's region and every fallback read failed) and a sweep that read a hundred
    thousand words and found every one constant produce the same empty diagnostic — and the second
    means "pivot the approach" while the first means "the read is broken". The census separates
    them, and ``direct_bytes_read`` records the width that actually succeeded rather than the one
    requested. Invariant: ``direct_pointers + hop_pointers`` is the number of sampled slot keys.
    """

    direct_bytes_read: int
    direct_slots_scanned: int
    direct_pointers: int
    sub_objects_read: int
    sub_objects_skipped: int
    hop_pointers: int

    @property
    def total_keys(self) -> int:
        """The number of slot keys the sweep yielded (direct + one hop out)."""
        return self.direct_pointers + self.hop_pointers

    def one_line(self) -> str:
        """A compact summary for the per-capture line (``read 0x4000, 412 direct ptrs, ...``)."""
        return (
            f"read 0x{self.direct_bytes_read:x}, {self.direct_pointers} direct ptrs, "
            f"{self.hop_pointers} one-hop ptrs"
        )

    def report(self) -> str:
        """The full multi-line census, for the preflight ``--census`` mode and a failed run."""
        return "\n".join(
            [
                f"  direct window read : 0x{self.direct_bytes_read:x} bytes",
                f"  direct slots scanned: {self.direct_slots_scanned}",
                f"  direct pointers    : {self.direct_pointers}",
                f"  sub-objects read   : {self.sub_objects_read} "
                f"(skipped {self.sub_objects_skipped})",
                f"  one-hop pointers   : {self.hop_pointers}",
                f"  total slot keys    : {self.total_keys}",
            ]
        )

    def starvation_hint(self, *, threshold: float = 0.01) -> str | None:
        """A warning when implausibly few direct slots passed validation, else ``None``.

        A live C++ game object is dense with pointers — vtable, components, owner back-references —
        so a plausible-pointer rate under ~1% of scanned slots means the *validity oracle* rejected
        them, not that the struct is empty. That is exactly how brief #24's starvation presented
        (13 of 2048), and it read as "nothing to correlate" rather than as a broken filter. Naming
        the cause here keeps the next occurrence from being re-diagnosed from scratch.
        """
        if self.direct_slots_scanned == 0 or self.direct_pointers == 0:
            return None
        if self.direct_pointers >= threshold * self.direct_slots_scanned:
            return None
        return (
            f"WARNING: only {self.direct_pointers} of {self.direct_slots_scanned} direct slots "
            "held a plausible pointer. A live game object should be far denser — suspect "
            "pointer-validation COVERAGE (the region map above), not the struct. Pointers into "
            "large arenas or module images are rejected if the map validated against is capped."
        )


EMPTY_CENSUS = SampleCensus(0, 0, 0, 0, 0, 0)


def _direct_width_candidates(requested: int, room: int) -> list[int]:
    """The widths to try for the direct window, best first (brief #23 step-down).

    The requested window first; then the room left in the struct's mapped region (a widened window
    can overrun it); then successive halvings, floored at :data:`MIN_DIRECT_WIDTH` — the #21 window
    known to read. All pointer-aligned and deduped, so a narrow request degrades to a single try.
    """
    widths: list[int] = []

    def add(width: int) -> None:
        width -= width % POINTER_SIZE
        if POINTER_SIZE <= width <= requested and width not in widths:
            widths.append(width)

    add(requested)
    if room > 0:
        add(min(requested, room))
    step = requested // 2
    while step > MIN_DIRECT_WIDTH:
        add(step)
        step //= 2
    add(min(requested, MIN_DIRECT_WIDTH))
    return widths


def sample_player_slots(
    source: MemorySource,
    player_base: int,
    regions: RegionIndex,
    *,
    direct_start: int = DEFAULT_SLOT_START,
    direct_end: int = DEFAULT_DIRECT_END,
    hop_start: int = 0x0,
    hop_end: int = DEFAULT_HOP_END,
) -> tuple[dict[SlotKey, int], SampleCensus]:
    """Sample every plausible pointer in the player struct **and one hop out** (composite keys).

    For each 8-aligned direct slot in ``[direct_start, direct_end)`` whose value is a plausible heap
    pointer, records it under ``(offset,)`` and then dereferences it, sampling the plausible
    pointers inside a bounded sub-window ``[hop_start, hop_end)`` of that object under
    ``(offset, sub_offset)`` (brief #22 — the current-move pointer is not a direct slot, so it is
    sought one hop out). Returns the slots **and** a :class:`SampleCensus` of what the sweep did, so
    an empty result can be told apart from a sweep that read nothing (brief #23).

    Read-only and total: the direct window steps down through
    :func:`_direct_width_candidates` when the requested width overruns the struct's region — every
    attempt guarded, so exhausting them yields empty slots and a zeroed census rather than raising —
    and an unreadable sub-object is simply skipped.
    """
    slots: dict[SlotKey, int] = {}
    requested = direct_end - direct_start
    room = regions.room_at(player_base + direct_start)
    block: bytes | None = None
    for width in _direct_width_candidates(requested, room):
        try:
            block = source.read(player_base + direct_start, width)
            break
        except MemoryReadError:
            continue  # the window overruns the mapping — try the next, narrower width
    if block is None:
        return slots, EMPTY_CENSUS

    scanned = pointers = subs_read = subs_skipped = hops = 0
    for offset, value in pointer_candidates(block, start=direct_start):
        scanned += 1
        if not is_plausible_pointer(value, regions):
            continue
        pointers += 1
        slots[(offset,)] = value
        sub = _read_sub_object(source, regions, value, hop_start, hop_end)
        if sub is None:
            subs_skipped += 1
            continue
        subs_read += 1
        for sub_offset, sub_value in pointer_candidates(sub, start=hop_start):
            if is_plausible_pointer(sub_value, regions):
                slots[(offset, sub_offset)] = sub_value
                hops += 1
    census = SampleCensus(
        direct_bytes_read=len(block),
        direct_slots_scanned=scanned,
        direct_pointers=pointers,
        sub_objects_read=subs_read,
        sub_objects_skipped=subs_skipped,
        hop_pointers=hops,
    )
    return slots, census


# ---------------------------------------------------------------------------
# Phase 1 diagnostic — when no slot solves, report what *did* track move_id (brief #22)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlotDescription:
    """How one sampled slot key behaved across the observed ids — the failed-solve diagnostic.

    ``kind`` is ``"constant"`` (never changed), ``"linear"`` (its best line covers **every** id it
    was present for), or ``"nonlinear"`` (its best line skips some ids — indirection or a variable
    stride). ``n_on_line`` / ``n_ids`` is the K/N of that best line; ``moves_base`` /
    ``move_stride`` are that line (``None`` for a constant slot, or when no integer line fits).
    """

    slot_key: SlotKey
    kind: str
    n_ids: int
    n_on_line: int
    moves_base: int | None
    move_stride: int | None
    points: tuple[tuple[int, int], ...]

    @property
    def _rank(self) -> tuple[int, int, int, SlotKey]:
        """Sort key: varying slots (most-linear first) lead; constant slots trail."""
        return (int(self.kind == "constant"), -self.n_on_line, -self.n_ids, self.slot_key)

    def describe(self) -> str:
        """A one-line human summary, including the ``(id -> value)`` points for a varying slot."""
        loc = format_slot_key(self.slot_key)
        if self.kind == "constant":
            return f"{loc}: constant 0x{self.points[0][1]:x} across {self.n_ids} ids"
        points = ", ".join(f"{move_id}->0x{value:x}" for move_id, value in self.points)
        if self.moves_base is None:
            return f"{loc}: varied non-linearly across {self.n_ids} ids ({points})"
        shape = "linear" if self.kind == "linear" else "non-linear"
        return (
            f"{loc}: {shape}, best line fits {self.n_on_line}/{self.n_ids} ids "
            f"(base=0x{self.moves_base:x} stride=0x{self.move_stride:x}) [{points}]"
        )


def describe_slots(samples: Sequence[MoveSample]) -> list[SlotDescription]:
    """Classify each sampled slot key by how its value tracked ``move_id`` (failed-solve report).

    For each slot key present in at least two (real, deduped) samples, reports whether its value was
    constant, varied non-linearly, or fit a line over K of N ids. Ranked so the varying slots come
    first (most ids on a line first) and constants trail — the caller prints the top few varying
    slots. This turns a failed solve into signal: a slot that varies with ``move_id`` but not
    linearly points at indirection/variable stride to chase next; **no** varying slot at all means
    nothing reachable this way caches a ``moves_base + id*stride`` pointer, so we pivot rather than
    search deeper blindly.
    """
    kept = _real_dedup(samples)
    per_key: dict[SlotKey, list[tuple[int, int]]] = {}
    for sample in kept:
        for key, value in sample.slots.items():
            per_key.setdefault(key, []).append((sample.move_id, value))

    descriptions: list[SlotDescription] = []
    for key, raw_points in per_key.items():
        points = tuple(sorted(raw_points))
        if len(points) < 2:
            continue  # a single observation says nothing about how the slot tracks move_id
        n_ids = len(points)
        if len({value for _, value in points}) == 1:
            descriptions.append(SlotDescription(key, "constant", n_ids, n_ids, None, None, points))
            continue
        line = _best_line(points)
        if line is None:
            descriptions.append(SlotDescription(key, "nonlinear", n_ids, 1, None, None, points))
            continue
        base, stride, on_line = line
        kind = "linear" if on_line == n_ids else "nonlinear"
        descriptions.append(SlotDescription(key, kind, n_ids, on_line, base, stride, points))

    descriptions.sort(key=lambda d: d._rank)
    return descriptions


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
