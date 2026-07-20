"""Ground the ``tk_moveset`` layout on the trusted live ``move_id`` (briefs #21-#25).

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

* **Phase 2, brief #25 — the per-move cancel run.** The 2026-07-19 run *solved* Phase 1 (the
  current-move pointer is ``player+0x3d8``, ``move_stride`` is ``0x448``) and its ``tk_move`` dumps
  produced a strong lead: a heap pointer at ``+0x098`` that differs per move, is ordered with
  ``move_id``, and whose gap between moves 1566 and 1695 is exactly ``853 * CANCEL_SIZE``. That fits
  one structure — *cancels stored contiguously, each move pointing at the start of its own run, the
  run ending where the next move's begins* — and :func:`cancel_range` tests it against its sharpest
  prediction: the span between consecutive pointers must be a whole multiple of ``CANCEL_SIZE``, so
  a non-multiple falsifies it outright. :func:`probe_cancel_run` then follows the pointer and gates
  what it finds against the character's known anchors, which would confirm the pointer's identity,
  the ``tk_cancel`` layout, and the command encoding together. If it holds, #18's open
  owner-attribution unknown closes. :func:`dump_move` also stops truncating (it defaults to the
  whole stride now — the 24-word dump showed 17% of a ``0x448`` record) and reports repeated small
  u32s with the ``move_id + K`` each implies, surfacing mechanically the ``+1893`` relationship that
  corroborated the solve by eye.

Read-only throughout (docs/02 §2): it resolves addresses, reads bytes, and follows pointers — the
grounding uses the game's own observable ``move_id`` as truth, and the dumps print only the target's
own bytes for our inspection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tekken_coach.framedata.moveset_decode import decode_command
from tekken_coach.reader.decode import read_scalar
from tekken_coach.reader.discovery.moveset_scan import _find_value_locations
from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.moveset import (
    CANCEL_COMMAND_OFFSET,
    CANCEL_MOVE_ID_OFFSET,
    CANCEL_SIZE,
    CANCELS_COUNT_MAX,
    CANCELS_COUNT_MIN,
    GateRow,
    KnownPair,
    MovesetHeader,
    RawCancel,
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

# The fallback word count for :func:`dump_move` when the stride is unknown. Brief #25 makes the
# **full stride** the default instead (``move_stride // 8`` words): the 2026-07-19 dump printed 24
# words = 0xc0 of a 0x448 record, i.e. 17% of it, and its trailing zeros at +0x0a0..+0x0b8 were
# where the *dump* stopped, not where the data did. A silently-truncated dump is how a second cancel
# reference or an explicit count would be missed.
DEFAULT_MOVE_WORDS = 24

# A u32 half is "small" — worth splitting out and testing for an id relationship — below this.
# Move ids and asset indices live in the low thousands; a pointer half or a float bit pattern does
# not, so this keeps the split annotation on the fields where it means something.
SMALL_U32_MAX = 0x10000

# How many distinct offsets a small u32 value must occupy within one record before it is reported as
# an id-related candidate. A value appearing once is unremarkable — every u64 trivially yields
# *some* ``move_id + K``. A value repeated **through** the record is the signal that surfaced the
# live ``+1893`` relationship (0x0e04 for move 1695, 0x0d83 for move 1566).
MIN_U32_REPEATS = 2

# The window :func:`locate_moves_base_holder` dumps around the word that stores ``moves_base`` —
# enough either side that the neighbouring ``cancels_ptr``/count pairs of the real header show up.
DEFAULT_HOLDER_BACK = 0x80
DEFAULT_HOLDER_FWD = 0x40


@dataclass(frozen=True)
class IdRelatedWord:
    """A small u32 repeated through one ``tk_move``, and its offset from that move's ``move_id``.

    The live dumps carried ``0x0e04`` (3588) through move 1695's record and ``0x0d83`` (3459)
    through move 1566's — both ``move_id + 1893``, the *same* ``K`` for two unrelated moves. That
    coincidence corroborated the solved ``moves_base``/``move_stride``, and it was found by eye.
    This surfaces it mechanically instead.

    One record alone cannot establish ``K``: every value yields some delta. What this reports is a
    **candidate** — a value repeated within the record, and what ``K`` it would imply. Dumping two
    different ids and seeing the same ``delta`` is the confirmation.
    """

    value: int
    delta: int  # value - move_id: the K this word would imply
    offsets: tuple[int, ...]  # every byte offset of the u32 half holding it

    def describe(self) -> str:
        """A one-line summary naming the implied ``K`` and where the value sat."""
        where = ", ".join(f"+0x{off:03x}" for off in self.offsets)
        return (
            f"0x{self.value:x} ({self.value}) = move_id {self.delta:+d} "
            f"at {len(self.offsets)} offsets [{where}]"
        )


def find_id_related_u32s(
    words: Sequence[tuple[int, int | None]],
    move_id: int,
    *,
    min_repeats: int = MIN_U32_REPEATS,
    max_value: int = SMALL_U32_MAX,
) -> list[IdRelatedWord]:
    """Small u32 halves repeated through a dumped record, with the ``K`` each would imply (pure).

    ``words`` is the dump's ``(byte_offset, u64 value or None)`` sequence. A word contributes only
    when **both** its u32 halves are below ``max_value`` — i.e. it is a pair of small fields rather
    than a pointer or float bits. That test is what keeps the report clean: a 64-bit heap pointer
    has a small *high* half (the live ones were ``0x2b4``), so collecting halves independently would
    surface every pointer's page bits as a spurious repeated value. Zero halves are skipped as
    padding. Values occupying at least ``min_repeats`` distinct offsets are returned, most-repeated
    first, ties by value — deterministic, so two dumps are directly comparable.

    Reports candidates, never a conclusion: see :class:`IdRelatedWord`.
    """
    offsets_by_value: dict[int, list[int]] = {}
    for offset, value in words:
        if value is None:
            continue
        halves = (value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF)
        if any(half >= max_value for half in halves):
            continue  # a pointer or a wide field, not a pair of small ones
        for half_index, half in enumerate(halves):
            if half > 0:
                offsets_by_value.setdefault(half, []).append(offset + half_index * 4)

    found = [
        IdRelatedWord(value=value, delta=value - move_id, offsets=tuple(offsets))
        for value, offsets in offsets_by_value.items()
        if len(offsets) >= min_repeats
    ]
    found.sort(key=lambda w: (-len(w.offsets), w.value))
    return found


def _read_move_words(
    source: MemorySource, move_addr: int, n_words: int
) -> list[tuple[int, int | None]]:
    """Read ``n_words`` u64s from ``move_addr`` as ``(byte_offset, value or None)`` (read-only).

    Total: an unreadable word yields ``None`` rather than raising, so a slightly-off stride or a
    record running past its mapping still produces a usable dump.
    """
    words: list[tuple[int, int | None]] = []
    for i in range(n_words):
        off = i * POINTER_SIZE
        try:
            words.append((off, _read_u64(source, move_addr + off)))
        except MemoryReadError:
            words.append((off, None))
    return words


def dump_move(
    source: MemorySource,
    moves: MovesArray,
    move_id: int,
    *,
    n_words: int | None = None,
) -> str:
    """Dump the raw words of ``move_id``'s ``tk_move`` at ``moves.move_addr(move_id)`` (read-only).

    Grounds the ``tk_move`` layout (#18's unconfirmed owner-attribution offsets): somewhere in these
    words is this move's cancel-list reference (a pointer into the cancels array), which is the path
    to the real cancels. ``n_words`` defaults to the **whole record** (``move_stride // 8``) so a
    dump is never silently truncated the way the 24-word 2026-07-19 dump was; pass it explicitly to
    override.

    Each word whose two u32 halves are both small is annotated with that split, and the trailing
    section reports which small u32s repeat through the record and what ``move_id + K`` each would
    imply (:func:`find_id_related_u32s`). Total: an unreadable word is shown as ``<unreadable>``,
    never raising.
    """
    if n_words is None:
        n_words = max(1, moves.move_stride // POINTER_SIZE)
    move_addr = moves.move_addr(move_id)
    words = _read_move_words(source, move_addr, n_words)

    lines = [
        f"tk_move dump for move_id {move_id} @ 0x{move_addr:x} "
        f"(moves_base=0x{moves.moves_base:x}, stride=0x{moves.move_stride:x}, "
        f"{n_words} words = 0x{n_words * POINTER_SIZE:x} bytes):"
    ]
    for off, value in words:
        if value is None:
            lines.append(f"    +0x{off:03x}: <unreadable>")
            continue
        low, high = value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF
        split = f"  (u32 {low}, {high})" if low < SMALL_U32_MAX and high < SMALL_U32_MAX else ""
        lines.append(f"    +0x{off:03x}: 0x{value:016x}{split}")

    related = find_id_related_u32s(words, move_id)
    if related:
        lines.append("  small u32s repeated through the record (each implies a candidate K):")
        lines.extend(f"    {word.describe()}" for word in related)
        lines.append(
            "  dump a second move_id: a K that is the SAME for both is a real id relationship."
        )
    else:
        lines.append("  no small u32 repeats through the record — no candidate K to report.")
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


# ---------------------------------------------------------------------------
# Phase 2 — the per-move cancel run (brief #25): test the contiguous-range hypothesis
# ---------------------------------------------------------------------------

# The ``tk_move`` word the 2026-07-19 dumps point at as this move's cancel-list pointer. A
# **hypothesis under test**, not a confirmed fact: both records held a heap pointer here, into an
# arena distinct from ``moves_base``, differing per move and ordered with ``move_id`` — and the gap
# between move 1566's and move 1695's (0x8548 across 129 moves) is exactly 853 * CANCEL_SIZE. Every
# entry point takes it as a parameter so a wrong guess is one flag away from corrected.
DEFAULT_CANCEL_PTR_OFFSET = 0x98

# How many rows :func:`probe_cancel_run` reads when no range bounds it (an unreadable neighbour, or
# an explicit override). Enough to reach a gate anchor if the pointer is real; small enough that a
# junk pointer costs nothing.
DEFAULT_PROBE_ROWS = 32

# The hard ceiling on rows read from one move's run, however large the computed span. A wrong
# ``ptr_offset`` can yield an enormous span; this keeps that a bounded, reportable non-result rather
# than a hang.
MAX_PROBE_ROWS = 4096


def read_cancel_ptr(
    source: MemorySource, moves: MovesArray, move_id: int, ptr_offset: int
) -> int | None:
    """Read ``move_id``'s cancel-run pointer at ``+ptr_offset``, or ``None`` if unreadable.

    Unreadable is **data** (the move index may not exist, or its record may run past the mapping),
    so the callers report it rather than raising.
    """
    try:
        return _read_u64(source, moves.move_addr(move_id) + ptr_offset)
    except MemoryReadError:
        return None


def read_cancel_row(source: MemorySource, row_addr: int) -> RawCancel | None:
    """Read one ``tk_cancel`` at ``row_addr`` in the confirmed layout, or ``None`` if unreadable."""
    try:
        command = _read_u64(source, row_addr + CANCEL_COMMAND_OFFSET)
        dest = int(read_scalar(source, row_addr + CANCEL_MOVE_ID_OFFSET, "u16"))
    except MemoryReadError:
        return None
    return RawCancel(command=command, dest_move_id=dest)


def _row_is_plausible(row: RawCancel) -> bool:
    """Whether a row decodes as a plausible ``tk_cancel`` — its direction bits are a modeled code.

    Deliberately structural rather than a notation check: a real cancel may legitimately decode to
    no notation (a motion input, a Heat engage), so requiring notation would reject genuine rows.
    What junk *cannot* do is land on a modeled direction code repeatedly — an arbitrary u64's low 32
    bits are almost never one of the eight documented codes or the two no-prefix values.
    """
    return not decode_command(row.command).unknown_direction


@dataclass(frozen=True)
class CancelRange:
    """Whether ``move_id``'s cancels are the contiguous run ``[ptr(N), ptr(N+1))`` (#25 Part C).

    The hypothesis: cancels are stored contiguously, each ``tk_move`` points at the start of its own
    run, and the run ends where the next move's run begins. Its sharpest prediction is that the span
    between consecutive moves' pointers is a **whole multiple of** :data:`CANCEL_SIZE` — a
    non-multiple falsifies it outright, which is why that is recorded separately from the count.

    Non-results are data, not errors: an unreadable pointer (``start``/``next_start`` ``None``), a
    zero span (a move with no cancels sharing its neighbour's pointer), and a negative span (the
    runs are not laid out in ascending ``move_id`` order) each get a verdict of their own.
    """

    move_id: int
    ptr_offset: int
    start: int | None
    next_start: int | None
    span: int | None
    count: int | None
    whole_multiple: bool
    n_rows_checked: int
    n_rows_plausible: int
    verdict: str

    @property
    def falsified(self) -> bool:
        """Whether this range **disproves** the contiguous-run hypothesis (a non-multiple span)."""
        return self.span is not None and not self.whole_multiple

    def report(self) -> str:
        """A multi-line summary: the two pointers, the span, and what it does to the hypothesis."""
        lines = [
            f"cancel range for move_id {self.move_id} (ptr at tk_move+0x{self.ptr_offset:x}):",
            f"  ptr(N)   = {'<unreadable>' if self.start is None else f'0x{self.start:x}'}",
            f"  ptr(N+1) = "
            f"{'<unreadable>' if self.next_start is None else f'0x{self.next_start:x}'}",
        ]
        if self.span is not None:
            multiple = "YES" if self.whole_multiple else "NO"
            lines.append(
                f"  span     = {self.span} bytes; whole multiple of 0x{CANCEL_SIZE:x}: {multiple}"
            )
        if self.count is not None:
            lines.append(f"  implied cancel count = {self.count}")
        if self.n_rows_checked:
            lines.append(
                f"  rows decoding as plausible tk_cancels: "
                f"{self.n_rows_plausible}/{self.n_rows_checked}"
            )
        lines.append(f"  {self.verdict}")
        return "\n".join(lines)


def cancel_range(
    source: MemorySource,
    moves: MovesArray,
    move_id: int,
    *,
    ptr_offset: int = DEFAULT_CANCEL_PTR_OFFSET,
    max_rows: int = MAX_PROBE_ROWS,
) -> CancelRange:
    """Test the contiguous-run hypothesis for ``move_id`` against move ``move_id + 1`` (read-only).

    Reads both moves' candidate cancel pointers, reports the byte span between them and whether it
    is a whole multiple of :data:`CANCEL_SIZE`, the implied cancel count, and — when the span is a
    positive whole multiple — how many of the rows in ``[ptr(N), ptr(N+1))`` decode as plausible
    ``tk_cancel``\\ s. Row reads are capped at ``max_rows`` so a wrong ``ptr_offset`` yielding an
    enormous span is a bounded non-result.

    Total: every failure mode above is returned as a :class:`CancelRange` with its own verdict.
    """
    start = read_cancel_ptr(source, moves, move_id, ptr_offset)
    next_start = read_cancel_ptr(source, moves, move_id + 1, ptr_offset)

    def result(
        span: int | None,
        count: int | None,
        whole: bool,
        checked: int,
        plausible: int,
        verdict: str,
    ) -> CancelRange:
        return CancelRange(
            move_id=move_id,
            ptr_offset=ptr_offset,
            start=start,
            next_start=next_start,
            span=span,
            count=count,
            whole_multiple=whole,
            n_rows_checked=checked,
            n_rows_plausible=plausible,
            verdict=verdict,
        )

    if start is None:
        return result(
            None, None, False, 0, 0, "NO VERDICT: this move's own pointer word is unreadable."
        )
    if next_start is None:
        return result(
            None,
            None,
            False,
            0,
            0,
            f"NO VERDICT: move {move_id + 1}'s pointer word is unreadable (a last move, or a "
            "record past the mapping) — the run's end is unknown, which bounds nothing.",
        )

    span = next_start - start
    whole = span % CANCEL_SIZE == 0
    if span < 0:
        return result(
            span,
            None,
            whole,
            0,
            0,
            "DATA, NOT A FALSIFICATION: the next move's run starts BEFORE this one's, so the runs "
            "are not laid out in ascending move_id order. The hypothesis needs a different "
            "end-of-run source.",
        )
    if not whole:
        return result(
            span,
            None,
            whole,
            0,
            0,
            f"FALSIFIED: {span} is not a whole multiple of 0x{CANCEL_SIZE:x} — consecutive "
            f"tk_cancel runs cannot start {span} bytes apart. Either +0x{ptr_offset:x} is not the "
            "cancel pointer, or the cancels are not stored contiguously.",
        )
    if span == 0:
        return result(
            span,
            0,
            whole,
            0,
            0,
            "EMPTY RUN: this move shares its pointer with the next, i.e. it owns no cancels. "
            "Consistent with the hypothesis; carries no evidence for it.",
        )

    count = span // CANCEL_SIZE
    checked = min(count, max_rows)
    plausible = 0
    for i in range(checked):
        row = read_cancel_row(source, start + i * CANCEL_SIZE)
        if row is not None and _row_is_plausible(row):
            plausible += 1
    if plausible == checked:
        verdict = (
            f"CONSISTENT: the span is exactly {count} tk_cancel rows and all {checked} checked "
            "decode plausibly."
        )
    elif plausible == 0:
        verdict = (
            f"INCONSISTENT: the span divides into {count} rows, but NONE of the {checked} checked "
            "decode as a plausible tk_cancel — a divisible span alone is weak evidence."
        )
    else:
        verdict = (
            f"PARTIAL: {plausible} of {checked} checked rows decode plausibly. Not a confirmation "
            "— report which rows, do not round up."
        )
    return result(span, count, whole, checked, plausible, verdict)


@dataclass(frozen=True)
class CancelRunProbe:
    """What the pointer at ``tk_move+ptr_offset`` actually points at — brief #25 Part B.

    The self-validating gate trick :func:`find_cancels_ptr_offset` uses, applied to a **per-move**
    pointer: if rows read at this pointer reproduce known ``(command -> destination move_id)``
    anchors, then the pointer's identity, the ``tk_cancel`` layout, and the T8 command encoding are
    confirmed together. Partial results are reported as partial — ``gate`` carries every anchor's
    own verdict, so a caller can never report a confirmation without naming which parts matched.
    """

    move_id: int
    ptr_offset: int
    ptr: int | None
    rows: tuple[RawCancel, ...]
    n_undecodable: int
    n_plausible: int
    gate: tuple[GateRow, ...]

    @property
    def matched(self) -> tuple[GateRow, ...]:
        """The anchors this run reproduced."""
        return tuple(row for row in self.gate if row.found)

    @property
    def unmatched(self) -> tuple[GateRow, ...]:
        """The anchors this run did not reproduce."""
        return tuple(row for row in self.gate if not row.found)

    @property
    def confirmed(self) -> bool:
        """Whether **every** anchor was reproduced — the only state that confirms the layout."""
        return bool(self.gate) and not self.unmatched

    def report(self) -> str:
        """A multi-line summary that states matched, unmatched, and undecodable separately."""
        if self.ptr is None:
            return (
                f"cancel run for move_id {self.move_id}: the pointer word at "
                f"tk_move+0x{self.ptr_offset:x} is unreadable — no probe possible."
            )
        lines = [
            f"cancel run for move_id {self.move_id} via tk_move+0x{self.ptr_offset:x} "
            f"-> 0x{self.ptr:x}:",
            f"  read {len(self.rows)} row(s); {self.n_plausible} decode as plausible tk_cancels, "
            f"{self.n_undecodable} yield no notation",
        ]
        for row in self.rows[:16]:
            note = decode_command(row.command).notation()
            lines.append(
                f"    command=0x{row.command:016x} -> dest {row.dest_move_id} "
                f"({note if note is not None else '<no notation>'})"
            )
        if len(self.rows) > 16:
            lines.append(f"    ... {len(self.rows) - 16} more")

        if not self.gate:
            lines.append("  no anchors supplied — nothing gated, so nothing is confirmed.")
            return "\n".join(lines)
        for anchor in self.gate:
            mark = "MATCH  " if anchor.found else "no     "
            lines.append(
                f"  {mark} move {anchor.move_id} expected {anchor.expected!r}; "
                f"this run decoded {anchor.decoded or '[]'}"
            )
        if self.confirmed:
            lines.append(
                f"  CONFIRMED: all {len(self.gate)} anchors reproduced — the pointer at "
                f"+0x{self.ptr_offset:x}, the tk_cancel layout, and the command encoding all hold."
            )
        elif self.matched:
            lines.append(
                f"  PARTIAL: {len(self.matched)}/{len(self.gate)} anchors reproduced. NOT a "
                "confirmation — the listed misses are unexplained."
            )
        else:
            lines.append(
                f"  NO MATCH: none of the {len(self.gate)} anchors were reproduced. Note that a "
                "per-move run holds that move's FOLLOW-UPS, so anchors are only expected in the "
                "neutral move's run — a miss here is weak evidence against the offset."
            )
        return "\n".join(lines)


def probe_cancel_run(
    source: MemorySource,
    moves: MovesArray,
    move_id: int,
    *,
    ptr_offset: int = DEFAULT_CANCEL_PTR_OFFSET,
    known_pairs: Sequence[KnownPair] = (),
    n_rows: int | None = None,
) -> CancelRunProbe:
    """Follow ``move_id``'s candidate cancel pointer and gate what it points at (read-only).

    Reads the pointer at ``tk_move+ptr_offset``, treats its target as a ``tk_cancel`` array using
    the existing :data:`CANCEL_SIZE` / offset constants and the #18 decoder, and gates the decoded
    ``(command -> dest)`` entries against ``known_pairs``. ``n_rows`` defaults to the count implied
    by :func:`cancel_range` (falling back to :data:`DEFAULT_PROBE_ROWS` when the range yields none),
    capped at :data:`MAX_PROBE_ROWS`.

    Total: an unreadable pointer or row ends the read and is reported, never raised — a wrong
    ``ptr_offset`` pointing into junk yields no matches rather than a crash.
    """
    ptr = read_cancel_ptr(source, moves, move_id, ptr_offset)
    if ptr is None:
        return CancelRunProbe(move_id, ptr_offset, None, (), 0, 0, ())

    if n_rows is None:
        implied = cancel_range(source, moves, move_id, ptr_offset=ptr_offset).count
        n_rows = implied if implied else DEFAULT_PROBE_ROWS
    n_rows = max(0, min(n_rows, MAX_PROBE_ROWS))

    rows: list[RawCancel] = []
    for i in range(n_rows):
        row = read_cancel_row(source, ptr + i * CANCEL_SIZE)
        if row is None:
            break  # the array ends (or was never one) — report what was readable
        rows.append(row)

    n_undecodable = sum(1 for row in rows if decode_command(row.command).notation() is None)
    n_plausible = sum(1 for row in rows if _row_is_plausible(row))
    gate = tuple(gate_pairs(rows, tuple(known_pairs))) if known_pairs else ()
    return CancelRunProbe(move_id, ptr_offset, ptr, tuple(rows), n_undecodable, n_plausible, gate)
