"""Pure pointer-slot classifier for ``probe-state --slots`` (brief #11 Stage 1).

Brief #10 swept the player struct ``0x0-0x1600`` byte-granular across two scripted live passes and
returned an evidenced negative: raw input is **not** stored flat on the entity struct. But that
sweep was structurally blind in a specific way, and this module exists to cover exactly that blind
spot:

    A change-sweep only finds *changing* bytes. A component object hangs off a **pointer**, and the
    pointer does not change. The region #10 dismissed as dead *because it never changed* —
    ``0x0-0x100`` — is precisely where a component slot would sit.

This is not speculation. ``players.components.transform`` is already in our own offset table at
``slot_offset`` **0x20** with ``pointer_path [8]``: a real component, reached by a constant pointer,
inside ``0x0-0x100``. Position is not in the entity struct for the same reason input may not be —
Tekken 8 is UE5, the entity is a pawn, and a pawn reaches its parts through pointers.

So Stage 1 asks a question a change-sweep cannot: *which 8-byte slots in the struct hold a plausible
heap pointer, and which of those look like a per-player component worth sweeping behind?*

Plausibility is deliberately mechanical (docs/02 §5 rule 2 — this module reports facts, never
meanings): a slot is a plausible pointer when its value is non-null, 8-byte aligned, and lands
inside a **committed readable region** per
:meth:`~tekken_coach.reader.memory_source.MemorySource.regions` (the ``VirtualQueryEx`` map — a
query that reads no contents and adds no write path, docs/02 §2).

What makes a plausible slot *worth chasing* is then three observations, all facts:

* **both** — it resolves for P1 *and* P2. The structs are symmetric (``transform`` does this), so a
  component slot should.
* **per-player** — P1 and P2 point at *different* objects. A slot where both players hold the same
  address is an engine-global (a world, a class descriptor), not per-player state, and cannot carry
  *this* player's input.
* **stable** — the value holds across polls. A slot that churns is not a component anchor.

The live enumeration is a thin shell in ``commands``; everything decidable is here, tested against a
:class:`~tekken_coach.reader.memory_source.FakeMemorySource` with planted regions.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from tekken_coach.reader.memory_source import MemoryRegion

# x86-64: pointers are 8 bytes, and a pointer slot in a struct is 8-byte aligned. The sweep steps by
# this, so a "pointer" straddling the alignment is not a pointer we care about.
POINTER_SIZE = 8

# Below this, an aligned integer is a small number that happens to look aligned (a count, a flags
# word), not an address. No region maps the null page, so the region check would reject these
# anyway — this is a cheap pre-filter that keeps the bisect off the hot path of a 5 KB sweep.
MIN_POINTER_VALUE = 0x10000

# The default region of the player struct to enumerate. #10 proved the *flat* bytes here are dead;
# that is exactly why the pointers here are unexplored, and where `transform`'s slot (0x20) lives.
DEFAULT_SLOT_START = 0x0
DEFAULT_SLOT_END = 0x1600


class RegionIndex:
    """A sorted, bisect-searchable view of the committed regions — the pointer-validity oracle.

    Built once per run from :meth:`~tekken_coach.reader.memory_source.MemorySource.regions`, then
    asked once per candidate slot. A struct sweep asks this hundreds of times, so the lookup is
    ``O(log n)`` over the region bases rather than a linear walk of a few thousand regions.

    Read-only by construction: it holds ``(base, size)`` spans only. It never reads their contents —
    that is the caller's job, through the source (docs/02 §2).
    """

    def __init__(self, regions: Sequence[MemoryRegion]) -> None:
        ordered = sorted(regions, key=lambda r: r.base)
        self._bases = [r.base for r in ordered]
        self._ends = [r.end for r in ordered]

    def _covering(self, address: int) -> int | None:
        """Index of the region containing ``address``, or ``None``."""
        # bisect_right finds the first region starting *after* the address; the only region that can
        # cover it is the one before that.
        i = bisect_right(self._bases, address) - 1
        if i < 0 or address >= self._ends[i]:
            return None
        return i

    def contains(self, address: int, size: int = POINTER_SIZE) -> bool:
        """Whether ``[address, address+size)`` lies wholly inside one committed region."""
        i = self._covering(address)
        return i is not None and address + size <= self._ends[i]

    def room_at(self, address: int) -> int:
        """Bytes from ``address`` to the end of its region (0 if unmapped).

        This bounds Stage 2 honestly: it is how far behind a slot a sweep can read without running
        off the mapping. It is *not* the object's size — the allocation almost certainly ends well
        before the region does — so it is an upper bound, reported as one.
        """
        i = self._covering(address)
        return 0 if i is None else self._ends[i] - address


def pointer_candidates(block: bytes, *, start: int = 0) -> Iterator[tuple[int, int]]:
    """Yield ``(offset, value)`` for every 8-byte-aligned slot in ``block``.

    ``start`` is the struct offset ``block[0]`` came from, so the yielded offsets are
    struct-relative (what the offset table and a ``--watch-behind`` spec name) rather than
    block-relative. Values are unsigned little-endian; plausibility is
    :func:`is_plausible_pointer`'s call, not this one's.
    """
    if start % POINTER_SIZE:
        raise ValueError(f"slot sweep must start 8-byte aligned, got 0x{start:x}")
    for i in range(0, len(block) - POINTER_SIZE + 1, POINTER_SIZE):
        yield start + i, int.from_bytes(block[i : i + POINTER_SIZE], "little")


def is_plausible_pointer(value: int, regions: RegionIndex) -> bool:
    """Whether ``value`` could be a heap pointer: non-null, aligned, and inside a mapped region.

    Deliberately three mechanical tests and no heuristics about what the target *contains* — this
    module reports that a slot holds a readable address, not that it holds a component.
    """
    if value < MIN_POINTER_VALUE or value % POINTER_SIZE:
        return False
    return regions.contains(value)


@dataclass(frozen=True)
class SlotFinding:
    """One struct slot's verdict across the polled samples, for both players.

    ``values`` and ``plausible`` are indexed by player slot (``[0]`` = P1, ``[1]`` = P2), holding
    that player's **last observed** value. ``stable`` says the value never moved across polls, for
    every player — a component anchor should not churn.
    """

    offset: int
    values: tuple[int, ...]
    plausible: tuple[bool, ...]
    stable: bool

    @property
    def both(self) -> bool:
        """Whether the slot is a plausible pointer for *every* player (structs are symmetric)."""
        return all(self.plausible)

    @property
    def per_player(self) -> bool:
        """Whether the players point at **different** objects — the per-player component signature.

        A slot plausible for both players but holding the *same* address is shared engine state (a
        world, a class descriptor). It cannot carry this player's input, so it ranks below a slot
        that differs.
        """
        return self.both and len(set(self.values)) == len(self.values)

    @property
    def chase(self) -> bool:
        """Whether Stage 2 should sweep behind this slot: per-player, both players, and stable.

        The brief's fan-out bound in one predicate. Everything else is reported but not chased.
        """
        return self.per_player and self.stable

    @property
    def rank_key(self) -> tuple[int, int, int, int]:
        """Sort key (descending): chase-worthiness first, then offset ascending."""
        return (-int(self.chase), -int(self.per_player), -int(self.both), self.offset)

    def label(self) -> str:
        """The ``--watch-behind`` slot label for this finding (``0x20``)."""
        return f"0x{self.offset:x}"


def classify_slots(
    samples: Sequence[Sequence[bytes]],
    regions: RegionIndex,
    *,
    start: int = DEFAULT_SLOT_START,
) -> list[SlotFinding]:
    """Classify every aligned slot across ``samples``, ranked with the chase-worthy first.

    ``samples`` is one entry per poll, each a per-player list of that player's struct **block** (one
    ``ReadProcessMemory`` per player per poll — see
    :func:`~tekken_coach.reader.probe.block_span`). All blocks must be the same length: they are the
    same region of two symmetric structs.

    Only slots plausible for at least one player are returned — a struct is mostly not pointers, and
    a table of 700 dead slots is not a table anyone reads.
    """
    if not samples:
        return []
    widths = {len(block) for poll in samples for block in poll}
    if len(widths) > 1:
        raise ValueError(f"slot samples must all be the same width, got {sorted(widths)}")
    players = len(samples[0])
    if any(len(poll) != players for poll in samples):
        raise ValueError("every poll must sample the same number of players")

    # offset -> per-player list of the values seen across polls, in poll order.
    seen: dict[int, list[list[int]]] = {}
    for poll in samples:
        for index, block in enumerate(poll):
            for offset, value in pointer_candidates(block, start=start):
                seen.setdefault(offset, [[] for _ in range(players)])[index].append(value)

    findings: list[SlotFinding] = []
    for offset, per_player in sorted(seen.items()):
        values = tuple(history[-1] for history in per_player)
        plausible = tuple(is_plausible_pointer(v, regions) for v in values)
        if not any(plausible):
            continue
        stable = all(len(set(history)) == 1 for history in per_player)
        findings.append(
            SlotFinding(offset=offset, values=values, plausible=plausible, stable=stable)
        )
    return sorted(findings, key=lambda f: f.rank_key)


def _flag(finding: SlotFinding) -> str:
    """The one-word verdict column: why (or why not) this slot is worth a Stage 2 sweep."""
    if finding.chase:
        return "CHASE"
    if finding.per_player:
        return "churns"
    if finding.both:
        return "shared"
    return "one-side"


def format_slot_table(
    findings: Sequence[SlotFinding], regions: RegionIndex, *, top: int = 0
) -> Iterator[str]:
    """Render the ranked slot table — Stage 1's whole output, and Stage 2's input.

    ``room`` is the bytes from the landing to the end of its mapped region: an upper bound on how
    far behind the slot a sweep can read (see :meth:`RegionIndex.room_at`), not the object's size.
    """
    shown = list(findings[:top] if top > 0 else findings)
    chase = [f for f in findings if f.chase]
    yield f"{len(findings)} plausible pointer slots; {len(chase)} worth chasing (CHASE)"
    yield ""
    yield f"{'slot':>8}  {'verdict':<8}  {'P1':>18}  {'P2':>18}  {'room(P1)':>10}"
    for f in shown:
        p1, p2 = (f"0x{v:x}" if ok else "-" for v, ok in zip(f.values, f.plausible, strict=True))
        room = regions.room_at(f.values[0]) if f.plausible[0] else 0
        yield f"{f.label():>8}  {_flag(f):<8}  {p1:>18}  {p2:>18}  {room:>10}"
    if top > 0 and len(findings) > top:
        yield f"... and {len(findings) - top} more (pass --top 0 for all)"
    yield ""
    if chase:
        spec = ",".join(f"{f.label()}:0x0-0x100:u8" for f in chase[:4])
        yield "Stage 2 — sweep behind the ranked slots while following `input-protocol`:"
        yield f'  probe-state --watch-behind "{spec}" --record debug/behind-1.jsonl'
    else:
        yield "No slot is per-player + stable. Nothing here is worth a Stage 2 press-through pass;"
        yield "input is likely off the pawn entirely (the UE PlayerController path)."
