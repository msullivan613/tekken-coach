"""Derive the Tekken 8 player layout from observed behavior — no seeded offsets (C4h, docs/02 §3).

C4d/C4e locate the heap-allocated entity struct by scanning static data for a pointer whose *seeded*
chain (``0x10→0x68→0x8→0x30``) lands on a struct with a *seeded* ``char_id`` at +0x168 and
``move_id``
at +0x528. On build 5.02.01 those seeds are stale — the fork they came from died Oct 2024, ~1.75y of
patches ago — and a fair windowed run (C4g) found 0 of 13 structural candidates behaving. The
tooling
was sound; only the seeded facts were wrong. C4h removes the dependence entirely:

* **Phase 2 (:func:`locate_entity_layout`)** finds the entity struct by *behavior*, not by a seeded
  offset. It sweeps the enumerated **heap** (C4h Phase 1 :meth:`~...MemorySource.regions`) for the
  ``char_id`` pair — the value 12 (Kazuya, our one retained C1 seed) and a plausible small int at a
  constant **stride** whose surrounding struct is byte-for-byte *similar* (two idle players differ
  only at ``char_id``/position/facing). It then confirms behaviorally with the C4g action window:
  a candidate is accepted only if some 4-byte field in the acting player's struct **changes** when
  they act. The outputs — ``char_id`` address, stride, **Jin's id** (discovered, not seeded), and a
  ``move_id`` offset — are all derived.
* **Phase 3 (:func:`reverse_pointer_paths` + :func:`confirm_across_realloc`)** finds a *static,
  reallocation-surviving* path to that heap address: a reverse pointer scan builds a value index of
  every stored pointer, BFS-walks backward from the struct until it reaches a module ``.data`` slot,
  and keeps only paths that still resolve after a round reset moves the struct. That durability is
  the exact property the raw heap address lacks.
* **Phase 4 (:func:`derive_layout_scan`)** derives ``move_id``/``damage_taken`` from the real base
  and hands ``pos``/global/state to the existing C4e machinery (transform-component scan, global
  behavioral oracle, seeded state words), now that it has a correct base to work from.

Everything is offline-testable against a planted process image with pymem absent, and read-only
throughout — it enumerates regions and reads bytes and follows pointers; it never writes (docs/02
§2). The encoded state-word *offsets* remain best-effort seeds (docs/02 §8): no round-start oracle
can prove where ``stun_type`` lives, because nobody is in stun at round start.
"""

from __future__ import annotations

import bisect
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from tekken_coach.reader.decode import resolve_anchor
from tekken_coach.reader.discovery.basescan import (
    Behavior,
    Progress,
    _derive_global,
    _emit,
    _plausible_pointer,
    _read_region,
    _read_scalar,
    extract_signature,
    find_transform_component,
)
from tekken_coach.reader.discovery.derive import Confidence, DerivationResult, DerivedField
from tekken_coach.reader.discovery.manifest import DeriveScanSpec, ProbeManifest
from tekken_coach.reader.discovery.pe import ModuleImage, parse_module_image
from tekken_coach.reader.discovery.scanners import Region, value_scan
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemoryRegion, MemorySource
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    Anchor,
    AobSignature,
    EncodedStateSpec,
    OffsetTable,
)

_PTR_SIZE = 8


# ---------------------------------------------------------------------------
# Phase 2: locate the entity struct by behavior, deriving offsets as outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityCandidate:
    """A structural char-id pair at round start: Kazuya at ``p2_char``, a plausible id at
    ``p1_char``.

    ``p1_char``/``p2_char`` are the **absolute addresses** of the two players' ``char_id`` fields;
    the
    struct is symmetric so ``stride = p2_char - p1_char`` is the array stride. ``jin_id`` is P1's
    discovered id (an output — community data says 6, but we verify, never seed). Nothing here is a
    within-struct offset: the pair is found by scanning the heap for the value, and the stride by
    the
    two structs reading *similar* at round start (:func:`_struct_similarity`).
    """

    p1_char: int
    p2_char: int
    stride: int
    jin_id: int


@dataclass(frozen=True)
class EntityLayout:
    """The behaviorally-confirmed entity layout — every field an output (C4h Phase 2).

    Addresses are absolute in the round-start capture; :func:`derive_layout_scan` rebases them to
    the
    struct base once Phase 3 has found the static pointer path. ``move_id_addr`` is the
    acting-correlated field the window revealed; its offset from the struct base is discovered as a
    byproduct of locating it, never seeded.
    """

    p1_char: int  # P1 (Jin, acting) char_id address
    p2_char: int  # P2 (Kazuya, dummy) char_id address
    stride: int
    jin_id: int
    kazuya_id: int
    move_id_addr: int  # the field that changed when the acting player acted
    behavior: Behavior
    accepted: int  # distinct structs that behaved (>1 means the pick is not fully trustworthy)
    considered: int  # structural candidates the behavioral oracle chose between

    @property
    def ambiguous(self) -> bool:
        return self.accepted > 1


def _region_buffers(
    source: MemorySource, regions: Sequence[MemoryRegion], *, progress: Progress | None
) -> list[Region]:
    """Read each committed region into a :class:`Region` byte buffer (bounded, read-only)."""
    buffers: list[Region] = []
    for region in regions:
        data = _read_region(source, region.base, region.size)
        if data:
            buffers.append(Region(base=region.base, data=data))
    _emit(
        progress,
        f"  heap: {len(buffers)} readable region(s), "
        f"{sum(len(b.data) for b in buffers) // 1024} KiB",
    )
    return buffers


def _buffer_covering(buffers: Sequence[Region], address: int, size: int) -> Region | None:
    for buffer in buffers:
        if buffer.covers(address, size):
            return buffer
    return None


# The pairing similarity looks only this far into each struct — enough to span the shared non-zero
# constants two idle players carry, but small enough that testing every candidate stride is cheap.
_SIMILARITY_SPAN = 0x800


def _struct_similarity(
    a: Region, b: Region, base_a: int, base_b: int, span: int, align: int, min_shared: int
) -> float:
    """Similarity of ``[base_a, +span)`` and ``[base_b, +span)`` over their **non-zero** content.

    Two symmetric player structs at round start are byte-for-byte alike except at ``char_id``,
    position and facing — both idle, same health regime, same state constants, same 0 damage — so
    they share a great many *non-zero* words. The metric is deliberately over the non-zero union,
    not all words: a pair of empty (all-zero) heap spans would otherwise score a perfect 1.0 and
    flood the candidate set, which is exactly what a naive "fraction of equal words" does. Below
    ``min_shared`` non-zero words there is not enough evidence to call two spans the same struct, so
    it scores 0. This is the structural discriminator that keeps the char-id pairing from exploding,
    and it seeds no offset.

    Operates on the raw byte slices (not per-word :meth:`Region.read`) and rejects an all-zero span
    up front, so the overwhelming majority of candidate strides — which land on empty heap — cost a
    single C-level ``any()`` rather than a full word walk.
    """
    length = min(span, _SIMILARITY_SPAN)
    da = a.data[base_a - a.base : base_a - a.base + length]
    db = b.data[base_b - b.base : base_b - b.base + length]
    n = min(len(da), len(db))
    n -= n % align
    if n < min_shared * align or not any(da[:n]) or not any(db[:n]):
        return 0.0  # nothing (or nothing non-zero) to compare — cannot be a shared struct
    shared = 0
    equal = 0
    for off in range(0, n, align):
        wa = da[off : off + align]
        wb = db[off : off + align]
        if wa == wb:
            if any(wa):
                shared += 1
                equal += 1
        elif any(wa) or any(wb):
            shared += 1
    if shared < min_shared:
        return 0.0
    return equal / shared


def entity_candidates(
    source: MemorySource,
    *,
    manifest: ProbeManifest,
    spec: DeriveScanSpec,
    buffers: Sequence[Region] | None = None,
    progress: Progress | None = None,
) -> list[EntityCandidate]:
    """Sweep the heap for (Kazuya id, plausible id at a similar-struct stride) pairs (C4h Phase 2).

    Generate-and-prune: every occurrence of ``kazuya_char_id`` is a candidate P2 ``char_id``; a
    constant stride below it holding a plausible id whose surrounding struct is *similar* (both
    idle)
    is a candidate pair. The behavioral confirmation (:func:`confirm_entity_layout`) is what
    actually
    accepts one. Bounded by ``max_char_id_hits`` / ``max_pairs`` so a heap full of 12s stays
    tractable.
    """
    if buffers is None:
        buffers = _region_buffers(source, source.regions(), progress=progress)
    align = manifest.scan_align
    span = spec.struct_span
    kaz = manifest.kazuya_char_id
    hits: list[int] = []
    for region in buffers:
        for hit in value_scan(region, kaz, manifest.char_id_kind, align=align):
            hits.append(hit)
            if len(hits) >= spec.max_char_id_hits:
                break
        if len(hits) >= spec.max_char_id_hits:
            _emit(progress, f"  char-id hit ceiling {spec.max_char_id_hits} reached")
            break

    candidates: list[EntityCandidate] = []
    seen: set[tuple[int, int]] = set()
    for p2_char in hits:
        buffer = _buffer_covering(buffers, p2_char, 4)
        if buffer is None:
            continue
        low = p2_char - manifest.stride_max
        high = p2_char - manifest.stride_min
        for p1_char in range(max(buffer.base, low), high + 1, align):
            if not buffer.covers(p1_char, 4):
                continue
            (jin_id,) = struct.unpack("<I", buffer.read(p1_char, 4))
            if not (manifest.char_id_min <= jin_id <= manifest.char_id_max):
                continue
            stride = p2_char - p1_char
            sim = _struct_similarity(
                buffer, buffer, p1_char, p2_char, span, align, spec.min_shared_words
            )
            if sim < spec.similarity_min:
                continue
            key = (p1_char, stride)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                EntityCandidate(p1_char=p1_char, p2_char=p2_char, stride=stride, jin_id=jin_id)
            )
            if len(candidates) >= spec.max_pairs:
                _emit(progress, f"  candidate-pair ceiling {spec.max_pairs} reached")
                return candidates
    _emit(
        progress, f"  {len(candidates)} structural char-id pair(s) from {len(hits)} Kazuya hit(s)"
    )
    return candidates


def _acting_correlated_offsets(
    before: Region, during: Sequence[Region], base: int, span: int, manifest: ProbeManifest
) -> list[int]:
    """Offsets in ``[base, base+span)`` whose value changed across the window, plausibly a move id.

    A field is acting-correlated if it differs from its round-start value in **at least one** sample
    (the C4g "any-sample" rule — ``move_id`` is transient and idles back) and stays within the
    move-id
    plausibility range in every readable sample. That range is a data bound, not a seeded offset.
    """
    align = manifest.scan_align
    out: list[int] = []
    for off in range(0, span - 4 + 1, align):
        base0 = before.read_scalar(base + off, manifest.move_id_kind)
        if base0 is None or not (manifest.move_id_min <= base0 < manifest.move_id_max):
            continue
        changed = False
        plausible = True
        for sample in during:
            value = sample.read_scalar(base + off, manifest.move_id_kind)
            if value is None:
                continue
            if not (manifest.move_id_min <= value < manifest.move_id_max):
                plausible = False
                break
            if value != base0:
                changed = True
        if plausible and changed:
            out.append(off)
    return out


def _opponent_damaged(
    before: Region, during: Sequence[Region], base: int, span: int, align: int
) -> bool:
    """Whether some i32 field in the opponent struct went 0 -> >0 across the window (a landed
    hit)."""
    for off in range(0, span - 4 + 1, align):
        base0 = before.read_scalar(base + off, "i32")
        if base0 != 0:
            continue
        for sample in during:
            value = sample.read_scalar(base + off, "i32")
            if value is not None and value > 0:
                return True
    return False


def confirm_entity_layout(
    before: MemorySource,
    during: Iterable[MemorySource],
    candidates: Sequence[EntityCandidate],
    *,
    manifest: ProbeManifest,
    spec: DeriveScanSpec,
    progress: Progress | None = None,
) -> EntityLayout | None:
    """Accept the candidate whose acting player's struct *behaved* (C4h Phase 2, C4g window).

    Reads each candidate's P1 and P2 struct spans once at round start and once per window sample —
    the struct does not reallocate *within* a round, so the fixed heap address is stable here (the
    reallocation Phase 3 survives happens on a round reset). A candidate is accepted only if a field
    in the acting player changed; among survivors the one with the strongest corroboration (the
    dummy
    also reacted / took damage) wins, and any residual ambiguity is reported.
    """
    align = manifest.scan_align
    span = spec.struct_span
    acting = manifest.moving_player

    before_p1 = {c.p1_char: _read_region(before, c.p1_char, span) for c in candidates}
    before_p2 = {c.p2_char: _read_region(before, c.p2_char, span) for c in candidates}
    during_p1: dict[int, list[Region]] = {c.p1_char: [] for c in candidates}
    during_p2: dict[int, list[Region]] = {c.p2_char: [] for c in candidates}
    samples = 0
    for source in during:
        samples += 1
        for c in candidates:
            d1 = _read_region(source, c.p1_char, span)
            d2 = _read_region(source, c.p2_char, span)
            if d1:
                during_p1[c.p1_char].append(Region(base=c.p1_char, data=d1))
            if d2:
                during_p2[c.p2_char].append(Region(base=c.p2_char, data=d2))

    accepted: list[tuple[EntityCandidate, int, Behavior]] = []
    for c in candidates:
        b1 = before_p1[c.p1_char]
        b2 = before_p2[c.p2_char]
        if not b1 or not b2:
            continue
        acting_base = c.p1_char if acting == 0 else c.p2_char
        opp_base = c.p2_char if acting == 0 else c.p1_char
        acting_before = Region(base=acting_base, data=b1 if acting == 0 else b2)
        opp_before = Region(base=opp_base, data=b2 if acting == 0 else b1)
        acting_during = during_p1[c.p1_char] if acting == 0 else during_p2[c.p2_char]
        opp_during = during_p2[c.p2_char] if acting == 0 else during_p1[c.p1_char]

        changed = _acting_correlated_offsets(
            acting_before, acting_during, acting_base, span, manifest
        )
        if not changed:
            continue
        move_addr = acting_base + changed[0]
        opp_changed = bool(
            _acting_correlated_offsets(opp_before, opp_during, opp_base, span, manifest)
        )
        damaged = _opponent_damaged(opp_before, opp_during, opp_base, span, align)
        behavior = Behavior(
            acting_move_changed=True,
            opponent_move_changed=opp_changed,
            opponent_damaged=damaged,
            p1_after_base=c.p1_char,
            samples=len(acting_during),
        )
        accepted.append((c, move_addr, behavior))

    if not accepted:
        _emit(
            progress,
            f"  none of the {len(candidates)} structural pair(s) changed the acting player's "
            f"move_id across {samples} window sample(s)",
        )
        return None
    landings = {c.p1_char for c, _, _ in accepted}
    best_c, best_move, best_behavior = max(accepted, key=lambda t: t[2].score)
    _emit(
        progress,
        f"  entity located: P1(Jin)={best_c.jin_id} @0x{best_c.p1_char:x}, "
        f"stride 0x{best_c.stride:x}, "
        f"move_id +0x{best_move - best_c.p1_char:x} "
        f"({len(landings)} of {len(candidates)} accepted; {best_behavior.describe()})",
    )
    return EntityLayout(
        p1_char=best_c.p1_char,
        p2_char=best_c.p2_char,
        stride=best_c.stride,
        jin_id=best_c.jin_id,
        kazuya_id=manifest.kazuya_char_id,
        move_id_addr=best_move,
        behavior=best_behavior,
        accepted=len(landings),
        considered=len(candidates),
    )


def locate_entity_layout(
    before: MemorySource,
    during: Iterable[MemorySource],
    *,
    manifest: ProbeManifest,
    spec: DeriveScanSpec,
    buffers: Sequence[Region] | None = None,
    progress: Progress | None = None,
) -> EntityLayout | None:
    """Compose Phase 2: structural char-id-pair sweep, then the behavioral action window."""
    candidates = entity_candidates(
        before, manifest=manifest, spec=spec, buffers=buffers, progress=progress
    )
    if not candidates:
        _emit(progress, "  no structural char-id pair found on the heap")
        return None
    return confirm_entity_layout(
        before, during, candidates, manifest=manifest, spec=spec, progress=progress
    )


# ---------------------------------------------------------------------------
# Phase 3: reverse pointer scan for a static, reallocation-surviving path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReversePath:
    """A static-slot pointer path that resolves to a heap target (C4h Phase 3).

    ``base_offset`` is the module-relative ``.data`` slot; ``offsets`` is the forward pointer chain
    (each entry dereferences and adds), so ``module_base + base_offset`` followed through
    ``offsets``
    lands on the Phase 2 target address. Durable across reallocation only after
    :func:`confirm_across_realloc` proves it still resolves once the struct has moved.
    """

    base_offset: int
    offsets: tuple[int, ...]

    def anchor(self, module: str) -> Anchor:
        return Anchor(module=module, base_offset=self.base_offset, pointer_path=list(self.offsets))


@dataclass(frozen=True)
class _PointerIndex:
    """A value-sorted index of every stored pointer in enumerated memory, for backward BFS.

    ``values`` is sorted; ``locations[i]`` is where ``values[i]`` is stored. A reverse hop needs
    "locations whose stored pointer falls in ``[X-M, X]``", which is a bisect slice over ``values``.
    ``by_location`` recovers a slot's stored value in O(1) when a hop is taken.
    """

    values: list[int]
    locations: list[int]
    by_location: dict[int, int]

    def locations_pointing_into(self, low: int, high: int) -> list[int]:
        lo = bisect.bisect_left(self.values, low)
        hi = bisect.bisect_right(self.values, high)
        return self.locations[lo:hi]


def _data_section_ranges(module_base: int, image: ModuleImage) -> list[tuple[int, int]]:
    """Absolute ``[start, end)`` spans of the module's readable data sections (where a slot
    lives)."""
    return [
        (module_base + s.rva, module_base + s.rva + s.virtual_size) for s in image.data_sections()
    ]


def _in_ranges(address: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= address < end for start, end in ranges)


def build_pointer_index(
    source: MemorySource,
    *,
    module_base: int,
    image: ModuleImage,
    heap: Sequence[Region],
    max_entries: int,
    progress: Progress | None = None,
) -> _PointerIndex:
    """Index every 8-aligned plausible pointer in the module data sections + enumerated heap.

    The reverse index the backward BFS walks: static slots (module ``.data``) hold the chain root;
    heap slots hold the intermediate links. Bounded by ``max_entries`` for tractability.
    """
    pairs: list[tuple[int, int]] = []

    def scan_bytes(base: int, data: bytes) -> bool:
        for off in range(0, len(data) - _PTR_SIZE + 1, _PTR_SIZE):
            (value,) = struct.unpack_from("<Q", data, off)
            if _plausible_pointer(value):
                pairs.append((value, base + off))
                if len(pairs) >= max_entries:
                    return False
        return True

    for start, end in _data_section_ranges(module_base, image):
        data = _read_region(source, start, end - start)
        if data and not scan_bytes(start, data):
            break
    else:
        for buffer in heap:
            if not scan_bytes(buffer.base, buffer.data):
                break
    pairs.sort(key=lambda p: p[0])
    _emit(progress, f"  reverse index: {len(pairs)} stored pointer(s)")
    by_location = {loc: value for value, loc in pairs}
    return _PointerIndex(
        values=[v for v, _ in pairs],
        locations=[loc for _, loc in pairs],
        by_location=by_location,
    )


def reverse_pointer_paths(
    index: _PointerIndex,
    *,
    target: int,
    module_base: int,
    data_ranges: Sequence[tuple[int, int]],
    spec: DeriveScanSpec,
    progress: Progress | None = None,
) -> list[ReversePath]:
    """BFS backward from ``target`` to a static ``.data`` slot, bounded by depth/offset (C4h Phase
    3).

    A reverse hop from address ``X`` finds every location ``A`` holding a pointer ``V`` in
    ``[X - M, X]`` (``M = reverse_max_offset``); the forward offset is ``X - V`` and ``A`` becomes
    the
    predecessor. When ``A`` is a module data slot the chain is rooted and a :class:`ReversePath` is
    emitted. Bounded by ``reverse_max_depth`` (chain length) and ``reverse_max_nodes`` (BFS size).
    """
    found: list[ReversePath] = []
    seen: set[tuple[int, int]] = set()  # (address, depth) — a node reached at a given depth
    frontier: list[tuple[int, tuple[int, ...]]] = [(target, ())]
    nodes = 0
    depth = 0
    while frontier and depth < spec.reverse_max_depth:
        depth += 1
        nxt: list[tuple[int, tuple[int, ...]]] = []
        for x_addr, offs in frontier:
            for a_loc in index.locations_pointing_into(x_addr - spec.reverse_max_offset, x_addr):
                nodes += 1
                if nodes > spec.reverse_max_nodes:
                    _emit(progress, f"  reverse BFS node ceiling {spec.reverse_max_nodes} reached")
                    frontier = []
                    break
                # A points at V; the forward hop that lands on x_addr adds (x_addr - V).
                v = index.by_location[a_loc]
                hop = x_addr - v
                new_offs = (hop, *offs)
                if _in_ranges(a_loc, data_ranges):
                    found.append(ReversePath(base_offset=a_loc - module_base, offsets=new_offs))
                    continue
                key = (a_loc, depth)
                if key in seen:
                    continue
                seen.add(key)
                nxt.append((a_loc, new_offs))
            else:
                continue
            break
        frontier = nxt
    _emit(progress, f"  reverse scan: {len(found)} candidate static path(s) to 0x{target:x}")
    return found


def _resolves_to_struct(
    source: MemorySource,
    address: int,
    *,
    jin_id: int,
    kazuya_id: int,
    stride: int,
    manifest: ProbeManifest,
) -> bool:
    """Whether ``address`` lands on the *same* struct: Jin's id here, Kazuya's id one stride over.

    Used to validate a path against the **reallocated** struct without pre-pinning its new address —
    the struct moved, but its char-id pair signature (which Phase 2 already derived) did not.
    """
    here = _read_scalar(source, address, manifest.char_id_kind)
    opposite = _read_scalar(source, address + stride, manifest.char_id_kind)
    return here == jin_id and opposite == kazuya_id


def confirm_across_realloc(
    paths: Sequence[ReversePath],
    source_before: MemorySource,
    source_after: MemorySource | None,
    *,
    module: str,
    target_before: int,
    layout: EntityLayout,
    manifest: ProbeManifest,
    progress: Progress | None = None,
) -> list[ReversePath]:
    """Keep only paths that resolve to the struct in **both** captures (C4h Phase 3, the hard part).

    A coincidental chain that happens to reach the struct in the first capture will not reach the
    *reallocated* struct in the second — the whole point of taking a capture after a round reset. A
    real game chain re-resolves because the module slot is static and the game rewrites the
    intermediate pointers. This is what makes the anchor durable, which the raw heap address is not.
    The second capture's struct is re-identified by its char-id pair signature, so no independent
    re-location is needed. When ``source_after`` is ``None`` the durability check is skipped and the
    caller flags the anchor as unconfirmed (a round reset was not captured).
    """
    survivors: list[ReversePath] = []
    for path in paths:
        anchor = path.anchor(module)
        try:
            if resolve_anchor(source_before, anchor) != target_before:
                continue
            if source_after is not None:
                landed = resolve_anchor(source_after, anchor)
                if not _resolves_to_struct(
                    source_after,
                    landed,
                    jin_id=layout.jin_id,
                    kazuya_id=layout.kazuya_id,
                    stride=layout.stride,
                    manifest=manifest,
                ):
                    continue
        except MemoryReadError:
            continue
        survivors.append(path)
    kept = "resolved in both captures" if source_after is not None else "resolved (UNCONFIRMED)"
    _emit(progress, f"  {len(survivors)} of {len(paths)} path(s) {kept}")
    return survivors


# ---------------------------------------------------------------------------
# Phase 4: field derivation from the real base, then hand off to C4e machinery
# ---------------------------------------------------------------------------


def _to_base_anchor(path: ReversePath, char_id_addr: int, module: str) -> tuple[Anchor, int]:
    """Rebase a char-id-targeting path to resolve to the **struct base**, returning char_id_offset.

    The reverse scan targets the ``char_id`` address; its final hop offset is exactly
    ``char_id_addr - struct_base``. Dropping that hop (replacing it with ``+0``) makes the anchor
    resolve to the struct base the game's pointer actually points at, so every derived and seeded
    field offset is positive and relative to the same base the fork's within-struct offsets assume.
    """
    if not path.offsets:  # a static struct (no chain): base == char_id address, offset 0
        return path.anchor(module), 0
    char_id_offset = path.offsets[-1]
    base_path = (*path.offsets[:-1], 0)
    anchor = Anchor(module=module, base_offset=path.base_offset, pointer_path=list(base_path))
    return anchor, char_id_offset


def _derive_damage_offset(
    before: MemorySource,
    during: Sequence[MemorySource],
    *,
    opp_base: int,
    span: int,
    align: int,
) -> int | None:
    """Offset of an opponent field that goes 0 -> >0 and stays when the dummy is hit (i32)."""
    b = _read_region(before, opp_base, span)
    if not b:
        return None
    before_region = Region(base=opp_base, data=b)
    during_regions = [
        Region(base=opp_base, data=d)
        for d in (_read_region(s, opp_base, span) for s in during)
        if d
    ]
    for off in range(0, span - 4 + 1, align):
        if before_region.read_scalar(opp_base + off, "i32") != 0:
            continue
        rose = False
        held = True
        for sample in during_regions:
            value = sample.read_scalar(opp_base + off, "i32")
            if value is None:
                continue
            if value < 0:
                held = False
                break
            if value > 0:
                rose = True
        if rose and held:
            return off
    return None


@dataclass
class DeriveInputs:
    """The captures the C4h pipeline folds together (kept in one place for the orchestration)."""

    before: MemorySource
    during: Sequence[MemorySource]
    after: MemorySource
    realloc: MemorySource | None = None  # a second capture after a round reset (Phase 3 confirm)
    heap_buffers: Sequence[Region] | None = None
    realloc_buffers: Sequence[Region] | None = None
    seeded_state: list[str] = field(default_factory=list)


def derive_layout_scan(
    inputs: DeriveInputs,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    seed: OffsetTable,
    state_map: EncodedStateSpec | None = None,
    progress: Progress | None = None,
) -> DerivationResult:
    """Run the full C4h derivation (Phases 2-4) into a :class:`DerivationResult` (docs/02 §3).

    Locates the entity struct by behavior, finds a reallocation-surviving static path to it, derives
    ``char_id``/``move_id``/``damage_taken`` relative to the real base, and hands
    ``pos``/global/state
    to the existing C4e machinery. Fails closed (a table-blocking note, no anchor) whenever a phase
    cannot prove its result, exactly as C4d/C4g do — a struct that merely looks right is never
    taken.
    """
    result = DerivationResult(module=module, module_base=module_base)
    result.encoded_state = state_map
    spec = manifest.derive_scan
    if spec is None:
        result.notes.append("no derive_scan spec in the probe manifest; cannot run the C4h scan.")
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x", "frame_counter"])
        return result

    # Global/match anchor: the C4e behavioral oracle, unchanged (it already derives its own base).
    _derive_global(
        result,
        inputs.before,
        inputs.after,
        module=module,
        module_base=module_base,
        manifest=manifest,
        seed=seed,
        global_located=None,
        sweep_global=True,
        progress=progress,
    )

    image = parse_module_image(lambda rva, n: inputs.before.read(module_base + rva, n))
    layout = locate_entity_layout(
        inputs.before,
        inputs.during,
        manifest=manifest,
        spec=spec,
        buffers=inputs.heap_buffers,
        progress=progress,
    )
    if layout is None:
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x"])
        result.notes.append(
            "no heap struct BEHAVED like the acting player (Kazuya=12 with a plausible id at a "
            "similar-struct stride, whose move_id changed across the action window). Either the "
            "action was not performed — walk P1 (Jin) forward, jab P2, and jump for the whole "
            "window — or the Jin-vs-Kazuya round-start setup is wrong (see runbook)."
        )
        return result

    static = _resolve_static_path(
        inputs,
        layout,
        module=module,
        module_base=module_base,
        image=image,
        manifest=manifest,
        spec=spec,
        progress=progress,
    )
    if static is None:
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x"])
        result.notes.append(
            f"located the entity struct behaviorally (P1 @0x{layout.p1_char:x}, "
            f"stride 0x{layout.stride:x}) but found NO static pointer path that survives a "
            "reallocation. Re-run with a round reset between captures so the reverse pointer scan "
            "can confirm a durable path, or widen derive_scan.reverse_max_depth / "
            "reverse_max_offset."
        )
        return result

    _fill_player_layout(
        result,
        inputs,
        layout,
        static=static,
        module=module,
        module_base=module_base,
        manifest=manifest,
        spec=spec,
        progress=progress,
    )
    return result


@dataclass(frozen=True)
class StaticPath:
    """The chosen static path plus how it was established (durability + ambiguity for the
    report)."""

    anchor: Anchor  # resolves to the STRUCT BASE (the pointer target), not char_id
    char_id_offset: int  # char_id_addr - struct_base (derived)
    survivors: int  # distinct static paths that resolved (>1 means the pick is a guess)
    confirmed: bool  # survived a reallocation (a round-reset capture was provided)


def _resolve_static_path(
    inputs: DeriveInputs,
    layout: EntityLayout,
    *,
    module: str,
    module_base: int,
    image: ModuleImage,
    manifest: ProbeManifest,
    spec: DeriveScanSpec,
    progress: Progress | None = None,
) -> StaticPath | None:
    """Phase 3: reverse-scan to a static slot and keep a reallocation-surviving path to
    ``char_id``."""
    heap = inputs.heap_buffers
    if heap is None:
        heap = _region_buffers(inputs.before, inputs.before.regions(), progress=progress)
    index = build_pointer_index(
        inputs.before,
        module_base=module_base,
        image=image,
        heap=heap,
        max_entries=spec.reverse_max_nodes,
        progress=progress,
    )
    data_ranges = _data_section_ranges(module_base, image)
    paths = reverse_pointer_paths(
        index,
        target=layout.p1_char,
        module_base=module_base,
        data_ranges=data_ranges,
        spec=spec,
        progress=progress,
    )
    paths = paths[: spec.max_paths]
    survivors = confirm_across_realloc(
        paths,
        inputs.before,
        inputs.realloc,
        module=module,
        target_before=layout.p1_char,
        layout=layout,
        manifest=manifest,
        progress=progress,
    )
    if not survivors:
        return None
    # Prefer the shortest chain, then the lowest slot — a stable, explainable pick.
    best = min(survivors, key=lambda p: (len(p.offsets), p.base_offset))
    anchor, char_id_offset = _to_base_anchor(best, layout.p1_char, module)
    return StaticPath(
        anchor=anchor,
        char_id_offset=char_id_offset,
        survivors=len(survivors),
        confirmed=inputs.realloc is not None,
    )


def _fill_player_layout(
    result: DerivationResult,
    inputs: DeriveInputs,
    layout: EntityLayout,
    *,
    static: StaticPath,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    spec: DeriveScanSpec,
    progress: Progress | None = None,
) -> None:
    """Phase 4: derive the player fields relative to the real base and hand off to C4e machinery."""
    anchor = static.anchor
    try:
        struct_base = resolve_anchor(inputs.before, anchor)
    except MemoryReadError:
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x"])
        result.notes.append("the derived static path stopped resolving; no table written.")
        return

    char_id_offset = static.char_id_offset
    move_id_offset = layout.move_id_addr - struct_base
    signature = extract_signature_for(inputs.before, module_base, anchor.base_offset, spec)
    result.player_anchor = Anchor(
        module=module,
        base_offset=anchor.base_offset,
        pointer_path=list(anchor.pointer_path),
        signature=signature,
    )
    result.stride = layout.stride
    result.player_char_ids = (layout.jin_id, layout.kazuya_id)

    result.fields.append(
        DerivedField(
            name="char_id",
            scope="player",
            offset=char_id_offset,
            kind=manifest.char_id_kind,
            example_address=struct_base + char_id_offset,
            confidence=Confidence.high,
            method=f"derived: Jin={layout.jin_id} at P1, Kazuya={layout.kazuya_id} at P2 "
            f"(stride 0x{layout.stride:x}), located by heap value + similar-struct stride",
        )
    )
    result.fields.append(
        DerivedField(
            name="move_id",
            scope="player",
            offset=move_id_offset,
            kind=manifest.move_id_kind,
            example_address=struct_base + move_id_offset,
            confidence=Confidence.high,
            method="derived: the acting player's field that changed across the action window",
        )
    )

    _derive_health_field(
        result,
        inputs,
        struct_base=struct_base,
        stride=layout.stride,
        char_id_offset=char_id_offset,
        manifest=manifest,
        spec=spec,
    )
    _seed_state_from_layout(result, manifest, struct_base, char_id_offset=char_id_offset)
    _behavior_notes(result, layout, static)
    _derive_position_field(
        result,
        inputs,
        struct_base=struct_base,
        stride=layout.stride,
        module=module,
        module_base=module_base,
        manifest=manifest,
        progress=progress,
    )


def extract_signature_for(
    source: MemorySource, module_base: int, base_offset: int, spec: DeriveScanSpec
) -> AobSignature | None:
    """AOB signature around the derived slot (reuses C4d :func:`extract_signature`,
    spec-adapted)."""
    from tekken_coach.reader.discovery.manifest import BaseScanSpec  # noqa: PLC0415

    shim = BaseScanSpec(
        pointer_path=[],
        char_id_offset=0,
        move_id_offset=0,
        damage_taken_offset=0,
        round_start_health=0,
        aob_window_before=spec.aob_window_before,
        aob_window_after=spec.aob_window_after,
    )
    return extract_signature(source, module_base, base_offset, shim)


def _derive_health_field(
    result: DerivationResult,
    inputs: DeriveInputs,
    *,
    struct_base: int,
    stride: int,
    char_id_offset: int,
    manifest: ProbeManifest,
    spec: DeriveScanSpec,
) -> None:
    """Derive ``damage_taken`` from the landed jab (computed health), or seed it flagged (C4h).

    A seeded fallback is fork-relative, so it is translated by ``char_id_offset - fork char_id
    offset`` onto the derived base (the reverse scan lands on the pointer target, which is not
    necessarily the fork's struct-base convention). A *derived* offset is absolute and needs none.
    """
    align = manifest.scan_align
    opp_base = struct_base + stride
    base_scan = manifest.base_scan
    max_health = (
        base_scan.round_start_health if base_scan is not None else manifest.round_start_health
    )
    damage_off = _derive_damage_offset(
        inputs.before, inputs.during, opp_base=opp_base, span=spec.struct_span, align=align
    )
    if damage_off is not None:
        result.fields.append(
            DerivedField(
                name="damage_taken",
                scope="player",
                offset=damage_off,
                kind="i32",
                example_address=struct_base + damage_off,
                confidence=Confidence.high,
                method="derived: opponent field that went 0 -> >0 and held when the dummy was hit; "
                "health computed as round_start_health - damage_taken",
            )
        )
        result.max_health = max_health
        result.notes.append(
            f"health computed as {max_health} - damage_taken (+0x{damage_off:x}), derived from the "
            "landed jab; no direct HP field (as expected on T8). Verify full HP is really "
            f"{max_health} for this build."
        )
        return
    if base_scan is not None:
        seeded_off = base_scan.damage_taken_offset + (char_id_offset - base_scan.char_id_offset)
        result.fields.append(
            DerivedField(
                name="damage_taken",
                scope="player",
                offset=seeded_off,
                kind="i32",
                example_address=struct_base + seeded_off,
                confidence=Confidence.seeded,
                method="SEEDED from the stale T8 layout (no jab connected), translated onto the "
                "derived base; health computed as round_start_health - damage_taken",
            )
        )
        result.max_health = max_health
        result.notes.append(
            "damage_taken offset SEEDED (no jab landed in the window) — computed health uses it, "
            "but re-run with a CONNECTING jab so it can be derived from behavior."
        )
        return
    result.unresolved.append("health")
    result.notes.append(
        "could not derive damage_taken (no connecting jab) and no base_scan layout to seed it "
        "from; "
        "health unresolved."
    )


def _seed_state_from_layout(
    result: DerivationResult, manifest: ProbeManifest, struct_base: int, *, char_id_offset: int
) -> None:
    """Seed the encoded state-word offsets best-effort, flagged for re-derivation (docs/02 §8, C4h).

    These are the one thing C4h does not derive: no round-start oracle can prove where ``stun_type``
    lives, because nobody is in stun at round start. They come from the (stale) layout facts, which
    are fork-relative, so each is translated by ``char_id_offset - fork char_id offset`` onto the
    derived base (the reverse scan lands on the pointer target, not necessarily the fork's
    struct-base convention). They are flagged loudly. When there is nowhere to seed them from, the
    encoded state map is dropped rather than left dangling (the builder would otherwise raise).
    """
    if result.encoded_state is None:
        return
    base_scan = manifest.base_scan
    if base_scan is None or not base_scan.state_fields:
        result.encoded_state = None
        result.notes.append(
            "encoded state words dropped: no base_scan.state_fields to seed their offsets; state "
            "needs full re-derivation by observation (docs/02 §8)."
        )
        return
    delta = char_id_offset - base_scan.char_id_offset
    for name, field_spec in base_scan.state_fields.items():
        offset = field_spec.offset + delta
        result.fields.append(
            DerivedField(
                name=name,
                scope="player",
                offset=offset,
                kind=field_spec.kind,
                example_address=struct_base + offset,
                confidence=Confidence.seeded,
                method="SEEDED from the stale T8 layout, translated onto the derived base; its "
                "offset AND its value meanings need the docs/02 §8 observation re-derivation",
            )
        )
    result.drop_player_fields.extend(base_scan.legacy_state_fields)
    result.notes.append(
        "encoded state-word offsets are SEEDED best-effort from the stale layout and flagged for "
        "re-derivation (docs/02 §8) — they came from the same seed C4h exists to replace."
    )


def _behavior_notes(result: DerivationResult, layout: EntityLayout, static: StaticPath) -> None:
    """Say, loudly, on what evidence the player anchor was accepted (C4h)."""
    result.notes.append(
        f"player anchor confirmed BEHAVIORALLY across the action window "
        f"({layout.behavior.describe()}); {layout.accepted} of {layout.considered} "
        f"structural candidate(s) accepted."
    )
    if not static.confirmed:
        result.notes.append(
            "the static pointer path was NOT confirmed across a reallocation (no round-reset "
            "capture): it resolves now but may not survive a round/character change. Re-run "
            "capturing a round reset between the two snapshots."
        )
    if static.survivors > 1:
        result.notes.append(
            f"{static.survivors} distinct static paths resolved to the struct; the shortest was "
            "taken. Confirm with the doctor."
        )
    if layout.ambiguous:
        result.notes.append(
            f"{layout.accepted} distinct structs behaved like the acting player; the strongest was "
            "taken. Confirm char ids / move_id with the doctor before trusting the table."
        )


def _derive_position_field(
    result: DerivationResult,
    inputs: DeriveInputs,
    *,
    struct_base: int,
    stride: int,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    progress: Progress | None = None,
) -> None:
    """Hand position to the C4e transform-component scan, now anchored on the real base (C4h Phase
    4)."""
    base_scan = manifest.base_scan
    component_spec = base_scan.component_scan if base_scan is not None else None
    if component_spec is None:
        result.unresolved.append("pos_x")
        result.notes.append(
            "no base_scan.component_scan spec to locate the transform component; position "
            "unresolved."
        )
        return
    assert result.player_anchor is not None
    try:
        after_base = resolve_anchor(inputs.after, result.player_anchor)
    except MemoryReadError:
        result.unresolved.append("pos_x")
        result.notes.append("chain did not re-resolve in the after snapshot; position unresolved.")
        return
    component = find_transform_component(
        inputs.before,
        inputs.after,
        p1_base=struct_base,
        p1_after_base=after_base,
        p2_base=struct_base + stride,
        spec=component_spec,
        manifest=manifest,
        progress=progress,
    )
    if component is None:
        result.unresolved.append("pos_x")
        result.notes.append(
            "no moving float triple in the entity struct or a component it points at; position "
            "unresolved. Walk P1 a real step, and widen base_scan.component_scan if needed."
        )
        return
    result.components[POSITION_COMPONENT] = component
    result.drop_player_fields.extend(["pos_x", "pos_y", "pos_z"])
    triple = component.fields["pos_x"].offset
    hops = " -> ".join(f"+0x{o:x}" for o in component.pointer_path) or "(direct)"
    result.notes.append(
        f"position lives in a transform component reached via +0x{component.slot_offset:x} {hops}, "
        f"triple at +0x{triple:x}; confirmed by resolving the same path from P2 to a distinct "
        f"coordinate. Carried as a {POSITION_COMPONENT!r} component anchor."
    )
