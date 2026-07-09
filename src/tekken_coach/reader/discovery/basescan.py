"""Locate Tekken 8's heap structs via static code/data pointers (C4d/C4e, docs/02 §3).

C4c's value-scan derivation finds *field offsets* but locates the player struct by scanning the heap
for a known value — which fails on Tekken 8 because the entity struct is heap-allocated and
**reallocates** on every character change / round (confirmed live: a found address went NaN after a
character swap). The robust technique the TekkenBot lineage uses, and the one this module
implements, is to scan the module's **static** data for the pointer that leads to the struct and
follow a **pointer chain** — a module-anchored base that the OS keeps valid across reallocations
(:func:`~tekken_coach.reader.decode.resolve_anchor` already follows such a chain, so once we derive
``base_offset`` + ``pointer_path`` the reader reaches the struct with no decode change).

The clean-room core is **candidate-generate-and-validate with the known field layout as the oracle**
(docs/02 §5 — the layout offsets are facts/data; this validation logic is original, not ported):

1. **Bound** the scan by parsing the in-memory PE header (:mod:`.pe`) — sweep only the readable
   initialized-data sections where global pointers live, not the whole image.
2. **Generate** candidates: every 8-aligned slot in those sections holding a plausible user-space
   pointer.
3. **Validate** against the oracle: follow the seed :attr:`~.manifest.BaseScanSpec.pointer_path`
   from the slot and accept only if the landing is struct-shaped for **both** players — a plausible
   ``char_id``, a plausible ``move_id``, ``damage_taken == 0`` at round start, and the two players'
   ids form ``{Jin, Kazuya=12}``. Mutual multi-field consistency across the two symmetric structs is
   the acceptance, anchored in code rather than the heap (the same philosophy as C4c's health/stride
   confirmation).
4. **Persist an AOB signature** around the accepted slot (pointer bytes wildcarded) so a re-run
   re-finds it fast (:func:`extract_signature` / :func:`find_by_signature`), stored in the table as
   facts/data (docs/02 §5).
5. **Fill in** health + position with C4c's value/position scans, now tractable *inside the located
   struct* rather than over the whole heap.

C4e applies the same shape to the two things the player scan left seeded, changing only the oracle:

* the **global/match struct** (:func:`locate_global_struct`) is behind its own static pointer, but a
  frame counter has no structural signature — nothing about one instant identifies a ``u32``. Its
  oracle is therefore *behavioral*, read across the two snapshots: one offset ticks up, one holds a
  round number steady. The offsets are seeded **unassigned**; :func:`assign_global_fields` decides
  which is which from behavior, so a reordering in the source data cannot mislabel the counter.
* the **transform component** (:func:`find_transform_component`) holds ``pos_{x,y,z}``, which is not
  in the entity struct at all. The entity's own stable pointer slots are the candidates, a moving
  float triple in the pointee is the oracle, and P2 resolving through the identical path to a
  different plausible coordinate is the acceptance — the same two-struct consistency argument.

What C4e does **not** derive: the *meanings* of the encoded state values. Their offsets are seeded
facts (:func:`_seed_state_fields`); no round-start oracle can prove what ``stun_type == 3`` means,
because nobody is in stun at round start. That is the observation protocol of docs/02 §8.

Read-only throughout: it reads process memory through the :class:`MemorySource` seam and follows
pointers; it never writes (docs/02 §2).
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from tekken_coach.reader.decode import resolve_anchor
from tekken_coach.reader.discovery.derive import (
    Confidence,
    DerivationResult,
    DerivedField,
    _plausible_coord,
)
from tekken_coach.reader.discovery.manifest import (
    BaseScanSpec,
    ComponentScanSpec,
    GlobalScanSpec,
    ProbeManifest,
)
from tekken_coach.reader.discovery.pe import ModuleImage, Reader, Section, parse_module_image
from tekken_coach.reader.discovery.scanners import Region, aob_scan, value_scan
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    Anchor,
    AobSignature,
    ComponentAnchor,
    EncodedStateSpec,
    FieldSpec,
    OffsetTable,
    ScalarKind,
)

# x64 Windows user-space bounds — a value outside this is not a live pointer, so it is not worth
# following a chain from. Deliberately generous; the oracle does the real rejection.
_MIN_USERSPACE = 0x10000
_MAX_USERSPACE = 0x7FFF_FFFF_FFFF
_PTR_SIZE = 8

# A progress sink so the long live sweep is observable (the command layer prints; the library stays
# silent by default, docs/02 §2). ``None`` means no reporting.
Progress = Callable[[str], None]
_PROGRESS_EVERY = 50_000  # emit a validation tally every N candidates


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


_SCALAR_FMT: dict[ScalarKind, tuple[str, int]] = {
    "u32": ("<I", 4),
    "i32": ("<i", 4),
    "u16": ("<H", 2),
    "u8": ("<B", 1),
    "ptr": ("<Q", 8),
}


def _read_scalar(source: MemorySource, address: int, kind: ScalarKind) -> int | None:
    """Read a scalar, returning ``None`` if the address is unreadable (a dead pointer branch)."""
    fmt, size = _SCALAR_FMT[kind]
    try:
        raw = source.read(address, size)
    except MemoryReadError:
        return None
    if len(raw) != size:
        return None
    return int(struct.unpack(fmt, raw)[0])


def _module_reader(source: MemorySource, module_base: int) -> Reader:
    """A :data:`~.pe.Reader` bound to ``source`` at ``module_base`` (pure byte slices)."""

    def read(rva: int, size: int) -> bytes:
        return source.read(module_base + rva, size)

    return read


def _plausible_pointer(value: int) -> bool:
    return _MIN_USERSPACE <= value <= _MAX_USERSPACE


def _read_bounded(source: MemorySource, base: int, size: int, page: int = 0x1000) -> bytes:
    """Read up to ``size`` bytes, stopping at the first unreadable page (a mapping boundary).

    A single big read can straddle the end of a mapped region (a section whose virtual size exceeds
    its mapped tail, or a heap window scanned past its allocation) and fail wholesale. Reading
    page-by-page and stopping at the first failure returns *what is actually mapped* — the scanners
    tolerate a short region. Used as the fallback in :func:`_read_region`.
    """
    chunks: list[bytes] = []
    got = 0
    while got < size:
        want = min(page, size - got)
        try:
            chunks.append(source.read(base + got, want))
        except MemoryReadError:
            break
        got += want
    return b"".join(chunks)


def _read_region(source: MemorySource, base: int, size: int) -> bytes:
    """Read ``[base, base+size)``, falling back to a page-bounded read at a mapping boundary.

    Fast path: one read (the whole span is mapped, as on the live game). Fallback: page-by-page up
    to the first unmapped page — so a scan window overshooting the mapped region still returns the
    mapped prefix instead of nothing (and lets the offline suite use a bounded heap segment).
    """
    try:
        return source.read(base, size)
    except MemoryReadError:
        return _read_bounded(source, base, size)


def _follow_chain(
    source: MemorySource, module: str, module_base: int, base_offset: int, pointer_path: list[int]
) -> int | None:
    """Resolve ``module_base+base_offset`` through ``pointer_path``; ``None`` if a deref is dead.

    Reuses the decoder's :func:`~tekken_coach.reader.decode.resolve_anchor` — the same resolution
    the live reader performs — so a candidate is validated through the exact code path that will
    later read it. A bad intermediate pointer raises :class:`MemoryReadError`, which we swallow into
    ``None`` to prune the candidate.
    """
    anchor = Anchor(module=module, base_offset=base_offset, pointer_path=pointer_path)
    try:
        return resolve_anchor(source, anchor)
    except MemoryReadError:
        return None


# ---------------------------------------------------------------------------
# Candidate generation + the layout oracle
# ---------------------------------------------------------------------------


def find_candidate_slots(
    source: MemorySource,
    module_base: int,
    sections: Sequence[Section],
    *,
    progress: Progress | None = None,
) -> list[int]:
    """Return module-relative RVAs of 8-aligned slots in ``sections`` holding a plausible pointer.

    The *generate* half — cheap and permissive; the oracle (:func:`validate_candidate`) does the
    real filtering by following the chain. The caller picks which ``sections`` to sweep (writable
    ``.data`` first, then ``.rdata``; see :func:`locate_player_struct`), keeping the sweep off
    ``.text`` and off the huge read-only data unless needed (docs/02 §3).
    """
    rvas: list[int] = []
    for section in sections:
        data = _read_region(source, module_base + section.rva, section.virtual_size)
        if not data:
            _emit(progress, f"    section {section.name!r}: unreadable, skipped")
            continue
        region = Region(base=section.rva, data=data)
        found = value_scan_pointers(region)
        _emit(
            progress,
            f"    section {section.name!r}: {len(data) // 1024} KiB -> {len(found)} candidates",
        )
        rvas.extend(found)
    return rvas


def value_scan_pointers(region: Region) -> list[int]:
    """Every 8-aligned offset in ``region`` (RVA) whose 8-byte value is a plausible user pointer."""
    data = region.data
    out: list[int] = []
    for off in range(0, len(data) - _PTR_SIZE + 1, _PTR_SIZE):
        (value,) = struct.unpack_from("<Q", data, off)
        if _plausible_pointer(value):
            out.append(region.base + off)
    return out


@dataclass(frozen=True)
class OracleMatch:
    """An accepted candidate: the located P1 base and (when it holds) the constant stride to P2.

    ``stride is None`` is the **two-level P2** case: P1's struct passed the oracle but no struct
    reading Kazuya's id sits at a constant offset within ``max_stride`` — P2 is a separate
    allocation behind its own pointer offset (the fork's two-level ``p2_data_offset``). The
    single-anchor + stride model of :class:`~tekken_coach.reader.offsets.PlayerStruct` cannot
    express that, so the derivation reports the P1 anchor and stops rather than inventing a stride.
    """

    base_offset: int  # the static slot RVA (the durable-but-per-build anchor)
    p1_base: int  # absolute P1 (Jin) struct base in the before snapshot
    char_id: int  # P1's discovered char id (Jin — an output, not an input)
    stride: int | None  # P2_base - P1_base, or None in the two-level case

    @property
    def strong(self) -> bool:
        """Whether both players resolved — the full two-struct oracle, not just P1's fields."""
        return self.stride is not None


def _player_oracle_ok(
    source: MemorySource, base: int, spec: BaseScanSpec, m: ProbeManifest
) -> int | None:
    """If ``base`` is a plausible player struct, return its ``char_id``; else ``None``.

    The per-struct half of the oracle: a plausible ``char_id``, a plausible ``move_id``, and
    ``damage_taken == 0`` at round start. Any dead read fails the candidate.
    """
    char_id = _read_scalar(source, base + spec.char_id_offset, m.char_id_kind)
    if char_id is None or not (m.char_id_min <= char_id <= m.char_id_max):
        return None
    move_id = _read_scalar(source, base + spec.move_id_offset, m.move_id_kind)
    if move_id is None or not (m.move_id_min <= move_id < m.move_id_max):
        return None
    damage = _read_scalar(source, base + spec.damage_taken_offset, "i32")
    if damage != 0:
        return None
    return char_id


def _find_stride(
    source: MemorySource, p1_base: int, spec: BaseScanSpec, m: ProbeManifest
) -> int | None:
    """Find the smallest constant stride to a second struct reading Kazuya's id (docs/02 §4).

    Reads one bounded window ``[p1_base, p1_base + max_stride)`` and value-scans it for Kazuya's
    char id at the ``char_id`` offset — tractable inside the located region (not over the heap).
    Returns the stride, or ``None`` when P2 is not at a constant offset from P1 (the two-level case
    the runbook flags: P2 is a separate allocation, a per-player-anchor schema change).
    """
    window = _read_region(source, p1_base, spec.max_stride)
    if not window:
        return None
    region = Region(base=p1_base, data=window)
    hits = value_scan(region, m.kazuya_char_id, m.char_id_kind, align=m.scan_align)
    for hit in sorted(hits):
        p2_base = hit - spec.char_id_offset
        stride = p2_base - p1_base
        if stride <= 0 or stride > spec.max_stride:
            continue
        if _player_oracle_ok(source, p2_base, spec, m) == m.kazuya_char_id:
            return stride
    return None


def validate_candidate(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    base_offset: int,
    spec: BaseScanSpec,
    manifest: ProbeManifest,
) -> OracleMatch | None:
    """Follow the chain from a candidate slot and accept it iff the layout oracle passes.

    Acceptance: the chain lands on a struct whose ``char_id`` is a plausible non-Kazuya id (P1/Jin)
    with ``damage_taken == 0`` and a plausible ``move_id``; a *strong* acceptance additionally finds
    a struct reading Kazuya's id at a constant stride (P2). That mutual two-struct consistency is
    the code-anchored analogue of C4c's health/stride confirmation. A P1-only (weak) match is
    returned with ``stride=None`` so the caller can report the two-level case instead of guessing;
    ``None`` rejects the candidate outright.
    """
    p1_base = _follow_chain(source, module, module_base, base_offset, spec.pointer_path)
    if p1_base is None:
        return None
    p1_id = _player_oracle_ok(source, p1_base, spec, manifest)
    if p1_id is None or p1_id == manifest.kazuya_char_id:
        return None
    stride = _find_stride(source, p1_base, spec, manifest)
    return OracleMatch(base_offset=base_offset, p1_base=p1_base, char_id=p1_id, stride=stride)


@dataclass(frozen=True)
class Located:
    """A successful locate: the accepted oracle match plus the parsed module image."""

    match: OracleMatch
    image: ModuleImage
    ambiguous_weak: bool = False  # >1 distinct P1-only landing, none confirmed by a P2 stride
    from_signature: bool = False  # re-found via the previous table's AOB, skipping the full sweep


class SweepSpec(Protocol):
    """The two knobs :func:`_section_passes` needs; both scan specs supply them (docs/02 §3)."""

    scan_data_only: bool
    scan_writable_first: bool


def _section_passes(image: ModuleImage, spec: SweepSpec) -> list[tuple[str, tuple[Section, ...]]]:
    """The ordered section sweeps: writable ``.data`` first, then read-only ``.rdata`` as fallback.

    The root pointer is a runtime-written global, so it lives in writable ``.data`` — usually far
    smaller than ``.rdata``, so sweeping it first is both likelier to hit and much cheaper. A second
    pass over ``.rdata`` only runs if ``.data`` yields no match, so correctness is unchanged if the
    assumption is ever wrong. ``scan_writable_first=False`` restores the single all-data sweep.

    Shared by the player-struct sweep (:func:`locate_player_struct`) and the global/match-struct
    sweep (:func:`global_candidates`) — both chase a runtime-written global pointer.
    """
    if not spec.scan_data_only:
        return [("all sections", image.sections)]
    if spec.scan_writable_first and image.writable_data_sections():
        return [
            ("writable .data", image.writable_data_sections()),
            ("read-only .rdata", image.readonly_data_sections()),
        ]
    return [("data sections", image.data_sections())]


def _validate_all(
    source: MemorySource,
    candidates: Sequence[int],
    *,
    module: str,
    module_base: int,
    spec: BaseScanSpec,
    manifest: ProbeManifest,
    progress: Progress | None,
) -> tuple[OracleMatch | None, list[OracleMatch]]:
    """Validate every candidate; return the first strong match (short-circuit) and any weak ones."""
    weak: list[OracleMatch] = []
    total = len(candidates)
    for i, base_offset in enumerate(candidates):
        if progress is not None and i and i % _PROGRESS_EVERY == 0:
            _emit(progress, f"    validated {i}/{total} candidates ...")
        match = validate_candidate(
            source,
            module=module,
            module_base=module_base,
            base_offset=base_offset,
            spec=spec,
            manifest=manifest,
        )
        if match is None:
            continue
        if match.strong:
            return match, weak
        weak.append(match)
    return None, weak


def locate_player_struct(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    hint: AobSignature | None = None,
    progress: Progress | None = None,
) -> Located | None:
    """Parse the PE bounds, sweep candidate slots, and return the accepted oracle match.

    ``hint`` is the previous table's AOB signature: if it re-matches to a unique slot whose chain
    still satisfies the oracle, that is the answer and the full sweep is skipped — the fast re-find
    path the signature exists for. It is a *hint*, never a shortcut around validation: a signature
    that matches but fails the oracle falls through to the sweep.

    A **strong** match (both players resolved at a constant stride) wins immediately — that is the
    two-struct consistency the acceptance rests on. The sweep runs in passes (writable ``.data``,
    then ``.rdata``); the first strong match ends it. Only if every pass produces none do we fall
    back to a **weak** P1-only match, and then only if every weak candidate lands on the *same*
    struct; otherwise the landing is ambiguous and we say so rather than pick one. The weak path
    exists to report the two-level-P2 case (see :class:`OracleMatch`), never to write a table.

    ``progress`` (optional) makes the long live sweep observable — the command layer supplies a
    printer; the library is silent without one (docs/02 §2). Factored out of
    :func:`derive_base_layout` so the live orchestration can run it once (to find where the struct
    lives), freeze that region, and re-drive the position scan against a frozen before-snapshot
    (:class:`LayeredMemorySource`). ``None`` when the manifest lacks a ``base_scan`` spec or nothing
    landed on a plausible player struct at all.
    """
    spec = manifest.base_scan
    if spec is None:
        return None
    image = parse_module_image(_module_reader(source, module_base))
    _emit(
        progress,
        f"  parsed PE: SizeOfImage {image.size_of_image // 1024} KiB, "
        f"{len(image.sections)} sections",
    )

    if hint is not None:
        hinted = find_by_signature(source, module_base, image, hint)
        if hinted is not None:
            match = validate_candidate(
                source,
                module=module,
                module_base=module_base,
                base_offset=hinted,
                spec=spec,
                manifest=manifest,
            )
            if match is not None and match.strong:
                _emit(progress, "  re-found via seed AOB signature (fast path); sweep skipped")
                return Located(match=match, image=image, from_signature=True)

    weak: list[OracleMatch] = []
    for label, sections in _section_passes(image, spec):
        if not sections:
            continue
        _emit(progress, f"  sweeping {label} ...")
        candidates = find_candidate_slots(source, module_base, sections, progress=progress)
        _emit(progress, f"  validating {len(candidates)} candidates in {label} ...")
        strong, pass_weak = _validate_all(
            source,
            candidates,
            module=module,
            module_base=module_base,
            spec=spec,
            manifest=manifest,
            progress=progress,
        )
        if strong is not None:
            _emit(
                progress,
                f"  strong match in {label}: base_offset 0x{strong.base_offset:x}, "
                f"stride 0x{strong.stride:x}",
            )
            return Located(match=strong, image=image)
        weak.extend(pass_weak)
    if not weak:
        _emit(progress, "  no candidate landed on a plausible player struct")
        return None
    landings = {m.p1_base for m in weak}
    return Located(match=weak[0], image=image, ambiguous_weak=len(landings) > 1)


# ---------------------------------------------------------------------------
# AOB signature around the accepted slot (the durable re-find artifact)
# ---------------------------------------------------------------------------


def extract_signature(
    source: MemorySource, module_base: int, base_offset: int, spec: BaseScanSpec
) -> AobSignature | None:
    """Build an AOB signature around the pointer slot, wildcarding the (per-build) pointer bytes.

    Captures ``aob_window_before`` bytes before the slot and ``aob_window_after`` after the 8
    pointer bytes; the pointer's own 8 bytes become wildcards (they shift every build) while the
    data stays fixed. A re-run (:func:`find_by_signature`) scans for the pattern and recovers the
    slot at ``match + slot_delta``. Returns ``None`` if the window is unreadable.
    """
    before = spec.aob_window_before
    after = spec.aob_window_after
    start = base_offset - before
    total = before + _PTR_SIZE + after
    try:
        window = source.read(module_base + start, total)
    except MemoryReadError:
        return None
    if len(window) != total:
        return None
    tokens: list[str] = []
    for i, byte in enumerate(window):
        if before <= i < before + _PTR_SIZE:
            tokens.append("??")
        else:
            tokens.append(f"{byte:02X}")
    return AobSignature(pattern=" ".join(tokens), slot_delta=before)


def find_by_signature(
    source: MemorySource, module_base: int, image: ModuleImage, signature: AobSignature
) -> int | None:
    """Re-find a slot's ``base_offset`` by scanning data sections for ``signature`` (docs/02 §3).

    The fast path a subsequent run takes instead of the full candidate sweep: one unique match
    yields ``base_offset = match_rva + slot_delta``. Returns ``None`` on no match or an ambiguous
    (multiple) match — an ambiguous signature falls back to the full oracle scan, never a guess.
    """
    matches: list[int] = []
    for section in image.data_sections():
        data = _read_region(source, module_base + section.rva, section.virtual_size)
        if not data:
            continue
        region = Region(base=section.rva, data=data)
        for hit in aob_scan(region, signature.pattern):
            matches.append(hit + signature.slot_delta)
    unique = sorted(set(matches))
    if len(unique) != 1:
        return None
    return unique[0]


# ---------------------------------------------------------------------------
# The global/match struct: the same technique, one struct over (C4e Phase 1)
# ---------------------------------------------------------------------------
#
# The player oracle is *structural* — a struct whose fields read plausibly at one instant. The
# global struct has no such signature: a frame counter is just a u32. Its oracle is therefore
# **behavioral**, and needs two snapshots taken seconds apart: one offset ticks up (the frame
# counter), one holds steady in 1..k (the round). Coincidence is cheap for either alone and
# expensive for both at once, at offsets the fork's data file already says carry match state.

# The fields the behavioral oracle can assign, in the order it assigns them: most constrained
# first, so a small-int round is never mistaken for a small-int phase code.
_GLOBAL_ASSIGN_ORDER = ("frame_counter", "round", "timer_ms", "match_phase")
_GLOBAL_REQUIRED = ("frame_counter", "round")


@dataclass(frozen=True)
class GlobalCandidate:
    """A resolved global-struct landing at the *before* instant, awaiting the temporal oracle."""

    base_offset: int  # the static slot RVA
    pointer_path: tuple[int, ...]  # the chain shape that resolved it
    gbase: int  # absolute struct base in the before snapshot
    before: dict[int, int]  # field_offset -> value at round start


@dataclass(frozen=True)
class GlobalMatch:
    """An accepted global anchor: the slot, the chain, and the behavior-assigned field offsets."""

    base_offset: int
    pointer_path: tuple[int, ...]
    gbase: int
    offsets: dict[str, int]  # frame_counter/round[/timer_ms/match_phase] -> within-struct offset
    before: dict[int, int]  # raw values at both instants, for the diagnostic report
    after: dict[int, int]


@dataclass(frozen=True)
class GlobalLocated:
    """The accepted global match plus how many distinct landings passed the oracle."""

    match: GlobalMatch
    accepted: int

    @property
    def ambiguous(self) -> bool:
        """More than one landing satisfied frame-counter + round; the pick is not trustworthy."""
        return self.accepted > 1


def assign_global_fields(
    before: dict[int, int], after: dict[int, int], spec: GlobalScanSpec
) -> dict[str, int]:
    """Assign the seeded offsets to match fields **by behavior** across two snapshots (pure).

    We know these offsets carry match state; we do not know which is which, and guessing would bake
    a coincidence into a data file. So each field is claimed by the behavior only it exhibits, most
    constrained first:

    * ``frame_counter`` — strictly increased, by a plausible delta (a live counter, not a checksum).
    * ``round`` — held **constant** in ``[round_min, round_max]`` (the round does not turn over
      while the user walks a step).
    * ``timer_ms`` — strictly **decreased** and within ``[0, timer_ms_max]``: a round clock counts
      down. Optional — practice mode often freezes the clock, and a frozen clock is not
      distinguishable from any other constant.
    * ``match_phase`` — whatever is left that reads as a small code. Weakest, hence last, hence
      optional.

    Returns the offsets it could claim. Missing ``frame_counter`` or ``round`` means this landing is
    not the global struct; the caller rejects it (see :data:`_GLOBAL_REQUIRED`).
    """
    claimed: dict[str, int] = {}
    taken: set[int] = set()

    def claim(name: str, predicate: Callable[[int, int], bool]) -> None:
        for offset in sorted(before):
            if offset in taken or offset not in after:
                continue
            if predicate(before[offset], after[offset]):
                claimed[name] = offset
                taken.add(offset)
                return

    for name in _GLOBAL_ASSIGN_ORDER:
        if name == "frame_counter":
            claim(name, lambda old, new: spec.frame_delta_min <= new - old <= spec.frame_delta_max)
        elif name == "round":
            claim(name, lambda old, new: old == new and spec.round_min <= old <= spec.round_max)
        elif name == "timer_ms":
            claim(name, lambda old, new: new < old and 0 <= old <= spec.timer_ms_max)
        else:
            claim(name, lambda old, _new: 0 <= old < spec.match_phase_max)
    return claimed


def _read_global_fields(
    source: MemorySource, gbase: int, spec: GlobalScanSpec
) -> dict[int, int] | None:
    """Read every seeded field offset at ``gbase``; ``None`` if any is unreadable (dead landing)."""
    values: dict[int, int] = {}
    for offset in spec.field_offsets:
        value = _read_scalar(source, gbase + offset, spec.field_kind)
        if value is None:
            return None
        values[offset] = value
    return values


def global_candidates(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    image: ModuleImage,
    spec: GlobalScanSpec,
    progress: Progress | None = None,
) -> list[GlobalCandidate]:
    """Sweep static data for pointer slots whose chain lands on a *plausible* global struct.

    The **instant-A** half of the oracle, run at round start: follow every seeded chain shape from
    every candidate slot and keep the landings where all seeded field offsets are readable and at
    least one reads as a plausible round number. That is a weak filter by design — it exists only to
    shrink the set the temporal oracle (:func:`confirm_global`) has to re-read after the user acts,
    and a weak filter cannot produce a false accept, only extra work.

    Split from :func:`confirm_global` because the two halves must observe **different instants**: a
    single live process handle only ever reads *now*, so the frame-counter delta the oracle needs
    only exists across the user's action prompt.
    """
    candidates: list[GlobalCandidate] = []
    seen: set[tuple[int, tuple[int, ...]]] = set()
    for label, sections in _section_passes(image, spec):
        if not sections:
            continue
        _emit(progress, f"  sweeping {label} for global-struct pointers ...")
        slots = find_candidate_slots(source, module_base, sections, progress=progress)
        for i, base_offset in enumerate(slots):
            if i >= spec.max_candidates:
                _emit(progress, f"    candidate ceiling {spec.max_candidates} reached; stopping")
                break
            for path in spec.pointer_paths:
                gbase = _follow_chain(source, module, module_base, base_offset, path)
                if gbase is None:
                    continue
                key = (gbase, tuple(path))
                if key in seen:
                    continue
                values = _read_global_fields(source, gbase, spec)
                if values is None:
                    continue
                if not any(spec.round_min <= v <= spec.round_max for v in values.values()):
                    continue
                seen.add(key)
                candidates.append(
                    GlobalCandidate(
                        base_offset=base_offset,
                        pointer_path=tuple(path),
                        gbase=gbase,
                        before=values,
                    )
                )
        if candidates:
            _emit(progress, f"  {len(candidates)} global candidates from {label}")
            break
    return candidates


def confirm_global(
    source_after: MemorySource,
    candidates: Sequence[GlobalCandidate],
    *,
    module: str,
    module_base: int,
    spec: GlobalScanSpec,
    progress: Progress | None = None,
) -> GlobalLocated | None:
    """The **instant-B** half: re-resolve each candidate and apply the temporal oracle.

    Re-resolves the chain from the *slot* rather than reusing the recorded ``gbase`` — the global
    struct may have been reallocated between the snapshots, exactly as the player struct is, and
    re-walking the chain is what makes the anchor survive that (docs/02 §3).

    Several distinct landings satisfying the oracle is reported (:attr:`GlobalLocated.ambiguous`)
    rather than silently resolved by taking the first.
    """
    accepted: list[GlobalMatch] = []
    for candidate in candidates:
        gbase = _follow_chain(
            source_after, module, module_base, candidate.base_offset, list(candidate.pointer_path)
        )
        if gbase is None:
            continue
        after = _read_global_fields(source_after, gbase, spec)
        if after is None:
            continue
        offsets = assign_global_fields(candidate.before, after, spec)
        if any(name not in offsets for name in _GLOBAL_REQUIRED):
            continue
        accepted.append(
            GlobalMatch(
                base_offset=candidate.base_offset,
                pointer_path=candidate.pointer_path,
                gbase=gbase,
                offsets=offsets,
                before=candidate.before,
                after=after,
            )
        )
    if not accepted:
        _emit(progress, "  no global candidate passed the frame-counter/round oracle")
        return None
    _emit(progress, f"  global anchor: +0x{accepted[0].base_offset:x} ({len(accepted)} accepted)")
    return GlobalLocated(match=accepted[0], accepted=len(accepted))


def locate_global_struct(
    before: MemorySource,
    after: MemorySource,
    *,
    module: str,
    module_base: int,
    spec: GlobalScanSpec,
    image: ModuleImage | None = None,
    progress: Progress | None = None,
) -> GlobalLocated | None:
    """Compose both halves against two snapshot sources (the offline/simple path).

    The live orchestration cannot use this — it must run :func:`global_candidates` at round start,
    prompt the user to act, and only then run :func:`confirm_global` — but every offline test and
    any caller holding two frozen snapshots can.
    """
    if image is None:
        image = parse_module_image(_module_reader(before, module_base))
    candidates = global_candidates(
        before, module=module, module_base=module_base, image=image, spec=spec, progress=progress
    )
    if not candidates:
        return None
    return confirm_global(
        after,
        candidates,
        module=module,
        module_base=module_base,
        spec=spec,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# In-struct health + position (tractable once the struct is located)
# ---------------------------------------------------------------------------


def _derive_health(
    source: MemorySource,
    match: OracleMatch,
    stride: int,
    spec: BaseScanSpec,
    m: ProbeManifest,
    span: int,
) -> DerivedField | None:
    """Find a direct health field: an offset reading full HP at round start in **both** players.

    Usually ``None`` on Tekken 8 — the struct has no direct HP field (docs/02 §3), so the caller
    falls back to computed health. Kept as a first try in case a build does expose one within
    ``span`` (bounded to the stride so the P2 cross-check stays inside P2's struct).
    """
    window = _read_region(source, match.p1_base, span)
    if not window:
        return None
    region = Region(base=match.p1_base, data=window)
    hits = value_scan(region, spec.round_start_health, m.health_kind, align=m.scan_align)
    for hit in sorted(hits):
        off = hit - match.p1_base
        p2_val = _read_scalar(source, match.p1_base + stride + off, m.health_kind)
        if p2_val == spec.round_start_health:
            return DerivedField(
                name="health",
                scope="player",
                offset=off,
                kind=m.health_kind,
                example_address=match.p1_base + off,
                confidence=Confidence.high,
                method=f"both players read round-start max {spec.round_start_health}",
            )
    return None


def _derive_position(
    before: MemorySource,
    after: MemorySource,
    match: OracleMatch,
    after_base: int,
    spec: BaseScanSpec,
    m: ProbeManifest,
    span: int,
) -> int | None:
    """Offset of a moving (x,y,z) float triple in P1's struct across the two snapshots.

    Reads P1 at its before base and its (re-resolved) after base, so it is correct even if the chain
    re-resolved to a moved allocation between snapshots — the whole point of anchoring in code.
    ``span`` bounds the search to P1's struct (the stride) so it never wanders into P2.
    """
    for off in range(0, span - 8, m.scan_align):
        triple_ok = True
        x_moved = False
        for k in range(3):
            va = _read_f32(before, match.p1_base + off + 4 * k)
            vb = _read_f32(after, after_base + off + 4 * k)
            if va is None or vb is None:
                triple_ok = False
                break
            if not _plausible_coord(va, m) or not _plausible_coord(vb, m):
                triple_ok = False
                break
            if k == 0 and va != vb:
                x_moved = True
        if triple_ok and x_moved:
            return off
    return None


def _read_f32(source: MemorySource, address: int) -> float | None:
    try:
        raw = source.read(address, 4)
    except MemoryReadError:
        return None
    if len(raw) != 4:
        return None
    return float(struct.unpack("<f", raw)[0])


# ---------------------------------------------------------------------------
# The transform component: position lives outside the entity struct (C4e Phase 3)
# ---------------------------------------------------------------------------
#
# `_derive_position` above scans the entity struct itself and, on the real Tekken 8 build, finds
# nothing: there is no moving float triple anywhere in the struct, and the fork's layout data has no
# position field either. Position lives in a separate Unreal transform component that the entity
# points at. So the search goes one (or two) pointers deep: sweep the entity's own pointer slots for
# one whose pointee holds a triple that moves when the player walks.


def _moving_triple(
    before: Region, after: Region, base: int, span: int, spec: ComponentScanSpec, m: ProbeManifest
) -> int | None:
    """Offset within ``[base, base+span)`` of an (x,y,z) f32 triple whose ``x`` really moved (pure).

    ``min_delta`` is what separates a coordinate from a float that merely jitters (an animation
    weight, a blend factor); ``_plausible_coord`` rejects ints reinterpreted as denormals.
    """
    for off in range(0, span - 8, spec.probe_align):
        values: list[tuple[float, float]] = []
        for k in range(3):
            va = before.read_scalar(base + off + 4 * k, "f32")
            vb = after.read_scalar(base + off + 4 * k, "f32")
            if va is None or vb is None:
                break
            if not _plausible_coord(float(va), m) or not _plausible_coord(float(vb), m):
                break
            values.append((float(va), float(vb)))
        if len(values) != 3:
            continue
        if abs(values[0][1] - values[0][0]) >= spec.min_delta:
            return off
    return None


def _probe_regions(
    before: MemorySource, after: MemorySource, address: int, span: int
) -> tuple[Region, Region] | None:
    """Bulk-read the same span from both snapshots, or ``None`` if either is too short to hold a
    triple. Bulk reads keep the two snapshots *coherent* — a per-float live read would sample each
    axis at a different instant."""
    data_b = _read_region(before, address, span)
    data_a = _read_region(after, address, span)
    usable = min(len(data_b), len(data_a))
    if usable < 12:
        return None
    return Region(base=address, data=data_b[:usable]), Region(base=address, data=data_a[:usable])


def _stable_pointers(
    before: Region, after: Region, span: int, align: int, limit: int
) -> list[tuple[int, int]]:
    """``(slot_offset, pointer)`` for slots holding the *same* plausible pointer in both snapshots.

    A component object does not reallocate while the player takes a step, so a slot whose pointer
    changed is not the one we want — and this prunes the overwhelming majority of slots before any
    further read happens.
    """
    out: list[tuple[int, int]] = []
    usable = min(len(before.data), len(after.data), span)
    for off in range(0, usable - _PTR_SIZE + 1, align):
        (pb,) = struct.unpack_from("<Q", before.data, off)
        (pa,) = struct.unpack_from("<Q", after.data, off)
        if pb != pa or not _plausible_pointer(pb):
            continue
        out.append((off, pb))
        if len(out) >= limit:
            break
    return out


def _triple_in_component(
    before: MemorySource,
    after: MemorySource,
    component: int,
    spec: ComponentScanSpec,
    m: ProbeManifest,
) -> int | None:
    """Scan one component object for the moving triple."""
    regions = _probe_regions(before, after, component, spec.probe_span)
    if regions is None:
        return None
    rb, ra = regions
    return _moving_triple(rb, ra, component, len(rb.data), spec, m)


def _confirm_on_p2(
    source: MemorySource,
    p2_base: int,
    slot_offset: int,
    path: Sequence[int],
    triple_offset: int,
    p1_pos: tuple[float, float, float],
    m: ProbeManifest,
) -> bool:
    """The same component path must resolve for P2 to a *different*, plausible position.

    Mutual two-struct consistency, exactly as the player oracle uses: a coincidental float triple in
    P1's neighborhood will not also be a plausible, distinct coordinate at the identical path from
    P2's base. The two characters stand apart at round start, so identical positions mean we
    resolved the same object twice — a path through a shared/global object, not a per-player one.
    """
    address = _read_scalar(source, p2_base + slot_offset, "ptr")
    if address is None or not _plausible_pointer(address):
        return False
    for offset in path:
        address = _read_scalar(source, address + offset, "ptr")
        if address is None or not _plausible_pointer(address):
            return False
    p2_pos: list[float] = []
    for k in range(3):
        value = _read_f32(source, address + triple_offset + 4 * k)
        if value is None or not _plausible_coord(value, m):
            return False
        p2_pos.append(value)
    return tuple(p2_pos) != p1_pos


def find_transform_component(
    before: MemorySource,
    after: MemorySource,
    *,
    p1_base: int,
    p1_after_base: int,
    p2_base: int,
    spec: ComponentScanSpec,
    manifest: ProbeManifest,
    progress: Progress | None = None,
) -> ComponentAnchor | None:
    """Locate the component holding ``pos_{x,y,z}`` behind a pointer in the entity struct.

    Candidate-generate-and-validate again, one indirection out: the entity's stable pointer slots
    are the candidates, a moving float triple inside the pointee is the oracle, and P2 resolving via
    the *same* path to a different plausible position is the acceptance (:func:`_confirm_on_p2`).
    ``max_depth >= 2`` also follows one more hop, for the Unreal ``actor -> object -> component``
    shape. Returns the :class:`ComponentAnchor` the decoder consumes, or ``None``.
    """
    structs = _probe_regions(before, after, p1_base, spec.slot_span)
    if structs is None:
        return None
    sb, sa = structs
    # `sa` is P1's struct in the after snapshot, which may sit at a different address; re-base it so
    # slot offsets line up with `sb`.
    if p1_after_base != p1_base:
        sa_data = _read_region(after, p1_after_base, len(sb.data))
        sa = Region(base=p1_base, data=sa_data[: len(sb.data)])
    slots = _stable_pointers(sb, sa, spec.slot_span, spec.slot_align, spec.max_slots)
    _emit(progress, f"  transform scan: {len(slots)} stable pointer slots in the entity struct")

    for slot_offset, pointer in slots:
        triple = _triple_in_component(before, after, pointer, spec, manifest)
        if triple is not None and _confirm_on_p2(
            before, p2_base, slot_offset, (), triple, _pos_at(before, pointer, triple), manifest
        ):
            return _component_anchor(slot_offset, (), triple, manifest)
        if spec.max_depth < 2:
            continue
        inner = _probe_regions(before, after, pointer, spec.inner_span)
        if inner is None:
            continue
        ib, ia = inner
        for inner_offset, inner_pointer in _stable_pointers(
            ib, ia, spec.inner_span, spec.slot_align, spec.max_slots
        ):
            triple = _triple_in_component(before, after, inner_pointer, spec, manifest)
            if triple is None:
                continue
            p1_pos = _pos_at(before, inner_pointer, triple)
            if _confirm_on_p2(
                before, p2_base, slot_offset, (inner_offset,), triple, p1_pos, manifest
            ):
                return _component_anchor(slot_offset, (inner_offset,), triple, manifest)
    return None


def _pos_at(source: MemorySource, component: int, triple: int) -> tuple[float, float, float]:
    """P1's coordinate triple, for the P2 distinctness check. Unreadable axes read as ``nan``."""
    values = [_read_f32(source, component + triple + 4 * k) for k in range(3)]
    return tuple(float("nan") if v is None else v for v in values)  # type: ignore[return-value]


def _component_anchor(
    slot_offset: int, path: Sequence[int], triple: int, m: ProbeManifest
) -> ComponentAnchor:
    return ComponentAnchor(
        slot_offset=slot_offset,
        pointer_path=list(path),
        fields={
            axis: FieldSpec(offset=triple + delta, kind=m.pos_kind)
            for axis, delta in (("pos_x", 0), ("pos_y", 4), ("pos_z", 8))
        },
    )


# ---------------------------------------------------------------------------
# Top-level: derive the whole player anchor + fields into a DerivationResult
# ---------------------------------------------------------------------------


def _derive_global(
    result: DerivationResult,
    source: MemorySource,
    source_after: MemorySource | None,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    seed: OffsetTable,
    global_located: GlobalLocated | None,
    sweep_global: bool,
    progress: Progress | None,
) -> None:
    """Fill ``result``'s global anchor + match fields, falling back to the seed anchor (C4e §1).

    The global struct is behind its own static pointer, so it gets the same treatment as the player
    struct: sweep, chain, oracle. Only the oracle differs (behavioral, not structural — see
    :func:`assign_global_fields`). Failing to locate it is **not** fatal: the seed anchor is carried
    and the field is flagged, exactly as C4d did for every global field.
    """
    spec = manifest.global_scan
    result.global_anchor = seed.global_struct.anchor
    if spec is None:
        result.notes.append(
            "no global_scan spec in the probe manifest; global/match anchor carried from the seed "
            "table and still needs calibration."
        )
        return
    if global_located is None and sweep_global and source_after is not None:
        global_located = locate_global_struct(
            source,
            source_after,
            module=module,
            module_base=module_base,
            spec=spec,
            progress=progress,
        )
    if global_located is None:
        result.unresolved.append("frame_counter")
        result.notes.append(
            "no static pointer slot's chain landed on a struct with a ticking frame counter and a "
            "plausible round; the global/match anchor is SEEDED (the reader's frame-monotonic "
            "check will fail). Widen global_scan.pointer_paths / field_offsets in the probe "
            "manifest — they are DATA (see runbook)."
        )
        return

    match = global_located.match
    result.global_anchor = Anchor(
        module=module, base_offset=match.base_offset, pointer_path=list(match.pointer_path)
    )
    chain = " -> ".join(f"+0x{o:x}" for o in match.pointer_path) or "(static)"
    for name, offset in match.offsets.items():
        result.fields.append(
            DerivedField(
                name=name,
                scope="global",
                offset=offset,
                kind=spec.field_kind,
                example_address=match.gbase + offset,
                confidence=Confidence.high,
                method=f"behavioral oracle: {match.before[offset]} -> {match.after[offset]} "
                f"across the two snapshots (chain {chain})",
            )
        )
    unassigned = [o for o in spec.field_offsets if o not in match.offsets.values()]
    if unassigned:
        raws = ", ".join(f"+0x{o:x}={match.before[o]}" for o in unassigned)
        result.notes.append(
            f"global offsets the behavioral oracle could not claim ({raws}) are left SEEDED. "
            "timer_ms is often unclaimable in practice mode (the round clock is frozen, so it is "
            "indistinguishable from any other constant); match_phase needs its state_codes map "
            "calibrated regardless."
        )
    if "match_phase" not in match.offsets:
        result.notes.append(
            "match_phase not assigned; the seeded offset + state_codes.match_phase map are "
            "unverified. Capture will decode a MatchState from whatever that offset holds."
        )
    if global_located.ambiguous:
        result.notes.append(
            f"{global_located.accepted} distinct landings satisfied the global oracle; the first "
            "was taken. Verify via the doctor's frame-monotonic check, and narrow "
            "global_scan.pointer_paths if it is wrong."
        )


def _seed_state_fields(result: DerivationResult, spec: BaseScanSpec, p1_base: int) -> None:
    """Write the encoded state words into the table at their known offsets (C4e §2, facts/data).

    These are **not derived**. No two-snapshot oracle can prove where ``stun_type`` lives — its
    value only changes when the player is *hit*, which the Jin-vs-Kazuya round-start setup
    deliberately never is. So the offsets come from the layout data (docs/02 §5 facts), are marked
    :attr:`~.derive.Confidence.seeded`, and the *meanings* of their values are calibrated by
    observation (docs/02 §8).

    The C4a placeholder's per-flag booleans are dropped **only when a state map is present to
    replace them**: they describe a struct that does not exist, and carrying them forward would look
    like working offsets — but removing them without installing the encoded path would leave the
    decoder with no way to read state at all.
    """
    if not spec.state_fields:
        return
    # Three kinds of seeded field, and the report must not blur them: an encoded state word whose
    # values need the §8 observation protocol; `counter_state`, whose values are read through the
    # table's own state_codes map (seeded, verifiable by inspection); and `move_frame`, a plain
    # integer where the offset is all there is to know. Saying "calibrate this" about the last one
    # sends the user chasing a meaning that does not exist.
    encoded = set(result.encoded_state.flags) if result.encoded_state is not None else set()
    for name, field_spec in spec.state_fields.items():
        if name in encoded:
            method = (
                "seeded from the T8 layout data; its VALUES need the docs/02 §8 observation "
                "calibration"
            )
        elif name == "counter_state":
            method = (
                "seeded from the T8 layout data; its values decode via state_codes.counter_state"
            )
        else:
            method = "seeded from the T8 layout data (the offset is the whole fact)"
        result.fields.append(
            DerivedField(
                name=name,
                scope="player",
                offset=field_spec.offset,
                kind=field_spec.kind,
                example_address=p1_base + field_spec.offset,
                confidence=Confidence.seeded,
                method=method,
            )
        )
    if result.encoded_state is not None:
        result.drop_player_fields.extend(spec.legacy_state_fields)


def derive_base_layout(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    seed: OffsetTable,
    source_after: MemorySource | None = None,
    located: Located | None = None,
    global_located: GlobalLocated | None = None,
    sweep_global: bool = True,
    state_map: EncodedStateSpec | None = None,
    progress: Progress | None = None,
) -> DerivationResult:
    """Run the full code-signature derivation and return a :class:`DerivationResult`.

    Reuses C4c's result/build/report ecosystem: the derived player :class:`Anchor` (with pointer
    chain + AOB signature), the stride, the char ids, and the derivable field offsets (``char_id``,
    ``move_id`` from the oracle; ``health``/``pos`` from the in-struct and component scans) go into
    the same :class:`DerivationResult` the builder consumes. C4e adds the **global/match anchor**
    (:func:`_derive_global`), the seeded **encoded state words** (:func:`_seed_state_fields`), and
    the **transform component** position lives in (:func:`find_transform_component`).

    ``located`` / ``global_located`` let the caller pass locations it already swept for; when given,
    the expensive sweeps are not repeated. The live orchestration **must** pre-sweep the global
    (its oracle needs two instants straddling the user's action, which a single live handle cannot
    manufacture after the fact) and therefore passes ``sweep_global=False`` — so that "swept, found
    nothing" is not confused with "not swept yet". ``state_map`` is the value -> meaning data the
    builder writes into the table. ``progress`` threads to the sweeps for a live-observable log.
    """
    result = DerivationResult(module=module, module_base=module_base)
    result.encoded_state = state_map
    spec = manifest.base_scan
    if spec is None:
        result.notes.append("no base_scan spec in the probe manifest; cannot run the code scan.")
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x", "frame_counter"])
        return result

    # The previous table's AOB signature is a fast re-find hint; it still has to pass the oracle.
    # A caller that already located the struct (live path) passes it in to avoid re-sweeping.
    if located is None:
        located = locate_player_struct(
            source,
            module=module,
            module_base=module_base,
            manifest=manifest,
            hint=seed.players.anchor.signature,
            progress=progress,
        )

    _derive_global(
        result,
        source,
        source_after,
        module=module,
        module_base=module_base,
        manifest=manifest,
        seed=seed,
        global_located=global_located,
        sweep_global=sweep_global,
        progress=progress,
    )
    if located is None:
        result.notes.append(
            "no static pointer slot's chain landed on a plausible player struct. Verify the "
            "Jin-vs-Kazuya round-start setup, widen the pointer_path/oracle in the manifest's "
            "base_scan, or check the module name (see runbook)."
        )
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x"])
        return result

    match, image = located.match, located.image
    if located.from_signature:
        result.notes.append(
            "player-struct slot re-found via the seed table's AOB signature (fast path), then "
            "re-validated against the layout oracle — the full candidate sweep was not needed."
        )
    signature = extract_signature(source, module_base, match.base_offset, spec)
    if signature is not None:
        rediscovered = find_by_signature(source, module_base, image, signature)
        if rediscovered != match.base_offset:
            result.notes.append(
                "AOB signature did not re-match to a unique slot; stored it but the fast re-find "
                "path will fall back to the full oracle scan (widen aob_window_* for more context)."
            )
    result.player_anchor = Anchor(
        module=module,
        base_offset=match.base_offset,
        pointer_path=list(spec.pointer_path),
        signature=signature,
    )

    if not match.strong:
        # P1 resolved but no Kazuya struct at a constant stride: P2 is a separate allocation behind
        # a two-level offset. PlayerStruct is a single anchor + stride, so we stop here rather than
        # emit a table with an invented stride — extending the schema to per-player anchors is a
        # reviewer's call, not something this scan should decide.
        result.unresolved.extend(["health", "pos_x"])
        result.notes.append(
            f"P1 (char_id {match.char_id}) located at 0x{match.p1_base:x} via "
            f"{module}+0x{match.base_offset:x} + chain, but NO struct reading Kazuya's id "
            f"({manifest.kazuya_char_id}) sits at a constant stride within 0x{spec.max_stride:x}. "
            "This is the TWO-LEVEL P2 case (the fork's p2_data_offset is two-level): P2 is a "
            "separate allocation. The single-anchor+stride PlayerStruct schema cannot express it — "
            "no table written. Either raise base_scan.max_stride if P2 is merely farther away, or "
            "extend PlayerStruct to per-player anchors (schema change, reviewer's call)."
        )
        if located.ambiguous_weak:
            result.notes.append(
                "multiple candidate slots landed on DIFFERENT structs and none was confirmed by a "
                "P2 stride — the landing is ambiguous; do not trust the reported P1 anchor."
            )
        return result

    stride = match.stride
    assert stride is not None  # match.strong
    result.stride = stride
    result.player_char_ids = (match.char_id, manifest.kazuya_char_id)

    result.fields.append(
        DerivedField(
            name="char_id",
            scope="player",
            offset=spec.char_id_offset,
            kind=manifest.char_id_kind,
            example_address=match.p1_base + spec.char_id_offset,
            confidence=Confidence.high,
            method=f"oracle: Jin={match.char_id} at P1, Kazuya={manifest.kazuya_char_id} at P2 "
            f"(stride 0x{stride:x})",
        )
    )
    result.fields.append(
        DerivedField(
            name="move_id",
            scope="player",
            offset=spec.move_id_offset,
            kind=manifest.move_id_kind,
            example_address=match.p1_base + spec.move_id_offset,
            confidence=Confidence.high,
            method="oracle: plausible move id for both players at round start",
        )
    )

    # Bound the in-struct scans to P1's struct: it spans exactly `stride` bytes (P2 begins at
    # base+stride), so a manifest struct_span larger than the stride is capped here rather than
    # bleeding into P2 and producing false matches.
    span = min(spec.struct_span, stride)

    health = _derive_health(source, match, stride, spec, manifest, span)
    if health is not None:
        result.fields.append(health)
    else:
        # Expected on Tekken 8: no direct HP field. Compute health = round_start_health -
        # damage_taken (the fork's model, docs/02 §3). Emit damage_taken as a field and set
        # max_health so the decoder computes health; this is a resolved field, not a gap.
        result.fields.append(
            DerivedField(
                name="damage_taken",
                scope="player",
                offset=spec.damage_taken_offset,
                kind="i32",
                example_address=match.p1_base + spec.damage_taken_offset,
                confidence=Confidence.high,
                method="oracle field; health computed as round_start_health - damage_taken",
            )
        )
        result.max_health = spec.round_start_health
        result.notes.append(
            f"no direct HP field in the struct (as expected on T8); health is computed as "
            f"{spec.round_start_health} - damage_taken (+0x{spec.damage_taken_offset:x}), the "
            "fork's model. Verify full HP is really "
            f"{spec.round_start_health} for this build."
        )

    # The encoded state words + move_frame + counter_state, seeded at their known offsets (C4e §2).
    _seed_state_fields(result, spec, match.p1_base)

    if source_after is None:
        result.unresolved.append("pos_x")
        result.notes.append("no second snapshot; position seeded (run with an act-then-capture).")
        return result

    after_base = _follow_chain(
        source_after, module, module_base, match.base_offset, spec.pointer_path
    )
    if after_base is None:
        result.unresolved.append("pos_x")
        result.notes.append("chain did not re-resolve in the second snapshot; position seeded.")
        return result
    if after_base != match.p1_base:
        result.notes.append(
            f"player struct reallocated between snapshots (0x{match.p1_base:x} -> "
            f"0x{after_base:x}); the static chain tracked it — this is exactly why C4d "
            "anchors in code, not the heap."
        )
    _derive_position_fields(
        result,
        source,
        source_after,
        match=match,
        after_base=after_base,
        spec=spec,
        manifest=manifest,
        span=span,
        progress=progress,
    )
    return result


def _derive_position_fields(
    result: DerivationResult,
    source: MemorySource,
    source_after: MemorySource,
    *,
    match: OracleMatch,
    after_base: int,
    spec: BaseScanSpec,
    manifest: ProbeManifest,
    span: int,
    progress: Progress | None,
) -> None:
    """Locate ``pos_{x,y,z}``: in the entity struct if it is there, else in a transform component.

    The in-struct scan runs first because it is cheap and, where it works, yields a flat field the
    decoder reads without an extra dereference. On the real Tekken 8 build it finds nothing, and the
    component scan (C4e Phase 3) takes over — position sits behind the entity's own pointer.
    """
    pos_off = _derive_position(source, source_after, match, after_base, spec, manifest, span)
    if pos_off is not None:
        for axis, delta in (("pos_x", 0), ("pos_y", 4), ("pos_z", 8)):
            result.fields.append(
                DerivedField(
                    name=axis,
                    scope="player",
                    offset=pos_off + delta,
                    kind=manifest.pos_kind,
                    example_address=match.p1_base + pos_off + delta,
                    confidence=Confidence.medium,
                    method=f"moving finite float triple at +0x{pos_off:x} (x/y/z)",
                )
            )
        return

    component_spec = spec.component_scan
    if component_spec is None:
        result.unresolved.append("pos_x")
        result.notes.append(
            f"no moving float triple found in P1's struct (scanned 0x{span:x} bytes = the full "
            "stride) and no base_scan.component_scan spec to look one pointer further; position is "
            "seeded."
        )
        return

    assert match.stride is not None  # only reached on a strong match
    component = find_transform_component(
        source,
        source_after,
        p1_base=match.p1_base,
        p1_after_base=after_base,
        p2_base=match.p1_base + match.stride,
        spec=component_spec,
        manifest=manifest,
        progress=progress,
    )
    if component is None:
        result.unresolved.append("pos_x")
        result.notes.append(
            f"no moving float triple in P1's struct (0x{span:x} bytes) and none in any component "
            f"it points at (depth {component_spec.max_depth}); position is seeded. Either P1 did "
            "not actually move between the two snapshots, or the transform is deeper / further "
            "into the component than base_scan.component_scan bounds allow — widen slot_span / "
            "probe_span / max_depth (they are DATA)."
        )
        return

    result.components[POSITION_COMPONENT] = component
    result.drop_player_fields.extend(["pos_x", "pos_y", "pos_z"])
    triple = component.fields["pos_x"].offset
    hops = " -> ".join(f"+0x{o:x}" for o in component.pointer_path) or "(direct)"
    result.notes.append(
        f"position is NOT in the entity struct; it lives in a transform component reached via "
        f"+0x{component.slot_offset:x} {hops}, triple at +0x{triple:x}. Confirmed by resolving the "
        "same path from P2's base to a different, plausible coordinate. The table carries it as a "
        f"{POSITION_COMPONENT!r} component anchor (schema change, docs/03 §1)."
    )


class LayeredMemorySource:
    """A read-only :class:`MemorySource` overlaying frozen regions on a live fallback source.

    The live position scan needs a *before* snapshot (round start) and an *after* snapshot (post
    action), but a single live process handle only ever reads the current instant. This layers a
    frozen :class:`~tekken_coach.reader.discovery.scanners.Region` (captured at round start) over
    the live source: reads wholly inside a frozen region are served from the freeze; everything
    else (the PE header, data sections, pointer-chain heap) falls through to the live source. It is
    read-only — no write/inject method — mirroring the seam (docs/02 §2).
    """

    def __init__(self, overlay: Sequence[Region], fallback: MemorySource) -> None:
        self._overlay = list(overlay)
        self._fallback = fallback

    def read(self, address: int, size: int) -> bytes:
        for region in self._overlay:
            if region.covers(address, size):
                return region.read(address, size)
        return self._fallback.read(address, size)

    def module_base(self, module: str) -> int:
        return self._fallback.module_base(module)
