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
3. **Validate** against the oracle, in two stages:

   * *structurally*, at round start: follow the seed :attr:`~.manifest.BaseScanSpec.pointer_path`
     from the slot and keep the landing only if it is struct-shaped for **both** players — a
     plausible ``char_id``, a plausible ``move_id``, ``damage_taken == 0``, and the two players'
     ids forming ``{Jin, Kazuya=12}``. Mutual multi-field consistency across the two symmetric
     structs, anchored in code rather than the heap.
   * *behaviorally*, across an **action window**: the acting player's ``move_id`` must **change**
     from its round-start value in at least one sample while they walk, jab and jump
     (:func:`confirm_players`). This is what a single instant cannot say. A coincidental struct
     whose fields merely *read* plausibly at round start is not rejected by any amount of
     structural checking — it is rejected the moment we ask it to move, because it does not
     (confirmed live on 5.02.01: the C4e scan locked onto a struct whose ``move_id`` stayed 0 while
     the player jumped). ``damage_taken`` rising on the *opponent* corroborates a connected jab, but
     is not required — the jab may whiff.
4. **Persist an AOB signature** around the accepted slot (pointer bytes wildcarded) so a re-run
   re-finds it fast (:func:`extract_signature` / :func:`find_by_signature`), stored in the table as
   facts/data (docs/02 §5).
5. **Fill in** health + position with C4c's value/position scans, now tractable *inside the located
   struct* rather than over the whole heap.

Both oracles are therefore temporal, and they read the **same** action the position scan needs.

**Why a window and not two instants** (C4g). C4f compared ``move_id`` at round start against
``move_id`` at the moment the user pressed Enter, and that oracle cannot be satisfied by a human:
``move_id`` is *transient*. A jab or a jump rewrites it for roughly half a second, after which the
character returns to idle and ``move_id`` returns to the value it held at round start. Alt-tabbing
from the game to the terminal takes longer than the animation, so the acting player's struct read
*identically* at both instants and was rejected as frozen — the live scan found zero behaving
candidates and wrote no table, not because the struct was absent but because the test was
unsatisfiable. So the oracle samples a series of snapshots spanning the action and accepts a
candidate whose ``move_id`` differed from its round-start value in **any** sample. The corroborators
become "ever changed" / "ever damaged" the same way. Nothing about the argument weakens: a frozen
decoy is frozen in every sample.

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

import math
import struct
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
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
from tekken_coach.reader.memory_source import MemoryRegion, MemorySource
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

    ``move_id`` / ``p2_move_id`` are the round-start readings every sample in the action window is
    compared against (:func:`confirm_players`); without them the match is a single frozen instant,
    which is exactly what accepted the wrong struct on the live game.
    """

    base_offset: int  # the static slot RVA (the durable-but-per-build anchor)
    p1_base: int  # absolute P1 (Jin) struct base in the before snapshot
    char_id: int  # P1's discovered char id (Jin — an output, not an input)
    stride: int | None  # P2_base - P1_base, or None in the two-level case
    move_id: int  # P1's move_id at the *before* instant (no default: 0 is a frozen field's value)
    p2_move_id: int | None = None  # P2's, when a stride resolved

    @property
    def strong(self) -> bool:
        """Whether both players resolved — the full two-struct oracle, not just P1's fields."""
        return self.stride is not None


@dataclass(frozen=True)
class _Observation:
    """One sample of a candidate struct taken during the action window (an instant, re-resolved)."""

    moves: tuple[int, int]  # (P1, P2) move_id as read in this sample
    damage: int | None  # the opponent's damage_taken, or None if unreadable
    p1_base: int  # where the chain resolved in this sample


@dataclass(frozen=True)
class Behavior:
    """What a candidate struct **did** across the action window (C4f/C4g — the decisive evidence).

    :attr:`accepted` is the whole oracle: the acting player's ``move_id`` differed from its
    round-start value in **at least one** sample. Not "differed at the end" — a jab and a jump each
    rewrite ``move_id`` for about half a second and then the character idles back to exactly the
    value it started at, so an end-of-window comparison rejects the real struct (C4g: this is the
    bug that made C4f's correct oracle unrunnable on the live game).

    Every other signal is corroboration used only to *rank* several accepted candidates, never to
    accept one — a jab may whiff (no opponent damage) and the dummy may stand still (no opponent
    ``move_id`` change), so requiring either would reject the real struct on a bad run. They are
    "ever" signals too: the dummy's hit reaction ends inside the window just as the jab does.
    """

    acting_move_changed: bool
    opponent_move_changed: bool
    opponent_damaged: bool
    p1_after_base: int  # where the chain resolved in the LAST sample (the position scan's "after")
    samples: int  # how many window samples this candidate was readable in

    @property
    def accepted(self) -> bool:
        return self.acting_move_changed

    @property
    def score(self) -> int:
        """0-3: how much of the expected action this struct actually exhibited."""
        return sum((self.acting_move_changed, self.opponent_move_changed, self.opponent_damaged))

    def describe(self) -> str:
        signals = [
            name
            for name, seen in (
                ("acting move_id changed", self.acting_move_changed),
                ("opponent move_id changed", self.opponent_move_changed),
                ("opponent damage_taken rose", self.opponent_damaged),
            )
            if seen
        ]
        return f"{', '.join(signals)} (over {self.samples} samples)"


def _player_oracle_ok(
    source: MemorySource, base: int, spec: BaseScanSpec, m: ProbeManifest
) -> int | None:
    """If ``base`` is a plausible player struct, return its ``char_id``; else ``None``.

    The per-struct, per-instant half of the oracle: a plausible ``char_id``, a plausible
    ``move_id``, and ``damage_taken == 0`` at round start. Any dead read fails the candidate.

    Necessary, never sufficient — and deliberately permissive at the bottom of the ``char_id``
    range. A page of **zeroed** memory reads as char id 0 / move id 0 / damage 0 and passes this
    whole function, which is why C4f set ``char_id_min`` to 1; C4g put it back to 0, because Jin's
    id may really be 0 on this build and a sieve that drops the answer cannot be repaired
    downstream. Zeroes still never reach a table: :func:`_find_stride` needs a struct reading
    Kazuya's id at a constant offset (nothing in a zeroed page reads 12), so a zeroed landing is at
    most a *weak* match, and :func:`confirm_players` then requires a ``move_id`` that changes, which
    zeroes cannot do. Both backstops hold independently.
    """
    char_id = _read_scalar(source, base + spec.char_id_offset, m.char_id_kind)
    if char_id is None or not (m.char_id_min <= char_id <= m.char_id_max):
        return None
    if _read_move_id(source, base, spec, m) is None:
        return None
    damage = _read_scalar(source, base + spec.damage_taken_offset, "i32")
    if damage != 0:
        return None
    return char_id


def _read_move_id(
    source: MemorySource, base: int, spec: BaseScanSpec, m: ProbeManifest
) -> int | None:
    """``base``'s ``move_id`` if it reads in the plausible range, else ``None``."""
    move_id = _read_scalar(source, base + spec.move_id_offset, m.move_id_kind)
    if move_id is None or not (m.move_id_min <= move_id < m.move_id_max):
        return None
    return move_id


def _find_stride(
    source: MemorySource, p1_base: int, spec: BaseScanSpec, m: ProbeManifest
) -> tuple[int, int] | None:
    """Find the smallest constant stride to a second struct reading Kazuya's id (docs/02 §4).

    Reads one bounded window ``[p1_base, p1_base + max_stride)`` and value-scans it for Kazuya's
    char id at the ``char_id`` offset — tractable inside the located region (not over the heap).
    Returns ``(stride, p2_move_id)``, or ``None`` when P2 is not at a constant offset from P1 (the
    two-level case the runbook flags: P2 is a separate allocation, a per-player-anchor schema
    change). P2's round-start ``move_id`` rides along because :func:`confirm_players` needs it to
    tell whether the dummy moved too.
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
        if _player_oracle_ok(source, p2_base, spec, m) != m.kazuya_char_id:
            continue
        p2_move_id = _read_move_id(source, p2_base, spec, m)
        if p2_move_id is not None:
            return stride, p2_move_id
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
    """Follow the chain from a candidate slot and accept it iff the *structural* oracle passes.

    Acceptance: the chain lands on a struct whose ``char_id`` is a plausible non-Kazuya id (P1/Jin)
    with ``damage_taken == 0`` and a plausible ``move_id``; a *strong* acceptance additionally finds
    a struct reading Kazuya's id at a constant stride (P2). That mutual two-struct consistency is
    the code-anchored analogue of C4c's health/stride confirmation. A P1-only (weak) match is
    returned with ``stride=None`` so the caller can report the two-level case instead of guessing;
    ``None`` rejects the candidate outright.

    This is one instant, so several unrelated structs can pass it. :func:`confirm_players` is what
    picks the real one.
    """
    p1_base = _follow_chain(source, module, module_base, base_offset, spec.pointer_path)
    if p1_base is None:
        return None
    p1_id = _player_oracle_ok(source, p1_base, spec, manifest)
    if p1_id is None or p1_id == manifest.kazuya_char_id:
        return None
    move_id = _read_move_id(source, p1_base, spec, manifest)
    assert move_id is not None  # _player_oracle_ok already read it in range
    found = _find_stride(source, p1_base, spec, manifest)
    stride, p2_move_id = found if found is not None else (None, None)
    return OracleMatch(
        base_offset=base_offset,
        p1_base=p1_base,
        char_id=p1_id,
        stride=stride,
        move_id=move_id,
        p2_move_id=p2_move_id,
    )


@dataclass(frozen=True)
class Located:
    """A successful locate: the accepted oracle match plus the parsed module image."""

    match: OracleMatch
    image: ModuleImage
    ambiguous_weak: bool = False  # >1 distinct P1-only landing, none confirmed by a P2 stride
    from_signature: bool = False  # re-found via the previous table's AOB, skipping the full sweep
    behavior: Behavior | None = None  # what the accepted struct did across the snapshots
    accepted: int = 1  # distinct landings that passed the oracle (>1 means the pick is a guess)
    considered: int = 1  # structural candidates the behavioral oracle had to choose between

    @property
    def ambiguous(self) -> bool:
        """More than one distinct struct satisfied the whole oracle; the pick is not trustworthy."""
        return self.accepted > 1


@dataclass(frozen=True)
class PlayerCandidates:
    """Structurally-plausible player landings at the *before* instant, awaiting the action.

    Split out of :func:`locate_player_struct` for the same reason :func:`global_candidates` is split
    out of :func:`locate_global_struct`: the two halves of the oracle must observe **different
    instants**, and a live process handle only ever reads *now*. The sweep therefore has to finish
    before the user is asked to act.
    """

    image: ModuleImage
    strong: list[OracleMatch]  # both players resolved at a constant stride
    weak: list[OracleMatch]  # P1 only — the two-level-P2 report path, never a written table


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
) -> tuple[list[OracleMatch], list[OracleMatch]]:
    """Validate every candidate; return ``(strong, weak)`` matches.

    C4e stopped at the first strong match. C4f cannot: "strong" now means *structurally* plausible
    at one instant, and the whole lesson of the live run is that several structs are. They all have
    to survive the sweep so :func:`confirm_players` can ask each one to move. ``max_strong`` caps
    the collection so a pathological build cannot make the post-prompt confirmation unbounded.
    """
    strong: list[OracleMatch] = []
    weak: list[OracleMatch] = []
    total = len(candidates)
    for i, base_offset in enumerate(candidates):
        if progress is not None and i and i % _PROGRESS_EVERY == 0:
            _emit(progress, f"    validated {i}/{total} candidates ({len(strong)} strong) ...")
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
        if not match.strong:
            weak.append(match)
            continue
        strong.append(match)
        if len(strong) >= spec.max_strong_candidates:
            _emit(progress, f"    strong-candidate ceiling {spec.max_strong_candidates} reached")
            break
    return strong, weak


def player_candidates(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    image: ModuleImage | None = None,
    progress: Progress | None = None,
) -> PlayerCandidates | None:
    """Sweep static data for pointer slots whose chain lands on a plausible player struct.

    The **instant-A** half, run at round start. Passes run writable ``.data`` first, then ``.rdata``
    as a fallback; the first pass yielding any strong match ends the sweep. ``None`` when the
    manifest carries no ``base_scan`` spec.

    Note there is no ``hint`` here. An AOB signature can skip the sweep only when nothing else has
    to happen before the answer is final — and with a behavioral oracle something does: the user has
    to act, which cannot happen until after the sweep. :func:`locate_player_struct` still takes the
    fast path when it has no second snapshot to confirm against.
    """
    spec = manifest.base_scan
    if spec is None:
        return None
    if image is None:
        image = parse_module_image(_module_reader(source, module_base))
        _emit(
            progress,
            f"  parsed PE: SizeOfImage {image.size_of_image // 1024} KiB, "
            f"{len(image.sections)} sections",
        )
    weak: list[OracleMatch] = []
    for label, sections in _section_passes(image, spec):
        if not sections:
            continue
        _emit(progress, f"  sweeping {label} ...")
        slots = find_candidate_slots(source, module_base, sections, progress=progress)
        _emit(progress, f"  validating {len(slots)} candidates in {label} ...")
        strong, pass_weak = _validate_all(
            source,
            slots,
            module=module,
            module_base=module_base,
            spec=spec,
            manifest=manifest,
            progress=progress,
        )
        weak.extend(pass_weak)
        if strong:
            _emit(
                progress,
                f"  {len(strong)} structural candidate(s) in {label}; the acting player's move_id "
                "must change across the action to accept one",
            )
            return PlayerCandidates(image=image, strong=strong, weak=weak)
    if not weak:
        _emit(progress, "  no candidate landed on a plausible player struct")
    return PlayerCandidates(image=image, strong=[], weak=weak)


def _observe(
    source: MemorySource,
    match: OracleMatch,
    *,
    module: str,
    module_base: int,
    spec: BaseScanSpec,
    manifest: ProbeManifest,
) -> _Observation | None:
    """Re-resolve ``match``'s chain in one window sample and read its fields (``None`` if dead).

    Re-resolves from the *slot*, never from the recorded ``p1_base``: the entity struct reallocates,
    which is the whole reason the anchor lives in code. A sample where the chain does not resolve,
    or where the move ids no longer read plausibly, contributes no evidence — it is dropped rather
    than counted as "did not move", so a momentary bad read cannot reject the real struct.
    """
    stride, p2_move_id = match.stride, match.p2_move_id
    if stride is None or p2_move_id is None:
        return None  # a weak (two-level-P2) match: nothing to compare the opponent against
    p1_base = _follow_chain(source, module, module_base, match.base_offset, spec.pointer_path)
    if p1_base is None:
        return None
    p1_move = _read_move_id(source, p1_base, spec, manifest)
    p2_move = _read_move_id(source, p1_base + stride, spec, manifest)
    if p1_move is None or p2_move is None:
        return None
    opponent = 1 - manifest.moving_player
    damage = _read_scalar(source, p1_base + opponent * stride + spec.damage_taken_offset, "i32")
    return _Observation(moves=(p1_move, p2_move), damage=damage, p1_base=p1_base)


def _behavior(
    match: OracleMatch, observations: Sequence[_Observation], *, acting: int
) -> Behavior | None:
    """Fold a candidate's window samples into what it **ever did** (pure; ``None`` if never read).

    The windowed oracle in one expression: every signal is an ``any`` over the samples, compared
    against the *round-start* reading on ``match``. Comparing only the final sample is what C4f did,
    and it is unsatisfiable in practice — see :class:`Behavior`.
    """
    if not observations or match.p2_move_id is None:
        return None
    before = (match.move_id, match.p2_move_id)
    opponent = 1 - acting
    return Behavior(
        acting_move_changed=any(o.moves[acting] != before[acting] for o in observations),
        opponent_move_changed=any(o.moves[opponent] != before[opponent] for o in observations),
        opponent_damaged=any(o.damage is not None and o.damage > 0 for o in observations),
        p1_after_base=observations[-1].p1_base,
        samples=len(observations),
    )


def confirm_players(
    during: Iterable[MemorySource],
    candidates: PlayerCandidates,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    from_signature: bool = False,
    progress: Progress | None = None,
) -> Located | None:
    """The **window** half: keep only the candidates that *behaved* like the acting player.

    ``during`` is the series of snapshots taken while the user walks, jabs and jumps — one
    :class:`MemorySource` per sample, read in order. Offline that is a list of planted images; live
    it is a generator yielding the same process handle on a fixed cadence, so a "snapshot" is
    simply the live process at that instant. Either way the loop is the same, and the seam is what
    makes a transient ``move_id`` testable without a running game.

    A real entity struct's ``move_id`` moves when the player jabs and jumps — for about half a
    second, and then it idles back. A coincidental struct's frozen field never moves, however
    plausible it looked at round start. So the test is "differed in **any** sample", which rejects
    the landing C4e accepted live *and* accepts the real struct the user cannot alt-tab fast enough
    to catch mid-animation. Among the survivors the strongest :attr:`Behavior.score` wins, and the
    count of distinct accepted landings rides on :attr:`Located.accepted` so an ambiguous pick is
    reported rather than quietly taken.

    ``during`` is consumed **whole**, even when there are no candidates to observe: live, iterating
    it is what makes the window take the time it is supposed to take, and the global oracle's
    frame-delta band is calibrated against that duration.

    ``None`` when nothing behaved: either the user did not act, or no swept slot leads to the player
    struct. Both are reported, and neither is resolved by falling back to a structural guess.
    """
    spec = manifest.base_scan
    if spec is None:
        return None
    observed: list[list[_Observation]] = [[] for _ in candidates.strong]
    samples = 0
    for source in during:
        samples += 1
        for slot, match in enumerate(candidates.strong):
            observation = _observe(
                source, match, module=module, module_base=module_base, spec=spec, manifest=manifest
            )
            if observation is not None:
                observed[slot].append(observation)

    behaved: list[tuple[OracleMatch, Behavior]] = []
    for slot, match in enumerate(candidates.strong):
        behavior = _behavior(match, observed[slot], acting=manifest.moving_player)
        if behavior is not None and behavior.accepted:
            behaved.append((match, behavior))
    if not behaved:
        if candidates.strong:
            # Fail closed. A structural candidate that never moved is precisely the decoy C4e
            # accepted; falling back to "well, it looked right" would reinstate the bug.
            _emit(
                progress,
                f"  none of the {len(candidates.strong)} structural candidates changed the acting "
                f"player's move_id in any of the {samples} window sample(s)",
            )
            return None
        if not candidates.weak:
            return None
        # P1-only landings carry no stride, so there is no opponent to compare and no behavioral
        # test to run. They exist to report the two-level-P2 case, never to write a table.
        landings = {m.p1_base for m in candidates.weak}
        return Located(
            match=candidates.weak[0],
            image=candidates.image,
            ambiguous_weak=len(landings) > 1,
            accepted=len(landings),
            considered=len(candidates.weak),
        )

    match, behavior = max(behaved, key=lambda pair: pair[1].score)
    landings = {m.p1_base for m, _ in behaved}
    _emit(
        progress,
        f"  player anchor: +0x{match.base_offset:x}, stride 0x{match.stride:x} "
        f"({len(landings)} of {len(candidates.strong)} structural candidates accepted; "
        f"behavior: {behavior.describe()})",
    )
    return Located(
        match=match,
        image=candidates.image,
        from_signature=from_signature,
        behavior=behavior,
        accepted=len(landings),
        considered=len(candidates.strong),
    )


def _structural_only(candidates: PlayerCandidates, progress: Progress | None) -> Located | None:
    """Fall back to C4e's single-instant acceptance when there is no action window to ask.

    Reached only by callers that pass no ``during`` samples — the derivation then also *says* the
    landing is unconfirmed (:func:`derive_base_layout`), because a struct that merely looks right at
    round start is exactly what the live run got wrong.
    """
    if candidates.strong:
        match = candidates.strong[0]
        _emit(
            progress,
            f"  strong (structural, UNCONFIRMED) match: base_offset 0x{match.base_offset:x}, "
            f"stride 0x{match.stride:x}",
        )
        landings = {m.p1_base for m in candidates.strong}
        return Located(
            match=match,
            image=candidates.image,
            accepted=len(landings),
            considered=len(candidates.strong),
        )
    if not candidates.weak:
        return None
    landings = {m.p1_base for m in candidates.weak}
    return Located(
        match=candidates.weak[0],
        image=candidates.image,
        ambiguous_weak=len(landings) > 1,
        accepted=len(landings),
        considered=len(candidates.weak),
    )


def locate_player_struct(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    during: Sequence[MemorySource] | None = None,
    hint: AobSignature | None = None,
    progress: Progress | None = None,
) -> Located | None:
    """Compose both halves of the player oracle against a round-start source and an action window.

    With ``during``, a candidate is accepted only if the acting player's ``move_id`` changed from
    its round-start value in at least one of those samples (:func:`confirm_players`). Without it,
    the scan can only apply C4e's single-instant structural check, and the caller is told so. A
    one-element ``during`` is the degenerate C4f oracle — correct, and unsatisfiable by a human.

    ``hint`` is the previous table's AOB signature: when it re-matches to a unique slot whose chain
    still satisfies the oracle, that is the answer and the full sweep is skipped. It is a *hint*,
    never a shortcut around validation — a signature that matches but fails the oracle falls through
    to the sweep, and behavioral confirmation applies to it exactly as to a swept slot.

    The live orchestration does **not** call this: it must sweep before prompting the user to act
    and sample the window afterwards, so it drives :func:`player_candidates` and
    :func:`confirm_players` directly (and freezes the round-start struct in between, via
    :class:`LayeredMemorySource`).

    ``progress`` (optional) makes the long live sweep observable — the command layer supplies a
    printer; the library is silent without one (docs/02 §2). ``None`` when the manifest lacks a
    ``base_scan`` spec or nothing landed on a plausible player struct at all.
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

    def accept(candidates: PlayerCandidates, *, from_signature: bool) -> Located | None:
        if during is None:
            located = _structural_only(candidates, progress)
            return None if located is None else replace(located, from_signature=from_signature)
        return confirm_players(
            during,
            candidates,
            module=module,
            module_base=module_base,
            manifest=manifest,
            from_signature=from_signature,
            progress=progress,
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
                # The only candidate is `match`, so any acceptance is an acceptance of it; a
                # rejection falls through to the sweep rather than trusting a stale signature.
                located = accept(
                    PlayerCandidates(image=image, strong=[match], weak=[]), from_signature=True
                )
                if located is not None:
                    _emit(progress, "  re-found via seed AOB signature (fast path); sweep skipped")
                    return located

    candidates = player_candidates(
        source,
        module=module,
        module_base=module_base,
        manifest=manifest,
        image=image,
        progress=progress,
    )
    if candidates is None:
        return None
    return accept(candidates, from_signature=False)


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
    """The accepted global match plus how many distinct landings passed the oracle.

    ``accepted`` counts **distinct structs** (by resolved ``gbase``), not accepting slots: many
    static slots legitimately point into the same global struct, and calling that ambiguity would
    cry wolf on every run. Two different *structs* both ticking a counter beside a steady round
    number is the real ambiguity, and it is what gets reported.
    """

    match: GlobalMatch
    accepted: int
    slots: int = 1  # accepting slots, before deduping to distinct landings

    @property
    def ambiguous(self) -> bool:
        """More than one landing satisfied frame-counter + round; the pick is not trustworthy."""
        return self.accepted > 1


def frame_delta_band(spec: GlobalScanSpec, seconds: float) -> tuple[int, int]:
    """How far a real frame counter may advance over a window of ``seconds`` (C4g Phase 3).

    The single strongest global discriminator available, and it costs nothing: over a *timed* window
    the frame counter advances by ``frame_rate * seconds``, whereas a coincidental ``u32`` that
    merely trends upward advances by whatever it advances by. The live runs bear this out — a delta
    of 96 over the prompt is a counter; a delta of 1 is not, and the unbanded oracle accepted both.

    Widened by :attr:`~.manifest.GlobalScanSpec.frame_delta_tolerance` because the frame rate is a
    guess (vsync, a paused practice mode, a loading hitch), and clamped into the spec's absolute
    bounds. A non-positive ``seconds`` means the caller cannot time the window, so the band opens
    back up to those absolute bounds.
    """
    if seconds <= 0:
        return spec.frame_delta_min, spec.frame_delta_max
    expected = spec.frame_rate * seconds
    low = max(spec.frame_delta_min, int(expected * (1.0 - spec.frame_delta_tolerance)))
    high = min(spec.frame_delta_max, math.ceil(expected * (1.0 + spec.frame_delta_tolerance)))
    return low, max(low, high)


def assign_global_fields(
    before: dict[int, int],
    after: dict[int, int],
    spec: GlobalScanSpec,
    *,
    frame_delta: tuple[int, int] | None = None,
) -> dict[str, int]:
    """Assign the seeded offsets to match fields **by behavior** across two snapshots (pure).

    We know these offsets carry match state; we do not know which is which, and guessing would bake
    a coincidence into a data file. So each field is claimed by the behavior only it exhibits, most
    constrained first:

    * ``frame_counter`` — strictly increased, by a delta inside ``frame_delta``. When the caller
      timed the two snapshots it passes a band from :func:`frame_delta_band` and this becomes a
      sharp test (a live counter at ~60fps, not a checksum that happened to tick); otherwise it
      falls back to the spec's absolute ``frame_delta_min``/``max``.
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
    low, high = (
        frame_delta
        if frame_delta is not None
        else (
            spec.frame_delta_min,
            spec.frame_delta_max,
        )
    )
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
            claim(name, lambda old, new: low <= new - old <= high)
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


def rebase_global_candidates(
    source: MemorySource,
    candidates: Sequence[GlobalCandidate],
    *,
    module: str,
    module_base: int,
    spec: GlobalScanSpec,
) -> list[GlobalCandidate]:
    """Re-read every candidate's ``before`` values *now*, dropping the ones that no longer resolve.

    The frame-delta band (:func:`frame_delta_band`) only means something if the two readings are a
    known duration apart. The sweep's ``before`` values were read while the sweep ran — minutes ago,
    and at an unknown offset from the user pressing Enter. Re-reading them at the instant the action
    window opens makes the delta span exactly the window, which is what turns "it ticked up" into
    "it ticked up at 60fps for five seconds".
    """
    fresh: list[GlobalCandidate] = []
    for candidate in candidates:
        gbase = _follow_chain(
            source, module, module_base, candidate.base_offset, list(candidate.pointer_path)
        )
        if gbase is None:
            continue
        values = _read_global_fields(source, gbase, spec)
        if values is None:
            continue
        fresh.append(replace(candidate, gbase=gbase, before=values))
    return fresh


def confirm_global(
    source_after: MemorySource,
    candidates: Sequence[GlobalCandidate],
    *,
    module: str,
    module_base: int,
    spec: GlobalScanSpec,
    frame_delta: tuple[int, int] | None = None,
    progress: Progress | None = None,
) -> GlobalLocated | None:
    """The **instant-B** half: re-resolve each candidate and apply the temporal oracle.

    Re-resolves the chain from the *slot* rather than reusing the recorded ``gbase`` — the global
    struct may have been reallocated between the snapshots, exactly as the player struct is, and
    re-walking the chain is what makes the anchor survive that (docs/02 §3).

    ``frame_delta`` narrows what counts as a frame counter to what the window's duration allows;
    the live path passes :func:`frame_delta_band` over its measured window (C4g Phase 3).

    Several distinct landings satisfying the oracle is reported (:attr:`GlobalLocated.ambiguous`)
    rather than silently resolved. Accepting slots are **deduped by resolved struct** (``gbase``):
    the live run's alarming "22 accepted" was largely 22 static slots into the same struct, and
    counting the field assignment alongside it would have split one struct back into several
    whenever an optional field (``timer_ms``, ``match_phase``) was claimable via one slot and not
    another. Among distinct landings the pick is the one whose behavior named the **most** fields —
    a struct where the clock also counts down is more likely the match struct than one where only
    the two required fields resolved — with ties going to the first swept.
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
        offsets = assign_global_fields(candidate.before, after, spec, frame_delta=frame_delta)
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
    landings = {m.gbase for m in accepted}
    best = max(accepted, key=lambda m: len(m.offsets))
    _emit(
        progress,
        f"  global anchor: +0x{best.base_offset:x} ({len(landings)} distinct struct(s) "
        f"accepted via {len(accepted)} static slot(s))",
    )
    return GlobalLocated(match=best, accepted=len(landings), slots=len(accepted))


def locate_global_struct(
    before: MemorySource,
    after: MemorySource,
    *,
    module: str,
    module_base: int,
    spec: GlobalScanSpec,
    image: ModuleImage | None = None,
    frame_delta: tuple[int, int] | None = None,
    progress: Progress | None = None,
) -> GlobalLocated | None:
    """Compose both halves against two snapshot sources (the offline/simple path).

    The live orchestration cannot use this — it must run :func:`global_candidates` at round start,
    prompt the user to act, re-read the ``before`` values as the window opens
    (:func:`rebase_global_candidates`) and only then run :func:`confirm_global` — but every offline
    test and any caller holding two frozen snapshots can.
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
        frame_delta=frame_delta,
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
            "match_phase not assigned: no seeded offset read as a small code, so the SEEDED offset "
            "is carried. decode_frame will report match_state='unknown' for every frame (it no "
            "longer raises), the doctor still validates the mechanical core, and live CAPTURE "
            "REFUSES to record — a gate that cannot recognize an online match must not run. "
            "game_mode is likewise carried: no round-start oracle can derive it (the mode does not "
            "change while the user takes a step), so it needs the docs/02 §4 calibration."
        )
    result.notes.append(
        f"global anchor accepted on {global_located.accepted} distinct landing(s) reached via "
        f"{global_located.slots} static slot(s)."
    )
    if global_located.ambiguous:
        result.notes.append(
            f"{global_located.accepted} DISTINCT structs satisfied the global oracle; the first "
            "was taken. The arbiter is the doctor's frame_monotonic check over several real "
            "frames — run it. Narrow global_scan.pointer_paths / field_offsets if it fails."
        )


def _note_player_behavior(result: DerivationResult, located: Located, had_window: bool) -> None:
    """Say, loudly, on what evidence the player anchor was accepted (C4f/C4g).

    Three things a reader of the report must be able to tell apart: a landing confirmed by the
    acting player actually moving; a landing accepted on one frozen instant because no action window
    was sampled; and a landing picked out of several that all moved.
    """
    behavior = located.behavior
    if behavior is None:
        if had_window:
            return  # a weak/two-level landing; `derive_base_layout` reports that case in full
        result.notes.append(
            "the player anchor was accepted on a SINGLE INSTANT (no action window): its fields "
            "read plausibly at round start, but nothing proved the struct tracks the acting "
            "player. That is how a coincidental struct was accepted on the live game. Re-run with "
            "an act-then-capture so the behavioral oracle can confirm the landing."
        )
        return
    result.notes.append(
        f"player anchor confirmed BEHAVIORALLY across the action window ({behavior.describe()}); "
        f"{located.accepted} of {located.considered} structurally-plausible landings passed."
    )
    if located.ambiguous:
        result.notes.append(
            f"{located.accepted} DISTINCT structs behaved like the acting player; the one with the "
            "strongest behavioral delta was taken. Confirm with the doctor (char ids, health, "
            "move_id stability) before trusting the table."
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


def _window_of(source_after: MemorySource | None) -> list[MemorySource] | None:
    """A snapshot pair's degenerate one-sample action window (``None`` when there is no pair).

    Offline callers that hold two frozen images get exactly the C4f oracle out of the C4g machinery,
    which is what most of the derivation tests want: their planted "after" image *is* the instant
    the action happened. Only the live path (and the transient-move_id regression) needs more.
    """
    return None if source_after is None else [source_after]


def derive_base_layout(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    seed: OffsetTable,
    source_after: MemorySource | None = None,
    during: Sequence[MemorySource] | None = None,
    located: Located | None = None,
    global_located: GlobalLocated | None = None,
    sweep_global: bool = True,
    sweep_player: bool = True,
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

    ``source_after`` is the snapshot at the *end* of the action window, used for the position
    derivation; ``during`` is the whole window the behavioral oracle folds over, defaulting to the
    single-sample ``[source_after]`` for callers that hold only a snapshot pair.

    ``located`` / ``global_located`` let the caller pass locations it already swept for; when given,
    the expensive sweeps are not repeated. The live orchestration **must** pre-sweep **both** (each
    oracle needs readings straddling the user's action, which a single live handle cannot
    manufacture after the fact) and therefore passes ``sweep_global=False`` / ``sweep_player=False``
    — so that "swept, nothing behaved" is not confused with "not swept yet" and re-run as a useless
    same-instant sweep. ``state_map`` is the value -> meaning data the builder writes into the
    table. ``progress`` threads to the sweeps for a live-observable log.
    """
    window = during if during is not None else _window_of(source_after)
    result = DerivationResult(module=module, module_base=module_base)
    result.encoded_state = state_map
    spec = manifest.base_scan
    if spec is None:
        result.notes.append("no base_scan spec in the probe manifest; cannot run the code scan.")
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x", "frame_counter"])
        return result

    # The previous table's AOB signature is a fast re-find hint; it still has to pass the oracle.
    # A caller that already located the struct (live path) passes it in to avoid re-sweeping.
    if located is None and sweep_player:
        located = locate_player_struct(
            source,
            module=module,
            module_base=module_base,
            manifest=manifest,
            during=window,
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
            "no static pointer slot's chain landed on a player struct that BEHAVED like the acting "
            "player (its move_id must change from its round-start value in at least one sample of "
            "the action window). Either the action was not performed — walk P1 (Jin) forward, jab "
            "P2, and jump, repeatedly, for the whole capture window — or the Jin-vs-Kazuya "
            "round-start setup / base_scan pointer_path / module name is wrong (see runbook). A "
            "struct that merely looks right at round start is deliberately NOT accepted."
        )
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x"])
        return result

    match, image = located.match, located.image
    if located.from_signature:
        result.notes.append(
            "player-struct slot re-found via the seed table's AOB signature (fast path), then "
            "re-validated against the layout oracle — the full candidate sweep was not needed."
        )
    _note_player_behavior(result, located, window is not None)
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

    def regions(self) -> Sequence[MemoryRegion]:
        """Enumeration is a property of the live map, so it comes from the fallback (read-only)."""
        return self._fallback.regions()

    def mapped_regions(self) -> Sequence[MemoryRegion]:
        """Likewise for the validation map — the overlay freezes bytes, not the map (brief #24)."""
        return self._fallback.mapped_regions()


def freeze_struct(
    live: MemorySource, base: int, span: int, *, floor: int = 0x1000
) -> Region | None:
    """Freeze the largest readable prefix of ``[base, base+span)`` as a :class:`Region`.

    The manifest's ``struct_span`` is a generous upper bound (0x40000), far larger than a player
    struct's actual heap allocation. :meth:`MemorySource.read` is **all-or-nothing**: a range that
    overruns the allocation's committed pages raises :class:`MemoryReadError` rather than returning
    a short buffer. A single full-span ``read(base, span)`` therefore *fails outright*, and where
    the caller swallows that (the freeze loops do), it silently drops the whole freeze and reads the
    live handle instead. That defeats the freeze: by the time the frozen samples are consulted the
    transient ``move_id`` has idled back and the behavioral oracle sees no change.

    Shrinking to the largest readable prefix (halve on failure) still captures every field the
    freeze is read for — ``move_id`` (0x550), ``damage_taken`` (0x1578), and the component pointer
    slots (``slot_span`` 0x4000) all lie in the first few KiB. Returns ``None`` only if even
    ``floor`` bytes are unreadable (a dead base).
    """
    size = span
    while size >= floor:
        try:
            return Region(base=base, data=live.read(base, size))
        except MemoryReadError:
            size //= 2
    return None
