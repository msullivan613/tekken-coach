"""Locate the heap-allocated player struct via a static code/data pointer (C4d, docs/02 §3).

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

Read-only throughout: it reads process memory through the :class:`MemorySource` seam and follows
pointers; it never writes (docs/02 §2).
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tekken_coach.reader.decode import resolve_anchor
from tekken_coach.reader.discovery.derive import (
    Confidence,
    DerivationResult,
    DerivedField,
    _plausible_coord,
)
from tekken_coach.reader.discovery.manifest import BaseScanSpec, ProbeManifest
from tekken_coach.reader.discovery.pe import ModuleImage, Reader, Section, parse_module_image
from tekken_coach.reader.discovery.scanners import Region, aob_scan, value_scan
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import Anchor, AobSignature, OffsetTable, ScalarKind

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


def _section_passes(
    image: ModuleImage, spec: BaseScanSpec
) -> list[tuple[str, tuple[Section, ...]]]:
    """The ordered section sweeps: writable ``.data`` first, then read-only ``.rdata`` as fallback.

    The root pointer is a runtime-written global, so it lives in writable ``.data`` — usually far
    smaller than ``.rdata``, so sweeping it first is both likelier to hit and much cheaper. A second
    pass over ``.rdata`` only runs if ``.data`` yields no match, so correctness is unchanged if the
    assumption is ever wrong. ``scan_writable_first=False`` restores the single all-data sweep.
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
# In-struct health + position (tractable once the struct is located)
# ---------------------------------------------------------------------------


def _derive_health(
    source: MemorySource, match: OracleMatch, stride: int, spec: BaseScanSpec, m: ProbeManifest
) -> DerivedField | None:
    """Find the health field: an offset reading full HP at round start in **both** players."""
    window = _read_region(source, match.p1_base, spec.struct_span)
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
) -> int | None:
    """Offset of a moving (x,y,z) float triple in P1's struct across the two snapshots.

    Reads P1 at its before base and its (re-resolved) after base, so it is correct even if the chain
    re-resolved to a moved allocation between snapshots — the whole point of anchoring in code.
    """
    for off in range(0, spec.struct_span - 8, m.scan_align):
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
# Top-level: derive the whole player anchor + fields into a DerivationResult
# ---------------------------------------------------------------------------


def derive_base_layout(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    seed: OffsetTable,
    source_after: MemorySource | None = None,
    located: Located | None = None,
    progress: Progress | None = None,
) -> DerivationResult:
    """Run the full C4d code-signature derivation and return a :class:`DerivationResult`.

    Reuses C4c's result/build/report ecosystem: the derived player :class:`Anchor` (with pointer
    chain + AOB signature), the stride, the char ids, and the derivable field offsets (``char_id``,
    ``move_id`` from the oracle; ``health``/``pos`` from the in-struct scans) go into the same
    :class:`DerivationResult` the builder consumes. The **global** anchor is carried from ``seed`` —
    C4d locates the heap player struct; global/match anchoring is a separate concern (see the
    runbook), so it is seeded and flagged for calibration, not re-derived here.

    ``located`` lets the caller pass a struct location it already swept for (the live orchestration
    runs :func:`locate_player_struct` once to know where to freeze the round-start snapshot); when
    given, the expensive sweep is not repeated here. ``progress`` threads to the sweep for a
    live-observable log.
    """
    result = DerivationResult(module=module, module_base=module_base)
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

    result.global_anchor = seed.global_struct.anchor
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
    result.notes.append(
        "global/match anchor carried from the seed table — C4d locates the heap PLAYER struct; "
        "global anchoring is a separate calibration (see runbook)."
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

    health = _derive_health(source, match, stride, spec, manifest)
    if health is not None:
        result.fields.append(health)
    else:
        result.unresolved.append("health")
        result.notes.append(
            f"no field read round-start max {spec.round_start_health} in both structs; health is "
            f"seeded. Fallback: derive health = {spec.round_start_health} - damage_taken "
            "(a computed field; reviewer's call)."
        )

    if source_after is not None:
        after_base = _follow_chain(
            source_after, module, module_base, match.base_offset, spec.pointer_path
        )
        if after_base is None:
            result.unresolved.append("pos_x")
            result.notes.append("chain did not re-resolve in the second snapshot; position seeded.")
        else:
            if after_base != match.p1_base:
                result.notes.append(
                    f"player struct reallocated between snapshots (0x{match.p1_base:x} -> "
                    f"0x{after_base:x}); the static chain tracked it — this is exactly why C4d "
                    "anchors in code, not the heap."
                )
            pos_off = _derive_position(source, source_after, match, after_base, spec, manifest)
            if pos_off is None:
                result.unresolved.append("pos_x")
                result.notes.append("no moving float triple found; position seeded (widen span).")
            else:
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
    else:
        result.unresolved.append("pos_x")
        result.notes.append("no second snapshot; position seeded (run with an act-then-capture).")

    return result


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
