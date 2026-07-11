"""Locate Tekken 8's players via the **holder** model: AoB code-sig -> holder -> two slots (C4i).

The C4d ``--base-scan`` finds a ``.data`` pointer slot and follows a chain to a single player
struct, then assumes P2 sits a constant ``stride`` later. Three current community tools (Irony,
opendojo — verified live on T8 v3.00.02 — and ParadiseAigo) agree the live game does something the
stride model cannot express: a **holder** object carries *two* per-player pointer slots
(``holder+0x30``, ``holder+0x38``) to **separate** allocations, and the holder's own ``.data`` slot
is found by an **AoB code signature** in ``.text`` with a RIP-relative displacement — the durable,
self-healing anchor the whole community relies on.

This module implements that derivation, mirroring :mod:`.basescan` but with a different *locating*
technique and the per-player-anchor schema (:attr:`~tekken_coach.reader.offsets.PlayerStruct.
player_slots`) instead of a stride:

1. **Find the holder slot** (:func:`find_holder_slot`) — AoB-scan the executable sections for the
   seeded pattern and RIP-decode the ``disp32`` embedded in the match to the module-relative slot
   RVA. A unique match is the answer; none or several fail closed (there is no candidate sweep — the
   code signature *is* the identity).
2. **Resolve + validate the holder** (:func:`resolve_holder`) — deref the slot to the holder, then
   each ``holder_slot`` to a player base, and require the round-start oracle: the two char ids form
   ``{jin, kazuya}``, move ids are plausible, ``damage_taken`` is 0.
3. **Confirm behaviorally** (:func:`confirm_holder`) — across the action window the acting player's
   ``move_id`` must change from its round-start value in at least one sample (the same C4f/C4g
   argument: a struct that merely reads right at round start is not proven to be the one the player
   controls, and ``move_id`` is transient so a single instant cannot see it).

The **global/match** anchor, the **encoded state words**, and the **transform component** position
lives in are the same facts the base scan already handles, so :func:`derive_holder_layout` reuses
:func:`~tekken_coach.reader.discovery.basescan._derive_global`,
:func:`~tekken_coach.reader.discovery.basescan._seed_state_fields`, and
:func:`~tekken_coach.reader.discovery.basescan.find_transform_component` unchanged.

Read-only throughout: it follows pointers through the :class:`MemorySource` seam and never writes
(docs/02 §2). All of it is offline-testable with ``pymem`` absent.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass

from tekken_coach.reader.discovery.basescan import (
    GlobalLocated,
    Progress,
    _derive_global,
    _emit,
    _module_reader,
    _plausible_pointer,
    _read_region,
    _read_scalar,
    _seed_state_fields,
    find_transform_component,
)
from tekken_coach.reader.discovery.derive import Confidence, DerivationResult, DerivedField
from tekken_coach.reader.discovery.manifest import HolderScanSpec, ProbeManifest
from tekken_coach.reader.discovery.pe import ModuleImage, parse_module_image
from tekken_coach.reader.discovery.scanners import Region, aob_scan
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    Anchor,
    AobSignature,
    ComponentAnchor,
    EncodedStateSpec,
    OffsetTable,
)

# ---------------------------------------------------------------------------
# Step 1: find the holder's .data slot by an AoB code signature (RIP-relative)
# ---------------------------------------------------------------------------


def decode_rip_relative(region: Region, match_rva: int, disp32_pos: int) -> int | None:
    """RIP-decode the ``.data`` slot an instruction match references (x64, pure).

    The matched instruction embeds a 32-bit signed displacement at byte ``disp32_pos``; on x64 the
    displacement is relative to the address of the *next* instruction, i.e. the 4 displacement bytes
    themselves end the reference. So the referenced slot RVA is
    ``match_rva + disp32_pos + 4 + disp32``. Returns ``None`` if the match's bytes are not fully
    covered (a truncated section tail).
    """
    try:
        raw = region.read(match_rva, disp32_pos + 4)
    except IndexError:
        return None
    (disp,) = struct.unpack_from("<i", raw, disp32_pos)
    return match_rva + disp32_pos + 4 + int(disp)


def find_holder_slot(
    source: MemorySource,
    module_base: int,
    image: ModuleImage,
    *,
    pattern: str,
    disp32_pos: int,
    progress: Progress | None = None,
) -> int | None:
    """AoB-scan the executable sections for ``pattern`` and RIP-decode the unique holder slot RVA.

    Unlike the base scan there is no candidate *sweep*: a code signature over the storing
    instruction identifies the holder outright. A unique match's ``disp32`` decodes to the slot;
    **no match or an ambiguous (multiple-distinct) match returns ``None``** — the caller fails
    closed rather than guessing, as
    :func:`~tekken_coach.reader.discovery.basescan.find_by_signature` does.
    """
    matches: list[int] = []
    for section in image.code_sections():
        data = _read_region(source, module_base + section.rva, section.virtual_size)
        if not data:
            continue
        region = Region(base=section.rva, data=data)
        for hit in aob_scan(region, pattern):
            slot = decode_rip_relative(region, hit, disp32_pos)
            if slot is not None:
                matches.append(slot)
    unique = sorted(set(matches))
    if len(unique) != 1:
        _emit(
            progress,
            f"  holder AoB matched {len(unique)} distinct slot(s); need exactly one"
            + (f" (candidates {[hex(u) for u in unique]})" if unique else ""),
        )
        return None
    _emit(progress, f"  holder slot re-found via AoB code signature: +0x{unique[0]:x}")
    return unique[0]


# ---------------------------------------------------------------------------
# Step 2: resolve the holder -> two player bases and apply the round-start oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HolderMatch:
    """A validated holder landing: the slot, the holder, and both players' bases/round-start fields.

    ``player_bases`` / ``char_ids`` / ``move_ids`` are in P1..P2 order (matching
    ``spec.holder_slots``). ``move_ids`` are the round-start readings the behavioral oracle compares
    every window sample against.
    """

    slot_rva: int
    holder_base: int
    player_bases: tuple[int, int]
    char_ids: tuple[int, int]
    move_ids: tuple[int, int]


def _read_player_fields(
    source: MemorySource, base: int, spec: HolderScanSpec, m: ProbeManifest
) -> tuple[int, int, int] | None:
    """Read (char_id, move_id, damage_taken) at a player base, or ``None`` if any is unreadable."""
    char_id = _read_scalar(source, base + spec.char_id_offset, m.char_id_kind)
    move_id = _read_scalar(source, base + spec.move_id_offset, m.move_id_kind)
    damage = _read_scalar(source, base + spec.damage_taken_offset, "i32")
    if char_id is None or move_id is None or damage is None:
        return None
    return char_id, move_id, damage


def _resolve_players(
    source: MemorySource, module_base: int, slot_rva: int, spec: HolderScanSpec
) -> tuple[int, tuple[int, int]] | None:
    """Re-resolve the holder -> ``(holder_base, (p1_base, p2_base))`` (``None`` if a deref is dead).

    Re-resolves from the *slot*, never a recorded base: the entity structs reallocate, which is the
    whole reason the anchor lives in code.
    """
    holder_base = _read_scalar(source, module_base + slot_rva, "ptr")
    if holder_base is None or not _plausible_pointer(holder_base):
        return None
    bases: list[int] = []
    for slot in spec.holder_slots:
        pb = _read_scalar(source, holder_base + slot, "ptr")
        if pb is None or not _plausible_pointer(pb):
            return None
        bases.append(pb)
    return holder_base, (bases[0], bases[1])


def resolve_holder(
    source: MemorySource, module_base: int, slot_rva: int, spec: HolderScanSpec, m: ProbeManifest
) -> HolderMatch | None:
    """Deref the slot -> holder -> each player base, and apply the round-start structural oracle.

    Acceptance: both player pointers are plausible; the two char ids form ``{jin, kazuya}``; both
    move ids are plausible; both ``damage_taken`` read 0 (round start). Necessary but not sufficient
    — :func:`confirm_holder` is what proves the landing tracks the acting player. ``None`` on any
    dead read or a failed structural check.
    """
    resolved = _resolve_players(source, module_base, slot_rva, spec)
    if resolved is None:
        return None
    holder_base, bases = resolved
    char_ids: list[int] = []
    move_ids: list[int] = []
    for base in bases:
        read = _read_player_fields(source, base, spec, m)
        if read is None:
            return None
        char_id, move_id, damage = read
        if not (m.move_id_min <= move_id < m.move_id_max):
            return None
        if damage != 0:
            return None
        char_ids.append(char_id)
        move_ids.append(move_id)
    if {spec.jin_char_id, m.kazuya_char_id} != set(char_ids):
        return None
    return HolderMatch(
        slot_rva=slot_rva,
        holder_base=holder_base,
        player_bases=bases,
        char_ids=(char_ids[0], char_ids[1]),
        move_ids=(move_ids[0], move_ids[1]),
    )


# ---------------------------------------------------------------------------
# Step 3: behavioral confirmation across the action window
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HolderBehavior:
    """What the holder's players **did** across the action window (C4f/C4g — the decisive evidence).

    :attr:`accepted` is the whole oracle: the acting player's ``move_id`` differed from its
    round-start value in at least one sample. ``opponent_damaged`` corroborates a connected jab but
    is not required (the jab may whiff). ``p1_after_base`` is where P1 re-resolved in the last
    sample — the position scan's "after".
    """

    acting_move_changed: bool
    opponent_damaged: bool
    p1_after_base: int
    samples: int

    @property
    def accepted(self) -> bool:
        return self.acting_move_changed

    def describe(self) -> str:
        signals = [
            name
            for name, seen in (
                ("acting move_id changed", self.acting_move_changed),
                ("opponent damage_taken rose", self.opponent_damaged),
            )
            if seen
        ]
        return f"{', '.join(signals) or 'nothing moved'} (over {self.samples} samples)"


def confirm_holder(
    during: Sequence[MemorySource],
    match: HolderMatch,
    *,
    module_base: int,
    spec: HolderScanSpec,
    manifest: ProbeManifest,
    progress: Progress | None = None,
) -> HolderBehavior | None:
    """Fold the window samples into what the holder's players **ever did** (``None`` if never read).

    ``during`` is the series of snapshots taken while the user acts — one source per sample. The
    acting player's ``move_id`` is compared against its round-start reading on ``match``; a change
    in **any** sample accepts (a jab/jump is transient and idles back, so an end-of-window compare
    would reject the real struct — the C4g lesson). ``opponent_damaged`` rides along as
    corroboration only.
    """
    acting = manifest.moving_player
    opponent = 1 - acting
    acting_changed = False
    opponent_damaged = False
    last_p1_base = match.player_bases[0]
    samples = 0
    for source in during:
        samples += 1
        resolved = _resolve_players(source, module_base, match.slot_rva, spec)
        if resolved is None:
            continue
        _holder_base, bases = resolved
        last_p1_base = bases[0]
        acting_move = _read_scalar(
            source, bases[acting] + spec.move_id_offset, manifest.move_id_kind
        )
        if acting_move is not None and acting_move != match.move_ids[acting]:
            acting_changed = True
        damage = _read_scalar(source, bases[opponent] + spec.damage_taken_offset, "i32")
        if damage is not None and damage > 0:
            opponent_damaged = True
    if samples == 0:
        return None
    behavior = HolderBehavior(
        acting_move_changed=acting_changed,
        opponent_damaged=opponent_damaged,
        p1_after_base=last_p1_base,
        samples=samples,
    )
    _emit(progress, f"  holder behavior: {behavior.describe()}")
    return behavior


# ---------------------------------------------------------------------------
# Top-level: derive the whole holder layout into a DerivationResult
# ---------------------------------------------------------------------------


def _emit_fields(
    result: DerivationResult, match: HolderMatch, spec: HolderScanSpec, m: ProbeManifest
) -> None:
    """Emit the validated player fields (char_id, move_id, damage_taken) as high-confidence."""
    p1_base = match.player_bases[0]
    result.fields.append(
        DerivedField(
            name="char_id",
            scope="player",
            offset=spec.char_id_offset,
            kind=m.char_id_kind,
            example_address=p1_base + spec.char_id_offset,
            confidence=Confidence.high,
            method=f"holder oracle: P1={match.char_ids[0]}, P2={match.char_ids[1]} "
            f"= {{Jin {spec.jin_char_id}, Kazuya {m.kazuya_char_id}}}",
        )
    )
    result.fields.append(
        DerivedField(
            name="move_id",
            scope="player",
            offset=spec.move_id_offset,
            kind=m.move_id_kind,
            example_address=p1_base + spec.move_id_offset,
            confidence=Confidence.high,
            method="holder oracle: plausible for both players and changed across the action window",
        )
    )
    # HP is encrypted on T8 (Irony/opendojo), so health is always computed from damage_taken.
    result.fields.append(
        DerivedField(
            name="damage_taken",
            scope="player",
            offset=spec.damage_taken_offset,
            kind="i32",
            example_address=p1_base + spec.damage_taken_offset,
            confidence=Confidence.high,
            method="holder oracle field; health computed as round_start_health - damage_taken "
            "(HP is encrypted on T8)",
        )
    )
    result.max_health = spec.round_start_health


def derive_holder_layout(
    source: MemorySource,
    *,
    module: str,
    module_base: int,
    manifest: ProbeManifest,
    seed: OffsetTable,
    source_after: MemorySource | None = None,
    during: Sequence[MemorySource] | None = None,
    holder: HolderMatch | None = None,
    slot_rva: int | None = None,
    global_located: GlobalLocated | None = None,
    sweep_global: bool = True,
    state_map: EncodedStateSpec | None = None,
    image: ModuleImage | None = None,
    progress: Progress | None = None,
) -> DerivationResult:
    """Run the full holder derivation and return a :class:`DerivationResult` (C4i).

    Locates the holder by AoB (:func:`find_holder_slot`), validates it structurally
    (:func:`resolve_holder`) and behaviorally (:func:`confirm_holder`), then assembles the
    per-player anchor + slots, the validated fields, the global anchor, the seeded state words, the
    transform component — reusing the base scan's global/state/component derivation unchanged.

    ``holder`` / ``slot_rva`` let the live orchestration pass a landing it already found before the
    action prompt (the AoB find and the structural oracle read only round start, but the behavioral
    oracle needs the window). ``during`` is the window the behavioral oracle folds over; without it
    a landing is accepted structurally and flagged as behaviorally unconfirmed.
    """
    result = DerivationResult(module=module, module_base=module_base)
    result.encoded_state = state_map
    spec = manifest.holder_scan
    if spec is None:
        result.notes.append(
            "no holder_scan spec in the probe manifest; cannot run the holder scan."
        )
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x", "frame_counter"])
        return result
    if manifest.base_scan is None:
        result.notes.append(
            "holder_scan needs base_scan for the state-word offsets and component scan; absent."
        )
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x", "frame_counter"])
        return result

    if image is None:
        image = parse_module_image(_module_reader(source, module_base))
        _emit(
            progress,
            f"  parsed PE: SizeOfImage {image.size_of_image // 1024} KiB, "
            f"{len(image.sections)} sections",
        )

    if holder is not None:
        slot_rva = holder.slot_rva  # a pre-prompt find the live path passes back
    elif slot_rva is None:
        slot_rva = find_holder_slot(
            source,
            module_base,
            image,
            pattern=spec.aob_pattern,
            disp32_pos=spec.disp32_pos,
            progress=progress,
        )
    if holder is None and slot_rva is not None:
        holder = resolve_holder(source, module_base, slot_rva, spec, manifest)

    # The global/match anchor is independent of the holder; derive it either way (C4e §1).
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

    if holder is None:
        result.notes.append(
            "the holder AoB signature did not match a unique slot, or the landing failed the "
            "round-start oracle (P1=Jin, P2=Kazuya, plausible move ids, damage_taken 0). Confirm "
            "the P1-Jin-vs-P2-Kazuya round-start setup and that holder_scan.aob_pattern still "
            "matches this build (re-source the pattern from a current tool if a patch moved the "
            "storing function; it is DATA)."
        )
        result.unresolved.extend(["char_id", "move_id", "health", "pos_x"])
        return result
    assert slot_rva is not None

    # Behavioral confirmation: the acting player's move_id must have moved.
    window = (
        during if during is not None else ([source_after] if source_after is not None else None)
    )
    behavior: HolderBehavior | None = None
    if window is not None:
        behavior = confirm_holder(
            window, holder, module_base=module_base, spec=spec, manifest=manifest, progress=progress
        )
        if behavior is None or not behavior.accepted:
            result.notes.append(
                "the holder was located but the acting player's move_id did NOT change in any "
                "window sample — the AoB found a holder, but nothing proved it tracks the player. "
                "Act the WHOLE window (walk P1/Jin forward, jab P2, jump, on repeat); move_id is "
                "transient, so a change in ANY sample suffices, but there has to be one."
            )
            result.unresolved.extend(["char_id", "move_id", "health", "pos_x"])
            return result

    # Assemble the per-player anchor + slots (the C4i schema) and the validated fields.
    result.player_anchor = Anchor(
        module=module,
        base_offset=slot_rva,
        pointer_path=[0],  # deref the .data slot -> the holder object
        signature=AobSignature(pattern=spec.aob_pattern, disp32_pos=spec.disp32_pos),
    )
    result.player_slots = [ComponentAnchor(slot_offset=s) for s in spec.holder_slots]
    result.player_char_ids = (spec.jin_char_id, manifest.kazuya_char_id)
    _emit_fields(result, holder, spec, manifest)
    if behavior is not None:
        result.notes.append(
            f"holder confirmed BEHAVIORALLY across the action window ({behavior.describe()}); "
            f"players at holder+{[hex(s) for s in spec.holder_slots]} -> "
            f"0x{holder.player_bases[0]:x} / 0x{holder.player_bases[1]:x} (separate allocations, "
            "NOT a stride)."
        )
    else:
        result.notes.append(
            "the holder was accepted on the round-start oracle only (no action window). Re-run "
            "with an act-then-capture so the behavioral oracle can confirm the players move."
        )

    # The encoded state words + move_frame + counter_state, seeded at their base_scan offsets.
    _seed_state_fields(result, manifest.base_scan, holder.player_bases[0])

    # Position: the transform component, reached relative to each player's base (C4e Phase 3).
    _derive_holder_position(
        result,
        source,
        source_after,
        holder=holder,
        slot_rva=slot_rva,
        module_base=module_base,
        spec=spec,
        manifest=manifest,
        behavior=behavior,
        progress=progress,
    )
    return result


def _derive_holder_position(
    result: DerivationResult,
    source: MemorySource,
    source_after: MemorySource | None,
    *,
    holder: HolderMatch,
    slot_rva: int,
    module_base: int,
    spec: HolderScanSpec,
    manifest: ProbeManifest,
    behavior: HolderBehavior | None,
    progress: Progress | None,
) -> None:
    """Locate ``pos_{x,y,z}`` in the transform component behind each player's own pointer.

    Position is not in the entity struct on Tekken 8 (docs/02 §3); it lives in a transform
    component reached relative to each player base — the *same* scan the base path uses, off the
    holder-resolved P1/P2 bases.
    """
    component_spec = manifest.base_scan.component_scan if manifest.base_scan else None
    if source_after is None or component_spec is None:
        result.unresolved.append("pos_x")
        result.notes.append(
            "no second snapshot or component_scan spec; position is seeded (run with an "
            "act-then-capture)."
        )
        return
    after = _resolve_players(source_after, module_base, slot_rva, spec)
    if after is None:
        result.unresolved.append("pos_x")
        result.notes.append("chain did not re-resolve in the second snapshot; position seeded.")
        return
    _after_holder, after_bases = after
    component = find_transform_component(
        source,
        source_after,
        p1_base=holder.player_bases[0],
        p1_after_base=after_bases[0],
        p2_base=holder.player_bases[1],
        spec=component_spec,
        manifest=manifest,
        progress=progress,
    )
    if component is None:
        result.unresolved.append("pos_x")
        result.notes.append(
            "no moving float triple in any component the players point at; position is seeded. "
            "Walk P1 a real step, or widen base_scan.component_scan (slot_span / probe_span / "
            "max_depth)."
        )
        return
    result.components[POSITION_COMPONENT] = component
    result.drop_player_fields.extend(["pos_x", "pos_y", "pos_z"])
    triple = component.fields["pos_x"].offset
    hops = " -> ".join(f"+0x{o:x}" for o in component.pointer_path) or "(direct)"
    result.notes.append(
        f"position lives in a transform component reached via +0x{component.slot_offset:x} {hops}, "
        f"triple at +0x{triple:x}; confirmed by resolving the same path from P2's base to a "
        "different plausible coordinate."
    )


__all__ = [
    "HolderBehavior",
    "HolderMatch",
    "confirm_holder",
    "decode_rip_relative",
    "derive_holder_layout",
    "find_holder_slot",
    "resolve_holder",
]
