"""Correlate scan candidates into a ``(base, stride, {field: offset})`` layout (docs/02 §3/§4).

This is the heart of the clean-room re-discovery: given the pure scan primitives run over two
snapshots of the P1-Jin-vs-P2-Kazuya setup, derive where the reader's fields live and express them
as **module-relative anchors + a stride + per-field offsets** (docs/02 §3 addressing).

The technique (docs/02 §4), stated as an algorithm:

1. **Stride** — full health is a *shared* value at round start (both players max), so the
   player-struct ``stride`` shows up as the distance between two health matches. Every health pair
   within the manifest's stride bounds is a stride hypothesis.
2. **Base** — the known Kazuya char id (12) locates player 2's ``char_id``; subtracting a stride
   hypothesis lands on player 1's ``char_id``, which must read as a *plausible* char id (Jin — an
   **output**, discovered as the P1 counterpart). That pins both player bases. The struct is
   anchored **at ``char_id``** (offset 0), matching the checked-in table layout.
3. **Confirm** — a hypothesis is accepted only if a *single* ``(base, stride, offset)`` also
   explains the other derivable fields for **both** players (health at a shared offset, a changed
   move id, a moving position triple). That mutual consistency across the two symmetric structs is
   the confirmation the summary technique relies on; the hypothesis resolving the most fields wins.

Only the fields the docs/02 §4 anchors actually support are derived with confidence here —
``char_id``, ``health``, ``move_id``, ``pos_{x,y,z}`` per player and the global ``frame_counter``.
These are exactly the fields the doctor self-check (docs/02 §6) validates. Everything else is
*seeded* by the builder from the previous known-good table and flagged for calibration — an honest
line between what one setup can prove and what needs a human in the loop.

Pure: every function here operates on :class:`~.scanners.Region` byte images. No memory access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from tekken_coach.reader.discovery.manifest import ProbeManifest
from tekken_coach.reader.discovery.scanners import Region, change_scan, value_scan
from tekken_coach.reader.offsets import Anchor, ScalarKind

# The fields one Jin-vs-Kazuya setup can prove (docs/02 §4 anchors); the rest are builder-seeded.
DERIVABLE_PLAYER_FIELDS = ("char_id", "health", "move_id", "pos_x", "pos_y", "pos_z")
DERIVABLE_GLOBAL_FIELDS = ("frame_counter",)


class Confidence(StrEnum):
    """How a field's location was established (drives the diagnostic report + calibration focus)."""

    high = "high"  # locked by a known-value or clear-behavior anchor across both players
    medium = "medium"  # derived but by a weaker signature (e.g. a moving float triple)
    seeded = "seeded"  # not derived this run — carried from the previous table, needs verification


@dataclass(frozen=True)
class DerivedField:
    """One field the scan located, with the evidence for the diagnostic report."""

    name: str
    scope: Literal["player", "global"]
    offset: int
    kind: ScalarKind
    example_address: int  # the resolved absolute address in player 0 / the global struct
    confidence: Confidence
    method: str


@dataclass
class DerivationResult:
    """Everything the scan derived (or failed to), for the builder and the report."""

    module: str
    module_base: int
    stride: int | None = None
    player_anchor: Anchor | None = None
    global_anchor: Anchor | None = None
    player_char_ids: tuple[int, int] | None = None  # (p0 = Jin, p1 = Kazuya)
    fields: list[DerivedField] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    frame_counter_candidates: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the confident core resolved (both struct anchors + a stride)."""
        return (
            self.player_anchor is not None
            and self.global_anchor is not None
            and self.stride is not None
            and not self.unresolved
        )

    def player_offsets(self) -> dict[str, DerivedField]:
        return {f.name: f for f in self.fields if f.scope == "player"}

    def global_offsets(self) -> dict[str, DerivedField]:
        return {f.name: f for f in self.fields if f.scope == "global"}


@dataclass(frozen=True)
class DiscoverySnapshots:
    """Two reads of each scan window — ``before`` (round start) and ``after`` (post-action).

    The ``after`` snapshot is taken once the moving player has walked and pressed a button, so the
    change scans see a moved position, a changed move id, and an advanced frame counter (§4).
    """

    player_before: Region
    player_after: Region
    global_before: Region
    global_after: Region


def _anchor(module: str, module_base: int, absolute: int) -> Anchor:
    """Convert an absolute address into a module-relative :class:`Anchor` (docs/02 §3).

    Produces a plain module-base + static-offset anchor (no pointer chain). A negative offset means
    the address sits below the module base — an absolute/heap address the caller pointed a window
    at, which is *not* stably anchorable without a pointer chain (out of scope; see the runbook).
    """
    return Anchor(module=module, base_offset=absolute - module_base, pointer_path=[])


def _resolve_player_fields(
    before: Region, after: Region, base0: int, base1: int, stride: int, m: ProbeManifest
) -> dict[str, DerivedField]:
    """Resolve the derivable player fields for one ``(base0, base1, stride)`` hypothesis.

    Returns the fields it could confirm (``char_id`` is always present as the anchor; the rest
    appear only when their signature resolves consistently). ``health`` present is the caller's
    acceptance gate for the hypothesis.
    """
    resolved: dict[str, DerivedField] = {}
    step = m.scan_align

    p0_id = before.read_scalar(base0, m.char_id_kind)
    p1_id = before.read_scalar(base1, m.char_id_kind)
    resolved["char_id"] = DerivedField(
        name="char_id",
        scope="player",
        offset=0,
        kind=m.char_id_kind,
        example_address=base0,
        confidence=Confidence.high,
        method=f"Kazuya id {int(p1_id or 0)} at P2; Jin id {int(p0_id or 0)} at P1 "
        f"(stride {stride})",
    )

    # health: the same value (round-start max) at the same offset in both structs.
    for off in range(0, stride, step):
        va = before.read_scalar(base0 + off, m.health_kind)
        vb = before.read_scalar(base1 + off, m.health_kind)
        if va is None or vb is None:
            continue
        if int(va) == m.round_start_health and int(vb) == m.round_start_health:
            resolved["health"] = DerivedField(
                name="health",
                scope="player",
                offset=off,
                kind=m.health_kind,
                example_address=base0 + off,
                confidence=Confidence.high,
                method=f"both players read round-start max {m.round_start_health}",
            )
            break

    mp = m.moving_player
    acting = base0 if mp == 0 else base1
    other = base1 if mp == 0 else base0

    # move_id: changed in the acting player, plausible in both snapshots, with a plausible value at
    # the same offset in the other player (the cross-struct consistency check).
    for off in range(0, stride, step):
        va = after.read_scalar(acting + off, m.move_id_kind)
        vb = before.read_scalar(acting + off, m.move_id_kind)
        ov = before.read_scalar(other + off, m.move_id_kind)
        if va is None or vb is None or ov is None:
            continue
        if (
            int(va) != int(vb)
            and m.move_id_min < int(va) < m.move_id_max
            and m.move_id_min < int(vb) < m.move_id_max
            and m.move_id_min <= int(ov) < m.move_id_max
        ):
            resolved["move_id"] = DerivedField(
                name="move_id",
                scope="player",
                offset=off,
                kind=m.move_id_kind,
                example_address=base0 + off,
                confidence=Confidence.high,
                method=f"changed {int(vb)}->{int(va)} in the acting player",
            )
            break

    # position: three consecutive finite f32s in game-unit range where at least x moved.
    pos_off = _find_position_triple(before, after, acting, stride, m)
    if pos_off is not None:
        for axis, delta in (("pos_x", 0), ("pos_y", 4), ("pos_z", 8)):
            resolved[axis] = DerivedField(
                name=axis,
                scope="player",
                offset=pos_off + delta,
                kind=m.pos_kind,
                example_address=base0 + pos_off + delta,
                confidence=Confidence.medium,
                method=f"moving finite float triple at +{pos_off} (x/y/z consecutive)",
            )
    return resolved


def _plausible_coord(value: float, m: ProbeManifest) -> bool:
    """A finite float that reads as a real game coordinate, not an int reinterpreted as a denormal.

    Zero is allowed (a resting axis); any other value must sit in ``[pos_abs_min, pos_abs_max)``.
    This rejects e.g. a move id (2145) whose f32 bit-pattern is a ~1e-42 denormal that "changes"
    between snapshots but is not a position.
    """
    if not math.isfinite(value):
        return False
    return value == 0.0 or (m.pos_abs_min <= abs(value) < m.pos_abs_max)


def _find_position_triple(
    before: Region, after: Region, acting: int, stride: int, m: ProbeManifest
) -> int | None:
    """Find the offset of a moving (x,y,z) float triple in the acting player's struct."""
    for off in range(0, stride - 8, m.scan_align):
        triple_ok = True
        x_moved = False
        for k in range(3):
            va = before.read_scalar(acting + off + 4 * k, m.pos_kind)
            vb = after.read_scalar(acting + off + 4 * k, m.pos_kind)
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


def _derive_player(snap: DiscoverySnapshots, m: ProbeManifest) -> tuple[
    int | None, int | None, int | None, tuple[int, int] | None, dict[str, DerivedField]
]:
    """Search for the best ``(base0, base1, stride)`` and its resolved fields.

    Returns ``(base0, base1, stride, (jin, kazuya), fields)`` or ``(None, ...)`` if no hypothesis
    both places Kazuya's id at a plausible-Jin counterpart *and* confirms health.
    """
    a, b = snap.player_before, snap.player_after
    health_hits = value_scan(a, m.round_start_health, m.health_kind, align=m.scan_align)
    kaz_hits = value_scan(a, m.kazuya_char_id, m.char_id_kind, align=m.scan_align)
    strides = sorted(
        {
            h2 - h1
            for h1 in health_hits
            for h2 in health_hits
            if m.stride_min <= h2 - h1 <= m.stride_max
        }
    )

    best_score = -1
    best: tuple[int, int, int, tuple[int, int], dict[str, DerivedField]] | None = None
    for stride in strides:
        for c1 in kaz_hits:
            c0 = c1 - stride
            jin = a.read_scalar(c0, m.char_id_kind)
            if jin is None or not (m.char_id_min <= int(jin) <= m.char_id_max):
                continue
            fields = _resolve_player_fields(a, b, c0, c1, stride, m)
            if "health" not in fields:  # health is the acceptance confirmation
                continue
            score = len(fields)
            if score > best_score:
                best_score = score
                best = (c0, c1, stride, (int(jin), m.kazuya_char_id), fields)

    if best is None:
        return None, None, None, None, {}
    base0, base1, stride, ids, fields = best
    return base0, base1, stride, ids, fields


def _derive_global(
    snap: DiscoverySnapshots, m: ProbeManifest
) -> tuple[int | None, dict[str, DerivedField], list[int]]:
    """Locate the global frame counter: a value that strictly advanced by a plausible delta."""
    a, b = snap.global_before, snap.global_after
    hits = change_scan(
        a,
        b,
        m.frame_counter_kind,
        align=m.scan_align,
        changed=True,
        predicate=lambda old, new: 0 < new - old <= m.frame_delta_max,
    )
    if not hits:
        return None, {}, []
    fc = hits[0]
    delta = int(b.read_scalar(fc, m.frame_counter_kind) or 0) - int(
        a.read_scalar(fc, m.frame_counter_kind) or 0
    )
    resolved = {
        "frame_counter": DerivedField(
            name="frame_counter",
            scope="global",
            offset=0,
            kind=m.frame_counter_kind,
            example_address=fc,
            confidence=Confidence.high,
            method=f"monotonic counter, +{delta} across the two snapshots",
        )
    }
    return fc, resolved, hits


def derive_layout(
    snap: DiscoverySnapshots, *, module: str, module_base: int, manifest: ProbeManifest
) -> DerivationResult:
    """Run the full docs/02 §4 derivation over two snapshots and return the result + diagnostics."""
    result = DerivationResult(module=module, module_base=module_base)

    base0, base1, stride, ids, pfields = _derive_player(snap, manifest)
    if base0 is not None and stride is not None and ids is not None:
        result.stride = stride
        result.player_char_ids = ids
        result.player_anchor = _anchor(module, module_base, base0)
        result.fields.extend(pfields.values())
        if base0 < module_base:
            result.notes.append(
                f"player struct anchored below module base (base_offset "
                f"{base0 - module_base}); an absolute/heap base is not stably anchorable without a "
                "pointer chain — calibrate manually (see runbook)."
            )
    for name in DERIVABLE_PLAYER_FIELDS:
        if name not in pfields:
            result.unresolved.append(name)

    fc, gfields, fc_candidates = _derive_global(snap, manifest)
    result.frame_counter_candidates = fc_candidates
    if fc is not None:
        result.global_anchor = _anchor(module, module_base, fc)
        result.fields.extend(gfields.values())
        if len(fc_candidates) > 1:
            result.notes.append(
                f"{len(fc_candidates)} monotonically-increasing candidates for frame_counter; "
                f"chose the lowest (0x{fc:x}). Verify via the doctor's frame-monotonic check."
            )
    for name in DERIVABLE_GLOBAL_FIELDS:
        if name not in gfields:
            result.unresolved.append(name)

    return result
