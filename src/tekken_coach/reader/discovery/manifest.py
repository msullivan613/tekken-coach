"""The probe manifest — the editable knob that drives offset re-discovery (docs/02 §4).

The re-discovery search is **not** hard-coded constants; it is driven by a data file describing
*what is provably true at the fixed calibration setup* — practice mode, **P1 Jin vs P2 Kazuya**,
round start. Calibration after a patch is then a **data edit** to this manifest (widen a scan
window, adjust a plausibility bound), not a source change — the same posture docs/02 §3/§4 takes
for the offset tables themselves.

What the manifest asserts about the setup (all facts/data, docs/02 §5):

* The known **Kazuya char id** (12, from the C1 move map) — the anchor that pins the player-struct
  base; Jin's id is *discovered* as the P1 counterpart at the same offset (an output, not an input).
* Full **health** at round start (a known max, identical for both players) — the shared-value
  anchor that pins the player-struct **stride**.
* Plausibility bounds (char-id range, move-id range, stride range, position magnitude) that keep
  the scanners from locking onto coincidental matches.
* The module-relative **scan windows** the live tool reads into :class:`Region`\\s.

The scalar **kinds** default to the standard T8 widths but are data too, so a layout change is a
manifest edit. A checked-in default lives at ``assets/offsets/probe-manifest.json``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from tekken_coach.reader.faults import OffsetTableError
from tekken_coach.reader.offsets import DEFAULT_OFFSETS_DIR, FieldSpec, ScalarKind

DEFAULT_MANIFEST_PATH = DEFAULT_OFFSETS_DIR / "probe-manifest.json"


class ScanWindow(BaseModel):
    """A span the live tool reads into a :class:`~.scanners.Region` before scanning.

    ``base_offset`` is module-relative by default (``module_base + base_offset``, the docs/02 §3
    anchoring); set ``absolute`` to point the window at an address the user located manually (e.g. a
    heap region behind a pointer chain, which module-relative windows cannot reach — see the
    calibration runbook).
    """

    base_offset: int
    size: int
    absolute: bool = False


class GlobalScanSpec(BaseModel):
    """The seed layout that drives C4e's global/match anchor derivation (docs/02 §3, facts/data).

    Same shape of argument as :class:`BaseScanSpec`, one level up: the global struct is *also*
    heap-allocated behind a static pointer, so the anchor is candidate-generate-and-validate against
    a **behavioral oracle** rather than a known address. What is seeded (facts/data, docs/02 §5):

    * ``pointer_paths`` — candidate chain shapes from a static pointer slot to the global struct.
      Several are allowed because the chain shape is the least certain seed; the first that
      validates wins.
    * ``field_offsets`` — within-struct offsets known to hold the match fields. The **assignment**
      of offset -> field is deliberately *not* seeded: we know these offsets carry frame counter /
      round / timer / phase, not which is which. The scan assigns them **by behavior** across two
      snapshots (see :func:`~.basescan.assign_global_fields`), which is both clean-room and robust
      to the fork's data being reordered.

    The rest are plausibility bounds keeping the oracle off coincidental matches. A frame counter
    that merely ticks is common in a game process; a struct where one offset ticks up, another holds
    a round number in 1..k, and a third counts a round clock *down* is not.
    """

    pointer_paths: list[list[int]]
    field_offsets: list[int]  # unassigned candidate offsets within the global struct
    field_kind: ScalarKind = "u32"
    frame_delta_min: int = 1  # a live frame counter advances by at least this between snapshots
    # ... and by at most this. The snapshots straddle one interactive prompt, so at 60fps a
    # plausible delta is seconds-to-minutes of frames. A u32 that jumped by more than ten minutes'
    # worth is a hash, a byte count, or an address — not this frame counter. Narrowing this narrows
    # the accept set, which is the point (C4f Phase 3).
    frame_delta_max: int = 36_000
    round_min: int = 1
    round_max: int = 8
    timer_ms_max: int = 300_000  # a round clock above this is not a round clock
    match_phase_max: int = 64  # a phase code at/above this is not a phase code
    max_candidates: int = 400_000  # hard ceiling on slots validated per sweep
    scan_data_only: bool = True  # sweep only readable initialized-data sections for slots
    scan_writable_first: bool = True  # writable .data first, .rdata as fallback


class ComponentScanSpec(BaseModel):
    """Locate the transform component holding ``pos_{x,y,z}`` (docs/02 §3, C4e Phase 3).

    Position is **not** in the entity struct: a full-struct scan across a walking snapshot pair
    finds no moving float triple, and the fork's layout data has no position field. It lives in a
    separate Unreal transform component the entity points at. So the search is one level indirect —
    sweep the entity struct's own pointer slots, follow each, and look for the moving triple inside
    the pointee.

    * ``slot_span`` — how far into the entity struct to look for the component pointer.
    * ``probe_span`` — how far into the component to scan for the triple.
    * ``max_depth`` — 1 follows ``entity -> component``; 2 additionally follows
      ``entity -> object -> component`` (Unreal nests a scene component under an actor), scanning
      ``inner_span`` bytes of the intermediate object for the second pointer.
    * ``min_delta`` — the acting player must have moved at least this far in ``x`` between the two
      snapshots, so a float that merely jitters is not mistaken for a coordinate.
    """

    slot_span: int = 0x2000
    slot_align: int = 8
    probe_span: int = 0x400
    probe_align: int = 4
    max_depth: int = 2
    inner_span: int = 0x200
    min_delta: float = 1.0e-4
    max_slots: int = 4096  # ceiling on pointer slots followed per level


class BaseScanSpec(BaseModel):
    """The seed layout that drives C4d's code-signature base derivation (docs/02 §3, facts/data).

    C4d locates the heap-allocated player struct by scanning the module's static data for the
    pointer that leads to it, then following a **seed pointer chain** and validating the landing
    against the **known field layout** (the oracle). Everything here is facts/data (docs/02 §5),
    seeded from the community-known Tekken 8 layout — the durable within-struct offsets and the
    stable-ish chain shape — never the base address itself (that is re-derived every build).

    * ``pointer_path`` — the chain from the static pointer slot to the player-data base. The chain
      *offsets* are more stable than the ``base_offset`` slot (which shifts every build), so they
      are seeded and the slot is discovered.
    * ``char_id_offset`` / ``move_id_offset`` / ``damage_taken_offset`` — the oracle fields: a
      candidate landing is accepted only if these read plausibly (char id in range, move id in
      range, damage 0 at round start) for **both** players.
    * ``struct_span`` — how far past the located base to scan for the in-struct health/position
      fields (tractable inside one located struct; intractable over the whole heap).
    * ``max_stride`` — the largest P1->P2 gap to accept as a constant stride before concluding P2 is
      a separate allocation (the two-level case the runbook flags).
    * ``aob_window_before`` / ``aob_window_after`` — bytes captured around the slot for the AOB
      signature (the pointer bytes themselves are wildcarded).
    * ``max_strong_candidates`` — how many structurally-plausible landings to carry into the
      behavioral confirmation (:func:`~.basescan.confirm_players`). The structural oracle accepts
      more than one struct on the real game, so the sweep may not stop at the first; this bounds how
      many it collects before the user is prompted to act.
    * ``state_fields`` — the encoded state words + ``move_frame`` + ``counter_state`` at their known
      within-struct offsets (facts/data). These are **seeded, not derived**: no two-snapshot oracle
      can prove where ``stun_type`` lives, so the scan writes them into the table and the report
      flags them for the observation-based calibration of docs/02 §8. Their *values*' meanings live
      in the separate state map (:func:`~tekken_coach.reader.offsets.load_state_map`).
    * ``legacy_state_fields`` — the C4a placeholder's per-flag boolean fields, dropped from the
      built table when ``state_fields`` supersedes them (they do not exist in the real struct, and
      carrying them forward would look like working offsets).
    """

    pointer_path: list[int]
    char_id_offset: int
    move_id_offset: int
    damage_taken_offset: int
    round_start_health: int  # full HP under this build's regime (T8 ~200); the in-struct anchor
    struct_span: int = 0x2000  # scan [base, base+span) for health/position
    max_stride: int = 0x40000  # P1->P2 gap ceiling for the constant-stride model
    aob_window_before: int = 16  # signature context bytes before the slot
    aob_window_after: int = 16  # ... and after (beyond the 8 wildcarded pointer bytes)
    max_strong_candidates: int = 32  # landings carried into the behavioral confirmation
    scan_data_only: bool = True  # sweep only readable initialized-data sections for slots
    scan_writable_first: bool = (
        True  # sweep writable .data first (likely + cheap), .rdata as fallback
    )
    state_fields: dict[str, FieldSpec] = Field(default_factory=dict)
    legacy_state_fields: list[str] = Field(default_factory=list)
    component_scan: ComponentScanSpec | None = None


class ProbeManifest(BaseModel):
    """Everything the re-discovery search needs, as an editable data file (docs/02 §4)."""

    module: str
    notes: str = ""

    # --- C4d code-signature base derivation (heap struct via a static pointer + chain) ---
    # Optional so the C4c value-scan path loads without it; required for the C4d base scan.
    base_scan: BaseScanSpec | None = None

    # --- C4e global/match anchor derivation (the same technique, one struct over) ---
    global_scan: GlobalScanSpec | None = None

    # The encoded-state value -> meaning map (docs/02 §8), resolved relative to the manifest's own
    # directory. Kept a separate file from the offset tables: `update-offsets` rewrites addresses
    # every build, but the state semantics are calibrated once and carried forward.
    state_map: str = "state-map.json"

    # --- Known anchors at the P1-Jin-vs-P2-Kazuya round-start setup (facts, docs/02 §5) ---
    kazuya_char_id: int  # 12 from the C1 move map; pins the player-struct base
    round_start_health: int  # full HP, identical for both players; pins the stride

    # --- Plausibility bounds (keep scanners off coincidental matches) ---
    # `char_id_min` is 1, not 0: a zeroed page reads char_id 0 / move_id 0 / damage_taken 0 and
    # satisfies the whole structural player oracle. No T8 character carries id 0 (Kazuya is 12), so
    # excluding it costs nothing and removes the cheapest false positive there is.
    char_id_min: int = 1
    char_id_max: int = 200
    move_id_min: int = 0
    move_id_max: int = 60000
    stride_min: int
    stride_max: int
    pos_abs_max: float = 1.0e6  # |position float| below this is plausible (game units)
    # A nonzero position float this small looks like an int reinterpreted as a denormal, not a real
    # coordinate — it lets the position finder reject move-id/health bytes that happen to change.
    pos_abs_min: float = 1.0e-3
    frame_delta_max: int = 100000  # the global frame counter advances by at most this between snaps

    # --- Which player performs the between-snapshots action (walks, jabs, jumps) ---
    # Indexes a two-element (P1, P2) tuple in the behavioral oracle, so it is bounded here rather
    # than crashing on an IndexError deep in a live sweep because someone typed a 2.
    moving_player: int = Field(default=0, ge=0, le=1)

    # --- Scan stepping + windows ---
    scan_align: int = 4
    global_window: ScanWindow
    player_window: ScanWindow

    # --- Scalar kinds (standard T8 widths; data so a layout change stays a manifest edit) ---
    char_id_kind: ScalarKind = "u32"
    health_kind: ScalarKind = "i32"
    move_id_kind: ScalarKind = "u32"
    pos_kind: ScalarKind = "f32"
    frame_counter_kind: ScalarKind = "u32"


def load_probe_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> ProbeManifest:
    """Load the probe manifest, mapping failures onto :class:`OffsetTableError` (docs/02 §7)."""
    try:
        return ProbeManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OffsetTableError(f"probe manifest not found: {path}") from exc
    except ValidationError as exc:
        raise OffsetTableError(f"malformed probe manifest {path}: {exc}") from exc


def state_map_path(
    manifest: ProbeManifest, manifest_path: str | Path = DEFAULT_MANIFEST_PATH
) -> Path:
    """Resolve ``manifest.state_map`` relative to the manifest file's own directory."""
    return Path(manifest_path).parent / manifest.state_map
