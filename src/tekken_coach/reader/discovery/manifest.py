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
    # ... and by at most this. The fallback bounds, used when the caller cannot say how long the two
    # snapshots were apart. A u32 that jumped by more than ten minutes' worth of frames is a hash, a
    # byte count, or an address — not this frame counter.
    frame_delta_max: int = 36_000
    # When the caller *can* say (the live path times its own sampling window), the accept band
    # narrows to `frame_rate * seconds`, give or take `frame_delta_tolerance` of it — which is the
    # real discriminator (C4g Phase 3). Two structs both "ticking up a little" over a 5-second
    # window are common; one advancing by ~300 frames is the frame counter. The tolerance is wide
    # because the frame rate is not guaranteed (vsync, a paused practice mode, a loading hitch) and
    # a band that rejects the real counter costs a re-run; a band that admits a coincidence costs
    # only a reported ambiguity, which the doctor then arbitrates.
    frame_rate: float = 60.0
    frame_delta_tolerance: float = 0.5
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


class DeriveScanSpec(BaseModel):
    """Drives C4h's **fully derived** layout scan — no seeded within-struct offsets (docs/02 §3).

    C4d/C4e seed the field offsets (``char_id`` at +0x168, the pointer chain ``0x10→0x68→0x8→0x30``)
    from the community layout and derive only the static ``base_offset``. On build 5.02.01 those
    seeds are stale (the fork died Oct 2024), and a fair windowed run found 0 of 13 structural
    candidates behaving. C4h removes the dependence: it locates the entity struct by **behavior**,
    derives the field offsets + stride + Jin's id + the pointer path as **outputs**, so a patch is a
    re-run rather than a re-seed. The only retained seed is our own C1 fact ``kazuya_char_id`` (12);
    even Jin's id is discovered.

    Everything here is a tractability **bound** on a standard differential/reverse-pointer scan, not
    a layout fact — the facts (Kazuya's id, the char/move plausibility ranges, the stride window,
    the scan alignment) are the manifest's existing top-level fields, reused here.

    * ``struct_span`` — how far past a candidate's ``char_id`` address to compare two structs / look
      for the acting-correlated field. Bounds Phase 2's differential scan and Phase 4's field scans.
    * ``similarity_min`` — the fraction of 4-byte words two symmetric player structs must share at
      round start to be a stride pair (both idle, same move/state, same 0 damage — they differ only
      at ``char_id``/position/facing). This is the structural discriminator that keeps the char-id
      pairing from exploding on small ints, and it seeds no offset.
    * ``max_char_id_hits`` / ``max_pairs`` / ``max_layouts`` — caps on the char-id-pair sweep and
      the behavioral confirmation, so a heap full of the value 12 cannot make the scan unbounded.
    * ``reverse_max_depth`` / ``reverse_max_offset`` — Phase 3's pointer-chain depth ceiling and the
      window ``[target-M, target]`` a stored pointer must fall in to count as a hop. Bound the
      backward BFS for tractability.
    * ``reverse_max_nodes`` / ``max_paths`` — the BFS node ceiling and how many candidate static
      paths to carry into the reallocation confirmation.
    * ``aob_window_before`` / ``aob_window_after`` — bytes captured around the derived slot for the
      AOB signature (the pointer bytes are wildcarded), as in C4d.
    """

    struct_span: int = 0x2000
    similarity_min: float = 0.5
    min_shared_words: int = 8  # non-zero words two spans must share before "similar" means anything
    max_char_id_hits: int = 4096
    max_pairs: int = 64
    max_layouts: int = 16
    reverse_max_depth: int = 4
    reverse_max_offset: int = 0x1000
    reverse_max_nodes: int = 400_000
    max_paths: int = 32
    aob_window_before: int = 16
    aob_window_after: int = 16


class HolderScanSpec(BaseModel):
    """Drives C4i's **holder** derivation — the model the live T8 game actually uses (docs/02 §3).

    Three independent community tools (Irony, opendojo verified on T8 v3.00.02, ParadiseAigo) agree
    that the player structs are **not** a single-anchor + stride array. They hang off a *holder*
    object by two per-player pointer slots, each to a **separate** allocation, and the holder's own
    ``.data`` slot is found by an **AoB code signature** with a RIP-relative displacement — the
    patch-durable, self-healing anchor every tool uses. Neither ``--base-scan`` (its stride
    validation cannot express two separate allocations) nor ``--derive`` can express this, so C4i
    adds it. Everything here is facts/data (docs/02 §5), attributable to those tools — not code.

    * ``aob_pattern`` — the wildcard AoB matching the *instruction* in ``.text`` that stores the
      holder pointer; the RIP-relative ``disp32`` is wildcarded. For v3.00.02 this is
      ``4C 89 35 ?? ?? ?? ?? 41 88 5E 28`` (``MOV [rip+disp32], r14 ; MOV [r14+0x28], BL``).
    * ``disp32_pos`` — the byte offset of the ``i32`` displacement within a match (3 for that
      pattern); the slot is at ``match_rva + disp32_pos + 4 + disp32`` (see :class:`AobSignature`).
    * ``holder_slots`` — the per-player pointer-slot offsets inside the holder, in P1..P2 order
      (``[0x30, 0x38]``). ``holder + slot`` dereferences to that player's struct base.
    * ``char_id_offset`` / ``move_id_offset`` / ``damage_taken_offset`` — the oracle fields inside
      each player struct: at round start P1 reads ``jin_char_id``, P2 reads
      :attr:`ProbeManifest.kazuya_char_id`, move ids are plausible, and ``damage_taken`` is 0.
    * ``round_gate_offset`` — ``frames_since_round_start`` (0 during the intro); a clean
      round-active gate. Read for validation, not emitted as a table field (``PlayerFrame`` has no
      home for it).
    * ``jin_char_id`` — Jin's id (6 on current builds). Unlike ``--base-scan`` (which *discovers*
      Jin's id), the current community sources state it, so it is validated rather than derived.
    * ``round_start_health`` — full HP; health is computed as ``round_start_health - damage_taken``
      because HP is encrypted on T8 (confirmed by Irony/opendojo).

    The within-struct **state-word offsets**, the **legacy** booleans they supersede, and the
    **transform component** scan are the *same* facts the base scan carries, so C4i reuses
    :attr:`ProbeManifest.base_scan`'s ``state_fields`` / ``legacy_state_fields`` /
    ``component_scan`` rather than duplicating them, and the **global/match** anchor reuses
    :attr:`ProbeManifest.global_scan`.
    """

    aob_pattern: str
    disp32_pos: int = 3
    holder_slots: list[int]
    char_id_offset: int
    move_id_offset: int
    damage_taken_offset: int
    round_gate_offset: int
    jin_char_id: int
    round_start_health: int = 200


class ProbeManifest(BaseModel):
    """Everything the re-discovery search needs, as an editable data file (docs/02 §4)."""

    module: str
    notes: str = ""

    # --- C4d code-signature base derivation (heap struct via a static pointer + chain) ---
    # Optional so the C4c value-scan path loads without it; required for the C4d base scan.
    base_scan: BaseScanSpec | None = None

    # --- C4i holder derivation (AoB code-sig -> holder object -> two per-player pointer slots) ---
    # Optional so the C4c/C4d/C4h manifests load without it; required for `update-offsets --holder-
    # scan`. Reuses base_scan.state_fields / .legacy_state_fields / .component_scan and global_scan
    # for the within-struct state, position, and match anchor (those are shared DATA facts).
    holder_scan: HolderScanSpec | None = None

    # --- C4e global/match anchor derivation (the same technique, one struct over) ---
    global_scan: GlobalScanSpec | None = None

    # --- C4h fully-derived layout scan (locate by behavior, derive every offset) ---
    # Optional so the C4c/C4d manifests load without it; required for `update-offsets --derive`.
    # Reuses base_scan.component_scan / .state_fields / .round_start_health and global_scan for the
    # Phase 4 handoff (those are DATA facts, not seeded locating offsets).
    derive_scan: DeriveScanSpec | None = None

    # The encoded-state value -> meaning map (docs/02 §8), resolved relative to the manifest's own
    # directory. Kept a separate file from the offset tables: `update-offsets` rewrites addresses
    # every build, but the state semantics are calibrated once and carried forward.
    state_map: str = "state-map.json"

    # --- Known anchors at the P1-Jin-vs-P2-Kazuya round-start setup (facts, docs/02 §5) ---
    kazuya_char_id: int  # 12 from the C1 move map; pins the player-struct base
    round_start_health: int  # full HP, identical for both players; pins the stride

    # --- Plausibility bounds (keep scanners off coincidental matches) ---
    # `char_id_min` is back to 0 (C4g). C4f raised it to 1 to keep a zeroed page — which reads
    # char_id 0 / move_id 0 / damage_taken 0 — out of the candidate set. But Jin's real char id may
    # genuinely BE 0 on this build, so that floor risks structurally excluding the very struct the
    # scan exists to find, and no amount of behavioral testing recovers a candidate the structural
    # pass never emitted. A zeroed page cannot become a *strong* candidate anyway: strong acceptance
    # requires a second struct reading Kazuya's id (12) at a constant stride, which zeroed memory
    # has nowhere to put. The behavioral oracle is the discriminator; this is only the sieve.
    char_id_min: int = 0
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

    # --- The action window the behavioral oracles observe (C4g) ---
    # C4f compared move_id at two instants: round start, and the moment the user pressed Enter. But
    # move_id is TRANSIENT — a jab or a jump lasts about half a second and the character then
    # returns to idle (move_id back to its round-start value). Alt-tabbing from the game to the
    # terminal takes longer than that, so the real struct read as frozen at both ends and was
    # rejected. The oracle is unchanged in principle; it now samples a *window* and accepts a
    # candidate whose move_id differed in ANY sample, which a human can actually satisfy.
    action_lead_in_seconds: float = Field(default=3.0, gt=0)  # time to alt-tab back to the game
    action_window_seconds: float = Field(default=5.0, gt=0)  # how long the window samples for
    action_sample_interval: float = Field(default=0.2, gt=0)  # cadence within the window

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
