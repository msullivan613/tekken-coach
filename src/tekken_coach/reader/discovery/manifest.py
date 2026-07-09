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

from pydantic import BaseModel, ValidationError

from tekken_coach.reader.faults import OffsetTableError
from tekken_coach.reader.offsets import DEFAULT_OFFSETS_DIR, ScalarKind

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
    scan_data_only: bool = True  # sweep only readable initialized-data sections for slots
    scan_writable_first: bool = (
        True  # sweep writable .data first (likely + cheap), .rdata as fallback
    )


class ProbeManifest(BaseModel):
    """Everything the re-discovery search needs, as an editable data file (docs/02 §4)."""

    module: str
    notes: str = ""

    # --- C4d code-signature base derivation (heap struct via a static pointer + chain) ---
    # Optional so the C4c value-scan path loads without it; required for the C4d base scan.
    base_scan: BaseScanSpec | None = None

    # --- Known anchors at the P1-Jin-vs-P2-Kazuya round-start setup (facts, docs/02 §5) ---
    kazuya_char_id: int  # 12 from the C1 move map; pins the player-struct base
    round_start_health: int  # full HP, identical for both players; pins the stride

    # --- Plausibility bounds (keep scanners off coincidental matches) ---
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

    # --- Which player performs the between-snapshots action (moves + presses a button) ---
    moving_player: int = 0

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
