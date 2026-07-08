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


class ProbeManifest(BaseModel):
    """Everything the re-discovery search needs, as an editable data file (docs/02 §4)."""

    module: str
    notes: str = ""

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
