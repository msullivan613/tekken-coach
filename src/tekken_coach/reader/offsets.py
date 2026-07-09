"""The offset model + typed loader + version detection (docs/02 §3).

Offsets are **data, not source** (docs/02 §5 licensing): the concrete field addresses live under
``assets/offsets/``, one file per game version, and are re-discovered by ``update-offsets`` (C4b,
a clean-room re-implementation of the Jin-vs-Kazuya technique). This module defines the *shape* of
that data and how the reader selects a table:

1. ``index.json`` maps a game version -> its offset file, with a ``detected_version`` marker
   recording which build the checked-in assets are aligned to.
2. Given a detected version (injected in C4a; read from the running process in C4b), the loader
   selects the matching table. **On an unknown version it fails closed** — raising
   :class:`~tekken_coach.reader.faults.UnknownGameVersionError` with the §4 runbook — and never
   falls back to a stale table, because a wrong offset silently yields garbage FrameRecords
   (docs/02 §3).

The addressing model is module-base + static offset, optionally followed by a pointer chain
(:class:`Anchor`), which survives minor relocations better than absolute addresses (docs/02 §3).
Each field is a ``(offset, kind)`` pair (:class:`FieldSpec`); the decoder (:mod:`.decode`) reads
them. Nothing here reads memory — this is pure data + selection logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from tekken_coach.reader.faults import OffsetTableError, UnknownGameVersionError

DEFAULT_OFFSETS_DIR = Path("assets/offsets")

# Scalar kinds a field may have. Fixed little-endian widths; the decoder maps these to struct
# formats and the (test-only) encoder inverts them.
ScalarKind = Literal["u8", "u16", "u32", "i32", "i64", "f32", "bool8", "ptr"]


class FieldSpec(BaseModel):
    """One field within a struct: a byte ``offset`` from the struct base and its scalar ``kind``."""

    offset: int
    kind: ScalarKind


class AobSignature(BaseModel):
    """A code/data-signature that re-finds an anchor's static ``base_offset`` after a patch.

    ``base_offset`` shifts every build, but the bytes *around* the static pointer slot are stable,
    so ``update-offsets`` stores the surrounding window as a wildcard AOB pattern (docs/02 §3, C4d).
    A re-run scans the module's data sections for ``pattern`` and recovers the slot at
    ``match_address + slot_delta`` — a fast path that skips the full candidate scan. This is
    facts/data (docs/02 §5), not code; the decoder ignores it (it resolves via ``base_offset``).
    """

    pattern: str  # wildcard AOB, e.g. "48 8B ?? ?? ?? ?? ?? 89" (the pointer bytes wildcarded)
    slot_delta: int  # offset from a pattern match to the pointer slot (the static base_offset)


class Anchor(BaseModel):
    """How to resolve a struct's base address (docs/02 §3 anchoring strategy).

    ``module_base(module) + base_offset`` gives the anchor; each entry in ``pointer_path`` then
    dereferences (reads a pointer) and adds its offset, giving a standard multi-level pointer.
    An empty ``pointer_path`` means the base is a plain static offset from the module.

    ``signature`` (optional) is the AOB that re-derives ``base_offset`` after a patch (C4d); it is
    metadata for ``update-offsets`` and is ignored by the decoder.
    """

    module: str
    base_offset: int
    pointer_path: list[int] = Field(default_factory=list)
    signature: AobSignature | None = None


class GlobalStruct(BaseModel):
    """Match/global fields: frame counter, phase, mode, round, timer (docs/03 §1)."""

    anchor: Anchor
    fields: dict[str, FieldSpec]


class PlayerStruct(BaseModel):
    """The per-player struct: a base anchor, the array ``stride`` between players, and fields."""

    anchor: Anchor
    stride: int
    fields: dict[str, FieldSpec]


class StateCodes(BaseModel):
    """Raw game code -> semantic mappings (kept as data so a patch is a data edit, docs/02 §4).

    Keys are the raw integers *as strings* (JSON object keys are strings); values are the
    corresponding enum/category names the decoder normalizes to.
    """

    match_phase: dict[str, str]  # raw -> MatchState value (pre_round|in_round|...)
    game_mode: dict[
        str, str
    ]  # raw -> mode category (idle|offline_match|online_match|replay|practice)
    counter_state: dict[str, str]  # raw -> CounterState value (none|counter_hit|punish_counter)
    simple_state: dict[str, str]  # raw -> neutral|attack|recovery (docs/02 §2 simple state)


class SanityBounds(BaseModel):
    """Plausibility bounds for the doctor self-check (docs/02 §6)."""

    round_start_health: int  # expected max HP at round start
    health_plausible_min: int
    health_plausible_max: int
    move_id_max: int  # a move id at/above this is treated as garbage


class OffsetTable(BaseModel):
    """A complete versioned offset table (one ``assets/offsets/<version>.json``)."""

    game_version: str
    discovered_at: str  # ISO-8601 timestamp of the update-offsets run that produced it
    notes: str  # run notes (docs/02 §3)
    endianness: Literal["little"] = "little"
    pointer_size: Literal[8] = 8
    global_struct: GlobalStruct = Field(alias="global")
    players: PlayerStruct
    state_codes: StateCodes
    sanity: SanityBounds

    model_config = {"populate_by_name": True}


class OffsetIndexEntry(BaseModel):
    """One version -> file binding in ``index.json``."""

    file: str


class OffsetIndex(BaseModel):
    """``assets/offsets/index.json`` (docs/02 §3): version -> file, plus the detected marker."""

    detected_version: str  # the build the checked-in assets are aligned to (a marker, docs/02 §3)
    versions: dict[str, OffsetIndexEntry]


# ---------------------------------------------------------------------------
# Loading + selection
# ---------------------------------------------------------------------------


def load_offset_index(offsets_dir: str | Path = DEFAULT_OFFSETS_DIR) -> OffsetIndex:
    """Load ``assets/offsets/index.json``."""
    path = Path(offsets_dir) / "index.json"
    try:
        return OffsetIndex.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OffsetTableError(f"offset index not found: {path}") from exc
    except ValidationError as exc:
        raise OffsetTableError(f"malformed offset index {path}: {exc}") from exc


def load_offset_table(path: str | Path) -> OffsetTable:
    """Load a single ``assets/offsets/<version>.json`` table."""
    try:
        return OffsetTable.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OffsetTableError(f"offset table not found: {path}") from exc
    except ValidationError as exc:
        raise OffsetTableError(f"malformed offset table {path}: {exc}") from exc


def select_offset_table(
    game_version: str,
    offsets_dir: str | Path = DEFAULT_OFFSETS_DIR,
) -> OffsetTable:
    """Select the offset table for ``game_version``, or fail closed (docs/02 §3/§7).

    This is the version-detection gate: the caller supplies the detected version (injected in
    C4a; read from the running process in C4b), and we load the matching table. An unknown
    version raises :class:`UnknownGameVersionError` (with the §4 runbook) rather than guessing
    with a stale table — a wrong offset silently produces garbage, which is worse than not
    running.
    """
    root = Path(offsets_dir)
    index = load_offset_index(root)
    entry = index.versions.get(game_version)
    if entry is None:
        raise UnknownGameVersionError(game_version, sorted(index.versions))
    table = load_offset_table(root / entry.file)
    if table.game_version != game_version:
        raise OffsetTableError(
            f"offset table {entry.file} declares version {table.game_version!r}, "
            f"expected {game_version!r}"
        )
    return table
