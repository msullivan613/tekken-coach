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

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from tekken_coach.reader.faults import OffsetTableError, UnknownGameVersionError

DEFAULT_OFFSETS_DIR = Path("assets/offsets")
DEFAULT_STATE_MAP_PATH = DEFAULT_OFFSETS_DIR / "state-map.json"

# Scalar kinds a field may have. Fixed little-endian widths; the decoder maps these to struct
# formats and the (test-only) encoder inverts them.
ScalarKind = Literal["u8", "u16", "u32", "i32", "i64", "f32", "bool8", "ptr"]

# The vocabulary an encoded state value may resolve to (:class:`EncodedStateSpec`). These are the
# *semantic facts* the decoder folds into a ``PlayerFrame``: the three mutually-exclusive "simple"
# states plus the situational flags. Kept here (not in ``decode``) so the offset loader can reject
# a typo in the data map at load time rather than silently dropping a flag at 60fps.
STATE_FLAGS: tuple[str, ...] = (
    "neutral",
    "attack",
    "recovery",
    "block_stun",
    "hit_stun",
    "stagger",
    "throw_active",
    "throw_tech",
    "thrown",
    "airborne",
    "juggle",
    "knockdown",
    "wakeup",
    "sidestep",
    "crouch",
)

# The component (docs/02 §3, C4e Phase 3) whose fields carry ``pos_{x,y,z}``. Tekken 8 keeps
# position in a separate Unreal transform component, not in the entity struct.
POSITION_COMPONENT = "transform"


class FieldSpec(BaseModel):
    """One field within a struct: a byte ``offset`` from the struct base and its scalar ``kind``."""

    offset: int
    kind: ScalarKind


class ComponentAnchor(BaseModel):
    """A struct hanging off the player struct by its own pointer (docs/02 §3, C4e Phase 3).

    Tekken 8 does not keep position in the entity struct: a full-struct scan finds no moving float
    triple, and the fork's layout data has no position field. It lives in a separate Unreal
    **transform component**, reached by a pointer stored *inside* the entity struct. A single
    ``anchor + stride + flat fields`` :class:`PlayerStruct` cannot express that, so a player may
    carry named components, each resolved **relative to that player's base**::

        address = deref(player_base + slot_offset)     # the component object
        for o in pointer_path:                         # further hops (a nested component)
            address = deref(address + o)
        # fields are read at address + field.offset

    ``pointer_path`` is empty for the common one-hop case. Both players use the same component
    layout (the structs are symmetric), so one :class:`ComponentAnchor` serves P1 and P2 — the
    derivation confirms that by resolving the component for *both* before accepting it.

    ``fields`` defaults to empty because the same "deref a pointer slot (+ optional chain)" shape
    also expresses a **per-player anchor** (C4i, docs/02 §3): Tekken 8's real model is a holder
    object with two pointer slots to *separate* P1/P2 allocations (``holder+0x30`` and
    ``holder+0x38``), not one array with a constant stride. There the landing *is* the player base
    and the fields are :attr:`PlayerStruct.fields`, so the slot carries no fields of its own (see
    :attr:`PlayerStruct.player_slots`).
    """

    slot_offset: int  # byte offset of the pointer slot within the player struct
    pointer_path: list[int] = Field(default_factory=list)
    fields: dict[str, FieldSpec] = Field(default_factory=dict)


class EncodedStateSpec(BaseModel):
    """Raw encoded state value -> the semantic flags it implies (docs/02 §3, C4e Phase 2).

    Tekken 8's entity struct does not carry the per-flag booleans ``PlayerFrame`` wants
    (``block_stun``, ``hit_stun``, ``airborne``, ...). It carries a handful of **encoded state
    words** — ``simple_move_state``, ``stun_type``, ``complex_move_state``, ... — whose integer
    values each denote a whole situation. ``flags`` is the value -> meaning map:

    * outer key: the player field name (which must exist in :attr:`PlayerStruct.fields`),
    * inner key: the raw integer value *as a string* (JSON object keys are strings),
    * value: the :data:`STATE_FLAGS` that value implies (possibly empty).

    The decoder reads every mapped field, unions the flags, and folds them into the ``PlayerFrame``
    — so the value -> meaning semantics are **data**, calibratable without a source change (docs/02
    §4/§8), and the decode logic stays clean-room. An unmapped raw value contributes no flags; the
    raw integers ride along on ``PlayerFrame.raw_state`` (docs/03 §1) so the calibration loop can
    see exactly what it has not mapped yet.

    ``calibrated`` records whether the map has actually been filled in by observation. A ``False``
    map decodes to ``neutral`` for everything — structurally valid, semantically empty — so the
    tooling flags it loudly rather than letting an uncalibrated map look like a working reader.
    """

    calibrated: bool = False
    notes: str = ""
    flags: dict[str, dict[str, list[str]]]

    @field_validator("flags")
    @classmethod
    def _known_flags(
        cls, value: dict[str, dict[str, list[str]]]
    ) -> dict[str, dict[str, list[str]]]:
        for field_name, codes in value.items():
            for raw, flags in codes.items():
                unknown = [f for f in flags if f not in STATE_FLAGS]
                if unknown:
                    raise ValueError(
                        f"state map {field_name}[{raw}] names unknown flag(s) {unknown}; "
                        f"known flags: {list(STATE_FLAGS)}"
                    )
        return value


class AobSignature(BaseModel):
    """A code/data-signature that re-finds an anchor's static ``base_offset`` after a patch.

    ``base_offset`` shifts every build, but the bytes *around* the static pointer slot are stable,
    so ``update-offsets`` stores the surrounding window as a wildcard AOB pattern (docs/02 §3, C4d).
    A re-run scans the module for ``pattern`` and recovers the slot — a fast path that skips the
    full candidate scan. This is facts/data (docs/02 §5), not code; the decoder ignores it (it
    resolves via ``base_offset``).

    Two re-find modes, one per re-discovery technique:

    * **data-adjacent** (``disp32_pos is None``, the C4d default): the pattern matches the ``.data``
      bytes surrounding the pointer slot and the slot sits at ``match_rva + slot_delta``.
    * **RIP-relative code** (``disp32_pos`` set, C4i): the pattern matches an *instruction* in
      ``.text`` that references the slot with a 32-bit RIP-relative displacement. The displacement
      is an ``i32`` embedded in the match at byte ``disp32_pos``, and the slot is at
      ``match_rva + disp32_pos + 4 + disp32`` (x64 RIP is the address of the *next* instruction, so
      the 4 displacement bytes end the reference). This is the durable, self-healing anchor every
      community tool uses — an AOB over a function body survives most patches (docs/02 §3, the T8
      holder model).
    """

    pattern: str  # wildcard AOB, e.g. "4C 89 35 ?? ?? ?? ?? 41 88 5E 28" (disp32 wildcarded)
    slot_delta: int = 0  # data-adjacent: offset from a match to the pointer slot
    disp32_pos: int | None = None  # RIP-relative: byte offset of the i32 displacement in the match


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
    """The per-player struct: how to reach each player's base, plus the shared field layout.

    Two addressing models, exactly one of which a table uses (enforced below):

    * **stride** (C4c/C4d, the fork's model): :attr:`anchor` resolves P1's base and P2 sits a
      constant ``stride`` bytes later — one array, two players.
    * **player_slots** (C4i, Tekken 8's real model): :attr:`anchor` resolves a **holder object**,
      and each player's base is dereferenced from its own pointer slot inside it (``holder+0x30``
      for P1, ``holder+0x38`` for P2 — two *separate* allocations, not an array). Each slot is a
      :class:`ComponentAnchor` (deref slot + optional chain) whose landing is the player base; its
      fields come from :attr:`fields`, so the slot itself carries none. This is what the live game
      does (Irony/opendojo, verified on T8 v3.00.02), and ``--base-scan``'s single-anchor+stride
      model cannot express it.

    ``max_health`` (optional) switches health to a **computed** field: Tekken 8's entity struct
    stores ``damage_taken`` (rising from 0), not current HP (docs/02 §3, confirmed live — HP is
    encrypted in a separate subsystem). When set, the decoder reads ``damage_taken`` and reports
    ``health = max_health - damage_taken``; when ``None`` (the C4c/legacy path) it reads a direct
    ``health`` field. This is the fork's own health model.

    ``components`` (optional) holds fields that are **not** in the entity struct but in a separate
    object it points at — on Tekken 8 that is ``pos_{x,y,z}``, in the
    :data:`POSITION_COMPONENT` transform (see :class:`ComponentAnchor`), resolved relative to each
    player's base. Empty on the C4c/legacy path, where position is a plain in-struct field.
    """

    anchor: Anchor
    stride: int | None = None
    player_slots: list[ComponentAnchor] = Field(default_factory=list)
    fields: dict[str, FieldSpec]
    max_health: int | None = None
    components: dict[str, ComponentAnchor] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_addressing_model(self) -> PlayerStruct:
        """Exactly one addressing model: a constant ``stride`` *or* per-player ``player_slots``.

        A table that sets neither cannot place P2 (or P1); a table that sets both is ambiguous about
        which the decoder should trust. Both are silent-garbage risks, so reject at load time.
        """
        if self.player_slots:
            if self.stride is not None:
                raise ValueError(
                    "PlayerStruct sets both stride and player_slots; use one addressing model "
                    "(stride for the C4c/C4d array layout, player_slots for the C4i holder layout)"
                )
        elif self.stride is None:
            raise ValueError(
                "PlayerStruct sets neither stride nor player_slots; one is required to locate the "
                "players (stride = P2 at P1+stride; player_slots = holder+slot per player)"
            )
        return self


class StateCodes(BaseModel):
    """Raw game code -> semantic mappings (kept as data so a patch is a data edit, docs/02 §4).

    Keys are the raw integers *as strings* (JSON object keys are strings); values are the
    corresponding enum/category names the decoder normalizes to.

    ``encoded_state`` (optional) switches the per-player state decode from the C4c/legacy
    one-boolean-per-flag layout to Tekken 8's real one: a few **encoded state words** whose values
    denote whole situations (:class:`EncodedStateSpec`). When present the decoder ignores the
    legacy ``simple_state``/``block_stun``/... fields entirely.
    """

    match_phase: dict[str, str]  # raw -> MatchState value (pre_round|in_round|...)
    game_mode: dict[
        str, str
    ]  # raw -> mode category (idle|offline_match|online_match|replay|practice)
    counter_state: dict[str, str]  # raw -> CounterState value (none|counter_hit|punish_counter)
    simple_state: dict[str, str]  # raw -> neutral|attack|recovery (docs/02 §2 simple state)
    encoded_state: EncodedStateSpec | None = None


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
    # The game-MEMORY char ids this table was calibrated with (P1=Jin, P2=Kazuya), recorded by the
    # discovery so the doctor's char-id check (docs/02 §6) validates against the memory id space the
    # reader actually reads — NOT the movemap/framedata id space (a separate space; the memory->slug
    # roster is a C5 gap). Empty on older/hand-written tables; the doctor then falls back to the
    # movemap index.
    known_char_ids: list[int] = Field(default_factory=list)
    # Game-MEMORY char id -> display name, established by observation (docs/02 §8) — the memory id
    # space the reader reads, NOT the movemap/framedata id space (a different space; see
    # ``known_char_ids``). Lets ``--char <name>`` validate against what the reader actually sees
    # (project memory ``t8-reader-model-holder-aob``). Deliberately partial: only observed
    # characters are baked, and an unobserved id falls back to ``char:<id>``. Keys are strings
    # (JSON object keys); :meth:`char_names_by_id` gives the int-keyed view the resolver wants.
    char_names: dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def char_names_by_id(self) -> dict[int, str]:
        """The ``char_names`` map keyed by int memory char id (JSON keys are strings)."""
        return {int(cid): name for cid, name in self.char_names.items()}


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


def load_state_map(path: str | Path = DEFAULT_STATE_MAP_PATH) -> EncodedStateSpec:
    """Load the encoded-state value -> meaning map (docs/02 §8), the calibratable data file.

    Kept out of the offset tables so it survives a re-discovery run: ``update-offsets`` rewrites the
    *addresses* every build, but the value -> meaning **semantics** are calibrated once by
    observation and carried forward. The builder copies the loaded map into the table it writes.
    """
    try:
        return EncodedStateSpec.model_validate_json(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OffsetTableError(f"state map not found: {path}") from exc
    except ValidationError as exc:
        raise OffsetTableError(f"malformed state map {path}: {exc}") from exc


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
