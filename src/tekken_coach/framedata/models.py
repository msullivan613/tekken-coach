"""Typed models for the two C1 assets: the move map and the frame-data snapshot.

These mirror the on-disk shapes defined in docs/05 §2.2 (move map) and §3.2 (frame-data
snapshot). They are the contract the frame-data cross-reference (C2) consumes, so field
names line up with the ``labels`` block in docs/03 §3.

Two assets, two keys (docs/05 §1):

* **Move map** — keyed by the game's memory ``move_id`` (e.g. ``2145``). Maps an id to a
  ``notation`` and, crucially, a ``framedata_key`` — the explicit join key into the
  frame-data table (kept separate because notation strings and table keys don't always
  match, docs/05 §2.2).
* **Frame-data snapshot** — keyed by ``framedata_key`` (the CSV ``Command`` column). Carries
  the per-move fields and, for multi-hit strings, a per-hit ``hits[]`` sequence with a
  per-hit ``hit_level`` (docs/05 §3.2), which feeds the duckable-high check (docs/06 §4.1).

Both loaders are miss-tolerant (docs/05 §2.3, §4.1, §6): an unknown ``move_id`` or an
unresolved ``framedata_key`` degrades to a "no match" result, never an exception.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from tekken_coach.schemas import MoveProperty, StringGap

# ---------------------------------------------------------------------------
# Move map (assets/movemap/), docs/05 §2.2
# ---------------------------------------------------------------------------


class MoveMapEntry(BaseModel):
    """One ``move_id`` -> notation binding in a character's move map (docs/05 §2.2)."""

    notation: str  # human notation, e.g. "df+2"
    aliases: list[str] = Field(default_factory=list)  # alternate spellings, e.g. "down-forward 2"
    framedata_key: str  # explicit join key into the frame-data table (docs/05 §2.2)


class CharMoveMap(BaseModel):
    """A single character's move map file, e.g. ``assets/movemap/kazuya.json`` (docs/05 §2.2).

    ``moves`` is keyed by the game's memory ``move_id`` as a **string** (JSON object keys are
    strings); callers look up with either ``str`` or ``int`` via :meth:`get`. The map is
    deliberately **partial** for v1 (docs/05 §2.3): only scoped matchups need a complete map,
    and the ``move_id`` -> ``framedata_key`` binding is a property of the game build that comes
    from the reader (C4), so it can only be seeded from spec-sourced example ids here. The
    ``framedata_keys`` list is the fully-derivable seed (every notation from the CSV ``Command``
    column); ``partial`` flags that ``moves`` is not yet exhaustive.
    """

    char_id: int | None = None  # integer character id; None when not yet sourced (C4)
    char_name: str
    game_version: str
    partial: bool = False  # True when the move_id table is not yet exhaustive (docs/05 §2.3)
    moves: dict[str, MoveMapEntry] = Field(default_factory=dict)  # move_id (str) -> entry
    # Fully-derivable seed: every notation from the CSV Command column. Lets C2/tooling see the
    # complete framedata_key set even before move_ids are bound (docs/05 §2.3 scope boundary).
    framedata_keys: list[str] = Field(default_factory=list)

    def get(self, move_id: int | str) -> MoveMapEntry | None:
        """Return the entry for ``move_id`` (int or str), or ``None`` on a miss (never raises)."""
        return self.moves.get(str(move_id))


class MoveMapIndexEntry(BaseModel):
    """One entry in the move-map index (docs/05 §2.2)."""

    char_id: int | None = None
    char_name: str
    file: str  # relative filename, e.g. "kazuya.json"


class MoveMapIndex(BaseModel):
    """``assets/movemap/index.json`` — char_id -> name -> file, plus a game_version stamp.

    docs/05 §2.2. ``char_id`` may be ``None`` for a character whose integer id is not yet
    sourced (it comes from the reader, C4); such entries are still addressable by ``char_name``.
    """

    game_version: str
    characters: list[MoveMapIndexEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Frame-data snapshot (assets/framedata/), docs/05 §3.2
# ---------------------------------------------------------------------------


class Hit(BaseModel):
    """One hit of a multi-hit string (docs/05 §3.2).

    ``hit_level`` is populated directly from the CSV ``Hit level`` column, split on commas.
    ``hit_level_raw`` preserves the exact CSV token (e.g. ``"m!"``, ``"sp"``, ``"th(h)"``)
    because the CSV vocabulary is richer than the five-value :class:`MoveProperty` enum; the
    enum is a best-effort mapping and is ``None`` when the token has no clean height mapping.
    ``startup`` is per-hit and may be ``None`` (the CSV occasionally omits per-hit startup,
    docs/05 §3.1) — C2 needs it to compute string gaps (docs/05 §4.1).
    """

    hit_level: MoveProperty | None = None  # mapped height; None when the token is unmappable
    hit_level_raw: str  # exact CSV token, preserved for fidelity/curation
    startup: int | None = None  # per-hit startup (lower bound); None if the CSV omits it
    startup_raw: str | None = None  # exact CSV startup token, e.g. "i22~23"
    damage: int | None = None


class DuckPunish(BaseModel):
    """A hand-curated duck-the-high marker for a string (docs/05 §3.2).

    Not present in the CSV — derived and curated against okizeme.gg for scoped matchups
    (docs/05 §3.1/§3.2). Absent (``None`` on the move) => no ``standing_duckable_high`` flag,
    which is a safe miss (docs/05 §4.1).
    """

    after_hit: int  # duck the high that is hit ``after_hit`` (1-based hit index)
    answer: str  # recommended punish after ducking, e.g. "df+1 (i13)"


class StringGapInfo(BaseModel):
    """A hand-curated string-gap (timing) annotation for a string (docs/05 §3.2, §4.1).

    Distinct from :class:`DuckPunish`, which is a *height* check. This is the *timing* gap
    between two hits of a string: whether the string jails (``true``), leaves an interruptible
    gap (``interruptible``), or a duckable window (``duckable``), and how large the gap is.

    Not derivable from the snapshot CSV — the CSV has no per-hit on-block frames, only per-hit
    startup (docs/05 §4.1, gap #3). It is hand-curated against okizeme.gg / Wavu the same way
    ``duck_punish`` is; absent (``None`` on the move) => ``string_gap`` stays null, a safe miss.
    """

    after_hit: int  # the gap sits after hit ``after_hit`` (1-based hit index)
    gap: StringGap  # duckable | interruptible | true (docs/03 §3 labels.string_gap)
    gap_size: int | None = None  # frames of the gap, if known (docs/03 §3 labels.gap_size)


class MoveCuration(BaseModel):
    """One move's curated overlay entry (brief #17 §A): the hand-curated fields only.

    A durable overlay that survives a raw re-scrape: :class:`DuckPunish`/:class:`StringGapInfo`
    are curated project annotations "not from the CSV" (see :class:`FrameDataMove`), so a fresh
    scrape drops them. The overlay lives outside the snapshot dirs and is merged back at load
    time (:func:`~tekken_coach.framedata.loader.apply_curation`). Only the curated fields live
    here; everything else on a move comes from the scrape.
    """

    duck_punish: DuckPunish | None = None  # curated height check (models.DuckPunish)
    string_gap: StringGapInfo | None = None  # curated timing gap (models.StringGapInfo)


class CharCuration(BaseModel):
    """A single character's curation overlay, e.g. ``assets/framedata/curation/paul.json``.

    Keyed by ``framedata_key`` (the same key as the snapshot's ``moves``). A character with no
    curation simply has no overlay file — a missing file is a no-op, not an error (brief #17 §A).
    """

    char_slug: str  # CSV Character id, e.g. "paul" — matches the snapshot's char_slug
    moves: dict[str, MoveCuration] = Field(default_factory=dict)  # framedata_key -> curated fields


class HeatOverride(BaseModel):
    """Heat-state overrides where a move differs in Heat (docs/05 §3.2, §04 §4.6).

    Not present in the base CSV columns — curated. All fields optional; only the ones that
    differ in Heat are set.
    """

    on_block: int | None = None
    on_hit: str | None = None
    on_ch: str | None = None


class FrameDataMove(BaseModel):
    """One move (or string) in a character's frame-data file (docs/05 §3.2).

    The scalar fields (``startup``, ``on_block``, ``hit_level`` …) describe the move as a whole
    and are always populated where the CSV has the data. ``hits`` is populated **only for
    multi-hit strings** (``is_string`` true), mirroring the two shapes in docs/05 §3.2: a single
    move carries scalar ``hit_level``; a string additionally carries the per-hit ``hits[]``.

    Frame values that carry CSV annotations (``+0c``, ``-11a``, ``-13~-8``, ``+32a (+24)``) are
    kept as raw strings in the ``*_raw`` fields; the parsed leading integer is exposed where it
    is clean. ``on_hit``/``on_ch`` stay strings because they carry launch/knockdown markers
    (docs/05 §3.2: "or a launch marker").
    """

    key: str  # framedata_key == the CSV Command notation
    is_string: bool = False  # True when the move has more than one hit level (docs/05 §3.2)

    startup: int | None = None  # first-hit startup (lower bound of any range)
    startup_raw: str | None = None
    on_block: int | None = None  # ground-truth on-block advantage (docs/03 §3 labels.on_block)
    block_raw: str | None = None  # raw CSV Block frame cell (may carry an annotation)
    on_hit: str | None = None  # raw CSV Hit frame cell (int-or-launch-marker)
    on_ch: str | None = None  # raw CSV Counter hit frame cell
    damage: int | None = None
    hit_level: MoveProperty | None = None  # scalar height for single moves; None for strings
    hit_level_raw: str | None = None
    properties: list[str] = Field(default_factory=list)  # from the CSV Tags column (raw codes)
    recovery: int | None = None  # parsed CSV Recovery when numeric
    recovery_raw: str | None = None  # raw CSV Recovery (may be a stance code, e.g. "FDFA")
    name: str | None = None  # move display name (CSV Name)
    wavu_id: str | None = None  # CSV Wavu id, e.g. "Kazuya-df+2"
    notes: str | None = None  # CSV Notes (may contain newlines)

    hits: list[Hit] = Field(default_factory=list)  # per-hit sequence; strings only (docs/05 §3.2)
    duck_punish: DuckPunish | None = None  # curated height check; not from the CSV (docs/05 §3.2)
    string_gap: StringGapInfo | None = None  # curated timing gap; not from the CSV (docs/05 §4.1)
    heat: HeatOverride | None = None  # curated Heat overrides; not from the base CSV


class CharFrameData(BaseModel):
    """A single character's frame-data file, e.g. ``snapshot-<date>/kazuya.json`` (docs/05 §3.2).

    Keyed by ``framedata_key``. ``char_slug`` is the CSV ``Character id`` (e.g. ``"kazuya"``);
    the integer ``char_id`` is intentionally absent here because it is a game-build property
    (move map / reader), not a frame-data property (docs/05 §1).
    """

    char_slug: str  # CSV Character id, e.g. "kazuya"
    char_name: str
    moves: dict[str, FrameDataMove] = Field(default_factory=dict)  # framedata_key -> move

    def get(self, framedata_key: str) -> FrameDataMove | None:
        """Return the move for ``framedata_key`` or ``None`` on a miss (never raises)."""
        return self.moves.get(framedata_key)


class SnapshotCharEntry(BaseModel):
    """Per-character provenance recorded in ``manifest.json`` (docs/05 §3.2)."""

    file: str  # relative filename, e.g. "kazuya.json"
    csv_files: list[str] = Field(default_factory=list)  # source CSVs, e.g. ["kazuya-special.csv"]
    checksum: str  # sha256 of the concatenated source CSV bytes, "sha256:<hex>"
    move_count: int


class SnapshotManifest(BaseModel):
    """``snapshot-<date>/manifest.json`` — the reproducibility record (docs/05 §3.2, §3.3).

    Records the pinned source commit SHA (so the snapshot is reproducible, not date-keyed),
    attribution, and per-character checksums. ``game_version`` is nullable: the CSV does not
    carry the balance-patch version, so it is curated at ingest time when known (docs/05 §3.2
    lists it, but it is not derivable from the source).
    """

    source_repo: str  # "pbruvoll/tekkendocs"
    source_commit: str  # pinned commit SHA — the snapshot key (docs/05 §3.1)
    source_path_template: str  # how each CSV was located, for reproducibility
    fetched_at: str  # ISO-8601 timestamp of the ingest run
    snapshot_date: str  # the date component of the snapshot dir name
    game_version: str | None = None  # balance-patch version; not in the CSV, curated
    attribution: list[str] = Field(default_factory=list)  # ["tekkendocs.com", "rbnorway.org"]
    characters: dict[str, SnapshotCharEntry] = Field(default_factory=dict)  # char_slug -> entry


class FrameDataSnapshot(BaseModel):
    """A loaded frame-data snapshot: manifest plus every character's frame data (docs/05 §3.2)."""

    manifest: SnapshotManifest
    characters: dict[str, CharFrameData] = Field(default_factory=dict)  # char_slug -> data

    def get_char(self, char_slug: str) -> CharFrameData | None:
        """Return a character's frame data by slug, or ``None`` on a miss (never raises)."""
        return self.characters.get(char_slug)
