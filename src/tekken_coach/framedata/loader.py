"""Typed, miss-tolerant loaders for the move map and the current frame-data snapshot.

Loads the two C1 assets from disk into the models in :mod:`tekken_coach.framedata.models`
and resolves a ``move_id`` through the explicit ``move_id -> framedata_key -> move`` join
(docs/05 §4.1). Every lookup is **miss-tolerant** (docs/05 §2.3, §4.1, §6): an unknown
``move_id`` or an unresolved ``framedata_key`` returns a :class:`MoveLookup` with
``matched=False`` (the ``frame_data_matched:false`` path) rather than raising. Callers (the
C2 xref) turn that into an unlabeled interaction, never a crash.

The ``current`` snapshot pointer (docs/05 §3.2) is resolved either as a symlink to a
``snapshot-<date>/`` directory (the spec's ``current -> snapshot-<date>/`` form) or, on
filesystems without symlinks, as a plain text file whose contents name the snapshot dir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    FrameDataSnapshot,
    MoveMapIndex,
    SnapshotManifest,
)

DEFAULT_MOVEMAP_DIR = Path("assets/movemap")
DEFAULT_FRAMEDATA_DIR = Path("assets/framedata")
CURRENT_POINTER = "current"


# ---------------------------------------------------------------------------
# Lookup result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveLookup:
    """The result of resolving a ``move_id`` against the move map + frame-data snapshot.

    On a miss (unknown ``move_id`` or unresolved ``framedata_key``), ``matched`` is ``False``
    and ``move`` is ``None`` — the ``frame_data_matched:false`` path (docs/05 §2.3/§4.1). An
    unknown ``move_id`` still gets a stable ``notation`` of ``"move_id:<n>"`` (docs/05 §2.3).
    """

    move_id: int
    matched: bool
    notation: str  # move-map notation, or "move_id:<n>" fallback for an unknown id
    framedata_key: str | None
    move: FrameDataMove | None
    char_name: str | None


# ---------------------------------------------------------------------------
# Move map loading
# ---------------------------------------------------------------------------


def load_move_map_index(movemap_dir: str | Path = DEFAULT_MOVEMAP_DIR) -> MoveMapIndex:
    """Load ``assets/movemap/index.json`` (docs/05 §2.2)."""
    path = Path(movemap_dir) / "index.json"
    return MoveMapIndex.model_validate_json(path.read_text(encoding="utf-8"))


def load_char_move_map(path: str | Path) -> CharMoveMap:
    """Load a single character's move-map file (docs/05 §2.2)."""
    return CharMoveMap.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_move_maps(movemap_dir: str | Path = DEFAULT_MOVEMAP_DIR) -> dict[str, CharMoveMap]:
    """Load every character move map referenced by the index, keyed by ``char_name``.

    Missing per-character files are skipped rather than fatal, so a partially-seeded
    ``assets/movemap/`` (docs/05 §2.3) still loads what exists.
    """
    root = Path(movemap_dir)
    index = load_move_map_index(root)
    maps: dict[str, CharMoveMap] = {}
    for entry in index.characters:
        file_path = root / entry.file
        if file_path.exists():
            maps[entry.char_name] = load_char_move_map(file_path)
    return maps


# ---------------------------------------------------------------------------
# Frame-data snapshot loading
# ---------------------------------------------------------------------------


def resolve_current_snapshot(framedata_dir: str | Path = DEFAULT_FRAMEDATA_DIR) -> Path:
    """Resolve the ``current`` pointer to a concrete ``snapshot-<date>/`` directory (docs/05 §3.2).

    Accepts either a symlink (``current -> snapshot-<date>/``) or a plain text file whose
    contents name the snapshot directory. Raises :class:`FileNotFoundError` if neither the
    pointer nor its target exists (a genuinely-missing snapshot is a setup error, not a
    per-lookup miss).
    """
    root = Path(framedata_dir)
    pointer = root / CURRENT_POINTER
    if pointer.is_symlink() or pointer.is_dir():
        target = pointer.resolve()
        if not target.is_dir():
            raise FileNotFoundError(f"current -> {target} is not a directory")
        return target
    if pointer.is_file():
        name = pointer.read_text(encoding="utf-8").strip()
        target = root / name
        if not target.is_dir():
            raise FileNotFoundError(f"current points at missing snapshot {name!r}")
        return target
    raise FileNotFoundError(f"no 'current' snapshot pointer in {root}")


def load_snapshot_manifest(snapshot_dir: str | Path) -> SnapshotManifest:
    """Load a snapshot's ``manifest.json`` (docs/05 §3.2)."""
    path = Path(snapshot_dir) / "manifest.json"
    return SnapshotManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_snapshot(snapshot_dir: str | Path) -> FrameDataSnapshot:
    """Load a specific frame-data snapshot directory into a :class:`FrameDataSnapshot`."""
    directory = Path(snapshot_dir)
    manifest = load_snapshot_manifest(directory)
    characters: dict[str, CharFrameData] = {}
    for slug, entry in manifest.characters.items():
        char_path = directory / entry.file
        char = CharFrameData.model_validate_json(char_path.read_text(encoding="utf-8"))
        characters[slug] = char
    return FrameDataSnapshot(manifest=manifest, characters=characters)


def load_current_framedata(framedata_dir: str | Path = DEFAULT_FRAMEDATA_DIR) -> FrameDataSnapshot:
    """Load the ``current`` frame-data snapshot (docs/05 §3.2)."""
    return load_snapshot(resolve_current_snapshot(framedata_dir))


# ---------------------------------------------------------------------------
# Miss-tolerant resolution (the frame_data_matched:false path, docs/05 §2.3/§4.1)
# ---------------------------------------------------------------------------


def resolve_move(
    move_id: int,
    move_map: CharMoveMap | None,
    char_framedata: CharFrameData | None,
) -> MoveLookup:
    """Resolve ``move_id`` through ``move_id -> framedata_key -> move`` (docs/05 §4.1).

    Miss-tolerant in three ways, each yielding ``matched=False`` (never an exception):

    * no move map for the character, or the ``move_id`` isn't in it -> unknown move
      (``notation="move_id:<n>"``, docs/05 §2.3);
    * the mapped ``framedata_key`` isn't in the snapshot -> unresolved key;
    * only when both resolve is ``matched=True`` with the concrete move record.
    """
    char_name = move_map.char_name if move_map is not None else None
    entry = move_map.get(move_id) if move_map is not None else None
    if entry is None:
        return MoveLookup(
            move_id=move_id,
            matched=False,
            notation=f"move_id:{move_id}",
            framedata_key=None,
            move=None,
            char_name=char_name,
        )
    move = char_framedata.get(entry.framedata_key) if char_framedata is not None else None
    return MoveLookup(
        move_id=move_id,
        matched=move is not None,
        notation=entry.notation,
        framedata_key=entry.framedata_key,
        move=move,
        char_name=char_name,
    )
