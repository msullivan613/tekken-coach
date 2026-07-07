"""Frame-data cross-reference package (docs/05).

C1 delivers the two assets and their typed, miss-tolerant loaders plus the ``fetch-framedata``
ingest. The pure cross-reference function (``Interaction`` -> ``LabeledInteraction``) and the
rubric machine layer land in C2 (docs/05 §4, docs/06 §4.1).

Public surface:

* **Models** — :class:`CharMoveMap`, :class:`MoveMapIndex`, :class:`FrameDataSnapshot`,
  :class:`CharFrameData`, :class:`FrameDataMove`, :class:`Hit`, :class:`SnapshotManifest`.
* **Loaders** — :func:`load_move_maps`, :func:`load_current_framedata`, :func:`resolve_move`,
  and the miss-tolerant :class:`MoveLookup` result.
* **Ingest** — :func:`fetch_framedata`, :func:`promote_snapshot`, :class:`CharSpec`.
"""

from __future__ import annotations

from tekken_coach.framedata.ingest import (
    CharDiff,
    CharSpec,
    Fetcher,
    IngestResult,
    UrllibFetcher,
    discover_char_csvs,
    fetch_framedata,
    promote_snapshot,
)
from tekken_coach.framedata.loader import (
    MoveLookup,
    load_char_move_map,
    load_current_framedata,
    load_move_map_index,
    load_move_maps,
    load_snapshot,
    load_snapshot_manifest,
    resolve_current_snapshot,
    resolve_move,
)
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    DuckPunish,
    FrameDataMove,
    FrameDataSnapshot,
    HeatOverride,
    Hit,
    MoveMapEntry,
    MoveMapIndex,
    MoveMapIndexEntry,
    SnapshotCharEntry,
    SnapshotManifest,
)

__all__ = [
    # models
    "CharFrameData",
    "CharMoveMap",
    "DuckPunish",
    "FrameDataMove",
    "FrameDataSnapshot",
    "HeatOverride",
    "Hit",
    "MoveMapEntry",
    "MoveMapIndex",
    "MoveMapIndexEntry",
    "SnapshotCharEntry",
    "SnapshotManifest",
    # loaders
    "MoveLookup",
    "load_char_move_map",
    "load_current_framedata",
    "load_move_map_index",
    "load_move_maps",
    "load_snapshot",
    "load_snapshot_manifest",
    "resolve_current_snapshot",
    "resolve_move",
    # ingest
    "CharDiff",
    "CharSpec",
    "Fetcher",
    "IngestResult",
    "UrllibFetcher",
    "discover_char_csvs",
    "fetch_framedata",
    "promote_snapshot",
]
