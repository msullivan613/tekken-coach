"""Frame-data cross-reference package (docs/05).

C1 delivers the two assets and their typed, miss-tolerant loaders plus the ``fetch-framedata``
ingest. C2 adds the pure cross-reference function (``Interaction`` -> ``LabeledInteraction``), the
machine-layer rubric, the curated punisher profiles, and the recurrence tally (docs/05 §4,
docs/06 §4.1, docs/03 §4).

Public surface:

* **Models** — :class:`CharMoveMap`, :class:`MoveMapIndex`, :class:`FrameDataSnapshot`,
  :class:`CharFrameData`, :class:`FrameDataMove`, :class:`Hit`, :class:`SnapshotManifest`.
* **Loaders** — :func:`load_move_maps`, :func:`load_current_framedata`, :func:`resolve_move`,
  and the miss-tolerant :class:`MoveLookup` result.
* **Ingest** — :func:`fetch_framedata`, :func:`promote_snapshot`, :class:`CharSpec`.
* **Xref (C2)** — :func:`label_interaction`, :func:`label_interactions`.
* **Rubric (C2)** — :data:`DEFAULT_RUBRIC`, :class:`RubricPattern`, :func:`evaluate_triggers`.
* **Punishers (C2)** — :func:`load_punisher_profiles`, :class:`PunisherProfiles`, :class:`Punisher`.
* **Tally (C2)** — :func:`build_tally`, :class:`KnowledgeCheckTally`, :class:`TallyEntry`.
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
    StringGapInfo,
)
from tekken_coach.framedata.punishers import (
    FALLBACK_STANDING_STARTUP,
    Punisher,
    PunisherProfile,
    PunisherProfiles,
    PunisherStance,
    load_punisher_profiles,
)
from tekken_coach.framedata.rubric import (
    DEFAULT_RUBRIC,
    RubricPattern,
    evaluate_triggers,
    thresholds,
)
from tekken_coach.framedata.tally import (
    KnowledgeCheckTally,
    TallyEntry,
    build_tally,
    matchup_of,
)
from tekken_coach.framedata.xref import label_interaction, label_interactions

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
    "StringGapInfo",
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
    # xref (C2)
    "label_interaction",
    "label_interactions",
    # rubric (C2)
    "DEFAULT_RUBRIC",
    "RubricPattern",
    "evaluate_triggers",
    "thresholds",
    # punishers (C2)
    "FALLBACK_STANDING_STARTUP",
    "Punisher",
    "PunisherProfile",
    "PunisherProfiles",
    "PunisherStance",
    "load_punisher_profiles",
    # tally (C2)
    "KnowledgeCheckTally",
    "TallyEntry",
    "build_tally",
    "matchup_of",
]
