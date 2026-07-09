"""Assemble, validate, and persist a candidate offset table from a derivation (docs/02 §3/§4).

The derivation (:mod:`.derive`) locates the confident core — the two struct anchors, the stride,
and the anchor fields the doctor validates. This module turns that into a **complete, schema-valid**
:class:`~tekken_coach.reader.offsets.OffsetTable` by overlaying the derived pieces on a **seed
table** (the previous known-good build) and then writing it to ``assets/offsets/<version>.json`` and
registering the version in ``index.json``.

Why seed from the previous table: one Jin-vs-Kazuya setup cannot prove every field (state-code
maps, the many boolean flags, heat/rage/input, the global phase/mode/round/timer offsets). Across a
*minor* patch these usually don't move, so carrying them forward from the last table and letting the
user calibrate the deltas is both honest and practical (docs/02 §4 — patch handling is a data
operation). The resulting file is a **candidate**: the derived fields are trustworthy (the doctor
gates exactly them), and everything seeded is flagged for verification in the diagnostic report.

The key is the **detected exe version** (:func:`~tekken_coach.reader.version.detect_running_version`
returns e.g. ``5.02.01`` — distinct from the balance-patch version the frame data uses); the table
records that as its ``game_version`` and the index binds it to the written file.

Pure: builds and writes files, never touches process memory (docs/02 §2).
"""

from __future__ import annotations

import json
from pathlib import Path

from tekken_coach.reader.discovery.derive import DerivationResult
from tekken_coach.reader.faults import OffsetTableError
from tekken_coach.reader.offsets import (
    FieldSpec,
    GlobalStruct,
    OffsetIndex,
    OffsetIndexEntry,
    OffsetTable,
    PlayerStruct,
    load_offset_index,
)


def build_offset_table(
    result: DerivationResult,
    seed: OffsetTable,
    *,
    game_version: str,
    discovered_at: str,
    notes: str,
) -> OffsetTable:
    """Overlay a derivation onto a seed table, producing a schema-valid candidate table.

    Derived anchors/stride/field offsets win; every other field, the state-code maps, and the
    sanity bounds are carried from ``seed``. Raises :class:`OffsetTableError` if the derivation is
    missing the confident core (no anchors/stride) — we never emit a table that omits the fields the
    doctor must validate.
    """
    if result.player_anchor is None or result.global_anchor is None or result.stride is None:
        raise OffsetTableError(
            "derivation did not resolve the confident core (player/global anchors + stride); "
            f"unresolved: {result.unresolved or 'anchors'}. Cannot build a candidate table — "
            "widen the scan windows in the probe manifest and re-run (see runbook)."
        )

    player_fields: dict[str, FieldSpec] = dict(seed.players.fields)
    for df in result.player_offsets().values():
        player_fields[df.name] = FieldSpec(offset=df.offset, kind=df.kind)
    # Fields the derivation *supersedes* are removed, not overwritten: the C4a placeholder's
    # per-flag booleans describe a struct that does not exist, and in-struct pos_{x,y,z} are wrong
    # once position is known to live in a component. Carrying them would read as working offsets.
    for name in result.drop_player_fields:
        player_fields.pop(name, None)
    # max_health (when set) makes the decoder compute health = max_health - damage_taken instead of
    # reading a direct HP field (docs/02 §3 — T8's struct has no HP field, only damage_taken).
    players = PlayerStruct(
        anchor=result.player_anchor,
        stride=result.stride,
        fields=player_fields,
        max_health=result.max_health if result.max_health is not None else seed.players.max_health,
        components=result.components or dict(seed.players.components),
    )

    global_fields: dict[str, FieldSpec] = dict(seed.global_struct.fields)
    for df in result.global_offsets().values():
        global_fields[df.name] = FieldSpec(offset=df.offset, kind=df.kind)
    global_struct = GlobalStruct(anchor=result.global_anchor, fields=global_fields)

    # The encoded-state value -> meaning map is data the *scan* never derives (docs/02 §8); it is
    # loaded from its own file and copied in, switching the decoder onto the encoded-state path.
    state_codes = seed.state_codes
    if result.encoded_state is not None:
        missing = sorted(set(result.encoded_state.flags) - set(player_fields))
        if missing:
            raise OffsetTableError(
                f"the state map names encoded field(s) {missing} that the derived player struct "
                "does not carry. Add them to base_scan.state_fields in the probe manifest (they "
                "are DATA), or drop them from the state map — the decoder would raise every frame."
            )
        state_codes = state_codes.model_copy(update={"encoded_state": result.encoded_state})

    return OffsetTable(
        game_version=game_version,
        discovered_at=discovered_at,
        notes=notes,
        global_struct=global_struct,  # populated by field name (alias "global" in JSON)
        players=players,
        state_codes=state_codes,
        sanity=seed.sanity,
    )


def write_offset_table(offsets_dir: str | Path, table: OffsetTable) -> Path:
    """Write ``table`` to ``<offsets_dir>/<game_version>.json`` (pretty JSON) and return the path.

    Uses the ``global`` alias so the file matches the checked-in table shape and re-loads through
    :func:`~tekken_coach.reader.offsets.load_offset_table`.
    """
    out_dir = Path(offsets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{table.game_version}.json"
    path.write_text(table.model_dump_json(indent=2, by_alias=True) + "\n", encoding="utf-8")
    return path


def register_version(
    offsets_dir: str | Path, game_version: str, *, file_name: str, set_detected: bool = True
) -> OffsetIndex:
    """Add/update the ``game_version -> file`` binding in ``index.json`` and return the new index.

    Creates ``index.json`` if absent. By default also updates ``detected_version`` to the new build
    (the marker recording which build the assets are aligned to, docs/02 §3). Preserves any existing
    entries — this is additive; other versions' tables remain selectable.
    """
    out_dir = Path(offsets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    if index_path.exists():
        index = load_offset_index(out_dir)
    else:
        index = OffsetIndex(detected_version=game_version, versions={})
    index.versions[game_version] = OffsetIndexEntry(file=file_name)
    if set_detected:
        index.detected_version = game_version
    index_path.write_text(json.dumps(index.model_dump(), indent=2) + "\n", encoding="utf-8")
    return index
