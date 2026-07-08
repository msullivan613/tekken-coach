"""Offset table format, typed loader, and version detection / fail-closed (docs/02 §3, §7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tekken_coach.reader.faults import OffsetTableError, UnknownGameVersionError
from tekken_coach.reader.offsets import (
    OffsetTable,
    load_offset_index,
    load_offset_table,
    select_offset_table,
)

REPO_OFFSETS = Path("assets/offsets")


def test_checked_in_index_loads() -> None:
    index = load_offset_index(REPO_OFFSETS)
    assert index.detected_version == "2.01.01"
    assert "2.01.01" in index.versions


def test_checked_in_table_aligns_with_c1_game_version() -> None:
    # The offset version must line up with the C1 movemap snapshot (game_version 2.01.01).
    table = select_offset_table("2.01.01", REPO_OFFSETS)
    assert table.game_version == "2.01.01"
    # The layout the decoder relies on is present.
    assert "frame_counter" in table.global_struct.fields
    assert {"char_id", "move_id", "health", "pos_x", "counter_state"} <= set(table.players.fields)
    assert table.players.stride > 0
    assert table.sanity.round_start_health > 0


def test_unknown_version_fails_closed_with_runbook() -> None:
    with pytest.raises(UnknownGameVersionError) as exc_info:
        select_offset_table("9.99.99", REPO_OFFSETS)
    err = exc_info.value
    assert err.version == "9.99.99"
    assert "2.01.01" in err.available  # never silently swaps in a known table
    # The §4 re-discovery runbook rides along so the caller can present it.
    assert "practice mode" in err.runbook.lower()
    assert "update-offsets" in err.runbook


def test_select_never_falls_back_to_a_stale_table(tmp_path: Path) -> None:
    # An index that knows only one version must refuse any other version outright — no
    # "closest match" / "latest known" fallback (a wrong offset yields garbage, docs/02 §3).
    (tmp_path / "index.json").write_text(
        json.dumps(
            {"detected_version": "2.01.01", "versions": {"2.01.01": {"file": "2.01.01.json"}}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "2.01.01.json").write_text(
        (REPO_OFFSETS / "2.01.01.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    # The known version resolves.
    assert select_offset_table("2.01.01", tmp_path).game_version == "2.01.01"
    # A newer/unknown version does not fall through to it.
    with pytest.raises(UnknownGameVersionError):
        select_offset_table("2.02.00", tmp_path)


def test_missing_index_raises_offset_table_error(tmp_path: Path) -> None:
    with pytest.raises(OffsetTableError):
        load_offset_index(tmp_path)


def test_malformed_table_raises_offset_table_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"game_version": "x"}', encoding="utf-8")  # missing required sections
    with pytest.raises(OffsetTableError):
        load_offset_table(bad)


def test_table_version_mismatch_is_rejected(tmp_path: Path) -> None:
    # index says 2.01.01 -> file, but the file declares a different version: reject, don't trust.
    table = json.loads((REPO_OFFSETS / "2.01.01.json").read_text(encoding="utf-8"))
    table["game_version"] = "1.23.45"
    (tmp_path / "index.json").write_text(
        json.dumps({"detected_version": "2.01.01", "versions": {"2.01.01": {"file": "t.json"}}}),
        encoding="utf-8",
    )
    (tmp_path / "t.json").write_text(json.dumps(table), encoding="utf-8")
    with pytest.raises(OffsetTableError):
        select_offset_table("2.01.01", tmp_path)


def test_offset_table_ignores_comment_key() -> None:
    # The `_comment` posture note in the JSON is not a model field; it must be ignored, not fatal.
    raw = json.loads((REPO_OFFSETS / "2.01.01.json").read_text(encoding="utf-8"))
    assert "_comment" in raw
    table = OffsetTable.model_validate(raw)
    assert table.game_version == "2.01.01"
