"""Session store tests: .jsonl write/read, round-end flush, schema-version gate (03 §5, §6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tekken_coach.schemas import CaptureMode, SessionHeader
from tekken_coach.session import (
    SCHEMA_VERSION,
    IncompatibleSchemaVersionError,
    SessionWriter,
    check_compatibility,
    iter_interactions,
    load_session,
    read_header,
)
from tests.factories import make_header, make_labeled_interaction


def test_writer_produces_header_then_interactions(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    header = make_header()
    a = make_labeled_interaction()
    b = make_labeled_interaction().model_copy(update={"id": "m3-r2-i018"})

    with SessionWriter(path, header) as w:
        w.append(a)
        w.flush()  # round-end flush
        w.append(b)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    first = json.loads(lines[0])
    assert first["record"] == "session_header"
    assert first["schema_version"] == SCHEMA_VERSION


def test_round_trip_session_lossless(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    header = make_header()
    interactions = [
        make_labeled_interaction(),
        make_labeled_interaction().model_copy(update={"id": "m3-r2-i018"}),
    ]
    with SessionWriter(path, header) as w:
        for i in interactions:
            w.append(i)

    loaded = load_session(path)
    assert loaded.header == header
    assert loaded.interactions == interactions

    # iterator path agrees with the eager loader.
    assert list(iter_interactions(path)) == interactions


def test_round_end_flush_persists_before_close(tmp_path: Path) -> None:
    """A crash after a flush but before close must not lose the flushed round."""
    path = tmp_path / "session.jsonl"
    w = SessionWriter(path, make_header())
    w.append(make_labeled_interaction())
    w.flush()
    # Simulate a crash: never call close(). The flushed line must already be on disk.
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


# --- schema-version compatibility gate (03 §6) --------------------------------


def test_unknown_major_version_rejected() -> None:
    with pytest.raises(IncompatibleSchemaVersionError):
        check_compatibility("2.0.0")


def test_additive_minor_bump_tolerated() -> None:
    # A newer additive minor (and patch) is accepted: same major.
    check_compatibility("1.4.0")
    check_compatibility("1.0.9")


def test_load_rejects_unknown_major_header(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    header = make_header(schema_version="2.0.0")
    with SessionWriter(path, header) as w:
        w.append(make_labeled_interaction())

    with pytest.raises(IncompatibleSchemaVersionError):
        read_header(path)
    with pytest.raises(IncompatibleSchemaVersionError):
        load_session(path)
    with pytest.raises(IncompatibleSchemaVersionError):
        list(iter_interactions(path))


def test_load_tolerates_additive_minor_header_and_extra_fields(tmp_path: Path) -> None:
    """An additive minor bump with unknown extra fields loads cleanly (forward-additive)."""
    path = tmp_path / "additive.jsonl"
    header_obj = make_header(schema_version="1.5.0")
    # Inject an unknown additive field into the header and a body line, as a future
    # minor version would; the current models must ignore them, not reject the log.
    header_raw = json.loads(header_obj.model_dump_json())
    header_raw["future_field"] = "ignore me"

    interaction = make_labeled_interaction()
    body_raw = json.loads(interaction.model_dump_json())
    body_raw["labels"]["future_label"] = 99

    path.write_text(
        json.dumps(header_raw) + "\n" + json.dumps(body_raw) + "\n",
        encoding="utf-8",
    )

    loaded = load_session(path)
    assert loaded.header.schema_version == "1.5.0"
    assert loaded.interactions == [interaction]


def test_malformed_version_rejected() -> None:
    with pytest.raises(IncompatibleSchemaVersionError):
        check_compatibility("not-a-version")


def test_default_header_uses_current_schema_version() -> None:
    header = SessionHeader(
        schema_version=SCHEMA_VERSION,
        created_at="2026-07-07T20:14:03Z",
        capture_mode=CaptureMode.clean,
        game_version="2.01.01",
        framedata_snapshot="2026-06-30",
        user_player=0,
        user_char="Jin",
    )
    check_compatibility(header.schema_version)
    assert header.matches == []
