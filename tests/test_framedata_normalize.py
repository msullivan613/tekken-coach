"""CSV normalization tests (docs/05 §3.2/§3.3): the pure parse/normalize core, offline.

Exercised against the recorded CSV fixtures in ``tests/fixtures/framedata/`` (a faithful,
minimal slice of the real ``pbruvoll/tekkendocs`` data) plus inline dicts for edge cells.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.framedata.csv_normalize import (
    CsvFormatError,
    normalize_char_csvs,
    normalize_row,
    parse_csv,
    parse_frames,
    parse_hit_level,
)
from tekken_coach.schemas import MoveProperty

FIXTURES = Path(__file__).parent / "fixtures" / "framedata"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --- scalar parsers -----------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("i15~16", 15),  # range + 'i' prefix -> lower bound
        ("i14", 14),
        ("+0c", 0),  # crouching annotation
        ("-11a", -11),  # airborne annotation
        ("-13~-8", -13),  # signed range -> first value
        ("+32a (+24)", 32),  # trailing parenthetical
        ("-9", -9),
        ("", None),
        ("FDFA", None),  # non-numeric stance code
    ],
)
def test_parse_frames(cell: str, expected: int | None) -> None:
    assert parse_frames(cell) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("h", MoveProperty.high),
        ("m", MoveProperty.mid),
        ("l", MoveProperty.low),
        ("L", MoveProperty.low),  # case-insensitive
        ("M", MoveProperty.mid),
        ("m!", MoveProperty.mid),  # power/break marker stripped
        ("h!", MoveProperty.high),
        ("t", MoveProperty.throw),
        ("th", MoveProperty.throw),
        ("th(h)", MoveProperty.throw),  # parenthetical suffix stripped
        ("sm", MoveProperty.mid),  # special mid
        ("sl", MoveProperty.low),
        ("sp", None),  # unmappable (stance/special) -> None, raw preserved by caller
        ("", None),
    ],
)
def test_parse_hit_level(token: str, expected: MoveProperty | None) -> None:
    assert parse_hit_level(token) == expected


# --- single moves -------------------------------------------------------------


def test_single_move_scalar_fields() -> None:
    rows = list(parse_csv(_read("kazuya-mini.csv")))
    df2 = normalize_row(next(r for r in rows if r["Command"] == "df+2"))
    assert df2 is not None
    assert df2.is_string is False
    assert df2.hits == []
    assert df2.hit_level is MoveProperty.mid
    assert df2.hit_level_raw == "m"
    assert df2.startup == 14
    assert df2.on_block == -12
    assert df2.on_hit == "+5"
    assert df2.on_ch == "+59a"  # launch marker kept as a string (docs/05 §3.2)
    assert df2.damage == 22
    assert df2.properties == ["hom"]
    assert df2.name == "Abolishing Fist"
    assert df2.wavu_id == "Kazuya-df+2"
    # Multi-line Notes must survive parsing (StringIO, not splitlines).
    assert df2.notes is not None
    assert "\n" in df2.notes
    assert df2.notes.startswith("* Homing")


def test_command_with_commas_is_not_a_string() -> None:
    """EWGF f,n,d,df+2 has commas in its command but a single hit level -> not a string."""
    rows = list(parse_csv(_read("kazuya-mini.csv")))
    ewgf = normalize_row(next(r for r in rows if r["Command"] == "f,n,d,df+2"))
    assert ewgf is not None
    assert ewgf.is_string is False
    assert ewgf.hits == []
    assert ewgf.hit_level is MoveProperty.high


def test_unmappable_hit_level_preserves_raw() -> None:
    rows = list(parse_csv(_read("paul-mini.csv")))
    stance = normalize_row(next(r for r in rows if r["Command"] == "f+3+4"))
    assert stance is not None
    assert stance.hit_level is None
    assert stance.hit_level_raw == "sp"


# --- strings: the per-hit hits[] sequence (docs/05 §3.2, the duck-the-high contract) ----------


def test_string_hits_sequence_with_per_hit_levels() -> None:
    """Paul df+1,1,2 -> mid, high, mid, with per-hit startup and the blank-cell miss."""
    rows = list(parse_csv(_read("paul-mini.csv")))
    move = normalize_row(next(r for r in rows if r["Command"] == "df+1,1,2"))
    assert move is not None
    assert move.is_string is True
    assert [h.hit_level for h in move.hits] == [
        MoveProperty.mid,
        MoveProperty.high,
        MoveProperty.mid,
    ]
    # per-hit startup: middle cell is blank in "i14, ,i22~23" -> None; last is a range.
    assert [h.startup for h in move.hits] == [14, None, 22]
    assert move.hits[2].startup_raw == "i22~23"
    assert [h.damage for h in move.hits] == [11, 9, 22]
    # string-level fields still populated
    assert move.on_block == -9
    assert move.startup == 14
    # duck_punish is NOT set by ingest — curated later (docs/05 §3.3)
    assert move.duck_punish is None
    assert move.heat is None


def test_multi_hit_string_from_kazuya() -> None:
    rows = list(parse_csv(_read("kazuya-mini.csv")))
    move = normalize_row(next(r for r in rows if r["Command"] == "1,1,2"))
    assert move is not None
    assert move.is_string is True
    assert [h.hit_level_raw for h in move.hits] == ["h", "h", "m"]
    assert move.hits[1].startup is None  # blank middle startup cell


# --- edge cells via inline dicts ---------------------------------------------


def test_block_range_and_annotations_via_row() -> None:
    move = normalize_row(
        {
            "Command": "test",
            "Hit level": "m",
            "Damage": "20",
            "Start up frame": "i13",
            "Block frame": "-13~-8",
            "Hit frame": "+0c",
            "Counter hit frame": "",
            "Notes": "",
            "Tags": "hom, bbr",
            "Name": "Test",
            "Recovery": "20",
            "Wavu id": "X-test",
            "Character id": "x",
        }
    )
    assert move is not None
    assert move.on_block == -13  # range -> first value
    assert move.block_raw == "-13~-8"
    assert move.on_hit == "+0c"  # raw kept, annotation intact
    assert move.properties == ["hom", "bbr"]
    assert move.recovery == 20


def test_blank_command_row_is_skipped() -> None:
    assert normalize_row({"Command": "  "}) is None


# --- parse_csv guardrails -----------------------------------------------------


def test_parse_csv_rejects_missing_columns() -> None:
    with pytest.raises(CsvFormatError):
        list(parse_csv("Command;Hit level\ndf+2;m\n"))


def test_parse_csv_rejects_empty() -> None:
    with pytest.raises(CsvFormatError):
        list(parse_csv(""))


# --- char-level merge ---------------------------------------------------------


def test_normalize_char_csvs_takes_slug_from_column() -> None:
    char = normalize_char_csvs("Paul", [_read("paul-mini.csv")])
    assert char.char_slug == "paul"
    assert char.char_name == "Paul"
    assert set(char.moves) == {"df+1,1,2", "df+2", "f+3+4"}
    assert char.get("df+1,1,2") is not None
    assert char.get("nonexistent") is None  # miss-tolerant, no raise
