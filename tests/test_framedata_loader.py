"""Loader tests (docs/05 §2.3/§3.2/§4.1): typed loading + the miss-tolerant resolve path.

Split between the committed real sample assets (proves the shipped files validate and that the
duck-the-high string is present) and tmp-built trees for the pointer-form and miss edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.framedata.loader import (
    load_char_move_map,
    load_current_framedata,
    load_move_maps,
    load_snapshot,
    resolve_current_snapshot,
    resolve_move,
)
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    MoveMapEntry,
)
from tekken_coach.schemas import MoveProperty

REPO_ROOT = Path(__file__).parent.parent
ASSETS = REPO_ROOT / "assets"


# --- the committed real sample (Paul + Kazuya) --------------------------------


def test_current_snapshot_loads_and_validates() -> None:
    snap = load_current_framedata(ASSETS / "framedata")
    assert snap.manifest.source_repo == "pbruvoll/tekkendocs"
    assert snap.manifest.source_commit  # a pinned SHA is recorded
    assert "tekkendocs.com" in snap.manifest.attribution
    assert set(snap.characters) == {"paul", "kazuya"}
    # per-char manifest checksums are recorded
    assert snap.manifest.characters["paul"].checksum.startswith("sha256:")


def test_sample_has_multi_hit_string_with_per_hit_levels() -> None:
    """Acceptance: the sample proves per-hit hits[] against real data (Paul df+1,1,2)."""
    snap = load_current_framedata(ASSETS / "framedata")
    paul = snap.get_char("paul")
    assert paul is not None
    string = paul.get("df+1,1,2")
    assert string is not None
    assert string.is_string is True
    assert [h.hit_level for h in string.hits] == [
        MoveProperty.mid,
        MoveProperty.high,
        MoveProperty.mid,
    ]
    # curated duck-the-high answer (docs/05 §3.2 example)
    assert string.duck_punish is not None
    assert string.duck_punish.after_hit == 2
    assert string.duck_punish.answer == "df+1 (i13)"


def test_committed_move_map_resolves_known_id() -> None:
    """A known move_id (Kazuya 2145 -> df+2) resolves with correct fields (acceptance)."""
    maps = load_move_maps(ASSETS / "movemap")
    snap = load_current_framedata(ASSETS / "framedata")
    kazuya_map = maps["Kazuya"]
    kazuya_fd = snap.get_char("kazuya")

    lookup = resolve_move(2145, kazuya_map, kazuya_fd)
    assert lookup.matched is True
    assert lookup.notation == "df+2"
    assert lookup.framedata_key == "df+2"
    assert lookup.move is not None
    assert lookup.move.on_block == -12
    assert lookup.move.hit_level is MoveProperty.mid
    assert lookup.char_name == "Kazuya"


def test_committed_move_map_is_marked_partial() -> None:
    kazuya = load_char_move_map(ASSETS / "movemap" / "kazuya.json")
    paul = load_char_move_map(ASSETS / "movemap" / "paul.json")
    assert kazuya.partial is True
    assert paul.partial is True
    # Paul's move_id table is intentionally empty (no id is spec-sourced), but the derivable
    # framedata_key seed is complete.
    assert paul.moves == {}
    assert "df+1,1,2" in paul.framedata_keys
    assert paul.char_id is None  # not yet sourced (comes from the reader, C4)


# --- miss-tolerance (docs/05 §2.3/§4.1): degrade, never raise ------------------


def _kazuya_map() -> CharMoveMap:
    return CharMoveMap(
        char_id=12,
        char_name="Kazuya",
        game_version="2.01.01",
        partial=True,
        moves={"2145": MoveMapEntry(notation="df+2", framedata_key="df+2")},
    )


def _kazuya_fd() -> CharFrameData:
    return CharFrameData(
        char_slug="kazuya",
        char_name="Kazuya",
        moves={"df+2": FrameDataMove(key="df+2", on_block=-12, hit_level=MoveProperty.mid)},
    )


def test_unknown_move_id_degrades_not_raises() -> None:
    lookup = resolve_move(999999, _kazuya_map(), _kazuya_fd())
    assert lookup.matched is False
    assert lookup.notation == "move_id:999999"  # docs/05 §2.3 fallback
    assert lookup.framedata_key is None
    assert lookup.move is None


def test_known_id_but_missing_framedata_key_degrades() -> None:
    bad_map = CharMoveMap(
        char_id=12,
        char_name="Kazuya",
        game_version="2.01.01",
        moves={"7000": MoveMapEntry(notation="qcf+9", framedata_key="qcf+9")},
    )
    lookup = resolve_move(7000, bad_map, _kazuya_fd())
    assert lookup.matched is False
    assert lookup.notation == "qcf+9"  # move-map name still resolved
    assert lookup.framedata_key == "qcf+9"
    assert lookup.move is None


def test_no_move_map_for_character_degrades() -> None:
    lookup = resolve_move(2145, None, _kazuya_fd())
    assert lookup.matched is False
    assert lookup.notation == "move_id:2145"
    assert lookup.char_name is None


def test_no_framedata_for_character_degrades() -> None:
    lookup = resolve_move(2145, _kazuya_map(), None)
    assert lookup.matched is False
    assert lookup.framedata_key == "df+2"
    assert lookup.move is None


# --- current-pointer resolution: symlink and plain-file forms (docs/05 §3.2) ---


def _write_min_snapshot(snap_dir: Path) -> None:
    snap_dir.mkdir(parents=True)
    (snap_dir / "manifest.json").write_text(
        '{"source_repo":"r","source_commit":"abc","source_path_template":"t",'
        '"fetched_at":"2026-07-07T00:00:00Z","snapshot_date":"2026-07-07","characters":{}}',
        encoding="utf-8",
    )


def test_resolve_current_via_file_pointer(tmp_path: Path) -> None:
    _write_min_snapshot(tmp_path / "snapshot-2026-07-07")
    (tmp_path / "current").write_text("snapshot-2026-07-07\n", encoding="utf-8")
    resolved = resolve_current_snapshot(tmp_path)
    assert resolved.name == "snapshot-2026-07-07"
    # and it loads
    assert load_snapshot(resolved).manifest.source_commit == "abc"


def test_resolve_current_via_symlink(tmp_path: Path) -> None:
    _write_min_snapshot(tmp_path / "snapshot-2026-07-07")
    (tmp_path / "current").symlink_to("snapshot-2026-07-07")
    resolved = resolve_current_snapshot(tmp_path)
    assert resolved.name == "snapshot-2026-07-07"


def test_missing_current_pointer_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_current_snapshot(tmp_path)
