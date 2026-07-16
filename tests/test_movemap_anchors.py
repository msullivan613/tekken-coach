"""Regression-anchor guard tests (brief #8 Layer 3).

Two jobs. First, the **live guard**: the committed anchors must conform to the committed movemaps
and the current framedata snapshot — so the day a build maps an anchored id to the wrong key (an
id-space shift after a patch), this test goes red. Second, the **guard-bites proof**: on synthetic
inputs, a wrong binding fails, a missing framedata key fails the existence check, and an *absent*
anchored id is skipped (a partial/empty movemap must never trip the guard).
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.framedata.anchors import Anchors, check_anchors, load_anchors
from tekken_coach.framedata.loader import load_current_framedata, load_move_maps
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    FrameDataSnapshot,
    MoveMapEntry,
    SnapshotManifest,
)

REPO_ROOT = Path(__file__).parent.parent
ASSETS = REPO_ROOT / "assets"


# --- synthetic builders -------------------------------------------------------


def _snapshot(slug: str, *keys: str) -> FrameDataSnapshot:
    """A minimal snapshot with one character carrying the given framedata keys (on_block -12)."""
    moves = {k: FrameDataMove(key=k, on_block=-12, startup=14) for k in keys}
    char = CharFrameData(char_slug=slug, char_name=slug.title(), moves=moves)
    manifest = SnapshotManifest(
        source_repo="pbruvoll/tekkendocs",
        source_commit="deadbeef",
        source_path_template="{slug}.csv",
        fetched_at="2026-07-15T00:00:00Z",
        snapshot_date="2026-07-15",
    )
    return FrameDataSnapshot(manifest=manifest, characters={slug: char})


def _movemap(char_name: str, moves: dict[int, str]) -> dict[str, CharMoveMap]:
    """A movemap dict keyed by char_name (as loader.load_move_maps returns it)."""
    entries = {
        str(mid): MoveMapEntry(notation=key, framedata_key=key) for mid, key in moves.items()
    }
    return {
        char_name: CharMoveMap(char_name=char_name, game_version="2.01.01", moves=entries),
    }


def _anchors(mapping: dict[str, dict[str, str]]) -> Anchors:
    return Anchors.model_validate(mapping)


# --- the live guard over committed assets -------------------------------------


def test_committed_anchors_conform_to_movemaps_and_framedata() -> None:
    """Every committed anchor: present ids map to the anchored key, and the key exists in framedata.

    This is the regression alarm — it fires the moment a rebuilt movemap binds an anchored id to a
    different key, or an anchored key falls out of the snapshot.
    """
    anchors = load_anchors(ASSETS / "movemap" / "anchors.json")
    move_maps = load_move_maps(ASSETS / "movemap")
    snapshot = load_current_framedata(ASSETS / "framedata")

    checks = check_anchors(anchors, move_maps, snapshot)
    assert checks  # the seed anchor is present
    for check in checks:
        assert check.key_in_framedata, (
            f"{check.char_slug} {check.anchor_key} missing from framedata"
        )
        assert check.map_conforms is not False, (
            f"{check.char_slug} {check.move_id} maps to {check.mapped_key!r}, "
            f"anchor says {check.anchor_key!r}"
        )


def test_seed_anchor_is_the_proven_kazuya_binding() -> None:
    """The committed anchors carry the one proven anchor, kazuya 2145 -> df+2 (brief #8 seed)."""
    anchors = load_anchors(ASSETS / "movemap" / "anchors.json")
    assert anchors.for_char("kazuya") == {2145: "df+2"}


# --- guard-bites proofs on synthetic inputs -----------------------------------


def test_correct_binding_conforms() -> None:
    anchors = _anchors({"kazuya": {"2145": "df+2"}})
    checks = check_anchors(anchors, _movemap("Kazuya", {2145: "df+2"}), _snapshot("kazuya", "df+2"))
    assert len(checks) == 1
    assert checks[0].map_conforms is True
    assert checks[0].ok


def test_wrong_binding_fails_the_guard() -> None:
    """A movemap that binds the anchored id to a *different* key fails — the guard bites."""
    anchors = _anchors({"kazuya": {"2145": "df+2"}})
    checks = check_anchors(
        anchors, _movemap("Kazuya", {2145: "hFC.4"}), _snapshot("kazuya", "df+2", "hFC.4")
    )
    assert checks[0].map_conforms is False
    assert not checks[0].ok


def test_absent_anchored_id_is_skipped_not_failed() -> None:
    """A movemap that has *not* mapped the anchored id yet is skipped (partial map stays green)."""
    anchors = _anchors({"kazuya": {"2145": "df+2"}})
    checks = check_anchors(anchors, _movemap("Kazuya", {}), _snapshot("kazuya", "df+2"))
    assert checks[0].present_in_map is False
    assert checks[0].map_conforms is None
    assert checks[0].ok  # absent id does not fail the guard


def test_anchor_key_missing_from_framedata_fails_existence() -> None:
    """An anchor whose key is not in the snapshot fails the existence check (typo/stale)."""
    anchors = _anchors({"kazuya": {"2145": "df+9999"}})
    checks = check_anchors(
        anchors, _movemap("Kazuya", {2145: "df+9999"}), _snapshot("kazuya", "df+2")
    )
    assert checks[0].key_in_framedata is False
    assert not checks[0].ok


def test_empty_char_anchor_block_yields_no_checks() -> None:
    """The ``bryan: {}`` placeholder (a place to add signatures) produces no checks, no fail."""
    anchors = _anchors({"kazuya": {"2145": "df+2"}, "bryan": {}})
    checks = check_anchors(anchors, _movemap("Kazuya", {2145: "df+2"}), _snapshot("kazuya", "df+2"))
    assert [c.char_slug for c in checks] == ["kazuya"]
