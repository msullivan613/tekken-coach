"""Ingest tests (docs/05 §3.3): fetch -> normalize -> diff -> dated snapshot -> approval gate.

HTTP is mocked (a :class:`FakeFetcher` serving the recorded CSV fixtures), so these run fully
offline. They assert the load-bearing invariants: an immutable dated snapshot, a manifest that
records the pinned SHA + attribution + per-char checksums, and ``current`` moving **only** on
approval.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tekken_coach.framedata.ingest import (
    RAW_URL,
    TREES_URL,
    CharSpec,
    fetch_framedata,
    promote_snapshot,
)
from tekken_coach.framedata.loader import load_current_framedata, resolve_current_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "framedata"
SHA = "0123456789abcdef0123456789abcdef01234567"
REPO = "pbruvoll/tekkendocs"

PAUL_CSV = "data/wavuConvertedCsv/paul/paul-special.csv"
KAZUYA_CSV = "data/wavuConvertedCsv/kazuya/kazuya-special.csv"


class FakeFetcher:
    """Serves recorded responses by URL — the mock-HTTP seam (docs/05 §3.3)."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def get_text(self, url: str) -> str:
        self.calls.append(url)
        try:
            return self._responses[url]
        except KeyError:  # pragma: no cover - test wiring error
            raise AssertionError(f"unexpected URL fetched: {url}") from None


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _fetcher(paul_csv: str = "paul-mini.csv") -> FakeFetcher:
    tree = {
        "tree": [
            {"path": PAUL_CSV, "type": "blob"},
            {"path": KAZUYA_CSV, "type": "blob"},
            {"path": "data/wavuConvertedCsv/paul/README.md", "type": "blob"},  # non-CSV ignored
        ]
    }
    return FakeFetcher(
        {
            TREES_URL.format(repo=REPO, sha=SHA): json.dumps(tree),
            RAW_URL.format(repo=REPO, sha=SHA, path=PAUL_CSV): _read(paul_csv),
            RAW_URL.format(repo=REPO, sha=SHA, path=KAZUYA_CSV): _read("kazuya-mini.csv"),
        }
    )


CHARS = [CharSpec("Paul", "paul"), CharSpec("Kazuya", "kazuya")]


# --- snapshot writing + manifest ---------------------------------------------


def test_ingest_writes_dated_snapshot_with_manifest(tmp_path: Path) -> None:
    result = fetch_framedata(
        CHARS,
        sha=SHA,
        fetcher=_fetcher(),
        dest_root=tmp_path,
        snapshot_date="2026-07-07",
        game_version="2.01.01",
    )
    snap_dir = tmp_path / "snapshot-2026-07-07"
    assert result.snapshot_dir == snap_dir
    assert (snap_dir / "manifest.json").exists()
    assert (snap_dir / "paul.json").exists()
    assert (snap_dir / "kazuya.json").exists()

    manifest = result.manifest
    assert manifest.source_commit == SHA  # pinned SHA recorded (docs/05 §3.2)
    assert manifest.source_repo == REPO
    assert manifest.attribution == ["tekkendocs.com", "rbnorway.org"]
    assert manifest.game_version == "2.01.01"
    assert set(manifest.characters) == {"paul", "kazuya"}
    assert manifest.characters["paul"].checksum.startswith("sha256:")
    assert manifest.characters["paul"].csv_files == ["paul-special.csv"]
    assert manifest.characters["kazuya"].move_count == 3


def test_ingest_normalizes_hits_and_leaves_duck_punish_null(tmp_path: Path) -> None:
    fetch_framedata(
        CHARS, sha=SHA, fetcher=_fetcher(), dest_root=tmp_path, snapshot_date="2026-07-07"
    )
    paul = json.loads((tmp_path / "snapshot-2026-07-07" / "paul.json").read_text())
    string = paul["moves"]["df+1,1,2"]
    assert [h["hit_level"] for h in string["hits"]] == ["mid", "high", "mid"]
    assert string["duck_punish"] is None  # curated later, not by ingest (docs/05 §3.3)


# --- the approval gate: current moves only on approval (docs/05 §3.3) ----------


def test_current_not_repointed_without_approval(tmp_path: Path) -> None:
    result = fetch_framedata(
        CHARS, sha=SHA, fetcher=_fetcher(), dest_root=tmp_path, snapshot_date="2026-07-07"
    )
    assert result.repointed is False
    with pytest.raises(FileNotFoundError):
        resolve_current_snapshot(tmp_path)  # snapshot exists, but current does not


def test_current_repointed_on_approval(tmp_path: Path) -> None:
    result = fetch_framedata(
        CHARS,
        sha=SHA,
        fetcher=_fetcher(),
        dest_root=tmp_path,
        snapshot_date="2026-07-07",
        repoint=True,
    )
    assert result.repointed is True
    assert resolve_current_snapshot(tmp_path).name == "snapshot-2026-07-07"
    # and the promoted snapshot loads back cleanly through the loader
    snap = load_current_framedata(tmp_path)
    assert snap.get_char("paul") is not None


def test_promote_snapshot_moves_pointer_only(tmp_path: Path) -> None:
    fetch_framedata(
        CHARS, sha=SHA, fetcher=_fetcher(), dest_root=tmp_path, snapshot_date="2026-07-07"
    )
    fetch_framedata(
        CHARS, sha=SHA, fetcher=_fetcher(), dest_root=tmp_path, snapshot_date="2026-07-08"
    )
    promote_snapshot(tmp_path, "snapshot-2026-07-07")
    assert resolve_current_snapshot(tmp_path).name == "snapshot-2026-07-07"
    # both immutable snapshot dirs remain on disk
    assert (tmp_path / "snapshot-2026-07-07").is_dir()
    assert (tmp_path / "snapshot-2026-07-08").is_dir()
    # re-promoting to the newer one just moves the pointer
    promote_snapshot(tmp_path, "snapshot-2026-07-08")
    assert resolve_current_snapshot(tmp_path).name == "snapshot-2026-07-08"


# --- discovery + explicit file lists ------------------------------------------


def test_discovery_finds_csvs_via_trees_api(tmp_path: Path) -> None:
    fetcher = _fetcher()
    result = fetch_framedata(
        CHARS, sha=SHA, fetcher=fetcher, dest_root=tmp_path, snapshot_date="2026-07-07"
    )
    assert TREES_URL.format(repo=REPO, sha=SHA) in fetcher.calls
    assert result.manifest.characters["paul"].csv_files == ["paul-special.csv"]


def test_explicit_file_list_skips_trees_api(tmp_path: Path) -> None:
    fetcher = _fetcher()
    fetch_framedata(
        CHARS,
        sha=SHA,
        fetcher=fetcher,
        dest_root=tmp_path,
        snapshot_date="2026-07-07",
        file_lists={"paul": [PAUL_CSV], "kazuya": [KAZUYA_CSV]},
    )
    assert TREES_URL.format(repo=REPO, sha=SHA) not in fetcher.calls


# --- diffing (docs/05 §3.3) ---------------------------------------------------


def test_first_ingest_diff_is_all_added(tmp_path: Path) -> None:
    result = fetch_framedata(
        CHARS, sha=SHA, fetcher=_fetcher(), dest_root=tmp_path, snapshot_date="2026-07-07"
    )
    paul_diff = next(d for d in result.diff if d.slug == "paul")
    assert set(paul_diff.added) == {"df+1,1,2", "df+2", "f+3+4"}
    assert paul_diff.removed == []
    assert paul_diff.changed == []
    assert result.has_changes is True


def test_diff_detects_changed_added_removed(tmp_path: Path) -> None:
    # First snapshot becomes current.
    fetch_framedata(
        CHARS,
        sha=SHA,
        fetcher=_fetcher(),
        dest_root=tmp_path,
        snapshot_date="2026-07-07",
        repoint=True,
    )
    # Second snapshot from a "patched" Paul CSV: df+2 block changed, f+3+4 removed, b+4 added.
    result = fetch_framedata(
        CHARS,
        sha=SHA,
        fetcher=_fetcher(paul_csv="paul-mini-patched.csv"),
        dest_root=tmp_path,
        snapshot_date="2026-07-08",
    )
    paul_diff = next(d for d in result.diff if d.slug == "paul")
    assert paul_diff.changed == ["df+2"]
    assert paul_diff.added == ["b+4"]
    assert paul_diff.removed == ["f+3+4"]
    # kazuya unchanged between the two ingests
    kaz_diff = next(d for d in result.diff if d.slug == "kazuya")
    assert kaz_diff.is_empty is True


def test_no_csvs_found_raises(tmp_path: Path) -> None:
    empty = FakeFetcher({TREES_URL.format(repo=REPO, sha=SHA): json.dumps({"tree": []})})
    with pytest.raises(FileNotFoundError):
        fetch_framedata(
            [CharSpec("Paul", "paul")],
            sha=SHA,
            fetcher=empty,
            dest_root=tmp_path,
            snapshot_date="2026-07-07",
        )
