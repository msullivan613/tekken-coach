"""``fetch-framedata`` ingest: pinned-commit CSV -> normalized dated snapshot (docs/05 §3.3).

The pipeline (docs/05 §3.3):

1. Fetch the scoped characters' ``data/wavuConvertedCsv/<char>/*.csv`` from
   ``pbruvoll/tekkendocs`` **at a pinned commit SHA** (raw file fetch — no scraping, no
   rate-limit dance). Files are discovered via the GitHub trees API at the SHA, or an explicit
   file list is accepted.
2. Parse and normalize into the docs/05 §3.2 schema (:mod:`tekken_coach.framedata.csv_normalize`).
3. Diff against ``current`` — surfacing balance-patch deltas for review.
4. Write a new ``snapshot-<date>/`` whose ``manifest.json`` records the pinned SHA + attribution
   + per-character checksums, and **repoint ``current`` only on approval**.

The source is treated as a pinned, reproducible commit — **not** a live API contract (docs/05
§3.1): URLs are verified at ingest time, never hard-coded as a stable interface elsewhere.

Ingestion is manual-triggered. C1 exposes :func:`fetch_framedata` and :func:`promote_snapshot`
as callables; the CLI registration (``tekken-coach fetch-framedata``) is wired in C6.

Network etiquette (docs/05 §3.1): a descriptive ``User-Agent`` is set and requests are serial.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tekken_coach.framedata.csv_normalize import normalize_char_csvs
from tekken_coach.framedata.loader import (
    DEFAULT_FRAMEDATA_DIR,
    load_snapshot,
    resolve_current_snapshot,
)
from tekken_coach.framedata.models import (
    CharFrameData,
    FrameDataMove,
    SnapshotCharEntry,
    SnapshotManifest,
)

DEFAULT_REPO = "pbruvoll/tekkendocs"
DEFAULT_CSV_ROOT = "data/wavuConvertedCsv"
RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"
TREES_URL = "https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"
USER_AGENT = "tekken-coach fetch-framedata (+https://github.com/; frame-data ingest, serial)"
ATTRIBUTION = ["tekkendocs.com", "rbnorway.org"]


# ---------------------------------------------------------------------------
# HTTP seam (injectable for offline tests — docs/05 §3.3 "mock the HTTP")
# ---------------------------------------------------------------------------


class Fetcher(Protocol):
    """Minimal HTTP seam so ingest can be unit-tested with a recorded CSV (no network)."""

    def get_text(self, url: str) -> str: ...


class UrllibFetcher:
    """Default :class:`Fetcher` over ``urllib`` with a descriptive User-Agent (docs/05 §3.1)."""

    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 30.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout

    def get_text(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            data: bytes = response.read()
        return data.decode("utf-8")


# ---------------------------------------------------------------------------
# Inputs / results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CharSpec:
    """A character to ingest: its display name and its repo directory slug."""

    char_name: str  # e.g. "Kazuya"
    slug: str  # e.g. "kazuya" (the data/wavuConvertedCsv/<slug>/ directory)


@dataclass
class CharDiff:
    """Per-character diff of a new snapshot vs ``current`` (docs/05 §3.3)."""

    slug: str
    added: list[str] = field(default_factory=list)  # framedata_keys only in the new snapshot
    removed: list[str] = field(default_factory=list)  # keys only in current
    changed: list[str] = field(default_factory=list)  # keys whose move record differs

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.changed)


@dataclass
class IngestResult:
    """Outcome of :func:`fetch_framedata`: where it wrote, its manifest, and the diff."""

    snapshot_dir: Path
    snapshot_name: str
    manifest: SnapshotManifest
    diff: list[CharDiff]
    repointed: bool  # whether ``current`` was moved to this snapshot

    @property
    def has_changes(self) -> bool:
        return any(not d.is_empty for d in self.diff)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_char_csvs(
    slug: str,
    *,
    repo: str,
    sha: str,
    fetcher: Fetcher,
    csv_root: str = DEFAULT_CSV_ROOT,
) -> list[str]:
    """Discover a character's CSV paths via the GitHub trees API at the pinned SHA (docs/05 §3.3).

    Returns repo-relative paths, sorted. A character may have several CSV files under its
    directory; all are returned.
    """
    tree_json = fetcher.get_text(TREES_URL.format(repo=repo, sha=sha))
    tree = json.loads(tree_json)
    prefix = f"{csv_root}/{slug}/"
    paths = [
        node["path"]
        for node in tree.get("tree", [])
        if node.get("path", "").startswith(prefix) and node["path"].endswith(".csv")
    ]
    return sorted(paths)


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def _dump_move(move: FrameDataMove) -> str:
    """Canonical JSON of a move for equality-diffing (ignores dict ordering)."""
    return json.dumps(move.model_dump(mode="json"), sort_keys=True)


def diff_char(new: CharFrameData, current: CharFrameData | None) -> CharDiff:
    """Diff a character's new frame data vs its current version (docs/05 §3.3)."""
    diff = CharDiff(slug=new.char_slug)
    if current is None:
        diff.added = sorted(new.moves)
        return diff
    new_keys, cur_keys = set(new.moves), set(current.moves)
    diff.added = sorted(new_keys - cur_keys)
    diff.removed = sorted(cur_keys - new_keys)
    diff.changed = sorted(
        k for k in (new_keys & cur_keys) if _dump_move(new.moves[k]) != _dump_move(current.moves[k])
    )
    return diff


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def _checksum(texts: list[str]) -> str:
    """Return ``sha256:<hex>`` of the concatenated source CSV bytes (docs/05 §3.2)."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


def _load_current_chars(framedata_dir: Path) -> dict[str, CharFrameData]:
    """Load the current snapshot's per-char data for diffing, or empty if there is none."""
    try:
        current_dir = resolve_current_snapshot(framedata_dir)
    except FileNotFoundError:
        return {}
    return load_snapshot(current_dir).characters


def fetch_framedata(
    characters: list[CharSpec],
    *,
    sha: str,
    repo: str = DEFAULT_REPO,
    fetcher: Fetcher | None = None,
    dest_root: str | Path = DEFAULT_FRAMEDATA_DIR,
    csv_root: str = DEFAULT_CSV_ROOT,
    snapshot_date: str | None = None,
    game_version: str | None = None,
    file_lists: dict[str, list[str]] | None = None,
    repoint: bool = False,
) -> IngestResult:
    """Fetch, normalize, diff, and write a dated frame-data snapshot (docs/05 §3.3).

    Writes an **immutable** ``snapshot-<date>/`` under ``dest_root``. ``current`` is repointed
    to it **only** when ``repoint=True`` (the approval gate, docs/05 §3.3) — the default writes
    the snapshot and reports the diff without adopting it.

    ``sha`` pins the source commit (the reproducible snapshot key, docs/05 §3.1). CSV files are
    discovered via the trees API unless an explicit ``file_lists`` (slug -> repo-relative paths)
    is supplied.
    """
    fetcher = fetcher if fetcher is not None else UrllibFetcher()
    dest_root = Path(dest_root)
    snapshot_date = snapshot_date or datetime.now(UTC).strftime("%Y-%m-%d")
    snapshot_name = f"snapshot-{snapshot_date}"
    snapshot_dir = dest_root / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    current_chars = _load_current_chars(dest_root)

    manifest_chars: dict[str, SnapshotCharEntry] = {}
    diffs: list[CharDiff] = []

    for spec in characters:
        if file_lists is not None and spec.slug in file_lists:
            csv_paths = sorted(file_lists[spec.slug])
        else:
            csv_paths = discover_char_csvs(
                spec.slug, repo=repo, sha=sha, fetcher=fetcher, csv_root=csv_root
            )
        if not csv_paths:
            raise FileNotFoundError(f"no CSV files found for {spec.slug!r} at {repo}@{sha}")

        # Serial fetch (docs/05 §3.1 etiquette).
        texts = [fetcher.get_text(RAW_URL.format(repo=repo, sha=sha, path=p)) for p in csv_paths]
        char_data = normalize_char_csvs(spec.char_name, texts)

        char_file = f"{char_data.char_slug}.json"
        (snapshot_dir / char_file).write_text(
            char_data.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

        manifest_chars[char_data.char_slug] = SnapshotCharEntry(
            file=char_file,
            csv_files=[Path(p).name for p in csv_paths],
            checksum=_checksum(texts),
            move_count=len(char_data.moves),
        )
        diffs.append(diff_char(char_data, current_chars.get(char_data.char_slug)))

    manifest = SnapshotManifest(
        source_repo=repo,
        source_commit=sha,
        source_path_template=RAW_URL.format(
            repo=repo, sha=sha, path=f"{csv_root}/<char>/<file>.csv"
        ),
        fetched_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        snapshot_date=snapshot_date,
        game_version=game_version,
        attribution=list(ATTRIBUTION),
        characters=manifest_chars,
    )
    (snapshot_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    repointed = False
    if repoint:
        promote_snapshot(dest_root, snapshot_name)
        repointed = True

    return IngestResult(
        snapshot_dir=snapshot_dir,
        snapshot_name=snapshot_name,
        manifest=manifest,
        diff=diffs,
        repointed=repointed,
    )


def promote_snapshot(
    framedata_dir: str | Path,
    snapshot_name: str,
) -> None:
    """Repoint ``current`` at ``snapshot_name`` — the approval step (docs/05 §3.3).

    Prefers a symlink (``current -> snapshot-<date>/``, docs/05 §3.2); if the platform rejects
    symlinks, falls back to a plain text pointer file naming the snapshot (the loader reads both).
    Snapshots are immutable; only this pointer moves.
    """
    root = Path(framedata_dir)
    target = root / snapshot_name
    if not target.is_dir():
        raise FileNotFoundError(f"cannot promote missing snapshot {snapshot_name!r} in {root}")
    pointer = root / "current"
    if pointer.is_symlink() or pointer.exists():
        if pointer.is_dir() and not pointer.is_symlink():
            raise FileExistsError(
                f"{pointer} is a real directory, not a pointer; refusing to remove"
            )
        pointer.unlink()
    try:
        pointer.symlink_to(snapshot_name)
    except (OSError, NotImplementedError):
        pointer.write_text(snapshot_name + "\n", encoding="utf-8")
