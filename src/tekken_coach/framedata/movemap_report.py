"""``map-moves --report`` — the eyeball aid over a built movemap (brief #8 Layer 5).

Pure, offline, no game. Loads the committed movemap(s) + the current framedata snapshot + the anchor
set and prints, per mapped ``move_id``, the joined line

    move_id -> notation -> (startup, on_block, hit_level, name)

sorted by ``move_id``, each carrying a **confidence tag**. A character expert scans this in seconds
and spots a wrong binding without touching the game.

The confidence tag composes the signals available *today* (no new calibration):

* **broken** — the entry's ``framedata_key`` resolves to no move in the snapshot (a dangling key: a
  notation typo or a stale entry after a framedata refresh). The loudest flag — the mapping is
  unusable as-is.
* **anchor** — the id is in the hand-trusted anchor set (:mod:`tekken_coach.framedata.anchors`) and
  the map agrees with it. Highest confidence. ``anchor!`` (conflict) if the map disagrees with the
  anchor — that is the id-shift alarm, and it should never appear while the anchor test is green.
* **unique** — exactly one framedata move sits at that ``on_block`` (within the join's ±1), so the
  fingerprint join could isolate it. Strong.
* **tie-broken** — two or more framedata moves share that ``on_block``; the binding was resolved by
  something other than on-block alone (a live startup read, or curation). Check startup/name.

When a session log is passed (``--from-log`` alongside ``--report``) each entry also shows how many
blocked samples backed it in that log — a mapping seen blocked 40 times is worth more than one seen
once.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from tekken_coach.framedata.anchors import AnchorCheck, Anchors, check_anchors
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    FrameDataSnapshot,
)
from tekken_coach.framedata.movemap_build import DEFAULT_BLOCK_TOL, build_fingerprint
from tekken_coach.framedata.movemap_miner import resolve_char_ids
from tekken_coach.schemas import Interaction
from tekken_coach.session.store import LoadedSession

# Confidence tags, loudest → strongest (report/sort order).
CONFIDENCE_BROKEN = "broken"
CONFIDENCE_ANCHOR_CONFLICT = "anchor!"
CONFIDENCE_ANCHOR = "anchor"
CONFIDENCE_UNIQUE = "unique"
CONFIDENCE_TIE = "tie-broken"


@dataclass(frozen=True)
class ReportEntry:
    """One mapped ``move_id`` joined to its framedata, with a confidence tag (brief #8 Layer 5)."""

    char_slug: str
    move_id: int
    notation: str
    framedata_key: str
    move: FrameDataMove | None  # None => broken (dangling framedata_key)
    is_anchor: bool
    anchor_conflict: bool  # anchored, but the map binds a *different* key (the id-shift alarm)
    on_block_unique: (
        bool | None
    )  # None when broken or on_block absent; else exactly-one-at-on_block
    rivals: int  # other framedata moves within ±tol of this move's on_block (0 => unique)
    blocked_samples: int | None  # blocked observations backing it in the log, when a log is passed
    confidence: str

    @property
    def startup(self) -> int | None:
        return self.move.startup if self.move is not None else None

    @property
    def on_block(self) -> int | None:
        return self.move.on_block if self.move is not None else None

    @property
    def hit_level(self) -> str | None:
        if self.move is None or self.move.hit_level is None:
            return None
        return self.move.hit_level.value

    @property
    def name(self) -> str | None:
        return self.move.name if self.move is not None else None


@dataclass(frozen=True)
class CharReport:
    """One character's report section: its entries plus this character's anchor checks."""

    char_slug: str
    char_name: str
    entries: list[ReportEntry]
    anchor_checks: list[AnchorCheck]


@dataclass(frozen=True)
class MovemapReport:
    """The full ``--report`` result (pure — nothing is written)."""

    chars: list[CharReport]
    anchor_checks: list[AnchorCheck]  # every anchor, across all characters

    @property
    def broken(self) -> list[ReportEntry]:
        """Every broken (dangling-key) entry, across characters — the must-fix list."""
        return [e for c in self.chars for e in c.entries if e.confidence == CONFIDENCE_BROKEN]


def _sample_counts(session: LoadedSession | None) -> dict[tuple[str, int], int]:
    """Blocked-sample count per ``(char_slug, move_id)`` from a log, or empty when no log.

    Uses the header-driven ``char_id -> name`` resolution (brief #6 §A.3), so a build whose reader
    could not name a character still attributes its samples correctly.
    """
    if session is None:
        return {}
    names = resolve_char_ids(session.header, session.interactions)
    grouped: dict[tuple[int | None, int], list[Interaction]] = defaultdict(list)
    for interaction in session.interactions:
        grouped[(interaction.attacker_char_id, interaction.attacker_move_id)].append(interaction)
    counts: dict[tuple[str, int], int] = {}
    for (char_id, move_id), items in grouped.items():
        if char_id is None or char_id not in names:
            continue
        slug = names[char_id].lower()
        fp = build_fingerprint(char_id, move_id, items)
        counts[(slug, move_id)] = fp.blocked_samples
    return counts


def _rivals_at_on_block(move: FrameDataMove, char_fd: CharFrameData, *, block_tol: int) -> int:
    """Count framedata moves *other than* ``move`` whose on_block is within ±tol of ``move``'s.

    Mirrors the join's on-block collision test (brief #6 §A.1): 0 rivals means the fingerprint join
    could have isolated this move on on-block alone (``unique``); ≥1 means it needed another signal.
    """
    if move.on_block is None:
        return 0
    return sum(
        1
        for other in char_fd.moves.values()
        if other.key != move.key
        and other.on_block is not None
        and abs(other.on_block - move.on_block) <= block_tol
    )


def _confidence(
    *, broken: bool, is_anchor: bool, anchor_conflict: bool, unique: bool | None
) -> str:
    """Pick the single confidence tag (loudest signal wins)."""
    if broken:
        return CONFIDENCE_BROKEN
    if anchor_conflict:
        return CONFIDENCE_ANCHOR_CONFLICT
    if is_anchor:
        return CONFIDENCE_ANCHOR
    if unique:
        return CONFIDENCE_UNIQUE
    return CONFIDENCE_TIE


def build_report(
    move_maps: dict[str, CharMoveMap],
    snapshot: FrameDataSnapshot,
    anchors: Anchors,
    *,
    session: LoadedSession | None = None,
    only_char: str | None = None,
    block_tol: int = DEFAULT_BLOCK_TOL,
) -> MovemapReport:
    """Join every mapped ``move_id`` to its framedata and tag its confidence (brief #8 Layer 5).

    ``move_maps`` is keyed by ``char_name`` (as :func:`loader.load_move_maps` returns it). A
    ``session``
    (optional) supplies blocked-sample counts. ``only_char`` (case-insensitive, matched on slug)
    restricts the report to one character. Pure — reads only what it is handed.
    """
    all_checks = check_anchors(anchors, move_maps, snapshot)
    checks_by_slug: dict[str, list[AnchorCheck]] = defaultdict(list)
    for check in all_checks:
        checks_by_slug[check.char_slug].append(check)

    samples = _sample_counts(session)
    want = only_char.lower() if only_char else None

    chars: list[CharReport] = []
    for move_map in sorted(move_maps.values(), key=lambda m: m.char_name.lower()):
        slug = move_map.char_name.lower()
        if want is not None and slug != want:
            continue
        char_fd = snapshot.get_char(slug)
        anchor_keys = anchors.for_char(slug)

        entries: list[ReportEntry] = []
        for move_id_str, mapped in sorted(move_map.moves.items(), key=lambda kv: int(kv[0])):
            move_id = int(move_id_str)
            move = char_fd.get(mapped.framedata_key) if char_fd is not None else None
            broken = move is None
            is_anchor = move_id in anchor_keys
            anchor_conflict = is_anchor and anchor_keys[move_id] != mapped.framedata_key
            rivals = (
                _rivals_at_on_block(move, char_fd, block_tol=block_tol)
                if move is not None and char_fd is not None
                else 0
            )
            unique = None if move is None or move.on_block is None else rivals == 0
            entries.append(
                ReportEntry(
                    char_slug=slug,
                    move_id=move_id,
                    notation=mapped.notation,
                    framedata_key=mapped.framedata_key,
                    move=move,
                    is_anchor=is_anchor,
                    anchor_conflict=anchor_conflict,
                    on_block_unique=unique,
                    rivals=rivals,
                    blocked_samples=samples.get((slug, move_id)),
                    confidence=_confidence(
                        broken=broken,
                        is_anchor=is_anchor,
                        anchor_conflict=anchor_conflict,
                        unique=unique,
                    ),
                )
            )
        chars.append(
            CharReport(
                char_slug=slug,
                char_name=move_map.char_name,
                entries=entries,
                anchor_checks=checks_by_slug.get(slug, []),
            )
        )

    kept_checks = [c for c in all_checks if want is None or c.char_slug == want]
    return MovemapReport(chars=chars, anchor_checks=kept_checks)


# ---------------------------------------------------------------------------
# Human-readable rendering (pure; the CLI just prints these lines)
# ---------------------------------------------------------------------------


def _fmt_int(value: int | None, *, signed: bool = False) -> str:
    """Format an optional int, ``?`` when unknown; ``+``-signed for advantage values."""
    if value is None:
        return "?"
    return f"{value:+d}" if signed else str(value)


def format_report(report: MovemapReport) -> list[str]:
    """Render a :class:`MovemapReport` as printable lines (brief #8 Layer 5 summary)."""
    lines: list[str] = []
    total = sum(len(c.entries) for c in report.chars)
    broken = report.broken
    conflicts = [c for c in report.anchor_checks if c.map_conforms is False]
    stale_anchors = [c for c in report.anchor_checks if not c.key_in_framedata]

    lines.append(
        f"map-moves --report: {total} mapped move-id(s) across {len(report.chars)} character(s) — "
        f"{len(broken)} broken, {len(conflicts)} anchor-conflict, {len(stale_anchors)} stale-anchor"
    )

    for char in report.chars:
        lines.append("")
        lines.append(f"[{char.char_slug}] {char.char_name} — {len(char.entries)} mapped")
        lines.extend(_format_anchor_lines(char.anchor_checks))
        if not char.entries:
            lines.append("  (no move_ids mapped yet)")
            continue
        for entry in char.entries:
            lines.append(_format_entry(entry))

    return lines


def _format_anchor_lines(checks: list[AnchorCheck]) -> list[str]:
    """One summary line per anchor for a character (conformance surfaced, brief #8 Layer 3)."""
    lines: list[str] = []
    for check in checks:
        if not check.key_in_framedata:
            state = "STALE — key missing from framedata"
        elif check.map_conforms is False:
            state = f"CONFLICT — map has {check.mapped_key!r}, anchor says {check.anchor_key!r}"
        elif check.map_conforms is True:
            state = "ok (map agrees)"
        else:
            state = "not yet mapped (skipped)"
        lines.append(f"  anchor {check.move_id} -> {check.anchor_key}: {state}")
    return lines


def _format_entry(entry: ReportEntry) -> str:
    """One ``move_id -> notation -> (startup, on_block, hit_level, name)`` line with its tag."""
    joined = (
        f"(i{_fmt_int(entry.startup)}, {_fmt_int(entry.on_block, signed=True)}, "
        f"{entry.hit_level or '?'}, {entry.name or '?'})"
    )
    tag = entry.confidence
    if entry.confidence == CONFIDENCE_TIE and entry.rivals:
        tag = f"{tag} (+{entry.rivals} at same on_block)"
    if entry.blocked_samples is not None:
        tag = f"{tag}, {entry.blocked_samples} blocked-sample(s)"
    return f"  {entry.move_id} -> {entry.notation} -> {joined}  [{tag}]"
