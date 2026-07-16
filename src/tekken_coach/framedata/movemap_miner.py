"""Passive movemap miner + merge (the C6 ``map-moves --from-log`` path, brief #6 §A.2-4).

Reads a recorded session log, groups its interactions by ``(attacker_char_id, attacker_move_id)``,
forms a consensus fingerprint per group, runs the pure join (:mod:`movemap_build`) against the
character's Wavu snapshot, and **merges** the unambiguous results into ``assets/movemap/``.
Everything here is pure and offline apart from the final file write — the miner returns a plan
(:class:`MineReport`) that the CLI renders, and the merge is idempotent and resumable.

**Character resolution (brief #6 §A.3), the honest version.** The brief assumed each interaction's
``attacker_char_name`` is resolved, but the live-run-1 log proves it is not — names read as
``"char_id:7"`` on that build (the memory char-name map was still uncalibrated). The reliable source
is the **session header**: it records the user's character and, per match, the opponent's, plus
which player index is the user. Cross that with each interaction's ``attacker`` index and
``attacker_char_id`` and we get an authoritative ``memory char_id -> name`` map without trusting the
(then-buggy) resolved-name field. The resolved ``attacker_char_name`` is used only as a fallback.

A character whose Wavu snapshot is absent (e.g. Bryan/Xiaoyu today) is reported as
``needs_framedata`` — "run ``fetch-framedata`` first" — never a crash (brief #6 §A.3).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tekken_coach.framedata.loader import (
    DEFAULT_MOVEMAP_DIR,
    load_char_move_map,
)
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataSnapshot,
    MoveMapEntry,
)
from tekken_coach.framedata.movemap_build import (
    DEFAULT_BLOCK_TOL,
    DEFAULT_STARTUP_TOL,
    JoinResult,
    MoveFingerprint,
    build_fingerprint,
    entry_for,
    join_move,
)
from tekken_coach.schemas import Interaction, LabeledInteraction, SessionHeader
from tekken_coach.session.store import LoadedSession

# A resolved char name that is really just a numeric placeholder ("char_id:7", "char:5") — i.e. the
# reader could not name that memory id on this build. Treated as unresolved for slug purposes.
_PLACEHOLDER_NAME = re.compile(r"^char(?:_id)?:\d+$")


# ---------------------------------------------------------------------------
# Report shapes
# ---------------------------------------------------------------------------

# Per-group status: the join statuses (auto_mapped/collision/no_candidate/no_signal) plus the two
# miner adds before the join can run: the character has no snapshot, or could not be named at all.
GroupStatus = str  # one of the JoinStatus values | "needs_framedata" | "unresolved_char"


@dataclass(frozen=True)
class GroupOutcome:
    """The mining outcome for one ``(char_id, move_id)`` group (brief #6 §A.2)."""

    char_id: int | None
    char_name: str | None  # display name resolved from the header, e.g. "paul"
    char_slug: str | None  # framedata slug (== name.lower()) when a snapshot exists
    move_id: int
    fingerprint: MoveFingerprint
    join: JoinResult | None  # None when the group never reached the join (unresolved / no snapshot)
    status: GroupStatus


@dataclass(frozen=True)
class MineReport:
    """The full mining plan over a session (brief #6 §A.2). Pure — no file has been written yet."""

    groups: list[GroupOutcome]
    game_version: str

    def with_status(self, status: GroupStatus) -> list[GroupOutcome]:
        """All groups with the given status, in report order."""
        return [g for g in self.groups if g.status == status]

    @property
    def auto_mapped(self) -> list[GroupOutcome]:
        return self.with_status("auto_mapped")


# ---------------------------------------------------------------------------
# Character resolution (header-driven, brief #6 §A.3)
# ---------------------------------------------------------------------------


def _is_placeholder(name: str | None) -> bool:
    """True when ``name`` is missing or a ``char_id:<n>`` numeric placeholder (unnamed id)."""
    return name is None or bool(_PLACEHOLDER_NAME.match(name))


def resolve_char_ids(
    header: SessionHeader, interactions: list[LabeledInteraction]
) -> dict[int, str]:
    """Map each memory ``char_id`` to a character name, header-first (brief #6 §A.3).

    The header pins, per match, the user's character (at ``user_player``) and the opponent's (the
    other slot). Crossed with each interaction's player index and ``char_id`` that yields an
    authoritative ``char_id -> name`` map. The interaction's own ``attacker_char_name`` /
    ``defender_char_name`` is used only when the header leaves a slot unnamed (a placeholder).
    """
    per_match: dict[str, dict[int, str]] = {}
    for match in header.matches:
        per_match[match.match_id] = {
            header.user_player: header.user_char,
            1 - header.user_player: match.opponent_char,
        }

    resolved: dict[int, str] = {}
    for interaction in interactions:
        slots = per_match.get(interaction.match_id, {})
        for index, char_id, fallback in (
            (interaction.attacker, interaction.attacker_char_id, interaction.attacker_char_name),
            (interaction.defender, interaction.defender_char_id, interaction.defender_char_name),
        ):
            if char_id is None or char_id in resolved:
                continue
            name = slots.get(index)
            if _is_placeholder(name):
                name = None if _is_placeholder(fallback) else fallback
            if name is not None:
                resolved[char_id] = name
    return resolved


# ---------------------------------------------------------------------------
# The miner
# ---------------------------------------------------------------------------


def mine_session(
    session: LoadedSession,
    snapshot: FrameDataSnapshot,
    *,
    only_char: str | None = None,
    block_tol: int = DEFAULT_BLOCK_TOL,
    startup_tol: int = DEFAULT_STARTUP_TOL,
) -> MineReport:
    """Mine a loaded session into a movemap plan (brief #6 §A.2). Pure; writes nothing.

    Groups interactions by ``(attacker_char_id, attacker_move_id)``, forms a consensus fingerprint
    per group, and joins it against the character's snapshot. ``only_char`` (case-insensitive)
    restricts mining to a single character (its groups are still reported, including
    ``needs_framedata`` when its snapshot is absent).
    """
    names = resolve_char_ids(session.header, session.interactions)
    want = only_char.lower() if only_char else None

    grouped: dict[tuple[int | None, int], list[Interaction]] = defaultdict(list)
    for interaction in session.interactions:
        grouped[(interaction.attacker_char_id, interaction.attacker_move_id)].append(interaction)

    outcomes: list[GroupOutcome] = []
    for (char_id, move_id), items in sorted(grouped.items(), key=_group_sort_key):
        name = names.get(char_id) if char_id is not None else None
        slug = name.lower() if name is not None else None
        if want is not None and slug != want:
            continue

        char_fd: CharFrameData | None = snapshot.get_char(slug) if slug is not None else None
        fingerprint = _fingerprint(char_id, move_id, items)

        join: JoinResult | None
        if char_id is None or slug is None:
            join, status = None, "unresolved_char"
        elif char_fd is None:
            join, status = None, "needs_framedata"
        else:
            join = join_move(fingerprint, char_fd, block_tol=block_tol, startup_tol=startup_tol)
            status = join.status

        outcomes.append(
            GroupOutcome(
                char_id=char_id,
                char_name=name,
                char_slug=slug,
                move_id=move_id,
                fingerprint=fingerprint,
                join=join,
                status=status,
            )
        )

    game_version = snapshot.manifest.game_version or session.header.game_version
    return MineReport(groups=outcomes, game_version=game_version)


def _fingerprint(char_id: int | None, move_id: int, items: list[Interaction]) -> MoveFingerprint:
    """Consensus fingerprint for a group; ``char_id`` may be unknown (kept as -1 sentinel)."""
    return build_fingerprint(char_id if char_id is not None else -1, move_id, items)


def _group_sort_key(item: tuple[tuple[int | None, int], object]) -> tuple[int, int]:
    """Deterministic report order: by char_id (unresolved last), then move_id."""
    (char_id, move_id), _ = item
    return (char_id if char_id is not None else 1_000_000, move_id)


# ---------------------------------------------------------------------------
# Merge / write (idempotent, resumable — brief #6 §A.4)
# ---------------------------------------------------------------------------


@dataclass
class MergeOutcome:
    """What a merge did to one character's movemap file (brief #6 §A.4)."""

    char_slug: str
    path: Path
    created: bool
    written: list[int] = field(default_factory=list)  # move_ids newly mapped
    overwritten: list[int] = field(default_factory=list)  # curated entries replaced (--overwrite)
    preserved: list[int] = field(default_factory=list)  # already mapped, kept (no --overwrite)


def merge_report(
    report: MineReport,
    snapshot: FrameDataSnapshot,
    *,
    movemap_dir: str | Path = DEFAULT_MOVEMAP_DIR,
    overwrite: bool = False,
) -> list[MergeOutcome]:
    """Merge the report's auto-mapped groups into per-character movemap files (brief #6 §A.4).

    Idempotent and resumable: an existing curated entry is preserved unless ``overwrite`` is set,
    re-running never duplicates, and a file is (re)written only when the character has at least one
    auto-mapped group. Preserves ``partial`` and refreshes ``framedata_keys`` from the snapshot.
    """
    by_slug: dict[str, list[GroupOutcome]] = defaultdict(list)
    for group in report.auto_mapped:
        assert group.char_slug is not None  # auto_mapped implies a resolved char with a snapshot
        by_slug[group.char_slug].append(group)

    outcomes: list[MergeOutcome] = []
    for slug in sorted(by_slug):
        char_fd = snapshot.get_char(slug)
        assert char_fd is not None
        groups = by_slug[slug]
        pairs = [(g.move_id, _key_of(g)) for g in groups]
        char_ids = {g.char_id for g in groups if g.char_id is not None}
        char_id = char_ids.pop() if len(char_ids) == 1 else None
        outcomes.append(
            merge_mappings(
                slug,
                char_fd,
                report.game_version,
                pairs,
                char_id=char_id,
                movemap_dir=movemap_dir,
                overwrite=overwrite,
            )
        )
    return outcomes


def _key_of(group: GroupOutcome) -> str:
    """The auto-mapped framedata_key for a group (guarded — only auto_mapped groups reach here)."""
    assert group.join is not None and group.join.framedata_key is not None
    return group.join.framedata_key


def merge_mappings(
    slug: str,
    char_fd: CharFrameData,
    game_version: str,
    mappings: list[tuple[int, str]],
    *,
    char_id: int | None = None,
    movemap_dir: str | Path = DEFAULT_MOVEMAP_DIR,
    overwrite: bool = False,
) -> MergeOutcome:
    """Merge ``move_id -> framedata_key`` pairs into one character's ``<slug>.json`` (§A.4).

    The shared merge core for both the passive miner and the Stage-B live confirm (which merges one
    pair at a time). Idempotent: an existing curated entry is preserved unless ``overwrite``; the
    file is rewritten deterministically (moves by numeric id, ``framedata_keys`` sorted) so a
    re-run with no new mappings is a byte-for-byte no-op. ``partial`` stays true.
    """
    path = Path(movemap_dir) / f"{slug}.json"
    created = not path.exists()
    if created:
        movemap = CharMoveMap(
            char_id=None,
            char_name=char_fd.char_name,
            game_version=game_version,
            partial=True,
        )
    else:
        movemap = load_char_move_map(path)

    result = MergeOutcome(char_slug=slug, path=path, created=created)
    for move_id, framedata_key in sorted(mappings):
        key = str(move_id)
        entry = entry_for(char_fd, framedata_key)
        if key in movemap.moves and not overwrite:
            result.preserved.append(move_id)
            continue
        if key in movemap.moves:
            result.overwritten.append(move_id)
        else:
            result.written.append(move_id)
        movemap.moves[key] = entry

    if movemap.char_id is None and char_id is not None:
        movemap.char_id = char_id
    movemap.framedata_keys = sorted(char_fd.moves)
    movemap.partial = True

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_movemap(movemap), encoding="utf-8")
    return result


def _dump_movemap(movemap: CharMoveMap) -> str:
    """Serialize a movemap deterministically (moves by numeric id, keys sorted) — stable on disk."""
    data = {
        "char_id": movemap.char_id,
        "char_name": movemap.char_name,
        "game_version": movemap.game_version,
        "partial": movemap.partial,
        "moves": {k: _dump_entry(movemap.moves[k]) for k in sorted(movemap.moves, key=int)},
        "framedata_keys": sorted(movemap.framedata_keys),
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _dump_entry(entry: MoveMapEntry) -> dict[str, object]:
    """One move-map entry as a plain dict, in the on-disk field order (docs/05 §2.2)."""
    return {
        "notation": entry.notation,
        "aliases": list(entry.aliases),
        "framedata_key": entry.framedata_key,
    }


# ---------------------------------------------------------------------------
# Human-readable report (pure; the CLI just prints these lines)
# ---------------------------------------------------------------------------


def format_report(report: MineReport, merges: list[MergeOutcome]) -> list[str]:
    """Render a mining report + merge result as printable lines (brief #6 §A.2 summary)."""
    lines: list[str] = []
    auto = report.auto_mapped
    collisions = report.with_status("collision")
    no_candidate = report.with_status("no_candidate")
    no_signal = report.with_status("no_signal")
    needs_fd = report.with_status("needs_framedata")
    unresolved = report.with_status("unresolved_char")

    lines.append(
        f"map-moves: {len(report.groups)} move-id groups — "
        f"{len(auto)} auto-mapped, {len(collisions)} collisions, "
        f"{len(no_candidate)} no-candidate, {len(no_signal)} no-signal, "
        f"{len(needs_fd)} need framedata, {len(unresolved)} unresolved-char"
    )

    if auto:
        lines.append("")
        lines.append("auto-mapped (Wavu-verified, written):")
        for g in auto:
            assert g.join is not None and g.join.framedata_key is not None
            lines.append(
                f"  [{g.char_slug}] {g.move_id} -> {g.join.framedata_key}  ({g.join.reason})"
            )

    if collisions:
        lines.append("")
        lines.append("collisions (need a live startup read to disambiguate — Stage B):")
        for g in collisions:
            assert g.join is not None
            keys = ", ".join(c.framedata_key for c in g.join.candidates[:8])
            more = "" if len(g.join.candidates) <= 8 else f", +{len(g.join.candidates) - 8} more"
            lines.append(
                f"  [{g.char_slug}] {g.move_id} on_block≈{g.fingerprint.on_block:+d}: {keys}{more}"
            )

    if no_candidate:
        lines.append("")
        lines.append("no candidate (observed on_block matches no Wavu move — check the read):")
        for g in no_candidate:
            lines.append(f"  [{g.char_slug}] {g.move_id} on_block≈{g.fingerprint.on_block:+d}")

    if no_signal:
        lines.append("")
        lines.append("no signal (no usable blocked sample to fingerprint on):")
        for g in no_signal:
            lines.append(
                f"  [{g.char_slug or f'char_id:{g.char_id}'}] {g.move_id} "
                f"({g.join.reason if g.join else 'no blocked sample'})"
            )

    if needs_fd:
        chars = sorted({g.char_slug for g in needs_fd if g.char_slug})
        lines.append("")
        lines.append(
            f"needs framedata — run `fetch-framedata {' '.join(chars)}` first "
            f"({len(needs_fd)} move-id groups skipped): " + ", ".join(f"{c}" for c in chars)
        )

    if unresolved:
        ids = sorted({g.char_id for g in unresolved if g.char_id is not None})
        lines.append("")
        lines.append(
            f"unresolved character — memory char_id(s) {ids} not named by the session header "
            f"({len(unresolved)} groups skipped)"
        )

    lines.append("")
    if merges:
        for m in merges:
            verb = "created" if m.created else "updated"
            lines.append(
                f"{verb} {m.path}: +{len(m.written)} new, "
                f"{len(m.overwritten)} overwritten, {len(m.preserved)} preserved"
            )
    else:
        lines.append("no files written (no unambiguous mappings to merge).")
    return lines
