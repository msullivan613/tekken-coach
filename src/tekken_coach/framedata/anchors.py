"""Regression anchors: hand-trusted ``move_id -> framedata_key`` bindings (brief #8, Layer 3).

The fingerprint join (#6) can only *maximize* confidence — a plausible-but-wrong live confirm or a
post-patch id shift can still slip a bad binding into a built movemap. Anchors are the cheap guard:
for a handful of signature moves per character we commit the id -> key we *know* is right (proven,
or hand-confirmed after a live pass), and a test (:mod:`tests.test_movemap_anchors`) fires the
moment a build maps an anchored id to a different key.

The anchor set is deliberately **partial** and **conservative** (docs/05 §2.3 miss-tolerance carries
over): only ids a human has verified belong here, and the guard is written so an *absent* id is
skipped, never failed — an empty or partial movemap must not trip it. The file is keyed by the
frame-data **slug** (``kazuya``), matching the snapshot and the ``CharFrameData`` slug (the
movemap's ``char_name`` is normalized to its slug here). Seeded with the one proven anchor
(``kazuya 2145 -> df+2``); ``bryan: {}`` marks where the user adds signatures after the live pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import RootModel

from tekken_coach.framedata.models import CharFrameData, CharMoveMap, FrameDataSnapshot

DEFAULT_ANCHORS_PATH = Path("assets/movemap/anchors.json")


class Anchors(RootModel[dict[str, dict[str, str]]]):
    """The committed anchor set: ``char_slug -> {move_id (str) -> framedata_key}`` (Layer 3).

    JSON object keys are strings, so ``move_id`` is stored as a string exactly like the movemap
    (:class:`~tekken_coach.framedata.models.CharMoveMap`); callers look up with either type via
    :meth:`for_char`.
    """

    def for_char(self, char_slug: str) -> dict[int, str]:
        """Return ``{move_id -> framedata_key}`` for a character (empty on an unanchored slug)."""
        return {int(move_id): key for move_id, key in self.root.get(char_slug, {}).items()}

    def slugs(self) -> list[str]:
        """The anchored character slugs, sorted (for deterministic reporting)."""
        return sorted(self.root)


def load_anchors(path: str | Path = DEFAULT_ANCHORS_PATH) -> Anchors:
    """Load ``assets/movemap/anchors.json`` (brief #8 Layer 3)."""
    return Anchors.model_validate_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Conformance check (shared by the anchor test, --report, and --audit)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorCheck:
    """The conformance of one anchor against a built movemap + the framedata snapshot (Layer 3).

    Two independent checks per anchor, each a fail the guard test asserts on:

    * **map conformance** — only when the anchored ``move_id`` is *present* in that character's
      committed movemap: does the map bind it to the anchored key? ``map_conforms`` is ``None`` when
      the id is absent (skipped — a partial map must not fail), ``True``/``False`` otherwise. A
      ``False`` is the id-space-shift alarm the whole layer exists for.
    * **key existence** — is the anchored ``framedata_key`` actually in the current snapshot?
      ``key_in_framedata`` catches a notation typo or a stale anchor after a framedata refresh.
    """

    char_slug: str
    move_id: int
    anchor_key: str
    present_in_map: bool  # the anchored move_id is present in the committed movemap
    mapped_key: str | None  # what the movemap binds that id to, if present
    map_conforms: bool | None  # None when absent (skipped); else mapped_key == anchor_key
    key_in_framedata: bool  # the anchored key exists in the snapshot

    @property
    def ok(self) -> bool:
        """True when nothing about this anchor is wrong (absent id counts as ok — skipped)."""
        return self.key_in_framedata and self.map_conforms is not False


def check_anchors(
    anchors: Anchors,
    move_maps: dict[str, CharMoveMap],
    snapshot: FrameDataSnapshot,
) -> list[AnchorCheck]:
    """Check every anchor against the built movemaps + the current framedata (brief #8 Layer 3).

    ``move_maps`` is keyed by ``char_name`` (as :func:`loader.load_move_maps` returns it); it is
    matched to the slug-keyed anchors by ``char_name.lower()``. Returns one :class:`AnchorCheck` per
    anchored ``(slug, move_id)``, in ``(slug, move_id)`` order — deterministic for tests/reports.
    """
    by_slug = {m.char_name.lower(): m for m in move_maps.values()}
    checks: list[AnchorCheck] = []
    for slug in anchors.slugs():
        move_map = by_slug.get(slug)
        char_fd: CharFrameData | None = snapshot.get_char(slug)
        for move_id, anchor_key in sorted(anchors.for_char(slug).items()):
            entry = move_map.get(move_id) if move_map is not None else None
            present = entry is not None
            mapped_key = entry.framedata_key if entry is not None else None
            checks.append(
                AnchorCheck(
                    char_slug=slug,
                    move_id=move_id,
                    anchor_key=anchor_key,
                    present_in_map=present,
                    mapped_key=mapped_key,
                    map_conforms=(mapped_key == anchor_key) if present else None,
                    key_in_framedata=char_fd is not None and anchor_key in char_fd.moves,
                )
            )
    return checks
