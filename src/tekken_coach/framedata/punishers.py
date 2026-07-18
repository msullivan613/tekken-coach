"""Curated per-character punisher profiles (C2 gap #2, docs/05 §4.1).

Punishability (``was_punishable``, ``correct_punish``, ``punish_window``) is judged against the
**defender's** fastest relevant punisher, but that data lives nowhere in the frame-data snapshot —
the snapshot is per-*move* frame data, not per-*character* "what can you punish with." So we
introduce a small curated asset: for each scoped defender character, the ordered list of
block-punish options with startup, notation, stance/context, and whether it launches (docs/05 §4.1).

These are frame-number facts (no license issue); specific recommendations are cross-checked against
okizeme.gg and wavu.wiki (docs/05 §3.1).

Miss-tolerant (docs/05 §6): a character with no profile is **not** fatal — the xref falls back to a
coarse ``on_block <= -10`` standing default and leaves ``correct_punish`` null (docs/05 §4.1, and
the fallback in :mod:`tekken_coach.framedata.xref`).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_PUNISHERS_DIR = Path("assets/punishers")

# The coarse standing block-punish threshold used when a character has no curated profile
# (docs/05 §4.1 gap #2): assume a generic i10 jab punish is available standing.
FALLBACK_STANDING_STARTUP = 10


class PunisherStance(StrEnum):
    """Which defensive stance a punish is launched from.

    A blocked **high or mid** is punished ``standing``; a blocked **low** is punished from crouch
    with a ``while_standing`` (ws) move (docs/05 §4.1: "standing block-punish; while-standing/crouch
    for lows").
    """

    standing = "standing"
    while_standing = "while_standing"


class Punisher(BaseModel):
    """One block-punish option for a character (docs/05 §4.1 gap #2)."""

    startup: int  # i-frames of the punish (its own startup)
    notation: str  # the input to feed back to the user, e.g. "f,F+2 (i15)"
    stance: PunisherStance = PunisherStance.standing
    launcher: bool = False  # does it launch (preferred as the "strongest" punish, docs/05 §4.1)
    damage: int | None = None  # raw damage, for ranking non-launcher options
    notes: str | None = None


class PunisherProfile(BaseModel):
    """A single character's curated punisher list, e.g. ``assets/punishers/kazuya.json``.

    ``char_id`` mirrors the move map: it may be ``None`` until sourced from the reader (C4); the
    profile is still addressable by ``char_name``. ``punishers`` is kept ordered by startup for
    readability, but the selection logic does not rely on the order.
    """

    char_id: int | None = None
    char_name: str
    punishers: list[Punisher] = Field(default_factory=list)

    def by_stance(self, stance: PunisherStance) -> list[Punisher]:
        """Return this character's punishers usable from ``stance``."""
        return [p for p in self.punishers if p.stance == stance]

    def fastest(self, stance: PunisherStance) -> Punisher | None:
        """Return the fastest punisher for ``stance``, or ``None`` if the profile has none."""
        candidates = self.by_stance(stance)
        return min(candidates, key=lambda p: p.startup) if candidates else None


class PunisherProfiles(BaseModel):
    """A collection of curated punisher profiles, keyed by ``char_name`` (docs/05 §4.1 gap #2)."""

    profiles: dict[str, PunisherProfile] = Field(default_factory=dict)

    def get(self, char_name: str | None) -> PunisherProfile | None:
        """Return the profile for ``char_name``, or ``None`` on a miss (never raises).

        Case-insensitive: profiles are keyed lowercase on load, so a capitalized ``"Paul"`` from a
        move map matches a lowercase snapshot name and vice versa (brief #17 §B latent-trap guard).
        """
        if char_name is None:
            return None
        return self.profiles.get(char_name.lower())


def load_punisher_profiles(punishers_dir: str | Path = DEFAULT_PUNISHERS_DIR) -> PunisherProfiles:
    """Load every ``*.json`` punisher profile from ``punishers_dir``, keyed by ``char_name``.

    Directory-driven (no index file): each ``<char>.json`` is one :class:`PunisherProfile`. A
    missing directory yields an empty collection rather than raising, so the xref simply falls
    back to the coarse default for every character (docs/05 §6 degrade-don't-crash).
    """
    root = Path(punishers_dir)
    profiles: dict[str, PunisherProfile] = {}
    if not root.is_dir():
        return PunisherProfiles(profiles=profiles)
    for path in sorted(root.glob("*.json")):
        profile = PunisherProfile.model_validate_json(path.read_text(encoding="utf-8"))
        # Normalize the key on load so lookup is case-insensitive (brief #17 §B); :meth:`get`
        # lowercases the query to match.
        profiles[profile.char_name.lower()] = profile
    return PunisherProfiles(profiles=profiles)
