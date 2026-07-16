"""Input-reconstruction feasibility: bucket notation by onset-reconstructability (brief #9 Stage 1).

The dual-key idea (brief #9): if we can recover the *notation a player actually input* at a move's
onset — a short ``(dir 1-9, buttons)`` window — we can cross-check the frame-fingerprint movemap
binding. Input says ``df+2`` AND the frames fit Wavu ``df+2`` → the binding is verified twice over.

This module answers the *offline ceiling* question before any live work: given a character's
``framedata_keys`` (the Wavu ``Command`` notation), **which moves can even be reconstructed from a
short onset window, and which need more than onset buttons?** It is pure, deterministic notation
analysis — no game, no memory reads — so the coverage estimate is computable and unit-testable.

Two pieces:

* :func:`classify` buckets one notation into :class:`Bucket` with a :class:`Reason`. *Easy* = a
  ``(dir, buttons)`` group at onset (single move) or a comma-string of such groups (needs a few
  follow-up frames). *Hard* = anything that needs more than onset buttons: a stance/positional
  context (``H.``/``FC.``/``ws``…), a motion (``qcf``/``d,df,f``), a hold/charge (``f,F``/``1+2*``),
  a just-frame (``:``), or a ``~`` cancel.
* :func:`onset_of` extracts the reconstruction primitive for an easy move — the numpad ``dir`` and
  button set of its first group — which is exactly what a live reconstructor would match against a
  decoded :class:`~tekken_coach.schemas.InputState` onset window, intersected with the char's keys.

:func:`coverage` rolls a set of keys into a :class:`Coverage` summary (the report's headline count).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

# Lowercase numpad directions, as Wavu writes them, -> numpad 1-9 (5 = neutral). Uppercase variants
# (``F``/``D``/``B``…) denote a *held* direction (a dash/charge) and are deliberately NOT here — a
# hold is not onset-reconstructable, so an uppercase direction token routes a move to Hard.
_DIR_TO_NUMPAD: dict[str, int] = {
    "n": 5,
    "u": 8,
    "d": 2,
    "f": 6,
    "b": 4,
    "df": 3,
    "db": 1,
    "uf": 9,
    "ub": 7,
}
_BUTTONS: frozenset[str] = frozenset({"1", "2", "3", "4"})
# Motion prefixes that are *not* a single direction press: a quarter/half circle. Written inline in
# a group (``qcf+2``) rather than as a comma-sequence, so they need their own detection.
_MOTIONS: frozenset[str] = frozenset({"qcf", "qcb", "hcf", "hcb"})


class Bucket(StrEnum):
    """Whether a move's notation is reconstructable from a short onset window."""

    easy_single = "easy_single"  # one (dir, buttons) group — reconstructable from the onset frame
    easy_string = (
        "easy_string"  # comma-string of (dir, buttons) groups — onset + a few follow frames
    )
    hard = "hard"  # needs more than onset buttons (motion / stance / hold / just-frame / cancel)


class Reason(StrEnum):
    """Why a move landed in its bucket — the sub-reason that drives the coverage breakdown."""

    single = "single"  # a lone dir+buttons press
    string = "string"  # a sequence of dir+buttons presses
    stance = "stance"  # requires a stance/positional context (H./FC./ws/BT/…) the onset can't show
    motion = "motion"  # a motion input (qcf/hcf) or a directional-only sequence (d,df,f / f,n,…)
    hold = "hold"  # a held/charged direction (uppercase F/D/…) or a held button (* / **)
    just_frame = "just_frame"  # a just-frame link (`:`)
    cancel = "cancel"  # a `~` cancel/transition
    throw = "throw"  # a named/positional throw with no button notation
    other = "other"  # unparseable under this grammar (should be rare — flagged for review)


@dataclass(frozen=True)
class Classification:
    """One move's bucket + reason + (for easy moves) its reconstructed onset."""

    key: str
    bucket: Bucket
    reason: Reason
    onset: (
        tuple[int, tuple[str, ...]] | None
    )  # (numpad dir, sorted buttons) for easy moves, else None

    @property
    def easy(self) -> bool:
        return self.bucket is not Bucket.hard


def _parse_group(group: str) -> tuple[str, tuple[int, tuple[str, ...]] | None]:
    """Parse one comma-group into ``(kind, onset)``.

    ``kind`` is ``"easy"`` (with an ``(dir, buttons)`` onset), or a Hard reason name
    (``"motion"``/``"hold"``/``"other"``). A group is a ``+``-joined token list: an optional leading
    direction token followed by one or more attack buttons (``df+1+2`` → dir ``df`` + buttons 1,2;
    ``2`` → button 2 only). Direction-only groups (``d``, ``n``) are motion fragments, not attacks.
    """
    parts = group.split("+")
    dir_tok: str | None = None
    if parts and parts[0] not in _BUTTONS:
        dir_tok, parts = parts[0], parts[1:]
    if dir_tok is not None:
        if dir_tok in _MOTIONS:
            return "motion", None
        if dir_tok != dir_tok.lower():  # any uppercase letter -> a held direction (dash/charge)
            return "hold", None
        if dir_tok not in _DIR_TO_NUMPAD:
            return "other", None
    if not parts:  # direction-only group: a dash/crouch-dash/neutral fragment, not a button press
        return "motion", None
    if any(btn not in _BUTTONS for btn in parts):
        return "other", None
    numpad = _DIR_TO_NUMPAD[dir_tok] if dir_tok is not None else 5
    return "easy", (numpad, tuple(sorted(parts)))


def classify(key: str) -> Classification:
    """Bucket one notation by onset-reconstructability (the Stage 1 core).

    Prefix/whole-string markers are checked first (they dominate the notation regardless of the
    buttons): a stance/positional context, a named throw, a just-frame, a ``~`` cancel, a hold.
    What remains is a comma-string of groups; if every group is a clean ``(dir, buttons)`` press the
    move is Easy (single or string), otherwise the first Hard group's reason wins.
    """
    k = key.strip()
    # Named/positional throws ("Left Throw", "Back Throw.1+3") carry no reconstructable buttons.
    if "throw" in k.lower() or "(" in k:
        return Classification(
            key, Bucket.hard, Reason.throw if "throw" in k.lower() else Reason.stance, None
        )
    # A stance/positional prefix (``H.``/``FC.``/``DPD.``/``ws``…): the onset buttons can't show the
    # stance the player was in, so the notation isn't reconstructable from the press alone.
    if "." in k or k.startswith("ws"):
        return Classification(key, Bucket.hard, Reason.stance, None)
    if ":" in k:
        return Classification(key, Bucket.hard, Reason.just_frame, None)
    if "~" in k:
        return Classification(key, Bucket.hard, Reason.cancel, None)
    if "*" in k:  # `*`/`**` = hold the last input
        return Classification(key, Bucket.hard, Reason.hold, None)

    groups = k.split(",")
    onset: tuple[int, tuple[str, ...]] | None = None
    for i, group in enumerate(groups):
        kind, parsed = _parse_group(group)
        if kind != "easy":
            return Classification(key, Bucket.hard, Reason(kind), None)
        if i == 0:
            onset = parsed
    bucket = Bucket.easy_single if len(groups) == 1 else Bucket.easy_string
    reason = Reason.single if bucket is Bucket.easy_single else Reason.string
    return Classification(key, bucket, reason, onset)


def onset_of(key: str) -> tuple[int, tuple[str, ...]] | None:
    """The reconstruction primitive: an easy move's onset ``(numpad dir, buttons)``, else None.

    This is what a live reconstructor matches against a decoded :class:`InputState` onset window:
    recover ``(dir, buttons)`` from memory, look up which of the char's easy keys share that onset,
    and intersect with the frame-fingerprint candidate. Hard moves return ``None`` (no onset match).
    """
    return classify(key).onset


@dataclass(frozen=True)
class Coverage:
    """The Stage 1 headline: how many of a character's moves are onset-reconstructable.

    ``by_reason`` counts every move's :class:`Reason`; ``onset_collisions`` maps each shared onset
    ``(dir, buttons)`` to the number of *easy* keys that share it — the ambiguity a reconstructor
    resolves by intersecting with the frame-fingerprint candidate (a single opinion, not the only
    one). ``unique_onsets`` is the count of easy moves whose onset is theirs alone.
    """

    total: int
    easy_single: int
    easy_string: int
    hard: int
    by_reason: dict[str, int] = field(default_factory=dict)
    onset_collisions: dict[str, int] = field(default_factory=dict)
    unique_onsets: int = 0

    @property
    def easy(self) -> int:
        return self.easy_single + self.easy_string

    @property
    def easy_fraction(self) -> float:
        return self.easy / self.total if self.total else 0.0


def coverage(keys: list[str]) -> Coverage:
    """Roll a character's ``framedata_keys`` into a :class:`Coverage` summary."""
    classifications = [classify(k) for k in keys]
    by_reason: Counter[str] = Counter(c.reason.value for c in classifications)
    onset_keys: Counter[tuple[int, tuple[str, ...]]] = Counter(
        c.onset for c in classifications if c.onset is not None
    )
    collisions = {
        f"dir{dir_}+{'+'.join(btns) if btns else 'n'}": count
        for (dir_, btns), count in onset_keys.items()
        if count > 1
    }
    unique_onsets = sum(1 for count in onset_keys.values() if count == 1)
    return Coverage(
        total=len(keys),
        easy_single=sum(1 for c in classifications if c.bucket is Bucket.easy_single),
        easy_string=sum(1 for c in classifications if c.bucket is Bucket.easy_string),
        hard=sum(1 for c in classifications if c.bucket is Bucket.hard),
        by_reason=dict(by_reason),
        onset_collisions=dict(sorted(collisions.items(), key=lambda kv: -kv[1])),
        unique_onsets=unique_onsets,
    )
