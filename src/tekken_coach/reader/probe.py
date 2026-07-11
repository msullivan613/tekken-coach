"""Pure observation core for ``probe-state`` (docs/02 §8 state-map calibration).

``probe-state`` streams the raw encoded state words while the owner performs each state in-game, so
the value -> meaning map (:class:`~tekken_coach.reader.offsets.EncodedStateSpec`) can be filled in
by **observation** — nothing but observation can say what ``stun_type == 3`` means, because nobody
is in stun at round start (docs/02 §8; contrast the round-start oracle that derives the *offsets*).

The live half is a ``while True`` loop reading a live process — untestable in CI. So, mirroring how
:func:`~tekken_coach.reader.decode.poll_frames` (live) splits from
:func:`~tekken_coach.reader.doctor.evaluate_frames` (pure), the two things worth testing live as
pure functions over already-polled samples:

* :func:`change_records` — turn a sequence of per-poll reads into **change records** (one row per
  player whenever that player's watched tuple changes), the JSONL the ``--record`` log persists.
* :func:`distinct_values` / :func:`build_skeleton` — list every distinct raw value actually observed
  per encoded field and emit a state-map **draft skeleton**, so the human annotates flags next to
  real values instead of transcribing the values by hand.

Clean-room boundary (docs/02 §5 rule 2): this emits observed integers only. It never maps a value to
a flag — that value -> meaning judgment is the human's, deliberately not sourced from any community
enum. The skeleton ships every flag list empty and ``calibrated: false``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast, get_args

from tekken_coach.reader.offsets import ScalarKind

# The state-map draft this chunk emits: an empty flag list per observed value, for a human to fill.
SKELETON_NOTES = (
    "Draft from probe-state observation. Fill each [] with flags from docs/02 §8 "
    "(known flags in offsets.STATE_FLAGS), then set calibrated:true."
)

# The scalar kinds `--watch` accepts (the reader's whole ScalarKind set, kept in sync via get_args).
WATCH_KINDS: frozenset[str] = frozenset(get_args(ScalarKind))


@dataclass(frozen=True)
class WatchPoint:
    """One ad-hoc player-struct offset to watch during ``probe-state`` exploration (docs/02 §8).

    The seeded state-word offsets can go stale across a build (as the fork's ``move_id`` did), so
    ``--watch`` lets a run observe *candidate* raw offsets directly — without editing the table —
    to find where the live state words moved to. ``name`` is the ``@0x<offset>`` label the columns
    and JSONL use.
    """

    name: str
    offset: int
    kind: ScalarKind


def parse_watch(spec: str) -> list[WatchPoint]:
    """Parse ``--watch "0x434:u32,0x510:u32"`` into watch points named ``@0x<offset>``.

    Offsets are hex (``0x…``) or decimal; kinds are the reader's :data:`ScalarKind` set. Raises
    ``ValueError`` with an actionable message on a malformed pair (missing ``:``, bad number,
    unknown kind) or an empty spec — the CLI turns that into a clean error, not a traceback.
    """
    points: list[WatchPoint] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"watch spec {part!r} must be OFFSET:KIND (e.g. 0x434:u32)")
        off_text, kind = (s.strip() for s in part.split(":", 1))
        try:
            offset = int(off_text, 0)
        except ValueError:
            raise ValueError(f"watch spec {part!r}: {off_text!r} is not a valid offset") from None
        if offset < 0:
            raise ValueError(f"watch spec {part!r}: offset must be non-negative")
        if kind not in WATCH_KINDS:
            raise ValueError(
                f"watch spec {part!r}: unknown kind {kind!r} (use one of {sorted(WATCH_KINDS)})"
            )
        points.append(WatchPoint(name=f"@0x{offset:x}", offset=offset, kind=cast(ScalarKind, kind)))
    if not points:
        raise ValueError("watch spec is empty (expected OFFSET:KIND pairs, e.g. 0x434:u32).")
    return points


@dataclass(frozen=True)
class PollSample:
    """One poll instant: elapsed ``t`` seconds plus each player's watched-field tuple.

    ``rows`` is indexed by player slot (``rows[0]`` = P1, ``rows[1]`` = P2); each entry is that
    player's watched values in the same order as the ``names`` handed to :func:`change_records`.
    The live loop builds these from :func:`~tekken_coach.reader.commands._probe_row`; a test scripts
    them directly, which is the whole point of the split.
    """

    t: float
    rows: tuple[tuple[int | float, ...], ...]


@dataclass(frozen=True)
class ChangeRecord:
    """One emitted change: a player's watched fields at the instant its tuple changed.

    ``player`` is 1-based (1 or 2, the on-screen labels). ``fields`` maps every watched field name
    to its raw integer — both the context fields (``move_id``/``move_frame``/``counter_state``) and
    the encoded-state words — so a later reviewer can see which move the player was in when a state
    word changed (docs/02 §8: "which move was I in when stun_type went to 3").
    """

    t: float
    player: int
    fields: dict[str, int | float]

    def to_jsonl(self) -> str:
        """Serialize as one JSONL object (``t`` rounded to match the console's 2-decimal column)."""
        return json.dumps({"t": round(self.t, 2), "player": self.player, "fields": self.fields})


def change_records(samples: Iterable[PollSample], names: Sequence[str]) -> Iterator[ChangeRecord]:
    """Yield a :class:`ChangeRecord` each time a player's watched tuple changes across ``samples``.

    Emits **only on change**: a tuple that holds steady across polls produces no row, so a state
    performed and held reads as a single event, not a flood. Each player is tracked independently
    (blocking changes P2's stun word while P1 sits neutral), so both streams interleave by time.
    This is the exact "print only when it changed" logic the live loop had inline, lifted out so it
    runs against a scripted :class:`~tekken_coach.reader.memory_source.FakeMemorySource`.
    """
    previous: dict[int, tuple[int | float, ...]] = {}
    for sample in samples:
        for index, values in enumerate(sample.rows):
            if previous.get(index) == values:
                continue
            previous[index] = values
            yield ChangeRecord(
                t=sample.t,
                player=index + 1,
                fields=dict(zip(names, values, strict=True)),
            )


def distinct_values(
    records: Iterable[ChangeRecord], fields: Sequence[str]
) -> dict[str, dict[str, list[str]]]:
    """List every distinct raw value seen per ``field``, each mapped to an empty flag list.

    ``fields`` is the **encoded-state** field set (the ``spec.flags`` keys), not the context fields,
    which are watched only for correlation. Values are the integers actually observed across both
    players, sorted, as string keys (JSON object keys are strings, matching
    :class:`~tekken_coach.reader.offsets.EncodedStateSpec`). The flag lists are always empty: the
    tool emits values, a human emits meanings (docs/02 §5 rule 2).
    """
    seen: dict[str, set[int | float]] = {field: set() for field in fields}
    for record in records:
        for field in fields:
            if field in record.fields:
                seen[field].add(record.fields[field])
    return {field: {str(value): [] for value in sorted(values)} for field, values in seen.items()}


def build_skeleton(records: Iterable[ChangeRecord], fields: Sequence[str]) -> dict[str, object]:
    """Assemble the state-map draft: ``calibrated: false`` + observed values with empty flag lists.

    The result loads unchanged through
    :func:`~tekken_coach.reader.offsets.load_state_map` — it is a valid ``EncodedStateSpec`` with
    empty flag lists — so the human edits it in place, fills the flags, and flips ``calibrated``.
    """
    return {
        "calibrated": False,
        "notes": SKELETON_NOTES,
        "flags": distinct_values(records, fields),
    }
