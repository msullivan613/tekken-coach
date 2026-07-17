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

from tekken_coach.reader.offsets import ComponentAnchor, ScalarKind

# The state-map draft this chunk emits: an empty flag list per observed value, for a human to fill.
SKELETON_NOTES = (
    "Draft from probe-state observation. Fill each [] with flags from docs/02 §8 "
    "(known flags in offsets.STATE_FLAGS), then set calibrated:true."
)

# The scalar kinds `--watch` accepts (the reader's whole ScalarKind set, kept in sync via get_args).
WATCH_KINDS: frozenset[str] = frozenset(get_args(ScalarKind))

# Byte width per kind — the stride a `START-END:KIND` range steps by.
_KIND_SIZE: dict[str, int] = {
    "u8": 1,
    "bool8": 1,
    "u16": 2,
    "u32": 4,
    "i32": 4,
    "f32": 4,
    "i64": 8,
    "ptr": 8,
}

# A sweep this wide almost certainly means a typo, not intent — fail loud rather than read forever.
_MAX_WATCH_POINTS = 8192

# Above this many watched points, the aligned per-change row stops being readable and starts being a
# denial of service on the terminal: a row is ~20 chars per column, so a whole-struct sweep prints a
# ~100 KB line per change at ~20 changes/s. Rendering that is slower than the game runs, so the pass
# the user is trying to perform gets wrecked by the tool meant to observe it. Wide sweeps print a
# heartbeat instead; the JSONL still records every column, which is what the analyzer reads.
WIDE_SWEEP_COLUMNS = 24

# How often a wide sweep prints its heartbeat.
HEARTBEAT_SECONDS = 1.0


def is_wide_sweep(names: Sequence[str]) -> bool:
    """Whether a sweep watches too many points to print rows (:data:`WIDE_SWEEP_COLUMNS`)."""
    return len(names) > WIDE_SWEEP_COLUMNS


def due_for_beat(last_beat: float | None, t: float, every: float = HEARTBEAT_SECONDS) -> bool:
    """Whether a wide sweep should print its heartbeat at ``t`` (always for the first change)."""
    return last_beat is None or t - last_beat >= every


@dataclass
class PollRate:
    """How fast the sweep is actually polling — a correctness property, not a vanity metric.

    #10's whole-struct sweep managed **4.7 Hz** (one read per watched offset, 10752 a poll). The
    script asks for 2-second holds, so at 4.7 Hz a hold is ~9 samples and a *tap* can fall between
    two polls and never be observed at all — the sweep would report "this offset never reacted"
    about a button that was pressed. A rate this loop cannot sustain silently corrupts the negative
    it reports, so the run measures and prints it rather than leaving the user to guess.
    """

    polls: int = 0
    elapsed: float = 0.0

    @property
    def hz(self) -> float:
        """Observed polls per second (0.0 before the first interval elapses)."""
        return self.polls / self.elapsed if self.elapsed > 0 else 0.0

    def summary(self) -> str:
        """The end-of-run line, against #10's baseline."""
        return f"poll rate: {self.hz:.1f} Hz over {self.polls} polls (#10's sweep managed 4.7 Hz)"


def heartbeat_line(t: float, changes: int, points: int, hz: float | None = None) -> str:
    """The wide-sweep console line: the probe's own clock, which is the checklist's clock.

    Deliberately shows ``t``: the input-protocol checklist is timestamped against exactly this
    elapsed-seconds clock, so a user watching this line can follow the script against it — which
    also gives the analyzer's alignment fit an easier job. ``hz`` rides along when known, so a
    sweep too slow to catch a tap is visible *during* the pass, not after it.
    """
    rate = f" @ {hz:.1f} Hz" if hz is not None else ""
    return f"{t:>7.2f}  watching {points} offsets{rate} — {changes} changes recorded"


@dataclass(frozen=True)
class SlotPath:
    """A pointer slot to dereference before sweeping — the ``--watch-behind`` target (brief #11).

    Mirrors :class:`~tekken_coach.reader.offsets.ComponentAnchor` exactly (``slot_offset`` + hops),
    because that is the shape a hit gets baked into: if a sweep behind ``0x38`` finds input, the
    result is a ``players.components.input`` ``ComponentAnchor`` — a **data** edit to the offset
    table, not a schema change. The table already knows how to express what we are looking for.
    """

    slot_offset: int
    pointer_path: tuple[int, ...] = ()

    def label(self) -> str:
        """The spec-shaped label (``0x38``, or ``0x20/0x8`` with hops)."""
        hops = "".join(f"/0x{o:x}" for o in self.pointer_path)
        return f"0x{self.slot_offset:x}{hops}"

    def to_component(self) -> ComponentAnchor:
        """As a :class:`~tekken_coach.reader.offsets.ComponentAnchor` — the deref stays the
        tested one.

        ``--watch-behind`` resolves its landing through
        :func:`~tekken_coach.reader.decode.resolve_component` — the same call the decoder uses for
        ``transform`` — rather than reimplementing a pointer walk. It also means a confirmed hit
        transcribes into the offset table verbatim: the thing that found it *is* the thing that
        reads it.
        """
        return ComponentAnchor(slot_offset=self.slot_offset, pointer_path=list(self.pointer_path))


@dataclass(frozen=True)
class WatchPoint:
    """One ad-hoc offset to watch during ``probe-state`` exploration (docs/02 §8, brief #11).

    The seeded state-word offsets can go stale across a build (as the fork's ``move_id`` did), so
    ``--watch`` lets a run observe *candidate* raw offsets directly — without editing the table —
    to find where the live state words moved to. ``name`` is the label the columns and JSONL use.

    ``slot`` is ``None`` for a plain player-struct offset (named ``@0x1c``). When set, the offset is
    read **behind** that pointer slot instead (named ``@0x38+0x1c``): the sweep resolves
    ``player_base + slot_offset``, walks the hops, and reads ``offset`` from the landing. The names
    are what ``analyze-input`` prints, so a slot stays legible in the ranking with no change to the
    analyzer — it scores fields by name and does not care what a name means.
    """

    name: str
    offset: int
    kind: ScalarKind
    slot: SlotPath | None = None


def _watch_offsets(off_text: str, kind: str, part: str) -> list[int]:
    """Resolve one spec's offset text to the offsets it names — a single value or a ``START-END``
    range stepped by ``kind``'s byte width (so a whole struct region can be swept to find an unknown
    field). Raises ``ValueError`` on a bad number, an empty/backwards range, or an absurd sweep.
    """
    if "-" in off_text:
        start_text, end_text = (s.strip() for s in off_text.split("-", 1))
        try:
            start, end = int(start_text, 0), int(end_text, 0)
        except ValueError:
            raise ValueError(
                f"watch spec {part!r}: {off_text!r} is not a valid START-END range"
            ) from None
        if start < 0:
            raise ValueError(f"watch spec {part!r}: offset must be non-negative")
        if end <= start:
            raise ValueError(f"watch spec {part!r}: range END must be greater than START")
        offsets = list(range(start, end, _KIND_SIZE[kind]))
        if len(offsets) > _MAX_WATCH_POINTS:
            raise ValueError(
                f"watch spec {part!r}: range expands to {len(offsets)} points "
                f"(>{_MAX_WATCH_POINTS}); narrow it or use a wider kind"
            )
        return offsets
    try:
        offset = int(off_text, 0)
    except ValueError:
        raise ValueError(f"watch spec {part!r}: {off_text!r} is not a valid offset") from None
    if offset < 0:
        raise ValueError(f"watch spec {part!r}: offset must be non-negative")
    return [offset]


def parse_watch(spec: str) -> list[WatchPoint]:
    """Parse ``--watch "0x434:u32,0x510:u32"`` into watch points named ``@0x<offset>``.

    Each comma-separated term is ``OFFSET:KIND`` or ``START-END:KIND`` (a range stepped by the
    kind's width — e.g. ``0xd2e0-0xd4c0:u32`` sweeps that region to locate an unknown field).
    Offsets are hex (``0x…``) or decimal; kinds are the reader's :data:`ScalarKind` set. Raises
    ``ValueError`` with an actionable message on a malformed term or an empty spec — the CLI turns
    that into a clean error, not a traceback.
    """
    points: list[WatchPoint] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"watch spec {part!r} must be OFFSET:KIND (e.g. 0x434:u32)")
        off_text, kind = (s.strip() for s in part.split(":", 1))
        if kind not in WATCH_KINDS:
            raise ValueError(
                f"watch spec {part!r}: unknown kind {kind!r} (use one of {sorted(WATCH_KINDS)})"
            )
        for offset in _watch_offsets(off_text, kind, part):
            points.append(
                WatchPoint(name=f"@0x{offset:x}", offset=offset, kind=cast(ScalarKind, kind))
            )
    if not points:
        raise ValueError("watch spec is empty (expected OFFSET:KIND pairs, e.g. 0x434:u32).")
    return points


def _parse_slot(text: str, part: str) -> SlotPath:
    """Parse a ``SLOT`` or ``SLOT/HOP/HOP`` term into a :class:`SlotPath`."""
    numbers: list[int] = []
    for piece in text.split("/"):
        try:
            numbers.append(int(piece.strip(), 0))
        except ValueError:
            raise ValueError(
                f"watch-behind spec {part!r}: {piece.strip()!r} is not a valid slot offset"
            ) from None
    if any(n < 0 for n in numbers):
        raise ValueError(f"watch-behind spec {part!r}: offsets must be non-negative")
    return SlotPath(slot_offset=numbers[0], pointer_path=tuple(numbers[1:]))


def parse_watch_behind(spec: str) -> list[WatchPoint]:
    """Parse ``--watch-behind "0x20/8:0x0-0x100:u8,0x38:0x0-0x100:u8"`` into watch points.

    Each comma-separated term is ``SLOT[/HOP...]:OFFSET-END:KIND`` — dereference
    ``player_base + SLOT``, walk each ``/HOP``, then sweep ``OFFSET-END`` behind the landing. The
    offset/kind half is :func:`parse_watch`'s exact grammar, reused rather than reimplemented, so a
    range, a bare offset and every :data:`WATCH_KINDS` kind mean here what they mean there. Points
    are named ``@0x38+0x1c``.

    This is the deref #10's sweep could not do. #10 proved the *flat* struct carries no raw input;
    it could not speak to what hangs off the struct's pointers, because a pointer slot never changes
    and a change-sweep only sees change (brief #11).
    """
    points: list[WatchPoint] = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        fields = [s.strip() for s in part.split(":")]
        if len(fields) != 3:
            raise ValueError(
                f"watch-behind spec {part!r} must be SLOT[/HOP]:OFFSET-END:KIND "
                "(e.g. 0x38:0x0-0x100:u8 or 0x20/8:0x0-0x100:u8)"
            )
        slot_text, off_text, kind = fields
        if kind not in WATCH_KINDS:
            raise ValueError(
                f"watch-behind spec {part!r}: unknown kind {kind!r} "
                f"(use one of {sorted(WATCH_KINDS)})"
            )
        slot = _parse_slot(slot_text, part)
        for offset in _watch_offsets(off_text, kind, part):
            points.append(
                WatchPoint(
                    name=f"@{slot.label()}+0x{offset:x}",
                    offset=offset,
                    kind=cast(ScalarKind, kind),
                    slot=slot,
                )
            )
    if not points:
        raise ValueError(
            "watch-behind spec is empty (expected SLOT:OFFSET-END:KIND terms, "
            "e.g. 0x38:0x0-0x100:u8)."
        )
    return points


@dataclass(frozen=True)
class ReadPlan:
    """One object to block-read per player per poll, and where its values land in the row.

    ``slot`` is ``None`` for the player struct itself, or the :class:`SlotPath` to dereference
    first. ``indices`` maps each of ``points`` to its column in the assembled row, so grouping by
    object does not reorder the columns the header and JSONL promised.
    """

    slot: SlotPath | None
    points: tuple[WatchPoint, ...]
    indices: tuple[int, ...]
    start: int
    size: int


def build_read_plan(points: Sequence[WatchPoint]) -> list[ReadPlan]:
    """Group ``points`` into one block read per object, preserving column order.

    The whole perf story in one function: N watched offsets become len(plan) reads per player per
    poll (see :func:`block_span`), and every point keeps the column index it had.
    """
    grouped: dict[SlotPath | None, list[int]] = {}
    for index, point in enumerate(points):
        grouped.setdefault(point.slot, []).append(index)
    plans: list[ReadPlan] = []
    for slot, indices in grouped.items():
        chosen = tuple(points[i] for i in indices)
        start, size = block_span(chosen)
        plans.append(
            ReadPlan(slot=slot, points=chosen, indices=tuple(indices), start=start, size=size)
        )
    return plans


def assemble_row(
    plans: Sequence[ReadPlan], blocks: Sequence[bytes], width: int
) -> tuple[int | float, ...]:
    """Slice each plan's block into the row, at the columns the plan reserved.

    Pure: the live half reads ``blocks`` (one per plan, in plan order); this decides what the values
    mean positionally. A missing/short block is the caller's error, not something to paper over —
    :func:`slice_point` raises rather than emit a zero that would read as an observation.
    """
    row: list[int | float] = [0] * width
    for plan, block in zip(plans, blocks, strict=True):
        for point, index in zip(plan.points, plan.indices, strict=True):
            row[index] = slice_point(block, plan.start, point)
    return tuple(row)


def block_span(points: Sequence[WatchPoint]) -> tuple[int, int]:
    """The ``(start, size)`` byte span covering every point — one read instead of ``len(points)``.

    Perf is load-bearing here, not a nicety. #10 measured its 5376-offset sweep at **4.7 Hz** doing
    10752 individual reads per poll, and a sweep that slow cannot resolve a 2-second button hold.
    Chasing K slots x 256 bytes x 2 players would multiply that until the pass is worthless. One
    block read per object per poll makes it Kx2 syscalls, and the slicing
    (:func:`slice_point`) is free by comparison.
    """
    if not points:
        raise ValueError("cannot compute a block span for zero watch points")
    start = min(p.offset for p in points)
    end = max(p.offset + _KIND_SIZE[p.kind] for p in points)
    return start, end - start


def slice_point(block: bytes, block_start: int, point: WatchPoint) -> int | float:
    """Decode one watch point from an already-read ``block`` that starts at ``block_start``.

    ``bool8`` folds to ``int`` (the JSONL and the analyzer want numbers); ``f32`` stays a float —
    matching what the per-read path produced, so a recorded log is identical either way.
    """
    from tekken_coach.reader.decode import unpack_scalar  # noqa: PLC0415

    lo = point.offset - block_start
    raw = block[lo : lo + _KIND_SIZE[point.kind]]
    value = unpack_scalar(raw, point.kind)
    return int(value) if isinstance(value, bool) else value


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
