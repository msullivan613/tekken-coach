"""Read a Tekken 8 character's live ``tk_moveset`` and turn its cancels into notation (brief #18).

Route B of the moveset-datamine build: instead of an offline extractor (TKMovesets2 is Tekken 7
only), our own **read-only** reader walks the live cancels array, decodes each cancel's ``command``
with the confirmed T8 encoding (:mod:`tekken_coach.framedata.moveset_decode`), and joins
``move_id -> notation`` off the cancel graph. This module is the memory side: the struct **format
facts**, the read helpers over a :class:`~tekken_coach.reader.memory_source.MemorySource`, the
Phase-1 discovery/validation of the player -> moveset pointer, and the Phase-2 read+attribute
pipeline that feeds the pure join.

Clean-room boundary (docs/02 §2/§5): every value here is a **published Tekken 8 structure fact**
used with attribution, no extractor code is vendored, and the source is read-only — this reader
resolves addresses and reads bytes, and can do nothing else.

Two calibration surfaces, both honest:

* **The moveset pointer** — how the player struct points at its ``tk_moveset`` header — is *not*
  documented. Phase 1 (``moveset-probe``) discovers it live by reusing the #11 pointer-slot
  enumerator and validating each candidate by moveset shape + a decoder gate. Until it is recorded,
  ``OffsetTable.players.moveset_slot`` is ``null`` and the build reports "not yet discovered".
* **Owner attribution** — which move's cancel-list a cancel belongs to — needs the per-move cancel
  range, i.e. the ``tk_move`` layout (stride + the cancel pointer/count offsets). The published
  tables we have specify ``tk_cancel`` and the ``tk_moveset`` header but not ``tk_move``'s cancel
  fields, so the :class:`MoveLayout` is supplied explicitly (the offline tests inject a synthetic
  one; the live build requires a confirmed one via the offset table and otherwise degrades cleanly).
  This was discovered during implementation — the brief called owner attribution "confirm the exact
  linkage during implementation" — and is reported rather than papered over: without owner
  attribution a string-only move would be mis-mapped to its final input, which the honest posture
  forbids (docs/05 §2.3: never emit a wrong mapping to hit coverage).
"""

from __future__ import annotations

from dataclasses import dataclass

from tekken_coach.framedata.moveset_decode import Cancel, JoinResult, decode_command, join_moves
from tekken_coach.reader.decode import read_scalar
from tekken_coach.reader.memory_source import MemorySource

# ---------------------------------------------------------------------------
# Confirmed Tekken 8 format facts (tekkenmods.com Tekken 8 moveset docs)
# ---------------------------------------------------------------------------

# tk_cancel — one row is 0x28 (40) bytes.
CANCEL_SIZE = 0x28
CANCEL_COMMAND_OFFSET = 0x00  # u64 — the encoded input that triggers the cancel
CANCEL_MOVE_ID_OFFSET = 0x24  # u16 — destination move index == live move_id @ 0x550

# tk_moveset header — the pointer/count pairs we need.
MOVESET_CANCELS_PTR_OFFSET = 0x1D0  # tk_cancel*
MOVESET_CANCELS_COUNT_OFFSET = 0x1D8  # u64
MOVESET_MOVES_PTR_OFFSET = 0x230  # tk_move*
MOVESET_MOVES_COUNT_OFFSET = 0x238  # u64
MOVESET_INPUT_SEQ_PTR_OFFSET = 0x250  # tk_input_sequence*
MOVESET_INPUT_SEQ_COUNT_OFFSET = 0x258  # u64

# Plausibility bounds for the header-shape validation (Phase 1). A real moveset has hundreds to a
# few thousand moves and more cancels than moves; a slot that is not a moveset reads counts that are
# either 0/huge or fail the readable-pointer check. These bound the Phase-1 gate, and they bound the
# Phase-2 read so a mis-identified slot can never spin reading millions of rows.
COUNT_MIN = 1
COUNT_MAX = 20000


def _read_u64(source: MemorySource, address: int) -> int:
    """Read an unsigned 64-bit value (counts, pointers, the command word) at ``address``.

    ``ptr`` is the reader's unsigned 8-byte kind (``<Q``), which is exactly a ``u64`` — the offset
    model has no separate ``u64`` kind because a pointer *is* one, so this reuses it.
    """
    return int(read_scalar(source, address, "ptr"))


def _read_u16(source: MemorySource, address: int) -> int:
    """Read an unsigned 16-bit value (the cancel's destination ``move_id``) at ``address``."""
    return int(read_scalar(source, address, "u16"))


# ---------------------------------------------------------------------------
# The tk_moveset header
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovesetHeader:
    """The ``tk_moveset`` header pointer/count pairs, read from a candidate moveset address."""

    cancels_ptr: int
    cancels_count: int
    moves_ptr: int
    moves_count: int
    input_sequences_ptr: int
    input_sequences_count: int


def read_moveset_header(source: MemorySource, moveset_ptr: int) -> MovesetHeader:
    """Read the six pointer/count pairs of the ``tk_moveset`` header at ``moveset_ptr``.

    Propagates :class:`~tekken_coach.reader.faults.MemoryReadError` if the header is unreadable; the
    caller (Phase-1 validation) treats an unreadable candidate as "not a moveset", never a crash.
    """
    return MovesetHeader(
        cancels_ptr=_read_u64(source, moveset_ptr + MOVESET_CANCELS_PTR_OFFSET),
        cancels_count=_read_u64(source, moveset_ptr + MOVESET_CANCELS_COUNT_OFFSET),
        moves_ptr=_read_u64(source, moveset_ptr + MOVESET_MOVES_PTR_OFFSET),
        moves_count=_read_u64(source, moveset_ptr + MOVESET_MOVES_COUNT_OFFSET),
        input_sequences_ptr=_read_u64(source, moveset_ptr + MOVESET_INPUT_SEQ_PTR_OFFSET),
        input_sequences_count=_read_u64(source, moveset_ptr + MOVESET_INPUT_SEQ_COUNT_OFFSET),
    )


def _counts_plausible(header: MovesetHeader) -> bool:
    """Whether the move/cancel counts look like a real moveset: both bounded and non-zero.

    A real moveset also has more cancels than moves, but that is only *typically* true (the T8 docs
    hedge it) and it is a weak signal next to the decoder gate, so it is **not** a hard reject here:
    a slot with an odd ratio still faces the decisive gate. The hard reject is a count that is zero
    or absurd, which is what a non-moveset slot reads.
    """
    return (
        COUNT_MIN <= header.moves_count <= COUNT_MAX
        and COUNT_MIN <= header.cancels_count <= COUNT_MAX
    )


def _pointers_readable(source: MemorySource, header: MovesetHeader) -> bool:
    """Whether the moves/cancels arrays dereference — read the first row of each through the source.

    A structural read-only check: if either array pointer is garbage the read faults and the slot is
    rejected. It reads one ``tk_cancel`` (0x28) and touches the first ``tk_move`` word, no more.
    """
    from tekken_coach.reader.faults import MemoryReadError  # noqa: PLC0415

    try:
        source.read(header.cancels_ptr, CANCEL_SIZE)
        source.read(header.moves_ptr, 8)
    except MemoryReadError:
        return False
    return True


# ---------------------------------------------------------------------------
# Reading the cancels array (Phase-1 gate + Phase-2 input)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawCancel:
    """One ``tk_cancel`` row from the global cancels array: its command + destination move."""

    command: int
    dest_move_id: int


def read_cancels(source: MemorySource, header: MovesetHeader) -> list[RawCancel]:
    """Read the whole global cancels array (``cancels_count`` rows of 0x28 bytes) from ``header``.

    Owner-agnostic: it yields ``(command, dest_move_id)`` per row, which is all the Phase-1 gate
    needs (a membership check over destinations). The count is bounded by :data:`COUNT_MAX` at the
    call site (Phase-1 validation runs :func:`_counts_plausible` first), so a mis-identified slot
    cannot make this loop unbounded.
    """
    out: list[RawCancel] = []
    for i in range(header.cancels_count):
        row = header.cancels_ptr + i * CANCEL_SIZE
        command = _read_u64(source, row + CANCEL_COMMAND_OFFSET)
        dest = _read_u16(source, row + CANCEL_MOVE_ID_OFFSET)
        out.append(RawCancel(command=command, dest_move_id=dest))
    return out


# ---------------------------------------------------------------------------
# Phase 1 — validate a candidate slot as the moveset pointer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownPair:
    """A ground-truth ``move_id -> notation`` anchor the Phase-1 gate reproduces from cancels."""

    move_id: int
    notation: str


# Bryan's committed anchors (assets/movemap/bryan.json) — the from-neutral single-input moves the
# decisive gate reproduces. Deliberately the clean single-direction + button cases the T8 encoding
# resolves without calibration (docs confirm dir=0x08 df, 0x04 d, no-prefix for the jab).
BRYAN_GATE_PAIRS: tuple[KnownPair, ...] = (
    KnownPair(1695, "1"),
    KnownPair(1628, "df+2"),
    KnownPair(1725, "d+4"),
    KnownPair(1573, "3"),
    KnownPair(1566, "2"),
)


@dataclass(frozen=True)
class GateRow:
    """One anchor's result in the Phase-1 decoder gate: expected vs. what the cancels decoded to."""

    move_id: int
    expected: str
    found: bool
    decoded: list[str]  # every notation any cancel to this dest decoded to (for a miss diagnosis)


def gate_pairs(cancels: list[RawCancel], pairs: tuple[KnownPair, ...]) -> list[GateRow]:
    """Check that each known ``move_id -> notation`` anchor is reproduced by some cancel (Phase 1).

    A **membership** check, needing no owner attribution: if this slot really is the moveset, a
    cancel exists whose destination is the anchor's move and whose command decodes to its notation.
    That validates the moveset pointer + the ``tk_cancel`` layout + the decoder at once, which is
    exactly what the discovery must confirm before the offset is recorded.
    """
    decoded_by_dest: dict[int, list[str]] = {}
    for c in cancels:
        note = decode_command(c.command).notation()
        if note is not None:
            decoded_by_dest.setdefault(c.dest_move_id, []).append(note)

    rows: list[GateRow] = []
    for pair in pairs:
        decoded = decoded_by_dest.get(pair.move_id, [])
        rows.append(
            GateRow(
                move_id=pair.move_id,
                expected=pair.notation,
                found=pair.notation in decoded,
                decoded=sorted(set(decoded)),
            )
        )
    return rows


@dataclass(frozen=True)
class SlotValidation:
    """The verdict for one candidate pointer slot as the ``tk_moveset`` header (Phase 1)."""

    slot_offset: int
    header: MovesetHeader | None  # None when the header was unreadable at this slot
    counts_plausible: bool
    pointers_readable: bool
    move_id_in_range: bool  # live move_id @ 0x550 < moves_count (the "strong check")
    gate: list[GateRow]

    @property
    def gate_passed(self) -> bool:
        """Whether every known anchor was reproduced (the decisive check)."""
        return bool(self.gate) and all(row.found for row in self.gate)

    @property
    def is_moveset(self) -> bool:
        """Whether this slot passes every check: shape, readable arrays, move_id range, and gate."""
        return (
            self.counts_plausible
            and self.pointers_readable
            and self.move_id_in_range
            and self.gate_passed
        )


def validate_slot(
    source: MemorySource,
    slot_target: int,
    *,
    live_move_id: int,
    pairs: tuple[KnownPair, ...] = BRYAN_GATE_PAIRS,
) -> SlotValidation:
    """Validate ``slot_target`` (a dereferenced pointer-slot value) as a ``tk_moveset`` header.

    Runs the four checks in cost order — cheap shape first, the decoder gate last — short-circuiting
    the costly cancels read when the shape is already implausible, so sweeping many candidate slots
    stays fast. Read-only throughout; an unreadable candidate yields an all-false verdict, not a
    crash.
    """
    from tekken_coach.reader.faults import MemoryReadError  # noqa: PLC0415

    try:
        header = read_moveset_header(source, slot_target)
    except MemoryReadError:
        return SlotValidation(slot_target, None, False, False, False, [])

    counts_ok = _counts_plausible(header)
    if not counts_ok:
        # A slot whose counts are garbage is not a moveset; skip the readable/gate work entirely.
        return SlotValidation(slot_target, header, False, False, False, [])

    pointers_ok = _pointers_readable(source, header)
    move_id_ok = 0 <= live_move_id < header.moves_count
    gate: list[GateRow] = []
    if pointers_ok:
        try:
            gate = gate_pairs(read_cancels(source, header), pairs)
        except MemoryReadError:
            pointers_ok = False
    return SlotValidation(slot_target, header, counts_ok, pointers_ok, move_id_ok, gate)


# ---------------------------------------------------------------------------
# Phase 2 — owner attribution + the notation build
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveLayout:
    """The ``tk_move`` fields needed to attribute cancels to their owner move (NEEDS-LIVE-CONFIRM).

    The published tables specify ``tk_cancel`` and the ``tk_moveset`` header but not ``tk_move``'s
    cancel range, so this is supplied rather than baked as a fact: the offline tests inject a
    synthetic layout, and the live build requires a confirmed one (recorded once the user's live run
    validates the Phase-2 self-check against ``bryan.json``). A move at index ``i`` (== its
    ``move_id``, since the live ``move_id`` is an index into the moves array) owns ``cancel_count``
    cancel rows starting at the pointer in ``cancel_ptr_offset``.
    """

    size: int  # tk_move stride
    cancel_ptr_offset: (
        int  # offset of the pointer to this move's first cancel (into the cancels array)
    )
    cancel_count_offset: int  # offset of this move's cancel count (u64)


def read_attributed_cancels(
    source: MemorySource, header: MovesetHeader, layout: MoveLayout
) -> list[Cancel]:
    """Read every move's cancel list, attributing each cancel to its owner move (Phase 2).

    Walks the moves array (``moves_count`` rows of ``layout.size``); for move index ``i`` it reads
    that move's cancel start pointer + count and emits one :class:`Cancel` per row with
    ``source_move_id = i``. Owner attribution is what lets the join tell a from-neutral canonical
    input from a mid-string follow-up — without it a string-only move would be mis-mapped to its
    final input (docs/05 §2.3).
    """
    cancels: list[Cancel] = []
    for i in range(header.moves_count):
        move_addr = header.moves_ptr + i * layout.size
        cancel_start = _read_u64(source, move_addr + layout.cancel_ptr_offset)
        cancel_count = _read_u64(source, move_addr + layout.cancel_count_offset)
        if not (0 <= cancel_count <= COUNT_MAX):
            continue  # a garbage count on one move never derails the whole read
        for j in range(cancel_count):
            row = cancel_start + j * CANCEL_SIZE
            command = _read_u64(source, row + CANCEL_COMMAND_OFFSET)
            dest = _read_u16(source, row + CANCEL_MOVE_ID_OFFSET)
            cancels.append(Cancel(source_move_id=i, dest_move_id=dest, command=command))
    return cancels


def build_notation_map(
    source: MemorySource, moveset_ptr: int, layout: MoveLayout, *, neutral_move_id: int
) -> JoinResult:
    """Read, attribute, join: reconstruct ``move_id -> notation`` from the live moveset (Phase 2).

    The whole read path in one call: read the header, walk the moves for owner-attributed cancels,
    and hand them to the pure :func:`~tekken_coach.framedata.moveset_decode.join_moves`. Returns the
    :class:`~tekken_coach.framedata.moveset_decode.JoinResult` (resolved notations + reported
    collisions + unresolved), which the build command turns into movemap entries and a hit/miss
    table against the character's committed ids.
    """
    header = read_moveset_header(source, moveset_ptr)
    cancels = read_attributed_cancels(source, header, layout)
    return join_moves(cancels, neutral_move_id=neutral_move_id)


# ---------------------------------------------------------------------------
# Phase 2 self-check — reproduce a character's committed ids (the build gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfCheckRow:
    """One committed ``move_id`` in the self-check: ground-truth notation vs. what we rebuilt."""

    move_id: int
    expected: str
    got: str | None
    status: (
        str  # "HIT" (rebuilt and matches) | "MISS" (rebuilt but differs) | "MISSING" (not rebuilt)
    )


def self_check(notation: dict[int, str], ground_truth: dict[int, str]) -> list[SelfCheckRow]:
    """Compare the rebuilt ``move_id -> notation`` map against a character's committed ground truth.

    The Phase-2 gate: running against Bryan should reproduce his committed ids (``assets/movemap/
    bryan.json``). A ``MISS`` (rebuilt a *different* notation) is the failure that matters: it means
    a wrong mapping — while a ``MISSING`` id is only out of v1 scope (a motion/stance/held-direction
    the decoder honestly declines), not a correctness bug.
    """
    rows: list[SelfCheckRow] = []
    for move_id in sorted(ground_truth):
        expected = ground_truth[move_id]
        got = notation.get(move_id)
        if got is None:
            status = "MISSING"
        elif got == expected:
            status = "HIT"
        else:
            status = "MISS"
        rows.append(SelfCheckRow(move_id, expected, got, status))
    return rows
