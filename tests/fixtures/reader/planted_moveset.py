"""A planted **tk_moveset** world for the brief #18 moveset-datamine reader tests.

Builds a :class:`~tests.fixtures.reader.flat_source.FlatMemorySource` holding a synthetic Bryan
``tk_moveset`` — a header, a cancels array, and a moves array — all in the *confirmed Tekken 8
layout* (``tk_cancel`` 0x28 rows, the header pointer/count pairs, commands encoded per the
documented ``direction | button<<32`` split). A green read/decode/join here proves the reader flows
end to end on data shaped exactly like the live game; the real Bryan moveset flows through the same
:func:`~tekken_coach.reader.moveset.build_notation_map` path.

Move index == move_id (the live ``move_id`` is an index into the moves array), so each owner move's
cancel list is planted at ``moves_ptr + move_id * MOVE_SIZE``. The from-neutral cancels live on the
neutral move (id 0); string follow-ups live on their prefix move; and the jab (1695) is *also*
planted as a mid-string follow-up off ``b+1`` so the "neutral command stays canonical" trap is
exercised on real memory. Two deliberate non-clean cases ride along: a collision (two from-neutral
cancels to one dest) and two unresolved cases (an unknown direction, and a Heat-only engage) the
decoder must degrade, never guess.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from tekken_coach.framedata.moveset_decode import MODE_NORMAL
from tekken_coach.reader.moveset import (
    CANCEL_COMMAND_OFFSET,
    CANCEL_MOVE_ID_OFFSET,
    CANCEL_SIZE,
    MOVESET_CANCELS_COUNT_OFFSET,
    MOVESET_CANCELS_PTR_OFFSET,
    MOVESET_INPUT_SEQ_COUNT_OFFSET,
    MOVESET_INPUT_SEQ_PTR_OFFSET,
    MOVESET_MOVES_COUNT_OFFSET,
    MOVESET_MOVES_PTR_OFFSET,
    MoveLayout,
)
from tekken_coach.reader.offsets import ComponentAnchor
from tests.fixtures.reader.flat_source import FlatMemorySource

MODULE = "Polaris-Win64-Shipping.exe"
MODULE_BASE = 0x140000000

# Heap segments — deliberately far apart, non-overlapping.
MOVESET_BASE = 0x300000000  # the tk_moveset header
CANCELS_BASE = 0x301000000  # the global cancels array (tk_cancel rows)
MOVES_BASE = 0x302000000  # the moves array (tk_move rows)
DECOY_BASE = 0x303000000  # a readable but non-moveset object (for the negative slot test)

# Brief #19: the header is NOT a direct player slot — it is reached player -> object -> header. The
# player struct holds a pointer to an intermediate object; a slot inside that object holds the
# header address. This is the shape the live run proved (no direct slot landed on a header).
PLAYER_BASE = 0x304000000  # the resolved player struct base (holder model gives this live)
PLAYER_STRUCT_SPAN = 0x200  # the window the reverse scan sweeps for player pointer slots
PLAYER_MOVESET_SLOT = 0x30  # the player slot that points at the intermediate object
PLAYER_DISTRACTOR_SLOT = 0x80  # a second pointer slot, to a nearby object (must NOT be chosen)
OBJECT_BASE = 0x305000000  # the intermediate object the player points at (too small to be a header)
OBJECT_SIZE = 0x40
MOVESET_REF_OFFSET = 0x18  # the slot inside the object that holds the header address

# A second, gate-FAILING moveset: valid header shape (survives the cheap filter) but cancels that do
# NOT reproduce Bryan's anchors, proving the gate — not the shape filter — is what accepts a header.
GATE_DECOY_BASE = 0x306000000
GATE_DECOY_CANCELS_BASE = 0x307000000
GATE_DECOY_MOVES_COUNT = 200
GATE_DECOY_CANCELS_COUNT = 250  # > moves -> passes the shape filter; padded to reach the gate

# Brief #20: the scan world's REAL header carries realistic large counts — cancels ABOVE the old
# 20000 cap (the ceiling that filtered the true Bryan header out on 2026-07-19) and far more cancels
# than moves. Its cancels array is physically padded to this length so the gate can read all of it.
REAL_CANCELS_COUNT = (
    22000  # > 20000 (old cap) and >> MOVES_COUNT (1720): the defining moveset shape
)

# Two shape-filter decoys that must be rejected BEFORE the gate (they own no cancels array):
#  - a near-equal-count array (the 8,703-strong live junk): fails the new ``cancels > moves`` check.
#  - a micro array: fails the raised count floor.
NEAR_EQUAL_DECOY_BASE = 0x308000000
NEAR_EQUAL_DECOY_CANCELS = 4880
NEAR_EQUAL_DECOY_MOVES = 4881  # cancels < moves -> rejected at the shape filter, never gated
MICRO_DECOY_BASE = 0x309000000
MICRO_DECOY_CANCELS = 6  # cancels > moves but both below the floor -> rejected at the shape filter
MICRO_DECOY_MOVES = 3

# The durable path the reverse scan must derive back from the confirmed header.
EXPECTED_MOVESET_ANCHOR = ComponentAnchor(
    slot_offset=PLAYER_MOVESET_SLOT, pointer_path=[MOVESET_REF_OFFSET]
)

NEUTRAL_MOVE_ID = 0

# The synthetic tk_move layout the offline reader is handed (the live one is discovered/confirmed —
# see MoveLayout). One move = [cancel_start_ptr(8), cancel_count(8)], stride 0x10.
MOVE_SIZE = 0x10
MOVE_CANCEL_PTR_OFFSET = 0x00
MOVE_CANCEL_COUNT_OFFSET = 0x08
MOVE_LAYOUT = MoveLayout(
    size=MOVE_SIZE,
    cancel_ptr_offset=MOVE_CANCEL_PTR_OFFSET,
    cancel_count_offset=MOVE_CANCEL_COUNT_OFFSET,
)

MOVES_COUNT = 1720  # > the largest owner move index (1705); dests beyond it are just u16 keys

# Direction bitfield codes (low 32 of the command) — the confirmed T8 values.
_DIR = {"": 0x00, "n": 0x20, "d": 0x04, "df": 0x08, "b": 0x10}
_UNKNOWN_DIR = 0x400  # not a modeled direction code -> decodes to unresolved

# Button field low byte (PP): notation buttons + the non-notation Heat bit.
_PP = {"1": 0x01, "2": 0x02, "3": 0x04, "4": 0x08}
_PP_HEAT = 0x10  # a special (non-notation) button -> special_only -> unresolved


def _command(direction: int, pp: int, mode: int = MODE_NORMAL) -> int:
    """Pack a ``tk_cancel.command`` uint64 from a direction code + pressed-button byte + mode."""
    button_field = (mode << 24) | pp
    return direction | (button_field << 32)


def _pp_of(buttons: str) -> int:
    """OR the PP bits for a '+'-joined notation button string (``"1+2"`` -> 0x03)."""
    pp = 0
    for b in buttons.split("+"):
        pp |= _PP[b]
    return pp


@dataclass(frozen=True)
class PlantedCancel:
    """One synthetic cancel to plant: its owner move, destination move, and raw command."""

    owner: int
    dest: int
    command: int


def _clean(owner: int, dest: int, direction: str, buttons: str) -> PlantedCancel:
    return PlantedCancel(owner, dest, _command(_DIR[direction], _pp_of(buttons)))


# The synthetic Bryan cancel graph. From-neutral canonical inputs on move 0, string follow-ups on
# their prefix, plus the jab-as-mid-string trap and the collision / unresolved cases.
PLANTED_CANCELS: tuple[PlantedCancel, ...] = (
    # from-neutral single-input moves (owner = the neutral move)
    _clean(NEUTRAL_MOVE_ID, 1695, "", "1"),
    _clean(NEUTRAL_MOVE_ID, 1628, "df", "2"),
    _clean(NEUTRAL_MOVE_ID, 1725, "d", "4"),
    _clean(NEUTRAL_MOVE_ID, 1573, "", "3"),
    _clean(NEUTRAL_MOVE_ID, 1566, "", "2"),
    _clean(NEUTRAL_MOVE_ID, 1574, "", "4"),
    _clean(NEUTRAL_MOVE_ID, 1705, "b", "1"),
    # unresolved: an unknown direction code — degrade, never a wrong guess
    PlantedCancel(NEUTRAL_MOVE_ID, 1990, _command(_UNKNOWN_DIR, _PP["1"])),
    # unresolved: a Heat-only engage (special button, no 1-4) — no clean notation
    PlantedCancel(NEUTRAL_MOVE_ID, 1992, _command(_DIR[""], _PP_HEAT)),
    # collision: two from-neutral cancels to one dest, different notations — reported, not guessed
    _clean(NEUTRAL_MOVE_ID, 1991, "", "1"),
    _clean(NEUTRAL_MOVE_ID, 1991, "", "2"),
    # string continuations (owner = the prefix move)
    _clean(1574, 1582, "", "3"),  # 4 -> 4,3
    _clean(1695, 1697, "", "2"),  # 1 -> 1,2
    # the jab is ALSO a mid-string cancel target off b+1 — its neutral "1" must stay canonical
    _clean(1705, 1695, "", "1"),
)

# The notation the join must reconstruct for each mapped move (the fixture's ground truth).
EXPECTED_NOTATION: dict[int, str] = {
    1695: "1",
    1628: "df+2",
    1725: "d+4",
    1573: "3",
    1566: "2",
    1574: "4",
    1705: "b+1",
    1582: "4,3",
    1697: "1,2",
}
EXPECTED_COLLISIONS: dict[int, list[str]] = {1991: ["1", "2"]}
EXPECTED_UNRESOLVED: frozenset[int] = frozenset({1990, 1992})


def _cancel_row(command: int, dest: int) -> bytes:
    """One 0x28-byte tk_cancel row with command @ 0x00 and dest move_id @ 0x24."""
    row = bytearray(CANCEL_SIZE)
    row[CANCEL_COMMAND_OFFSET : CANCEL_COMMAND_OFFSET + 8] = struct.pack("<Q", command)
    row[CANCEL_MOVE_ID_OFFSET : CANCEL_MOVE_ID_OFFSET + 2] = struct.pack("<H", dest)
    return bytes(row)


def _build_arrays() -> tuple[bytearray, bytearray, int]:
    """Lay the cancels array (grouped by owner) + the moves array; return them and the cancel count.

    Cancels are grouped by owner move so each move's ``cancel_start`` points at a contiguous run;
    the grouped runs concatenated *are* the global cancels array the Phase-1 gate reads.
    """
    owners = sorted({c.owner for c in PLANTED_CANCELS})
    cancels_blob = bytearray()
    moves_blob = bytearray(MOVES_COUNT * MOVE_SIZE)
    total = 0
    for owner in owners:
        rows = [c for c in PLANTED_CANCELS if c.owner == owner]
        start_index = total
        for c in rows:
            cancels_blob += _cancel_row(c.command, c.dest)
        total += len(rows)
        move_addr = owner * MOVE_SIZE
        cancel_start_ptr = CANCELS_BASE + start_index * CANCEL_SIZE
        moves_blob[move_addr + MOVE_CANCEL_PTR_OFFSET : move_addr + MOVE_CANCEL_PTR_OFFSET + 8] = (
            struct.pack("<Q", cancel_start_ptr)
        )
        moves_blob[
            move_addr + MOVE_CANCEL_COUNT_OFFSET : move_addr + MOVE_CANCEL_COUNT_OFFSET + 8
        ] = struct.pack("<Q", len(rows))
    return cancels_blob, moves_blob, total


def _header_blob(cancels_count: int) -> bytes:
    """The tk_moveset header: the six pointer/count pairs at their documented offsets."""
    size = MOVESET_INPUT_SEQ_COUNT_OFFSET + 8
    buf = bytearray(size)

    def put(off: int, value: int) -> None:
        buf[off : off + 8] = struct.pack("<Q", value)

    put(MOVESET_CANCELS_PTR_OFFSET, CANCELS_BASE)
    put(MOVESET_CANCELS_COUNT_OFFSET, cancels_count)
    put(MOVESET_MOVES_PTR_OFFSET, MOVES_BASE)
    put(MOVESET_MOVES_COUNT_OFFSET, MOVES_COUNT)
    put(MOVESET_INPUT_SEQ_PTR_OFFSET, 0)  # input_sequences ignored for v1
    put(MOVESET_INPUT_SEQ_COUNT_OFFSET, 0)
    return bytes(buf)


def planted_moveset_source() -> tuple[FlatMemorySource, int]:
    """A source with the synthetic Bryan moveset planted; returns it and the moveset header address.

    The header address is what the player's ``moveset_slot`` would dereference to; the Phase-1 tests
    validate this address, and the Phase-2 build reads from it.
    """
    cancels_blob, moves_blob, cancels_count = _build_arrays()
    source = FlatMemorySource(
        [
            (MODULE_BASE, b"\x00" * 0x1000),
            (MOVESET_BASE, _header_blob(cancels_count)),
            (CANCELS_BASE, bytes(cancels_blob)),
            (MOVES_BASE, bytes(moves_blob)),
            # A decoy heap object: readable, but its "counts" are garbage -> not a moveset.
            (DECOY_BASE, b"\xff" * 0x400),
        ],
        module_bases={MODULE: MODULE_BASE},
    )
    return source, MOVESET_BASE


def _put_ptr(buf: bytearray, off: int, value: int) -> None:
    buf[off : off + 8] = struct.pack("<Q", value)


def _player_blob() -> bytes:
    """The player struct: a slot pointing at the intermediate object, plus a distractor slot.

    No slot holds the header address directly (that is exactly the live finding), so the reverse
    scan must take the one-hop player -> object -> header route.
    """
    buf = bytearray(PLAYER_STRUCT_SPAN)
    _put_ptr(buf, PLAYER_MOVESET_SLOT, OBJECT_BASE)
    _put_ptr(buf, PLAYER_DISTRACTOR_SLOT, DECOY_BASE)  # points elsewhere; must not be chosen
    return bytes(buf)


def _object_blob() -> bytes:
    """The intermediate object: too small to be a header, holds the header address at one slot."""
    buf = bytearray(OBJECT_SIZE)
    _put_ptr(buf, MOVESET_REF_OFFSET, MOVESET_BASE)
    return bytes(buf)


def _shape_header(cancels_ptr: int, cancels_count: int, moves_ptr: int, moves_count: int) -> bytes:
    """A tk_moveset header with only the four cheap-filter words set (for shape-filter decoys)."""
    header = bytearray(MOVESET_INPUT_SEQ_COUNT_OFFSET + 8)
    _put_ptr(header, MOVESET_CANCELS_PTR_OFFSET, cancels_ptr)
    _put_ptr(header, MOVESET_CANCELS_COUNT_OFFSET, cancels_count)
    _put_ptr(header, MOVESET_MOVES_PTR_OFFSET, moves_ptr)
    _put_ptr(header, MOVESET_MOVES_COUNT_OFFSET, moves_count)
    return bytes(header)


def _pad_cancels(cancels: bytearray, target_count: int) -> bytearray:
    """Extend a cancels blob with zero rows to ``target_count`` rows (zero commands decode to None).

    A physically full array so the gate can read every ``cancels_count`` row; the trailing zero rows
    decode to no notation and go to dest 0, so they change no anchor's verdict.
    """
    needed = target_count * CANCEL_SIZE - len(cancels)
    if needed > 0:
        cancels += bytearray(needed)
    return cancels


def _gate_decoy_blobs() -> tuple[bytes, bytes]:
    """A header with a valid *shape* (cancels > moves, in range) but cancels reproducing NO anchor.

    Its counts and pointers survive the cheap shape filter — including the new ``cancels > moves``
    check (brief #20) — so it reaches the decoder gate, where it is rejected because its cancels go
    to unrelated destinations. This proves the gate, not the shape filter, is the decisive
    discriminator (brief #19 test intent).
    """
    decoy_cancels = (
        _clean(0, 900, "", "1"),
        _clean(0, 901, "", "2"),
        _clean(0, 902, "", "3"),
    )
    cancels = bytearray()
    for c in decoy_cancels:
        cancels += _cancel_row(c.command, c.dest)
    cancels = _pad_cancels(cancels, GATE_DECOY_CANCELS_COUNT)

    header = _shape_header(
        GATE_DECOY_CANCELS_BASE, GATE_DECOY_CANCELS_COUNT, MOVES_BASE, GATE_DECOY_MOVES_COUNT
    )
    return header, bytes(cancels)


def planted_moveset_scan_source() -> FlatMemorySource:
    """A world for the brief #19 heap shape+gate scan: real header off a path, plus a gate decoy.

    Superset of :func:`planted_moveset_source` — the real Bryan header at :data:`MOVESET_BASE` is
    now reachable only via ``PLAYER_BASE -> OBJECT_BASE -> header`` (no direct slot), it carries
    realistic large counts (cancels ABOVE the old 20000 cap, cancels >> moves — brief #20), and
    three decoys ride along: a shape-valid gate decoy (rejected at the gate) plus a near-equal-count
    decoy and a micro decoy (both rejected at the shape filter). Regions are auto-derived per
    segment (see :class:`~tests.fixtures.reader.flat_source.FlatMemorySource`), so the scan sweeps
    every planted heap segment exactly as it would the live regions.
    """
    cancels_blob, moves_blob, _ = _build_arrays()
    cancels_blob = _pad_cancels(cancels_blob, REAL_CANCELS_COUNT)
    decoy_header, decoy_cancels = _gate_decoy_blobs()
    return FlatMemorySource(
        [
            (MODULE_BASE, b"\x00" * 0x1000),
            (MOVESET_BASE, _header_blob(REAL_CANCELS_COUNT)),
            (CANCELS_BASE, bytes(cancels_blob)),
            (MOVES_BASE, bytes(moves_blob)),
            (DECOY_BASE, b"\xff" * 0x400),
            (PLAYER_BASE, _player_blob()),
            (OBJECT_BASE, _object_blob()),
            (GATE_DECOY_BASE, decoy_header),
            (GATE_DECOY_CANCELS_BASE, decoy_cancels),
            # Shape-filter decoys: valid pointers into real regions, but counts the new filter cuts.
            (
                NEAR_EQUAL_DECOY_BASE,
                _shape_header(
                    CANCELS_BASE, NEAR_EQUAL_DECOY_CANCELS, MOVES_BASE, NEAR_EQUAL_DECOY_MOVES
                ),
            ),
            (
                MICRO_DECOY_BASE,
                _shape_header(CANCELS_BASE, MICRO_DECOY_CANCELS, MOVES_BASE, MICRO_DECOY_MOVES),
            ),
        ],
        module_bases={MODULE: MODULE_BASE},
    )
