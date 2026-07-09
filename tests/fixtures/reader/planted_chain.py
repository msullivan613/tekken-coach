"""Planted static-pointer -> chain -> struct layouts for the C4d/C4e base-scan tests.

The scans must recover heap-allocated structs reachable only through a *static* pointer in the
module's data plus a pointer chain. To prove that offline we plant exactly such a world in a
:class:`~tests.fixtures.reader.flat_source.FlatMemorySource`:

* a **module image** with a real (minimal) PE header and a ``.data`` section holding, at
  deliberately non-round RVAs, the root pointers of the player chain and the global chain, each
  wrapped in distinctive context bytes so the derived AOB signature is unique;
* a **heap** with the chain nodes wired so ``module+BASE_OFFSET`` dereferenced through
  ``POINTER_PATH`` lands on P1's struct base;
* two **player structs** — P1 = Jin, P2 = Kazuya — with the oracle fields (``char_id``,
  ``move_id``, ``damage_taken``) at the seed layout offsets and health/position at *discoverable*
  offsets;
* a **global/match struct** behind its own static pointer, whose frame counter ticks, whose round
  holds steady, and whose round clock counts down — the three behaviors C4e's global oracle assigns
  field names by. Their offsets are planted in a deliberately *scrambled* order relative to the
  field list, so a passing derivation proves the assignment came from behavior, not from position.

The base offsets, the stride, and the health/position offsets are all planted at values the manifest
does not know a priori, so a passing derivation proves the scan *found* them (not that they were
seeded). The before/after pair moves P1's ``pos_x`` so the position scan has a real delta.

:func:`planted_component` is the same world minus any in-struct position, plus a **transform
component** behind a pointer in each entity struct — the real Tekken 8 shape (C4e Phase 3), where
the in-struct scan must come up empty and the component scan must take over.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from tekken_coach.reader.offsets import OffsetTable
from tekken_coach.schemas import FrameRecord
from tests.fixtures.reader.flat_source import FlatMemorySource

MODULE = "Polaris-Win64-Shipping.exe"
MODULE_BASE = 0x140000000

# --- module layout (PE) ---
_E_LFANEW = 0x80
_SIZEOF_OPT = 0xF0
_TEXT_RVA = 0x1000
_DATA_RVA = 0x3000
_DATA_VSIZE = 0x1000
_SIZE_OF_IMAGE = 0x4000
_MODULE_SPAN = 0x4000

# The static pointer slot (the per-build base the scan derives) lives in .data at this RVA.
BASE_OFFSET = 0x3100
_SIG_BEFORE = 16
_SIG_AFTER = 16

# --- the pointer chain: module+BASE_OFFSET -> +0x10 -> +0x68 -> +0x8 -> +0x30 -> P1 base ---
POINTER_PATH = [0x10, 0x68, 0x8, 0x30]
_HEAP_BASE = 0x200001000
_ROOT_PTR = 0x200001000  # value at the static slot
_NODE1 = 0x200002000  # value at _ROOT_PTR + 0x10
_NODE2 = 0x200003000  # value at _NODE1 + 0x68
P1_BASE = 0x200010000
_NODE3 = P1_BASE - 0x30  # value at _NODE2 + 0x8; +0x30 lands on P1_BASE
STRIDE = 0x4000
P2_BASE = P1_BASE + STRIDE
_HEAP_SPAN = 0x28000

# --- planted field layout (oracle offsets are the seed facts; health/pos are discoverable) ---
CHAR_ID_OFFSET = 0x168
MOVE_ID_OFFSET = 0x528
DAMAGE_OFFSET = 0x1260
HEALTH_OFFSET = 0x60
POS_OFFSET = 0x80

JIN = 1
KAZUYA = 12
ROUND_START_HEALTH = 200
P1_MOVE_ID = 2145
P2_MOVE_ID = 800
P1_POS_BEFORE = (1.5, 0.0, -0.31)
P1_POS_AFTER = (2.0, 0.0, -0.31)  # x moved
P2_POS = (-1.0, 0.0, 0.5)

# --- the global/match struct: its own static slot, its own one-hop chain ---
GLOBAL_BASE_OFFSET = 0x3200  # static slot RVA in .data (the scan derives this)
GLOBAL_POINTER_PATH = [0x1468]  # module+GLOBAL_BASE_OFFSET -> deref -> +0x1468
_GLOBAL_SEG_BASE = 0x280000000
_GLOBAL_SEG_SPAN = 0x10000
_GLOBAL_ROOT = _GLOBAL_SEG_BASE  # value at the static slot
GLOBAL_BASE = _GLOBAL_ROOT + GLOBAL_POINTER_PATH[0]

# The three seeded offsets, planted so that NO offset holds the field its list position suggests:
# the oracle must assign them by behavior (ticks up / holds / counts down), never by order.
GLOBAL_TIMER_OFFSET = 0xD260  # counts down
GLOBAL_FRAME_OFFSET = 0xD2D8  # ticks up
GLOBAL_ROUND_OFFSET = 0xD4A8  # holds steady, in [1, 8]

FRAME_BEFORE = 128472
FRAME_STEP = 28
TIMER_BEFORE = 41200
TIMER_STEP = 1200
ROUND_NO = 2

# --- the transform component (C4e Phase 3): position lives behind a pointer, not in the struct ---
COMPONENT_SLOT_OFFSET = 0x100  # pointer slot inside the entity struct
COMPONENT_TRIPLE_OFFSET = 0x20  # (x,y,z) offset inside the component object
_P1_COMPONENT = 0x200020000
_P2_COMPONENT = 0x200021000


def _u16(v: int) -> bytes:
    return struct.pack("<H", v)


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _blit(buf: bytearray, off: int, data: bytes) -> None:
    buf[off : off + len(data)] = data


def _pe_module() -> bytearray:
    """A minimal but real PE32+ image: DOS + NT headers, a ``.text`` and a ``.data`` section."""
    buf = bytearray(_MODULE_SPAN)
    # DOS header
    _blit(buf, 0, b"MZ")
    _blit(buf, 0x3C, _u32(_E_LFANEW))
    # NT headers
    _blit(buf, _E_LFANEW, _u32(0x00004550))  # "PE\0\0"
    _blit(buf, _E_LFANEW + 4, _u16(0x8664))  # Machine (AMD64)
    _blit(buf, _E_LFANEW + 4 + 2, _u16(2))  # NumberOfSections
    _blit(buf, _E_LFANEW + 4 + 16, _u16(_SIZEOF_OPT))  # SizeOfOptionalHeader
    opt = _E_LFANEW + 4 + 20
    _blit(buf, opt, _u16(0x20B))  # PE32+ magic
    _blit(buf, opt + 56, _u32(_SIZE_OF_IMAGE))  # SizeOfImage
    # Section table
    sect = opt + _SIZEOF_OPT
    # .text — code/execute/read
    _blit(buf, sect, b".text\x00\x00\x00")
    _blit(buf, sect + 8, _u32(0x1000))  # VirtualSize
    _blit(buf, sect + 12, _u32(_TEXT_RVA))  # VirtualAddress
    _blit(buf, sect + 36, _u32(0x60000020))  # CODE|EXECUTE|READ
    # .data — initialized-data/read/write (scanned for slots)
    _blit(buf, sect + 40, b".data\x00\x00\x00")
    _blit(buf, sect + 40 + 8, _u32(_DATA_VSIZE))
    _blit(buf, sect + 40 + 12, _u32(_DATA_RVA))
    _blit(buf, sect + 40 + 36, _u32(0xC0000040))  # INITIALIZED_DATA|READ|WRITE
    # The static pointer slots + distinctive signature context in .data. The context bytes are
    # chosen so no 8-aligned qword over them reads as a plausible user-space pointer — otherwise
    # they would themselves be swept as candidates.
    _blit(buf, BASE_OFFSET - _SIG_BEFORE, bytes(range(0xA0, 0xA0 + _SIG_BEFORE)))
    _blit(buf, BASE_OFFSET, _u64(_ROOT_PTR))
    _blit(buf, BASE_OFFSET + 8, bytes(range(0xB0, 0xB0 + _SIG_AFTER)))
    _blit(buf, GLOBAL_BASE_OFFSET - _SIG_BEFORE, bytes(range(0x11, 0x11 + _SIG_BEFORE)))
    _blit(buf, GLOBAL_BASE_OFFSET, _u64(_GLOBAL_ROOT))
    _blit(buf, GLOBAL_BASE_OFFSET + 8, bytes(range(0x21, 0x21 + _SIG_AFTER)))
    return buf


def _put_player(
    buf: bytearray,
    off: int,
    char_id: int,
    move_id: int,
    pos: tuple[float, float, float] | None,
    component: int | None = None,
) -> None:
    """Write one player struct at ``off`` within ``buf`` (oracle fields + health + position).

    ``pos=None`` plants **no** in-struct position triple — the real Tekken 8 case, where the
    in-struct scan must find nothing. ``component`` plants a pointer to a transform object instead.
    """
    _blit(buf, off + CHAR_ID_OFFSET, _u32(char_id))
    _blit(buf, off + MOVE_ID_OFFSET, _u32(move_id))
    _blit(buf, off + DAMAGE_OFFSET, struct.pack("<i", 0))  # round start: nothing taken yet
    _blit(buf, off + HEALTH_OFFSET, struct.pack("<i", ROUND_START_HEALTH))
    if pos is not None:
        for k, v in enumerate(pos):
            _blit(buf, off + POS_OFFSET + 4 * k, struct.pack("<f", v))
    if component is not None:
        _blit(buf, off + COMPONENT_SLOT_OFFSET, _u64(component))


def _put_triple(buf: bytearray, off: int, pos: tuple[float, float, float]) -> None:
    for k, v in enumerate(pos):
        _blit(buf, off + 4 * k, struct.pack("<f", v))


def _chain_nodes(buf: bytearray) -> None:
    _blit(buf, (_ROOT_PTR + 0x10) - _HEAP_BASE, _u64(_NODE1))
    _blit(buf, (_NODE1 + 0x68) - _HEAP_BASE, _u64(_NODE2))
    _blit(buf, (_NODE2 + 0x8) - _HEAP_BASE, _u64(_NODE3))


def _heap(pos: tuple[float, float, float], *, with_p2: bool = True) -> bytearray:
    """The heap: chain nodes + P1 (Jin) and P2 (Kazuya) structs; ``pos`` is P1's position."""
    buf = bytearray(_HEAP_SPAN)
    _chain_nodes(buf)
    _put_player(buf, P1_BASE - _HEAP_BASE, JIN, P1_MOVE_ID, pos)
    if with_p2:
        _put_player(buf, P2_BASE - _HEAP_BASE, KAZUYA, P2_MOVE_ID, P2_POS)
    return buf


def _component_heap(p1_pos: tuple[float, float, float]) -> bytearray:
    """A heap where position lives in a per-player transform component, not in the entity struct."""
    buf = bytearray(_HEAP_SPAN)
    _chain_nodes(buf)
    _put_player(buf, P1_BASE - _HEAP_BASE, JIN, P1_MOVE_ID, None, component=_P1_COMPONENT)
    _put_player(buf, P2_BASE - _HEAP_BASE, KAZUYA, P2_MOVE_ID, None, component=_P2_COMPONENT)
    _put_triple(buf, (_P1_COMPONENT - _HEAP_BASE) + COMPONENT_TRIPLE_OFFSET, p1_pos)
    _put_triple(buf, (_P2_COMPONENT - _HEAP_BASE) + COMPONENT_TRIPLE_OFFSET, P2_POS)
    return buf


def _global_segment(*, step: int = 0) -> bytearray:
    """The global/match struct. ``step`` advances the frame counter and winds the clock down."""
    buf = bytearray(_GLOBAL_SEG_SPAN)
    base = GLOBAL_BASE - _GLOBAL_SEG_BASE
    _blit(buf, base + GLOBAL_FRAME_OFFSET, _u32(FRAME_BEFORE + step * FRAME_STEP))
    _blit(buf, base + GLOBAL_TIMER_OFFSET, _u32(TIMER_BEFORE - step * TIMER_STEP))
    _blit(buf, base + GLOBAL_ROUND_OFFSET, _u32(ROUND_NO))
    return buf


def _source(pos: tuple[float, float, float], *, step: int = 0) -> FlatMemorySource:
    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(_pe_module())),
            (_HEAP_BASE, bytes(_heap(pos))),
            (_GLOBAL_SEG_BASE, bytes(_global_segment(step=step))),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


def _component_source(pos: tuple[float, float, float], *, step: int = 0) -> FlatMemorySource:
    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(_pe_module())),
            (_HEAP_BASE, bytes(_component_heap(pos))),
            (_GLOBAL_SEG_BASE, bytes(_global_segment(step=step))),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


@dataclass(frozen=True)
class PlantedChain:
    """The two snapshots (round start / post-action) for the base-scan derivation."""

    before: FlatMemorySource
    after: FlatMemorySource


def planted_chain() -> PlantedChain:
    """Build the before/after flat sources for the planted static-pointer-chain layout."""
    return PlantedChain(before=_source(P1_POS_BEFORE), after=_source(P1_POS_AFTER, step=1))


def planted_component() -> PlantedChain:
    """The Tekken 8 shape: no in-struct position, a transform component behind a pointer."""
    return PlantedChain(
        before=_component_source(P1_POS_BEFORE),
        after=_component_source(P1_POS_AFTER, step=1),
    )


def relocated_pointer_source(root_ptr: int) -> FlatMemorySource:
    """The same module image with a **different** value in the static slot.

    Models the next build/run: the slot's contents (the heap pointer) move, but the surrounding
    ``.data`` bytes do not — which is exactly what the AOB signature wildcards, so it must still
    re-find the slot.
    """
    buf = _pe_module()
    _blit(buf, BASE_OFFSET, _u64(root_ptr))
    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(buf)),
            (_HEAP_BASE, bytes(_heap(P1_POS_BEFORE))),
            (_GLOBAL_SEG_BASE, bytes(_global_segment())),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


# P2 as a *separate allocation*, far outside any plausible constant stride from P1 — the fork's
# two-level `p2_data_offset` reality, which the single-anchor+stride PlayerStruct cannot express.
P2_FAR_BASE = 0x300000000


def two_level_source() -> FlatMemorySource:
    """A world where P1 is reachable but P2 is a separate allocation (no constant stride)."""
    far = bytearray(0x2000)
    _put_player(far, 0, KAZUYA, P2_MOVE_ID, P2_POS)
    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(_pe_module())),
            (_HEAP_BASE, bytes(_heap(P1_POS_BEFORE, with_p2=False))),
            (_GLOBAL_SEG_BASE, bytes(_global_segment())),
            (P2_FAR_BASE, bytes(far)),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


def _encode_globals(
    module: bytearray, glob: bytearray, table: OffsetTable, fr: FrameRecord, *, frame: int
) -> None:
    """Lay ``fr``'s global fields out at the *derived* offsets, wherever the anchor now points."""
    from tests.fixtures.reader.encode import _invert, pack_scalar

    g = table.global_struct
    if g.anchor.pointer_path:
        base, buf = GLOBAL_BASE - _GLOBAL_SEG_BASE, glob
    else:  # a seeded (static) global anchor: the struct sits inside the module image
        base, buf = g.anchor.base_offset, module
    phase = _invert(table.state_codes.match_phase)[fr.match_state.value]
    mode = _invert(table.state_codes.game_mode)["practice"]
    for name, value in (
        ("frame_counter", frame),
        ("match_phase", phase),
        ("game_mode", mode),
        ("round", fr.round),
        ("timer_ms", fr.timer_ms),
    ):
        spec = g.fields[name]
        _blit(buf, base + spec.offset, pack_scalar(spec.kind, value))


def decode_source(table: OffsetTable) -> FlatMemorySource:
    """A flat image laid out per ``table`` so ``decode_frame`` round-trips through the chain anchor.

    Places the global struct behind the derived global anchor, the static slots at the derived
    ``base_offset``s, and both player structs at the chain-resolved bases — proving the *derived
    table* (pointer chains + discovered field offsets) is consumable by the real C4a decoder.
    """
    from tests.fixtures.reader.encode import encode_player_into

    module = _pe_module()
    heap = bytearray(_HEAP_SPAN)
    glob = _global_segment()
    _chain_nodes(heap)

    fr = expected_frame()
    _encode_globals(module, glob, table, fr, frame=fr.frame)
    for pf, base in ((fr.players[0], P1_BASE), (fr.players[1], P2_BASE)):
        image: dict[int, bytes] = {}
        encode_player_into(image, table, base, pf)
        for addr, chunk in image.items():
            _blit(heap, addr - _HEAP_BASE, chunk)

    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(module)),
            (_HEAP_BASE, bytes(heap)),
            (_GLOBAL_SEG_BASE, bytes(glob)),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


def component_frame() -> FrameRecord:
    """:func:`expected_frame` at round-start health — what the doctor's health check expects."""
    fr = expected_frame()
    players = [p.model_copy(update={"health": ROUND_START_HEALTH}) for p in fr.players]
    return fr.model_copy(update={"players": players})


def component_decode_source(table: OffsetTable, *, step: int = 0) -> FlatMemorySource:
    """``decode_source`` for a table whose ``pos`` lives in a transform component (C4e Phase 3).

    ``step`` advances the global frame counter and walks P1 forward, so a sequence of these replays
    as successive frames for the doctor's monotonic-frame and moving-position checks.
    """
    from tests.fixtures.reader.encode import encode_player_into

    module = _pe_module()
    heap = bytearray(_HEAP_SPAN)
    glob = _global_segment(step=step)
    _chain_nodes(heap)

    fr = component_frame()
    _encode_globals(module, glob, table, fr, frame=fr.frame + step)
    for pf, base, component, pos in (
        (fr.players[0], P1_BASE, _P1_COMPONENT, (P1_POS_BEFORE[0] + step, *P1_POS_BEFORE[1:])),
        (fr.players[1], P2_BASE, _P2_COMPONENT, P2_POS),
    ):
        image: dict[int, bytes] = {}
        encode_player_into(image, table, base, pf)
        for addr, chunk in image.items():
            _blit(heap, addr - _HEAP_BASE, chunk)
        _blit(heap, (base - _HEAP_BASE) + COMPONENT_SLOT_OFFSET, _u64(component))
        _put_triple(heap, (component - _HEAP_BASE) + COMPONENT_TRIPLE_OFFSET, pos)

    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(module)),
            (_HEAP_BASE, bytes(heap)),
            (_GLOBAL_SEG_BASE, bytes(glob)),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


def expected_frame() -> FrameRecord:
    """The FrameRecord planted into :func:`decode_source` (kept in one place for the assertion)."""
    from tests.factories import make_frame_record

    return make_frame_record()
