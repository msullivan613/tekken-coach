"""A planted static-pointer -> chain -> player-struct layout for the C4d base-scan tests.

C4d must recover a heap-allocated player struct reachable only through a *static* pointer in the
module's data plus a pointer chain. To prove that offline we plant exactly such a world in a
:class:`~tests.fixtures.reader.flat_source.FlatMemorySource`:

* a **module image** with a real (minimal) PE header and a ``.data`` section holding, at a
  deliberately non-round RVA, the root pointer of the chain, wrapped in distinctive context bytes
  so the derived AOB signature is unique;
* a **heap** with the chain nodes wired so ``module+BASE_OFFSET`` dereferenced through
  ``POINTER_PATH`` lands on P1's struct base;
* two **player structs** — P1 = Jin, P2 = Kazuya — with the oracle fields (``char_id``,
  ``move_id``, ``damage_taken``) at the seed layout offsets and health/position at *discoverable*
  offsets.

The base offset, stride, and health/position offsets are all planted at values the manifest does
not know a priori, so a passing derivation proves the scan *found* them (not that they were
seeded). The before/after pair moves P1's ``pos_x`` so the position scan has a real delta.
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
_HEAP_SPAN = 0x18000

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
    # The static pointer slot + distinctive signature context in .data.
    slot = BASE_OFFSET
    _blit(buf, slot - _SIG_BEFORE, bytes(range(0xA0, 0xA0 + _SIG_BEFORE)))
    _blit(buf, slot, _u64(_ROOT_PTR))
    _blit(buf, slot + 8, bytes(range(0xB0, 0xB0 + _SIG_AFTER)))
    return buf


def _put_player(
    buf: bytearray, off: int, char_id: int, move_id: int, pos: tuple[float, float, float]
) -> None:
    """Write one player struct at ``off`` within ``buf`` (oracle fields + health + position)."""
    _blit(buf, off + CHAR_ID_OFFSET, _u32(char_id))
    _blit(buf, off + MOVE_ID_OFFSET, _u32(move_id))
    _blit(buf, off + DAMAGE_OFFSET, struct.pack("<i", 0))  # round start: nothing taken yet
    _blit(buf, off + HEALTH_OFFSET, struct.pack("<i", ROUND_START_HEALTH))
    for k, v in enumerate(pos):
        _blit(buf, off + POS_OFFSET + 4 * k, struct.pack("<f", v))


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


def _source(pos: tuple[float, float, float]) -> FlatMemorySource:
    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(_pe_module())),
            (_HEAP_BASE, bytes(_heap(pos))),
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
    return PlantedChain(before=_source(P1_POS_BEFORE), after=_source(P1_POS_AFTER))


def relocated_pointer_source(root_ptr: int) -> FlatMemorySource:
    """The same module image with a **different** value in the static slot.

    Models the next build/run: the slot's contents (the heap pointer) move, but the surrounding
    ``.data`` bytes do not — which is exactly what the AOB signature wildcards, so it must still
    re-find the slot.
    """
    buf = _pe_module()
    _blit(buf, BASE_OFFSET, _u64(root_ptr))
    return FlatMemorySource(
        [(MODULE_BASE, bytes(buf)), (_HEAP_BASE, bytes(_heap(P1_POS_BEFORE)))],
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
            (P2_FAR_BASE, bytes(far)),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


def decode_source(table: OffsetTable) -> FlatMemorySource:
    """A flat image laid out per ``table`` so ``decode_frame`` round-trips through the chain anchor.

    Places the global struct at ``module_base + global.base_offset``, the static slot at the
    derived ``base_offset``, and both player structs at the chain-resolved bases — proving the
    *derived table* (pointer chain + discovered field offsets) is consumable by the real C4a
    decoder. Returns a source carrying :func:`decode_frame`-ready bytes for :func:`expected_frame`.
    """
    from tests.fixtures.reader.encode import _encode_player, _invert, pack_scalar

    module = bytearray(_MODULE_SPAN)
    _blit(module, BASE_OFFSET, _u64(_ROOT_PTR))
    heap = bytearray(_HEAP_SPAN)
    _blit(heap, (_ROOT_PTR + 0x10) - _HEAP_BASE, _u64(_NODE1))
    _blit(heap, (_NODE1 + 0x68) - _HEAP_BASE, _u64(_NODE2))
    _blit(heap, (_NODE2 + 0x8) - _HEAP_BASE, _u64(_NODE3))

    fr = expected_frame()
    # Globals at module_base + global anchor (base_offset 0, no chain).
    g = table.global_struct
    gbase = g.anchor.base_offset  # relative to module_base -> into `module` buffer
    phase = _invert(table.state_codes.match_phase)[fr.match_state.value]
    mode = _invert(table.state_codes.game_mode)["practice"]
    for name, value in (
        ("frame_counter", fr.frame),
        ("match_phase", phase),
        ("game_mode", mode),
        ("round", fr.round),
        ("timer_ms", fr.timer_ms),
    ):
        spec = g.fields[name]
        _blit(module, gbase + spec.offset, pack_scalar(spec.kind, value))

    # Players at the chain-resolved bases.
    for pf, base in ((fr.players[0], P1_BASE), (fr.players[1], P2_BASE)):
        image: dict[int, bytes] = {}
        _encode_player(image, table, base, pf)
        for addr, chunk in image.items():
            _blit(heap, addr - _HEAP_BASE, chunk)

    return FlatMemorySource(
        [(MODULE_BASE, bytes(module)), (_HEAP_BASE, bytes(heap))],
        module_bases={MODULE: MODULE_BASE},
    )


def expected_frame() -> FrameRecord:
    """The FrameRecord planted into :func:`decode_source` (kept in one place for the assertion)."""
    from tests.factories import make_frame_record

    return make_frame_record()
