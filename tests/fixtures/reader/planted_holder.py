"""A planted **holder-model** world for the C4i holder-scan tests.

The live Tekken 8 game does not lay the players out as a single-anchor + stride array. They hang off
a *holder* object by two per-player pointer slots (``holder+0x30``, ``holder+0x38``) to **separate**
allocations, and the holder's own ``.data`` slot is found by an **AoB code signature** in ``.text``
that references it RIP-relative. This fixture plants exactly that world in a
:class:`~tests.fixtures.reader.flat_source.FlatMemorySource`:

* a ``.text`` section holding, at a non-round RVA, the storing instruction
  ``4C 89 35 <disp32> 41 88 5E 28`` whose 32-bit RIP-relative ``disp32`` points at
* a ``.data`` **holder slot** holding a pointer to
* a **holder object** whose ``+0x30`` / ``+0x38`` slots point at
* two **separate** player allocations (P1 = Jin, P2 = Kazuya), far apart in the address space — no
  constant stride relates them, which is the whole point: the stride model cannot express this.

Each player struct carries the oracle fields at the community layout offsets (``char_id`` 0x168,
``move_id`` 0x550, ``damage_taken`` 0x1578) and a pointer to a per-player **transform component**
holding the moving position triple (position is not in the entity struct on T8). The global/match
struct is the same one the base-scan fixtures use, behind its own static slot + chain, so the global
oracle is exercised unchanged.

Nothing tells the scan where the holder slot is: it must AoB-match the instruction and RIP-decode
the ``disp32`` to find it, so a passing derivation proves the code-signature path works. The
before/after/window sources model the user's action: P1's ``move_id`` changes (jab + jump), P2's
``damage_taken`` rises (the jab connects), and P1's ``pos_x`` moves (a step).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from tekken_coach.reader.offsets import POSITION_COMPONENT, OffsetTable
from tests.fixtures.reader.flat_source import FlatMemorySource
from tests.fixtures.reader.planted_chain import (
    CHAR_ID_OFFSET,
    COMPONENT_SLOT_OFFSET,
    COMPONENT_TRIPLE_OFFSET,
    DAMAGE_OFFSET,
    GARBAGE_MATCH_PHASE,
    HEALTH_OFFSET,
    KAZUYA,
    MODULE,
    MODULE_BASE,
    MOVE_ID_OFFSET,
    P1_MOVE_ID,
    P1_MOVE_ID_AFTER,
    P1_POS_AFTER,
    P1_POS_BEFORE,
    P1_POS_MID,
    P2_DAMAGE_AFTER,
    P2_MOVE_ID,
    P2_MOVE_ID_AFTER,
    P2_POS,
    ROUND_START_HEALTH,
    _blit,
    _encode_globals,
    _global_segment,
    _pe_module,
    _put_triple,
    _u32,
    _u64,
    component_frame,
)

# --- the AoB code site (in .text) and the .data holder slot it references RIP-relative -----------
JIN = 6  # v3.00.02 fact (Irony/opendojo); the holder scan VALIDATES this, unlike --base-scan
ROUND_GATE_OFFSET = 0x15C0  # frames_since_round_start (0 during intro) — a round-active gate

AOB_PATTERN = "4C 89 35 ?? ?? ?? ?? 41 88 5E 28"  # MOV [rip+disp32], r14 ; MOV [r14+0x28], BL
DISP32_POS = 3
HOLDER_SLOTS = [0x30, 0x38]  # holder+0x30 -> P1, holder+0x38 -> P2

_TEXT_SITE_RVA = 0x1500  # where the storing instruction sits inside .text (RVA 0x1000, size 0x1000)
HOLDER_SLOT_RVA = 0x3400  # the .data slot the disp32 resolves to (free of the base/global slots)
# x64 RIP is the address of the NEXT instruction: slot = site + disp32_pos + 4 + disp32.
_DISP32 = HOLDER_SLOT_RVA - (_TEXT_SITE_RVA + DISP32_POS + 4)

# --- the heap: holder object + two SEPARATE player allocations + their transform components -------
_HOLDER_SEG = 0x220000000
HOLDER_BASE = _HOLDER_SEG  # the .data slot points here
P1_BASE = 0x221000000  # separate allocations, deliberately far apart (no constant stride)
P2_BASE = 0x222000000
_P1_COMPONENT = 0x223000000
_P2_COMPONENT = 0x224000000
_PLAYER_SPAN = 0x2000
_HOLDER_SPAN = 0x1000
_COMPONENT_SPAN = 0x1000

# Reuse the base-scan fixtures' global/match struct verbatim (its own static slot + chain).
from tests.fixtures.reader.planted_chain import (  # noqa: E402
    _GLOBAL_SEG_BASE,
)


def _text_instruction() -> bytes:
    """The 11-byte storing instruction with the RIP-relative disp32 spliced in."""
    return b"\x4c\x89\x35" + struct.pack("<i", _DISP32) + b"\x41\x88\x5e\x28"


def _holder_module() -> bytearray:
    """The base-scan module image plus the AoB site in .text and the holder pointer in .data."""
    buf = _pe_module()
    _blit(buf, _TEXT_SITE_RVA, _text_instruction())
    _blit(buf, HOLDER_SLOT_RVA, _u64(HOLDER_BASE))
    return buf


def _holder_object() -> bytearray:
    """The holder: two per-player pointer slots to separate allocations."""
    buf = bytearray(_HOLDER_SPAN)
    _blit(buf, HOLDER_SLOTS[0], _u64(P1_BASE))
    _blit(buf, HOLDER_SLOTS[1], _u64(P2_BASE))
    return buf


def _put_holder_player(
    char_id: int, move_id: int, *, component: int, damage: int = 0, gate: int = 300
) -> bytearray:
    """One player struct: the oracle fields + health + a component pointer + the round gate."""
    buf = bytearray(_PLAYER_SPAN)
    _blit(buf, CHAR_ID_OFFSET, _u32(char_id))
    _blit(buf, MOVE_ID_OFFSET, _u32(move_id))
    _blit(buf, DAMAGE_OFFSET, struct.pack("<i", damage))
    _blit(buf, HEALTH_OFFSET, struct.pack("<i", ROUND_START_HEALTH))
    _blit(buf, ROUND_GATE_OFFSET, _u32(gate))
    _blit(buf, COMPONENT_SLOT_OFFSET, _u64(component))
    return buf


def _component(pos: tuple[float, float, float]) -> bytearray:
    buf = bytearray(_COMPONENT_SPAN)
    _put_triple(buf, COMPONENT_TRIPLE_OFFSET, pos)
    return buf


def _holder_source(
    p1_pos: tuple[float, float, float],
    *,
    step: int = 0,
    acted: bool = False,
    jab_connected: bool = True,
) -> FlatMemorySource:
    """A round-start / action snapshot of the holder world (module + holder + players + globals)."""
    p1_move = P1_MOVE_ID_AFTER if acted else P1_MOVE_ID
    hit = acted and jab_connected
    p2_move = P2_MOVE_ID_AFTER if hit else P2_MOVE_ID
    p2_damage = P2_DAMAGE_AFTER if hit else 0
    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(_holder_module())),
            (_HOLDER_SEG, bytes(_holder_object())),
            (P1_BASE, bytes(_put_holder_player(JIN, p1_move, component=_P1_COMPONENT))),
            (
                P2_BASE,
                bytes(
                    _put_holder_player(KAZUYA, p2_move, component=_P2_COMPONENT, damage=p2_damage)
                ),
            ),
            (_P1_COMPONENT, bytes(_component(p1_pos))),
            (_P2_COMPONENT, bytes(_component(P2_POS))),
            (_GLOBAL_SEG_BASE, bytes(_global_segment(step=step))),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


@dataclass(frozen=True)
class PlantedHolder:
    """Round-start + action-window captures for the holder derivation.

    ``during[-1]`` doubles as the position scan's "after", exactly as the last live sample does.
    """

    before: FlatMemorySource
    after: FlatMemorySource
    during: list[FlatMemorySource]


def planted_holder() -> PlantedHolder:
    """The holder world where P1 (Jin) acts across the window — move_id changes, jab lands, walk."""
    during = [
        _holder_source(P1_POS_BEFORE, step=1),  # still idle
        _holder_source(P1_POS_MID, step=2, acted=True),  # mid-action: move_id changed, jab landed
        _holder_source(P1_POS_AFTER, step=3),  # landed, idle again (transient move_id, C4g)
    ]
    return PlantedHolder(
        before=_holder_source(P1_POS_BEFORE),
        after=during[-1],
        during=during,
    )


def planted_holder_idle() -> PlantedHolder:
    """The same world where the user never acted: no move_id change, so nothing is confirmed.

    The holder is found and structurally plausible at round start (Jin+Kazuya behind the two slots),
    but its acting player's ``move_id`` never moves — the behavioral oracle must fail closed.
    """
    during = [_holder_source(P1_POS_BEFORE, step=i) for i in (1, 2, 3)]
    return PlantedHolder(
        before=_holder_source(P1_POS_BEFORE),
        after=during[-1],
        during=during,
    )


def no_holder_source(*, step: int = 0) -> FlatMemorySource:
    """The world with the AoB instruction removed — the code signature must find no holder.

    The global/match struct is still present (behind its own static slot + chain), so the global
    oracle can be shown to resolve independently of the missing holder — ``step`` advances its frame
    counter so a before/after pair exercises that.
    """
    buf = _pe_module()  # no _text_instruction blitted, no holder slot
    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(buf)),
            (_GLOBAL_SEG_BASE, bytes(_global_segment(step=step))),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


def holder_decode_source(
    table: OffsetTable, *, step: int = 0, phase_raw: int | None = GARBAGE_MATCH_PHASE
) -> FlatMemorySource:
    """A flat image laid out per a holder ``table`` so ``decode_frame`` round-trips through it.

    Places the holder behind the derived anchor, each player behind its slot, and position in the
    per-player component — proving the *holder table* (per-player pointer slots + discovered field
    offsets) drives the real decoder. ``step`` walks P1 and advances the frame counter so a sequence
    replays as successive frames for the doctor's monotonic-frame and moving-position checks.
    """
    from tests.fixtures.reader.encode import encode_player_into

    module = _holder_module()
    holder = _holder_object()
    glob = _global_segment(step=step)

    fr = component_frame()
    _encode_globals(module, glob, table, fr, frame=fr.frame + step, phase_raw=phase_raw)
    component = table.players.components[POSITION_COMPONENT]

    player_bufs = {P1_BASE: bytearray(_PLAYER_SPAN), P2_BASE: bytearray(_PLAYER_SPAN)}
    comp_bufs = {
        _P1_COMPONENT: bytearray(_COMPONENT_SPAN),
        _P2_COMPONENT: bytearray(_COMPONENT_SPAN),
    }
    for pf, base, comp_ptr, pos in (
        (fr.players[0], P1_BASE, _P1_COMPONENT, (P1_POS_BEFORE[0] + step, *P1_POS_BEFORE[1:])),
        (fr.players[1], P2_BASE, _P2_COMPONENT, P2_POS),
    ):
        image: dict[int, bytes] = {}
        encode_player_into(image, table, base, pf)
        for addr, chunk in image.items():
            _blit(player_bufs[base], addr - base, chunk)
        _blit(player_bufs[base], COMPONENT_SLOT_OFFSET, _u64(comp_ptr))
        _put_triple(comp_bufs[comp_ptr], component.fields["pos_x"].offset, pos)

    return FlatMemorySource(
        [
            (MODULE_BASE, bytes(module)),
            (_HOLDER_SEG, bytes(holder)),
            (P1_BASE, bytes(player_bufs[P1_BASE])),
            (P2_BASE, bytes(player_bufs[P2_BASE])),
            (_P1_COMPONENT, bytes(comp_bufs[_P1_COMPONENT])),
            (_P2_COMPONENT, bytes(comp_bufs[_P2_COMPONENT])),
            (_GLOBAL_SEG_BASE, bytes(glob)),
        ],
        module_bases={MODULE: MODULE_BASE},
    )


def resolved_player_bases(table: OffsetTable) -> tuple[int, int]:
    """P1/P2 struct bases as the decoder resolves them from ``table`` (for test assertions)."""
    source = holder_decode_source(table)
    from tekken_coach.reader.decode import resolve_player_base

    return resolve_player_base(source, table, 0), resolve_player_base(source, table, 1)
