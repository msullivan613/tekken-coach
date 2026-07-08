"""Test-support encoder: lay a ``FrameRecord`` out as raw bytes for a ``FakeMemorySource``.

The inverse of :mod:`tekken_coach.reader.decode`, used *only by tests* to script memory images
from high-level ``FrameRecord``s (so the doctor / gap / state tests can build many frames without
hand-packing bytes). It is deliberately not part of the shipped reader package — the reader reads,
never writes — and it packs into a plain ``dict`` byte buffer, not process memory.

Fidelity note: ``action_state`` is a *derived* field (docs/03 §1), so the encoder reconstructs the
raw flags that would fold to it. For single-flag states this round-trips exactly; the golden
decode test (``tests/test_reader_decode.py``) instead packs bytes explicitly to verify the raw
bytes -> record mapping without going through this helper.
"""

from __future__ import annotations

import struct

from tekken_coach.reader.decode import _FORMATS
from tekken_coach.reader.memory_source import MemoryImage
from tekken_coach.reader.offsets import OffsetTable, ScalarKind
from tekken_coach.schemas import ActionState, FrameRecord, PlayerFrame

DEFAULT_MODULE_BASE = 0x140000000

# action_state -> the raw boolean flag that folds to it (docs/03 §1 thin normalization).
_STATE_TO_FLAG: dict[ActionState, str] = {
    ActionState.thrown: "thrown",
    ActionState.throw_tech_window: "throw_tech",
    ActionState.blockstun: "block_stun",
    ActionState.hitstun: "hit_stun",
    ActionState.stagger: "stagger",
    ActionState.knockdown: "knockdown",
    ActionState.wakeup: "wakeup",
    ActionState.airborne: "airborne",
    ActionState.sidestep: "sidestep",
    ActionState.crouch: "crouch",
}
_BUTTON_BIT = {"1": 0, "2": 1, "3": 2, "4": 3}


def pack_scalar(kind: ScalarKind, value: int | float | bool) -> bytes:
    fmt, _size = _FORMATS[kind]
    if kind == "bool8":
        return struct.pack(fmt, 1 if value else 0)
    if kind == "f32":
        return struct.pack(fmt, float(value))
    return struct.pack(fmt, int(value))


def _invert(codes: dict[str, str]) -> dict[str, int]:
    return {name: int(raw) for raw, name in codes.items()}


def _encode_player(
    image: dict[int, bytes],
    table: OffsetTable,
    base: int,
    pf: PlayerFrame,
) -> None:
    fields = table.players.fields
    counter_code = _invert(table.state_codes.counter_state)[pf.counter_state.value]
    simple_code = _invert(table.state_codes.simple_state)
    if pf.action_state is ActionState.attack:
        simple = simple_code["attack"]
    elif pf.action_state is ActionState.recovery:
        simple = simple_code["recovery"]
    else:
        simple = simple_code["neutral"]

    flags = {
        "block_stun": pf.block_stun,
        "hit_stun": pf.hit_stun,
        "throw_active": pf.throw_active,
        "airborne": pf.airborne,
        "juggle": pf.juggle,
        "stagger": False,
        "throw_tech": False,
        "thrown": False,
        "knockdown": False,
        "wakeup": False,
        "sidestep": False,
        "crouch": False,
        "heat_active": pf.heat.active,
        "heat_engager_used": pf.heat.engager_used,
        "rage": pf.rage,
    }
    extra = _STATE_TO_FLAG.get(pf.action_state)
    if extra is not None:
        flags[extra] = True

    scalars: dict[str, tuple[ScalarKind, int | float | bool]] = {
        "char_id": ("u32", pf.char_id),
        "move_id": ("u32", pf.move_id),
        "move_frame": ("u32", pf.move_frame),
        "health": ("i32", pf.health),
        "pos_x": ("f32", pf.pos[0]),
        "pos_y": ("f32", pf.pos[1]),
        "pos_z": ("f32", pf.pos[2]),
        "facing": ("i32", pf.facing),
        "simple_state": ("u32", simple),
        "counter_state": ("u32", counter_code),
        "heat_timer_ms": ("u32", pf.heat.timer_ms),
    }
    for name, (kind, value) in scalars.items():
        image[base + fields[name].offset] = pack_scalar(kind, value)
    for name, flag in flags.items():
        image[base + fields[name].offset] = pack_scalar("bool8", flag)

    # Input group (optional): valid flag gates dir + button bitmask (docs/03 §1 input may be null).
    image[base + fields["input_valid"].offset] = pack_scalar("bool8", pf.input is not None)
    if pf.input is not None:
        image[base + fields["input_dir"].offset] = pack_scalar("u8", pf.input.dir)
        mask = 0
        for btn in pf.input.buttons:
            mask |= 1 << _BUTTON_BIT[btn]
        image[base + fields["input_buttons"].offset] = pack_scalar("u16", mask)
    else:
        image[base + fields["input_dir"].offset] = pack_scalar("u8", 0)
        image[base + fields["input_buttons"].offset] = pack_scalar("u16", 0)


def encode_frame(
    fr: FrameRecord,
    table: OffsetTable,
    *,
    module_base: int = DEFAULT_MODULE_BASE,
    game_mode: str = "practice",
) -> MemoryImage:
    """Encode a whole ``FrameRecord`` into a ``{address: bytes}`` image for a fake source."""
    image: dict[int, bytes] = {}
    g = table.global_struct
    gbase = module_base + g.anchor.base_offset
    phase_code = _invert(table.state_codes.match_phase)[fr.match_state.value]
    mode_code = _invert(table.state_codes.game_mode)[game_mode]
    globals_: dict[str, tuple[ScalarKind, int]] = {
        "frame_counter": ("u32", fr.frame),
        "match_phase": ("u32", phase_code),
        "game_mode": ("u32", mode_code),
        "round": ("u32", fr.round),
        "timer_ms": ("u32", fr.timer_ms),
    }
    for name, (kind, value) in globals_.items():
        image[gbase + g.fields[name].offset] = pack_scalar(kind, value)

    pbase = module_base + table.players.anchor.base_offset
    for idx, pf in enumerate(fr.players):
        _encode_player(image, table, pbase + idx * table.players.stride, pf)
    return image


def module_base_for(table: OffsetTable, module_base: int = DEFAULT_MODULE_BASE) -> dict[str, int]:
    """The ``module_bases`` mapping a :class:`FakeMemorySource` needs for ``table``."""
    return {table.global_struct.anchor.module: module_base}


def advance_on_for(table: OffsetTable, module_base: int = DEFAULT_MODULE_BASE) -> int:
    """The frame-counter absolute address that ticks a :class:`FakeMemorySource` forward."""
    g = table.global_struct
    return module_base + g.anchor.base_offset + g.fields["frame_counter"].offset
