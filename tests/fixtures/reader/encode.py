"""Test-support encoder: lay a ``FrameRecord`` out as raw bytes for a ``FakeMemorySource``.

The inverse of :mod:`tekken_coach.reader.decode`, used *only by tests* to script memory images
from high-level ``FrameRecord``s (so the doctor / gap / state tests can build many frames without
hand-packing bytes). It is deliberately not part of the shipped reader package — the reader reads,
never writes — and it packs into a plain ``dict`` byte buffer, not process memory.

It follows the table it is handed, so it covers both worlds the decoder does:

* the C4c/legacy layout (one ``bool8`` per flag, in-struct ``pos_{x,y,z}``, a direct ``health``);
* Tekken 8's real layout (encoded state words, computed ``health = max_health - damage_taken``,
  position in a transform component — which the *caller* plants, since it lives outside the struct).

Fidelity note: ``action_state`` is a *derived* field (docs/03 §1), so the encoder reconstructs the
raw state that would fold to it. For single-flag states this round-trips exactly; the golden decode
test (``tests/test_reader_decode.py``) instead packs bytes explicitly to verify the raw bytes ->
record mapping without going through this helper.
"""

from __future__ import annotations

import struct

from tekken_coach.reader.decode import _FORMATS
from tekken_coach.reader.memory_source import MemoryImage
from tekken_coach.reader.offsets import EncodedStateSpec, OffsetTable, ScalarKind
from tekken_coach.schemas import ActionState, FrameRecord, PlayerFrame

DEFAULT_MODULE_BASE = 0x140000000

# action_state -> the raw semantic flag that folds to it (docs/03 §1 thin normalization).
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


def _desired_flags(pf: PlayerFrame) -> set[str]:
    """The semantic flags a decoder would have to see to reproduce ``pf``."""
    flags = {
        name
        for name, on in (
            ("block_stun", pf.block_stun),
            ("hit_stun", pf.hit_stun),
            ("throw_active", pf.throw_active),
            ("airborne", pf.airborne),
            ("juggle", pf.juggle),
        )
        if on
    }
    extra = _STATE_TO_FLAG.get(pf.action_state)
    if extra is not None:
        flags.add(extra)
    if pf.action_state is ActionState.attack:
        flags.add("attack")
    elif pf.action_state is ActionState.recovery:
        flags.add("recovery")
    else:
        flags.add("neutral")
    return flags


def _raw_for_field(codes: dict[str, list[str]], desired: set[str]) -> int:
    """The raw value whose flags best fit ``desired`` without implying anything extra.

    Picks the value contributing the most desired flags among those whose flag set is a *subset* of
    what we want — so the encoder never plants a state that would decode to a flag the FrameRecord
    does not have. Falls back to 0 (which an uncalibrated map leaves meaningless anyway).
    """
    best_raw, best_score = 0, -1
    for raw, flags in codes.items():
        if not set(flags) <= desired:
            continue
        if len(flags) > best_score:
            best_raw, best_score = int(raw), len(flags)
    return best_raw


def _encode_state(
    image: dict[int, bytes], table: OffsetTable, base: int, pf: PlayerFrame, spec: EncodedStateSpec
) -> None:
    """Plant Tekken 8's encoded state words by inverting the value -> meaning map."""
    desired = _desired_flags(pf)
    for name, codes in spec.flags.items():
        field = table.players.fields[name]
        image[base + field.offset] = pack_scalar(field.kind, _raw_for_field(codes, desired))


def _encode_legacy_state(
    image: dict[int, bytes], table: OffsetTable, base: int, pf: PlayerFrame
) -> None:
    """Plant the C4c/legacy layout: a ``simple_state`` code plus one ``bool8`` per flag."""
    fields = table.players.fields
    simple_code = _invert(table.state_codes.simple_state)
    if pf.action_state is ActionState.attack:
        simple = simple_code["attack"]
    elif pf.action_state is ActionState.recovery:
        simple = simple_code["recovery"]
    else:
        simple = simple_code["neutral"]
    image[base + fields["simple_state"].offset] = pack_scalar("u32", simple)

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
    }
    extra = _STATE_TO_FLAG.get(pf.action_state)
    if extra is not None:
        flags[extra] = True
    for name, flag in flags.items():
        image[base + fields[name].offset] = pack_scalar("bool8", flag)


def encode_player_into(
    image: dict[int, bytes],
    table: OffsetTable,
    base: int,
    pf: PlayerFrame,
) -> None:
    """Write one ``PlayerFrame`` into ``image`` at ``base``, following ``table``'s layout."""
    fields = table.players.fields
    counter_code = _invert(table.state_codes.counter_state)[pf.counter_state.value]

    scalars: dict[str, tuple[ScalarKind, int | float | bool]] = {
        "char_id": ("u32", pf.char_id),
        "move_id": ("u32", pf.move_id),
        "move_frame": ("u32", pf.move_frame),
        "facing": ("i32", pf.facing),
        "counter_state": ("u32", counter_code),
        "heat_timer_ms": ("u32", pf.heat.timer_ms),
        # The per-round frame counter the phase deriver reads (docs/02 §8); packed only when the
        # table declares the field (the real T8 layout), skipped on the legacy layout below.
        "frames_since_round_start": ("u32", pf.frames_since_round_start),
    }
    # health is either a direct field or computed from damage_taken (docs/02 §3).
    if table.players.max_health is not None:
        scalars["damage_taken"] = ("i32", table.players.max_health - pf.health)
    else:
        scalars["health"] = ("i32", pf.health)
    # pos_{x,y,z} exist in-struct only on the legacy layout; a component table's caller plants them.
    for axis, value in zip(("pos_x", "pos_y", "pos_z"), pf.pos, strict=True):
        scalars[axis] = ("f32", value)

    for name, (kind, value) in scalars.items():
        if name in fields:
            image[base + fields[name].offset] = pack_scalar(kind, value)

    for name, flag in (
        ("heat_active", pf.heat.active),
        ("heat_engager_used", pf.heat.engager_used),
        ("rage", pf.rage),
    ):
        if name in fields:
            image[base + fields[name].offset] = pack_scalar("bool8", flag)

    encoded = table.state_codes.encoded_state
    if encoded is not None:
        _encode_state(image, table, base, pf, encoded)
    else:
        _encode_legacy_state(image, table, base, pf)

    # Input group (optional): valid flag gates dir + button bitmask (docs/03 §1 input may be null).
    if "input_valid" not in fields:
        return
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
    stride = table.players.stride
    assert stride is not None, "encode fixtures use the legacy stride model"
    for idx, pf in enumerate(fr.players):
        encode_player_into(image, table, pbase + idx * stride, pf)
    return image


def module_base_for(table: OffsetTable, module_base: int = DEFAULT_MODULE_BASE) -> dict[str, int]:
    """The ``module_bases`` mapping a :class:`FakeMemorySource` needs for ``table``."""
    return {table.global_struct.anchor.module: module_base}


def advance_on_for(table: OffsetTable, module_base: int = DEFAULT_MODULE_BASE) -> int:
    """The frame-counter absolute address that ticks a :class:`FakeMemorySource` forward."""
    g = table.global_struct
    return module_base + g.anchor.base_offset + g.fields["frame_counter"].offset
