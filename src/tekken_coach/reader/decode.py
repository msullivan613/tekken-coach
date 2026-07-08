"""Decode raw process memory into ``FrameRecord``s (docs/02 §2, emitting docs/03 §1).

Given a read-only :class:`~tekken_coach.reader.memory_source.MemorySource` and an
:class:`~tekken_coach.reader.offsets.OffsetTable`, this resolves the global and per-player struct
bases (module-base + static offset, optional pointer chain) and reads the fixed field set for one
frame into a :class:`~tekken_coach.schemas.FrameRecord`.

Two things the spec is emphatic about (docs/03 §1 Notes):

* ``action_state`` is a **thin** normalization derived cheaply from raw flags — the segmenter does
  the real interpretation and *does not trust ``action_state`` alone* (docs/04 §4.1). So the raw
  flags (``block_stun``, ``hit_stun``, ``counter_state``, ``throw_active``, ``airborne``,
  ``juggle``) are carried **separately** on the ``PlayerFrame``, not folded away.
* ``input`` may be ``null`` (unresolvable during replay playback). We honor that: no input group,
  or a false ``input_valid`` flag, yields ``input=None``, and the segmenter never requires it.

:class:`FrameReader` wraps :func:`decode_frame` with the dropped-frame handling of docs/02 §7:
when the global frame counter jumps, it records a ``gap-tolerated:N`` marker whose ``N`` matches
the segmenter's own gap accounting exactly (docs/04 §4.7 uses ``missed = frame - prev - 1``).

Nothing here writes, injects, or renders — it only reads and returns data (docs/02 §2).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from tekken_coach.reader.faults import DecodeError, MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import Anchor, FieldSpec, OffsetTable, ScalarKind
from tekken_coach.reader.state import StateSignal, classify_state
from tekken_coach.schemas import (
    ActionState,
    CounterState,
    FrameRecord,
    HeatState,
    InputState,
    MatchState,
    PlayerFrame,
)

# Little-endian struct format + width per scalar kind (docs/02 §3; endianness/pointer_size are
# fixed on the OffsetTable). ``bool8`` reads one byte as a truthy flag; ``ptr`` is an 8-byte
# pointer for pointer-chain dereferences.
_FORMATS: dict[ScalarKind, tuple[str, int]] = {
    "u8": ("<B", 1),
    "u16": ("<H", 2),
    "u32": ("<I", 4),
    "i32": ("<i", 4),
    "i64": ("<q", 8),
    "f32": ("<f", 4),
    "bool8": ("<B", 1),
    "ptr": ("<Q", 8),
}


def read_scalar(source: MemorySource, address: int, kind: ScalarKind) -> int | float | bool:
    """Read one scalar of ``kind`` at ``address`` through the read-only source."""
    fmt, size = _FORMATS[kind]
    raw = source.read(address, size)
    if len(raw) != size:
        raise DecodeError(f"short read at 0x{address:x}: got {len(raw)}, need {size}")
    (value,) = struct.unpack(fmt, raw)
    if kind == "bool8":
        return bool(value)
    return value  # type: ignore[no-any-return]


def _read_int(source: MemorySource, address: int, kind: ScalarKind) -> int:
    value = read_scalar(source, address, kind)
    return int(value)


def _read_float(source: MemorySource, address: int, kind: ScalarKind) -> float:
    value = read_scalar(source, address, kind)
    return float(value)


def resolve_anchor(source: MemorySource, anchor: Anchor) -> int:
    """Resolve a struct base address from its :class:`Anchor` (docs/02 §3 anchoring).

    ``module_base + base_offset`` is the start; each ``pointer_path`` offset dereferences an
    8-byte pointer and adds the offset (a standard multi-level pointer). An empty path is a plain
    static offset.
    """
    address = source.module_base(anchor.module) + anchor.base_offset
    for offset in anchor.pointer_path:
        address = _read_int(source, address, "ptr") + offset
    return address


def _field(fields: dict[str, FieldSpec], name: str) -> FieldSpec:
    spec = fields.get(name)
    if spec is None:
        raise DecodeError(f"offset table is missing required field {name!r}")
    return spec


@dataclass(frozen=True)
class _RawPlayer:
    """Raw per-player reads before normalization (the flags feeding ``action_state``)."""

    simple_state: int
    block_stun: bool
    hit_stun: bool
    stagger: bool
    throw_active: bool
    throw_tech: bool
    thrown: bool
    airborne: bool
    juggle: bool
    knockdown: bool
    wakeup: bool
    sidestep: bool
    crouch: bool


def _derive_action_state(table: OffsetTable, raw: _RawPlayer) -> ActionState:
    """The *thin* ``action_state`` normalization (docs/03 §1 Notes).

    A cheap, fixed-priority fold of the raw flags into the coarse ``action_state`` enum. It is
    deliberately shallow: the segmenter re-derives the truth from the raw flags carried alongside
    (docs/04 §4.1), so this only needs to be a useful hint, not authoritative. Priority runs from
    the most specific/overriding states (thrown, stuns) down to the simple attack/recovery/neutral
    code that the game itself exposes (docs/02 §2).
    """
    if raw.thrown:
        return ActionState.thrown
    if raw.throw_tech:
        return ActionState.throw_tech_window
    if raw.block_stun:
        return ActionState.blockstun
    if raw.hit_stun:
        return ActionState.hitstun
    if raw.stagger:
        return ActionState.stagger
    if raw.knockdown:
        return ActionState.knockdown
    if raw.wakeup:
        return ActionState.wakeup
    if raw.airborne:
        return ActionState.airborne
    if raw.sidestep:
        return ActionState.sidestep
    simple = table.state_codes.simple_state.get(str(raw.simple_state))
    if simple == "attack":
        return ActionState.attack
    if simple == "recovery":
        return ActionState.recovery
    if raw.crouch:
        return ActionState.crouch
    return ActionState.neutral


_BUTTON_BITS: tuple[tuple[int, str], ...] = ((0, "1"), (1, "2"), (2, "3"), (3, "4"))


def _decode_input(
    source: MemorySource, base: int, fields: dict[str, FieldSpec]
) -> InputState | None:
    """Read the optional per-frame input (docs/03 §1).

    Returns ``None`` when the table has no input group or the ``input_valid`` flag is false —
    inputs are not always resolvable (e.g. during replay playback), and the segmenter must not
    require them.
    """
    if "input_valid" not in fields or "input_dir" not in fields or "input_buttons" not in fields:
        return None
    valid = read_scalar(source, base + _field(fields, "input_valid").offset, "bool8")
    if not valid:
        return None
    dir_spec = _field(fields, "input_dir")
    btn_spec = _field(fields, "input_buttons")
    direction = _read_int(source, base + dir_spec.offset, dir_spec.kind)
    mask = _read_int(source, base + btn_spec.offset, btn_spec.kind)
    buttons = [name for bit, name in _BUTTON_BITS if mask & (1 << bit)]
    return InputState(dir=direction, buttons=buttons)


def _decode_player(source: MemorySource, table: OffsetTable, index: int) -> PlayerFrame:
    players = table.players
    fields = players.fields
    base = resolve_anchor(source, players.anchor) + index * players.stride

    def i(name: str) -> int:
        spec = _field(fields, name)
        return _read_int(source, base + spec.offset, spec.kind)

    def f(name: str) -> float:
        spec = _field(fields, name)
        return _read_float(source, base + spec.offset, spec.kind)

    def b(name: str) -> bool:
        spec = _field(fields, name)
        return bool(read_scalar(source, base + spec.offset, "bool8"))

    raw = _RawPlayer(
        simple_state=i("simple_state"),
        block_stun=b("block_stun"),
        hit_stun=b("hit_stun"),
        stagger=b("stagger"),
        throw_active=b("throw_active"),
        throw_tech=b("throw_tech"),
        thrown=b("thrown"),
        airborne=b("airborne"),
        juggle=b("juggle"),
        knockdown=b("knockdown"),
        wakeup=b("wakeup"),
        sidestep=b("sidestep"),
        crouch=b("crouch"),
    )

    counter_raw = i("counter_state")
    counter_name = table.state_codes.counter_state.get(str(counter_raw), CounterState.none.value)

    facing_raw = i("facing")
    facing = 1 if facing_raw >= 0 else -1

    return PlayerFrame(
        char_id=i("char_id"),
        move_id=i("move_id"),
        move_frame=i("move_frame"),
        action_state=_derive_action_state(table, raw),
        health=i("health"),
        pos=(f("pos_x"), f("pos_y"), f("pos_z")),
        facing=facing,
        block_stun=raw.block_stun,
        hit_stun=raw.hit_stun,
        counter_state=CounterState(counter_name),
        throw_active=raw.throw_active,
        airborne=raw.airborne,
        juggle=raw.juggle,
        heat=HeatState(
            active=b("heat_active"),
            timer_ms=i("heat_timer_ms"),
            engager_used=b("heat_engager_used"),
        ),
        rage=b("rage"),
        input=_decode_input(source, base, fields),
    )


def decode_state(
    table: OffsetTable, phase_raw: int, mode_raw: int
) -> tuple[MatchState, str, StateSignal]:
    """Normalize the raw phase/mode codes into a ``MatchState`` and a :class:`StateSignal`."""
    phase_name = table.state_codes.match_phase.get(str(phase_raw))
    if phase_name is None:
        raise DecodeError(f"unknown match_phase code {phase_raw}")
    match_state = MatchState(phase_name)
    mode_name = table.state_codes.game_mode.get(str(mode_raw), "idle")
    return match_state, mode_name, classify_state(match_state, mode_name)


def decode_frame(source: MemorySource, table: OffsetTable) -> FrameRecord:
    """Decode one complete :class:`FrameRecord` from the source at the current instant.

    Reads the global frame counter **first** (it is the frame boundary), then the phase/mode and
    both player structs. Raises :class:`~tekken_coach.reader.faults.DecodeError` /
    :class:`~tekken_coach.reader.faults.MemoryReadError` on malformed data or unreadable memory —
    it never returns a partially-filled or guessed record.
    """
    g = table.global_struct
    gbase = resolve_anchor(source, g.anchor)

    def gi(name: str) -> int:
        spec = _field(g.fields, name)
        return _read_int(source, gbase + spec.offset, spec.kind)

    frame = gi("frame_counter")
    phase_raw = gi("match_phase")
    mode_raw = gi("game_mode")
    round_no = gi("round")
    timer_ms = gi("timer_ms")

    match_state, _mode, _signal = decode_state(table, phase_raw, mode_raw)

    players = [_decode_player(source, table, 0), _decode_player(source, table, 1)]
    return FrameRecord(
        frame=frame,
        match_state=match_state,
        round=round_no,
        timer_ms=timer_ms,
        players=players,
    )


def read_state_signal(source: MemorySource, table: OffsetTable) -> StateSignal:
    """Read just the match/replay-state signal (docs/01 §4.3) without decoding a full frame.

    Lets an armed-but-not-recording capture (docs/01 §3.1) and clean mode's online-refusal gate
    watch state flags cheaply, without paying for both player structs every poll.
    """
    g = table.global_struct
    gbase = resolve_anchor(source, g.anchor)
    phase_raw = _read_int(source, gbase + _field(g.fields, "match_phase").offset, "u32")
    mode_raw = _read_int(source, gbase + _field(g.fields, "game_mode").offset, "u32")
    _match_state, _mode, signal = decode_state(table, phase_raw, mode_raw)
    return signal


@dataclass(frozen=True)
class FrameRead:
    """One frame plus the dropped-frame accounting for it (docs/02 §7, docs/04 §4.7).

    ``gap`` is the number of frames missed since the previous read — ``frame - prev - 1``, clamped
    at 0 — matching the segmenter's own gap computation exactly (docs/04 §4.7). ``gap_note`` is the
    ``"gap-tolerated:N"`` marker string the segmenter emits into ``Interaction.notes``, or ``None``
    when no frames were dropped.
    """

    frame: FrameRecord
    gap: int
    gap_note: str | None


class FrameReader:
    """Stateful wrapper over :func:`decode_frame` that marks dropped frames (docs/02 §7).

    A single reader instance is polled repeatedly against a live source. Between polls the game
    advances; if we polled slower than 60fps the global frame counter jumps, and we surface that
    as a ``gap-tolerated:N`` marker in the stream — the exact signal the segmenter's §4.7 tolerance
    consumes. A backwards/equal frame counter (e.g. a new match resetting the counter) reports
    ``gap=0``; match-boundary handling is the segmenter's concern (docs/04 §4.8), not the reader's.
    """

    def __init__(self) -> None:
        self._prev_frame: int | None = None

    def read_frame(self, source: MemorySource, table: OffsetTable) -> FrameRead:
        record = decode_frame(source, table)
        gap = 0
        if self._prev_frame is not None:
            delta = record.frame - self._prev_frame - 1
            gap = delta if delta > 0 else 0
        self._prev_frame = record.frame
        note = f"gap-tolerated:{gap}" if gap > 0 else None
        return FrameRead(frame=record, gap=gap, gap_note=note)


def poll_frames(source: MemorySource, table: OffsetTable, count: int) -> list[FrameRead]:
    """Poll ``count`` successive frames from ``source`` through a fresh :class:`FrameReader`.

    The unit the doctor (docs/02 §6) uses to gather several frames over time. Propagates
    :class:`~tekken_coach.reader.faults.MemoryReadError` if the process becomes unreadable
    mid-poll (a process-lost fault, docs/02 §7); it does not swallow it, so callers can classify.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    reader = FrameReader()
    frames: list[FrameRead] = []
    for _ in range(count):
        frames.append(reader.read_frame(source, table))
    return frames


# ``MemoryReadError`` is re-exported for callers that catch it around a poll loop.
__all__ = [
    "FrameRead",
    "FrameReader",
    "MemoryReadError",
    "decode_frame",
    "decode_state",
    "poll_frames",
    "read_scalar",
    "read_state_signal",
    "resolve_anchor",
]
