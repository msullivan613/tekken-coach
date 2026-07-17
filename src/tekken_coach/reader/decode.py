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
  or a table that declares an ``input_valid`` flag which reads false, yields ``input=None``, and the
  segmenter never requires it.

Two layout realities are decoded through one seam (:class:`PlayerStateFacts`). The C4c/legacy table
carries one ``bool8`` per flag; the real Tekken 8 entity struct carries a few **encoded state
words** whose integer values denote whole situations. When the table declares an
:class:`~tekken_coach.reader.offsets.EncodedStateSpec`, the value -> meaning mapping is looked up in
that **data** map (docs/02 §8) and the raw words ride out on ``PlayerFrame.raw_state`` so the map
stays debuggable. Likewise ``pos_{x,y,z}`` come from a separate transform
:class:`~tekken_coach.reader.offsets.ComponentAnchor` when the table declares one, and from plain
in-struct fields when it does not.

One boundary runs through this module and is worth naming up front: :func:`decode_frame` *describes*
a frame and :func:`read_state_signal` *decides* whether to record one. So an uncalibrated
``match_phase`` decodes to ``MatchState.unknown`` in the former (the doctor still gets to check the
mechanical core) and raises in the latter (a gate that cannot recognize an online match refuses).
:func:`decode_state` takes the ``strict`` flag that separates them.

:class:`FrameReader` wraps :func:`decode_frame` with the dropped-frame handling of docs/02 §7:
when the global frame counter jumps, it records a ``gap-tolerated:N`` marker whose ``N`` matches
the segmenter's own gap accounting exactly (docs/04 §4.7 uses ``missed = frame - prev - 1``).

Nothing here writes, injects, or renders — it only reads and returns data (docs/02 §2).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

from tekken_coach.reader.faults import DecodeError, MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import (
    POSITION_COMPONENT,
    Anchor,
    ComponentAnchor,
    FieldSpec,
    OffsetTable,
    ScalarKind,
)
from tekken_coach.reader.state import MODE_OFFLINE, StateSignal, classify_state
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


def resolve_component(source: MemorySource, player_base: int, component: ComponentAnchor) -> int:
    """Resolve a :class:`ComponentAnchor` against one player's struct base (docs/02 §3).

    Dereferences the pointer slot inside the entity struct, then takes each ``pointer_path`` hop.
    This is how ``pos_{x,y,z}`` are reached on Tekken 8, where position lives in a separate Unreal
    transform component rather than in the entity struct itself.
    """
    address = _read_int(source, player_base + component.slot_offset, "ptr")
    for offset in component.pointer_path:
        address = _read_int(source, address + offset, "ptr")
    return address


def _field(fields: dict[str, FieldSpec], name: str) -> FieldSpec:
    spec = fields.get(name)
    if spec is None:
        raise DecodeError(f"offset table is missing required field {name!r}")
    return spec


@dataclass(frozen=True)
class PlayerStateFacts:
    """The per-player situation, normalized away from *how* the game encoded it.

    Both layouts converge here before anything is derived: the C4c/legacy table's
    one-boolean-per-flag fields (:func:`_legacy_facts`), and Tekken 8's real encoded state words
    (:func:`_encoded_facts`). ``simple`` is the mutually-exclusive attack/recovery/neutral posture;
    everything else is a situational flag that may overlap. Keeping this seam means
    :func:`_derive_action_state` and the ``PlayerFrame`` fold are written once, against meaning
    rather than against a byte layout.
    """

    simple: str  # "neutral" | "attack" | "recovery"
    block_stun: bool = False
    hit_stun: bool = False
    stagger: bool = False
    throw_active: bool = False
    throw_tech: bool = False
    thrown: bool = False
    airborne: bool = False
    juggle: bool = False
    knockdown: bool = False
    wakeup: bool = False
    sidestep: bool = False
    crouch: bool = False


def _derive_action_state(facts: PlayerStateFacts) -> ActionState:
    """The *thin* ``action_state`` normalization (docs/03 §1 Notes).

    A cheap, fixed-priority fold of the state facts into the coarse ``action_state`` enum. It is
    deliberately shallow: the segmenter re-derives the truth from the raw flags carried alongside
    (docs/04 §4.1), so this only needs to be a useful hint, not authoritative. Priority runs from
    the most specific/overriding states (thrown, stuns) down to the simple attack/recovery/neutral
    posture the game itself exposes (docs/02 §2).
    """
    if facts.thrown:
        return ActionState.thrown
    if facts.throw_tech:
        return ActionState.throw_tech_window
    if facts.block_stun:
        return ActionState.blockstun
    if facts.hit_stun:
        return ActionState.hitstun
    if facts.stagger:
        return ActionState.stagger
    if facts.knockdown:
        return ActionState.knockdown
    if facts.wakeup:
        return ActionState.wakeup
    if facts.airborne:
        return ActionState.airborne
    if facts.sidestep:
        return ActionState.sidestep
    if facts.simple == "attack":
        return ActionState.attack
    if facts.simple == "recovery":
        return ActionState.recovery
    if facts.crouch:
        return ActionState.crouch
    return ActionState.neutral


# The three mutually-exclusive postures, in the priority a flag union resolves them by: a state that
# implies both "attack" and "recovery" (a move's active frames bleeding into recovery) reads as the
# more specific "attack".
_SIMPLE_PRIORITY: tuple[str, ...] = ("attack", "recovery", "neutral")


def _facts_from_flags(flags: set[str]) -> PlayerStateFacts:
    """Fold a union of :data:`~tekken_coach.reader.offsets.STATE_FLAGS` into the facts record."""
    simple = next((name for name in _SIMPLE_PRIORITY if name in flags), "neutral")
    return PlayerStateFacts(
        simple=simple,
        block_stun="block_stun" in flags,
        hit_stun="hit_stun" in flags,
        stagger="stagger" in flags,
        throw_active="throw_active" in flags,
        throw_tech="throw_tech" in flags,
        thrown="thrown" in flags,
        airborne="airborne" in flags,
        juggle="juggle" in flags,
        knockdown="knockdown" in flags,
        wakeup="wakeup" in flags,
        sidestep="sidestep" in flags,
        crouch="crouch" in flags,
    )


def _legacy_facts(
    source: MemorySource, table: OffsetTable, base: int, fields: dict[str, FieldSpec]
) -> PlayerStateFacts:
    """Read the C4c/legacy layout: one ``bool8`` per flag plus a ``simple_state`` code."""

    def b(name: str) -> bool:
        spec = _field(fields, name)
        return bool(read_scalar(source, base + spec.offset, "bool8"))

    simple_spec = _field(fields, "simple_state")
    simple_raw = _read_int(source, base + simple_spec.offset, simple_spec.kind)
    simple = table.state_codes.simple_state.get(str(simple_raw), "neutral")
    return PlayerStateFacts(
        simple=simple,
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


def _encoded_facts(
    source: MemorySource, table: OffsetTable, base: int, fields: dict[str, FieldSpec]
) -> tuple[PlayerStateFacts, dict[str, int]]:
    """Read Tekken 8's encoded state words and map them to facts (docs/02 §3/§8, C4e Phase 2).

    Each mapped field holds an integer denoting a whole situation; the data map
    (:class:`~tekken_coach.reader.offsets.EncodedStateSpec`) says which flags each value implies,
    and the union across fields is the frame's state. An **unmapped** value contributes nothing —
    the raw integers are returned alongside and ride out on ``PlayerFrame.raw_state`` (docs/03 §1),
    which is how the calibration protocol (docs/02 §8) discovers what it still has to map.
    """
    spec = table.state_codes.encoded_state
    assert spec is not None  # only called when the table declares an encoded-state map
    flags: set[str] = set()
    raw_state: dict[str, int] = {}
    for name, codes in spec.flags.items():
        field_spec = _field(fields, name)
        raw = _read_int(source, base + field_spec.offset, field_spec.kind)
        raw_state[name] = raw
        flags.update(codes.get(str(raw), ()))
    return _facts_from_flags(flags), raw_state


def _read_position(
    source: MemorySource, table: OffsetTable, base: int
) -> tuple[float, float, float]:
    """Read ``pos_{x,y,z}``, from the transform component when the table declares one (§3).

    Tekken 8 keeps position outside the entity struct, behind the struct's own pointer to an Unreal
    transform component; the C4c/legacy tables keep it as plain in-struct fields. The table says
    which world it is in, and the decoder follows.
    """
    component = table.players.components.get(POSITION_COMPONENT)
    if component is None:
        fields = table.players.fields
        return tuple(  # type: ignore[return-value]
            _read_float(source, base + _field(fields, axis).offset, _field(fields, axis).kind)
            for axis in ("pos_x", "pos_y", "pos_z")
        )
    cbase = resolve_component(source, base, component)
    return tuple(  # type: ignore[return-value]
        _read_float(
            source,
            cbase + _field(component.fields, axis).offset,
            _field(component.fields, axis).kind,
        )
        for axis in ("pos_x", "pos_y", "pos_z")
    )


_BUTTON_BITS: tuple[tuple[int, str], ...] = ((0, "1"), (1, "2"), (2, "3"), (3, "4"))


def _decode_input(
    source: MemorySource, base: int, fields: dict[str, FieldSpec]
) -> InputState | None:
    """Read the optional per-frame input (docs/03 §1).

    Returns ``None`` when the table has no ``input_dir``/``input_buttons``, or when it declares an
    ``input_valid`` flag that reads false — inputs are not always resolvable (e.g. during replay
    playback), and the segmenter must not require them.

    ``input_valid`` is **optional**: a table that omits it decodes input unconditionally. Requiring
    it is what silently killed the 5.02.01 read — the seeded ``input_valid@55`` was a fork-era
    leftover that reads false forever, so every frame decoded to ``None`` no matter how good the
    other two offsets were. A validity gate is only worth having if a real one is found; binding the
    decode to a bogus field is strictly worse than not gating at all.
    """
    if "input_dir" not in fields or "input_buttons" not in fields:
        return None
    if "input_valid" in fields and not read_scalar(
        source, base + _field(fields, "input_valid").offset, "bool8"
    ):
        return None
    dir_spec = _field(fields, "input_dir")
    btn_spec = _field(fields, "input_buttons")
    direction = _read_int(source, base + dir_spec.offset, dir_spec.kind)
    mask = _read_int(source, base + btn_spec.offset, btn_spec.kind)
    buttons = [name for bit, name in _BUTTON_BITS if mask & (1 << bit)]
    return InputState(dir=direction, buttons=buttons)


def resolve_player_base(source: MemorySource, table: OffsetTable, index: int) -> int:
    """Resolve one player's struct base under whichever addressing model the table uses (§3).

    * ``player_slots`` (C4i holder model): :attr:`~.offsets.PlayerStruct.anchor` resolves the holder
      object and the player base is dereferenced from that player's slot (``holder+0x30`` /
      ``holder+0x38`` — separate allocations).
    * ``stride`` (C4c/C4d array model): the anchor resolves P1 and P2 sits ``index * stride`` later.

    The schema guarantees exactly one model is set (:meth:`PlayerStruct._one_addressing_model`), so
    the ``player_slots`` branch is taken iff the table carries per-player slots.
    """
    players = table.players
    if players.player_slots:
        holder_base = resolve_anchor(source, players.anchor)
        return resolve_component(source, holder_base, players.player_slots[index])
    assert players.stride is not None  # validated: stride set when there are no player_slots
    return resolve_anchor(source, players.anchor) + index * players.stride


def _decode_player(source: MemorySource, table: OffsetTable, index: int) -> PlayerFrame:
    players = table.players
    fields = players.fields
    base = resolve_player_base(source, table, index)

    def i(name: str) -> int:
        spec = _field(fields, name)
        return _read_int(source, base + spec.offset, spec.kind)

    def b(name: str) -> bool:
        spec = _field(fields, name)
        return bool(read_scalar(source, base + spec.offset, "bool8"))

    # Two state layouts converge on PlayerStateFacts: T8's encoded state words when the table
    # carries a value -> meaning map, else the C4c/legacy one-boolean-per-flag fields (docs/02 §3).
    raw_state: dict[str, int] | None = None
    if table.state_codes.encoded_state is not None:
        facts, raw_state = _encoded_facts(source, table, base, fields)
    else:
        facts = _legacy_facts(source, table, base, fields)

    counter_raw = i("counter_state")
    counter_name = table.state_codes.counter_state.get(str(counter_raw), CounterState.none.value)

    facing_raw = i("facing")
    facing = 1 if facing_raw >= 0 else -1

    # Health: computed from damage_taken when the table says so (T8's struct has no direct HP field,
    # docs/02 §3), else a direct field read (C4c/legacy). Clamped to [0, max_health].
    if players.max_health is not None:
        health = max(0, players.max_health - i("damage_taken"))
    else:
        health = i("health")

    # The per-round frame counter (docs/02 §8, Stage 1 round-gating) — present on the real T8 table,
    # absent on the legacy layout (defaults to 0 there, which the phase deriver never consults since
    # legacy tables keep the calibrated global match_phase instead).
    counter = i("frames_since_round_start") if "frames_since_round_start" in fields else 0

    return PlayerFrame(
        char_id=i("char_id"),
        move_id=i("move_id"),
        move_frame=i("move_frame"),
        action_state=_derive_action_state(facts),
        health=health,
        pos=_read_position(source, table, base),
        facing=facing,
        frames_since_round_start=counter,
        block_stun=facts.block_stun,
        hit_stun=facts.hit_stun,
        counter_state=CounterState(counter_name),
        throw_active=facts.throw_active,
        airborne=facts.airborne,
        juggle=facts.juggle,
        heat=HeatState(
            active=b("heat_active"),
            timer_ms=i("heat_timer_ms"),
            engager_used=b("heat_engager_used"),
        ),
        rage=b("rage"),
        input=_decode_input(source, base, fields),
        raw_state=raw_state,
    )


def decode_state(
    table: OffsetTable, phase_raw: int, mode_raw: int, *, strict: bool
) -> tuple[MatchState, str, StateSignal]:
    """Normalize the raw phase/mode codes into a ``MatchState`` and a :class:`StateSignal`.

    ``strict`` is the **diagnostic/capture boundary** (docs/01 §4.3, docs/02 §6), and every caller
    states which side it is on:

    * ``strict=False`` — *describe* the frame. An unrecognized ``match_phase`` code becomes
      :attr:`~tekken_coach.schemas.MatchState.unknown` rather than raising, so
      :func:`decode_frame` completes and the doctor can validate the mechanical core (char ids,
      health, frame monotonicity, move ids, positions) on a build whose ``match_phase`` offset has
      not been calibrated. Tolerating the code is *not* trusting it: ``unknown`` is not an active
      phase, so it never reads as "a match is being played".
    * ``strict=True`` — *decide* whether to record. :func:`read_state_signal`, the gate clean mode's
      online-refusal and live mode's arm/disarm consult, refuses an unknown phase outright: a gate
      that cannot tell an online ranked match from a practice round must not run.

    ``game_mode`` has no strict/tolerant split — an unmapped mode already falls back to ``"idle"``,
    which is refusal-shaped for both gates (never online, never buffering).
    """
    phase_name = table.state_codes.match_phase.get(str(phase_raw))
    if phase_name is None:
        if strict:
            raise DecodeError(f"unknown match_phase code {phase_raw}")
        phase_name = MatchState.unknown.value
    match_state = MatchState(phase_name)
    mode_name = table.state_codes.game_mode.get(str(mode_raw), "idle")
    return match_state, mode_name, classify_state(match_state, mode_name)


def decode_frame(source: MemorySource, table: OffsetTable) -> FrameRecord:
    """Decode one complete :class:`FrameRecord` from the source at the current instant.

    Reads the global frame counter **first** (it is the frame boundary), then the phase/mode and
    both player structs. Raises :class:`~tekken_coach.reader.faults.DecodeError` /
    :class:`~tekken_coach.reader.faults.MemoryReadError` on malformed data or unreadable memory —
    it never returns a partially-filled or guessed record.

    An unrecognized ``match_phase`` code is the one thing it does **not** treat as malformed: it
    decodes to :attr:`~tekken_coach.schemas.MatchState.unknown` (``strict=False``) so a build whose
    phase offset is still seeded can be read and diagnosed at all. Deciding whether to *record* such
    a frame is :func:`read_state_signal`'s job, and it refuses.
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

    match_state, _mode, _signal = decode_state(table, phase_raw, mode_raw, strict=False)

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

    **Strict, deliberately.** This is the path that decides whether to record, so an unrecognized
    ``match_phase`` raises :class:`~tekken_coach.reader.faults.DecodeError` rather than decoding to
    :attr:`~tekken_coach.schemas.MatchState.unknown`. :func:`decode_frame` tolerates that code so
    the doctor can still validate the mechanical core on an uncalibrated build; a *diagnostic*
    tolerating a phase it cannot read must never loosen the gate that keeps clean mode off a ranked
    match (docs/01 §4.3 defense-in-depth).
    """
    g = table.global_struct
    gbase = resolve_anchor(source, g.anchor)
    phase_raw = _read_int(source, gbase + _field(g.fields, "match_phase").offset, "u32")
    mode_raw = _read_int(source, gbase + _field(g.fields, "game_mode").offset, "u32")
    _match_state, _mode, signal = decode_state(table, phase_raw, mode_raw, strict=True)
    return signal


def read_match_flag(source: MemorySource, table: OffsetTable) -> int:
    """Read the global ``match_flag`` word cheaply (Stage 2 round-gating, docs/02 §8).

    The in-stage-vs-menu gate input for :class:`MatchPhaseTracker`: the ``@0xd444`` global word
    that *holds* a single (mode-dependent) value for the whole time a stage is loaded and *churns*
    through low UI values in a menu. Read like :func:`read_state_signal`'s global fields, off the
    same anchor as ``frame_counter``. Like the strict signal it is an internal gate input — it is
    **not** persisted on :class:`~tekken_coach.schemas.FrameRecord`.
    """
    g = table.global_struct
    gbase = resolve_anchor(source, g.anchor)
    spec = _field(g.fields, "match_flag")
    return _read_int(source, gbase + spec.offset, spec.kind)


# ---------------------------------------------------------------------------
# Derived round phase (Stage 1 round-gating, docs/02 §8)
# ---------------------------------------------------------------------------
#
# The real Tekken 8 build exposes no usable *global* match-phase enum — the seeded global
# match_phase/game_mode offsets read a stale constant and an inert word (project memory
# ``capture-round-gating-deferred``). But every player struct carries ``frames_since_round_start``,
# a per-round frame counter that resets to ~0 at each round start and rises at 60 fps during play.
# :class:`RoundPhaseTracker` *derives* the match phase + round index from that counter (plus each
# player's damage for the KO/round-decided edge), replacing the bogus global reads. This is the one
# stateful thing in the read path: :func:`decode_frame` stays mechanical, and the capture sources
# thread a single tracker across polls.

# A per-round counter drop larger than this marks a round reset. Within a round the counter only
# rises (to ~1500); observed resets drop by well over a thousand (e.g. 1381 -> 2), so 50 cleanly
# separates a round boundary from a mere poll gap or a paused (frozen) counter.
ROUND_RESET_DROP = 50


@dataclass(frozen=True)
class DerivedPhase:
    """The round phase derived from the per-player frame counter (Stage 1 round-gating).

    ``match_state`` is one of :attr:`~tekken_coach.schemas.MatchState.pre_round` /
    ``in_round`` / ``round_over`` — never ``match_over`` (menu/results/match-over detection is
    Stage 2, keyed on the in-match flag, not derivable from this counter). ``round`` is the 1-based
    round index, incremented on each detected reset.
    """

    match_state: MatchState
    round: int


class RoundPhaseTracker:
    """Derive the match phase + round index from ``frames_since_round_start`` (docs/02 §8).

    Fed successive frames' counter and both players' damage, it emits a :class:`DerivedPhase` per
    frame. It needs the *previous* counter to spot a reset, so it is stateful — but it is the only
    stateful thing here, and the mechanical :func:`decode_frame` stays pure.

    The rules (calibrated by observation, offline Bryan vs Paul on 5.02.01):

    * **reset** — the first frame, or the counter dropping by more than :data:`ROUND_RESET_DROP` —
      begins a new round: increment ``round``, clear the round-over latch, emit ``pre_round``. The
      KO check is skipped on the boundary frame, where damage is mid-reset/stale.
    * **round over** — either player's damage reaching ``round_start_health`` (a KO) latches
      ``round_over`` until the next reset. The round winner's damage never approaches the threshold,
      so it separates them cleanly. A frozen counter (a pause) is *not* a reset, so it never
      misfires a boundary.

    It deliberately cannot tell a results screen from a fresh round: after the final KO the counter
    resets and climbs again just like a new round, so a post-match reset reads as a (spurious) new
    round start. Distinguishing that is Stage 2 (the in-match flag), not derivable from the counter.
    """

    def __init__(self, round_start_health: int) -> None:
        self._ko_threshold = round_start_health
        self._prev_counter: int | None = None
        self._round = 0
        self._round_over = False

    def update(self, counter: int, p1_damage: int, p2_damage: int) -> DerivedPhase:
        """Advance the tracker by one frame and return the derived phase for it."""
        reset = self._prev_counter is None or (self._prev_counter - counter > ROUND_RESET_DROP)
        self._prev_counter = counter
        if reset:
            self._round += 1
            self._round_over = False
            return DerivedPhase(MatchState.pre_round, self._round)
        if p1_damage >= self._ko_threshold or p2_damage >= self._ko_threshold:
            self._round_over = True
        state = MatchState.round_over if self._round_over else MatchState.in_round
        return DerivedPhase(state, self._round)


def table_derives_round_phase(table: OffsetTable) -> bool:
    """Whether this build's match phase must be *derived* from the per-player counter (Stage 1).

    True when the player struct exposes ``frames_since_round_start`` — the real T8 holder builds,
    whose global match_phase/game_mode are un-calibratable. False for the legacy tables that carry
    real global phase codes, which keep the :func:`read_state_signal` global path unchanged.
    """
    return "frames_since_round_start" in table.players.fields


def derive_phase(
    tracker: RoundPhaseTracker, table: OffsetTable, frame: FrameRecord
) -> DerivedPhase:
    """Derive one frame's round phase by feeding the counter + both damages to ``tracker``.

    The counter is mirrored on both players, so P1's is read. Damage is reconstructed from the
    decoded (computed) health — ``round_start_health - health`` equals ``damage_taken`` below the
    KO threshold and saturates at it once a player is KO'd, which is exactly what the KO edge needs.
    """
    hp = table.sanity.round_start_health
    p0, p1 = frame.players
    return tracker.update(p0.frames_since_round_start, hp - p0.health, hp - p1.health)


def phase_signal(phase: DerivedPhase) -> StateSignal:
    """Build the capture :class:`StateSignal` from a derived round phase (Stage 1 round-gating).

    ``game_mode`` is **user-driven** — we do not detect online vs. offline — so ``online`` is always
    False and an active round reads as a live match. Menu/results/idle detection is Stage 2 (the
    in-match flag), so Stage 1 cannot yet report ``idle`` on the results screen.
    """
    return classify_state(phase.match_state, MODE_OFFLINE)


def stamp_phase(frame: FrameRecord, phase: DerivedPhase) -> FrameRecord:
    """Return ``frame`` with its ``match_state`` + ``round`` replaced by the derived phase (§8).

    The seeded global match_phase/round reads are bogus on the real build; the capture path stamps
    the tracker's verdict over them so the persisted FrameRecord carries a correct phase and round.
    """
    return frame.model_copy(update={"match_state": phase.match_state, "round": phase.round})


# ---------------------------------------------------------------------------
# Derived match phase (Stage 2 round-gating, docs/02 §8)
# ---------------------------------------------------------------------------
#
# Stage 1 (:class:`RoundPhaseTracker`) derives the round *arc* (pre_round/in_round/round_over) from
# the per-player counter but deliberately never emits ``match_over``/``menu``: the counter resets
# identically for a new round and for the post-match results screen, so a match-end is not derivable
# from it alone. Stage 2 adds the missing edge from a second signal — the **global** ``match_flag``
# word (``@0xd444``, project memory ``capture-round-gating-deferred``). Its observed behavior is
# noisier than a clean enum, and the gate is built around that shape (re-mined from ``debug/
# phase.jsonl``, a menu -> practice -> menu -> VS-5-round -> results -> menu arc, offline, 5.02.01):
#
# * In a **menu** it CHURNS — changes on nearly every poll, cycling low values (16/18/40/44/56...).
# * In a **loaded stage** it HOLDS one value for the whole match (all rounds *and* results): VS held
#   73 for 193 s; practice held 127. The held value is MODE-DEPENDENT and not even constant within a
#   stage (practice jumped 73 -> 127 mid-session then re-held), so the gate is **value-agnostic**:
#   it keys on hold-vs-churn, never on a specific value. (The held value in ranked / replay is
#   unconfirmed — no capture exists — so this stage ships best-effort pending a live check.)
#
# Two real false-positives the gate must survive, both present in the capture: a ~37 s stable hold
# at 40 in the pre-match setup menus (defeated by *arming on a real round* — none ran there), and
# the single 73 -> 127 substate change inside practice (defeated by requiring *churn* — many changes
# over a window — with a debounce a lone change can't meet).

# Thresholds as elapsed-poll counts (the live capture cadence is 0.05 s, DEFAULT_POLL_INTERVAL, so
# these map to ~0.5-1.0 s of wall time; the calibration probe polled slower, at ~0.25 s).
#
# Consecutive unchanged polls before ``match_flag`` counts as a loaded-stage HOLD (~0.75 s).
STAGE_HOLD_POLLS = 15
# Sliding window over which flag changes are counted to detect menu CHURN (~1.0 s).
MENU_CHURN_POLLS = 20
# Flag changes within that window required to declare churn (leave the stage). The leave debounce:
# a lone substate change (practice 73 -> 127) contributes a single change and can never reach it, so
# it does not flip the stage off; only a real return-to-menu churns enough.
MENU_LEAVE_CHANGES = 3
# The per-round counter must climb by at least this since the previous in-stage poll for a reported
# ``in_round`` to ARM a match. A real round's counter rises at 60 fps; a menu whose counter is idle
# (frozen at 0 or a stale value) never advances, so a menu hold never arms — this is what makes the
# 37 s menu-40 hold a non-event, and it is robust to *whatever* stale value the menu counter holds.
ARMING_COUNTER_ADVANCE = 1


class MatchPhaseTracker:
    """Derive the full match phase (menu … match_over) from the counter + ``match_flag`` (§8).

    Owns a :class:`RoundPhaseTracker` for the round arc and layers the menu/match-over edges on top,
    consuming ``(counter, p1_damage, p2_damage, match_flag)`` per poll and emitting a
    :class:`DerivedPhase` whose ``match_state`` now spans the full
    ``{menu, pre_round, in_round, round_over, match_over}`` set. It is the one stateful thing in the
    read path; :func:`decode_frame` stays mechanical.

    The rules (all calibrated by observation — see the module notes above):

    * **hold vs churn** — ``match_flag`` holding one value for ``STAGE_HOLD_POLLS`` polls means a
      loaded stage (``in_stage``); ``MENU_LEAVE_CHANGES`` changes within the last
      ``MENU_CHURN_POLLS`` polls means a menu (leave the stage). Between the two the previous
      verdict holds (hysteresis), which is what lets a lone substate change ride through without
      flipping the stage off.
    * **arm on a real round** — ``armed`` becomes True the first time, while ``in_stage``, the owned
      :class:`RoundPhaseTracker` reports ``in_round`` *and* the counter actually advanced (the round
      clock is running). A menu whose counter is idle never advances, so a menu hold — even the 37 s
      hold at 40 — never arms and is never a match.
    * **emit** — not armed → ``menu``; armed and in-stage → the round arc stamped through; armed and
      the stage unloads (in_stage -> menu churn) → ``match_over`` exactly once, then disarm and read
      ``menu`` until the next arm.

    A fresh :class:`RoundPhaseTracker` is started on each stage load, so the round index restarts at
    1 per match rather than climbing across the session.

    **Timing:** ``match_over`` fires on **stage-unload** (results -> menu), *not* at the match-point
    KO — the round-win target (first-to-2/-3/…) varies by mode/settings and is not reliably known,
    whereas stage-unload is a clean, mode-agnostic edge and is exactly the between-matches downtime
    C6's coach wants. Firing earlier is out of scope.
    """

    def __init__(self, round_start_health: int) -> None:
        self._round_start_health = round_start_health
        # match_flag hold/churn detection.
        self._prev_flag: int | None = None
        self._hold_run = 0  # consecutive polls the flag has held its current value
        self._recent_changes: list[bool] = []  # last MENU_CHURN_POLLS polls: did the flag change?
        self._in_stage = False
        # Round arc + arming (reset per stage load).
        self._rounds: RoundPhaseTracker | None = None
        self._prev_counter: int | None = None
        self._armed = False
        self._round = 0  # last derived round index, for the match_over/menu emit

    def update(self, counter: int, p1_damage: int, p2_damage: int, match_flag: int) -> DerivedPhase:
        """Advance the tracker by one poll and return the derived match phase for it."""
        in_stage = self._update_stage(match_flag)

        if not in_stage:
            # Menu churn (or a menu hold we never armed on). Tear down the per-stage round state so
            # the next stage load starts a fresh match (round index restarts at 1). If we were
            # mid-match, the stage just unloaded: emit match_over exactly once on this edge, then
            # idle in the menu until the next arm.
            was_armed = self._armed
            self._reset_stage()
            if was_armed:
                return DerivedPhase(MatchState.match_over, self._round)
            return DerivedPhase(MatchState.menu, 0)

        # In a loaded stage. Run the round arc off a per-stage RoundPhaseTracker and arm on a real
        # round (in_round with the counter actually advancing — a menu's idle counter never does).
        if self._rounds is None:
            self._rounds = RoundPhaseTracker(self._round_start_health)
        phase = self._rounds.update(counter, p1_damage, p2_damage)
        advanced = (
            self._prev_counter is not None
            and counter - self._prev_counter >= ARMING_COUNTER_ADVANCE
        )
        self._prev_counter = counter
        if phase.match_state is MatchState.in_round and advanced:
            self._armed = True
        if not self._armed:
            return DerivedPhase(MatchState.menu, 0)
        self._round = phase.round
        return phase

    def _update_stage(self, match_flag: int) -> bool:
        """Fold one ``match_flag`` reading into the hold-vs-churn verdict (``in_stage``)."""
        changed = self._prev_flag is not None and match_flag != self._prev_flag
        self._prev_flag = match_flag
        self._hold_run = 1 if changed else self._hold_run + 1
        self._recent_changes.append(changed)
        if len(self._recent_changes) > MENU_CHURN_POLLS:
            self._recent_changes.pop(0)
        changes = sum(self._recent_changes)
        # Hold wins over churn: right after a stage loads the window still holds the pre-load menu
        # churn, but a long enough unchanged run is a stage regardless.
        if self._hold_run >= STAGE_HOLD_POLLS:
            self._in_stage = True
        elif changes >= MENU_LEAVE_CHANGES:
            self._in_stage = False
        # else: neither condition met -> keep the previous verdict (hysteresis across the debounce).
        return self._in_stage

    def _reset_stage(self) -> None:
        """Tear down the per-match round state so the next stage load starts a fresh match.

        Idempotent — run on every not-in-stage poll — so a stage left *un-armed* (a menu hold that
        never became a match) also drops its round tracker, and the next real stage restarts at 1.
        """
        self._armed = False
        self._rounds = None
        self._prev_counter = None


def derive_match_phase(
    tracker: MatchPhaseTracker, table: OffsetTable, frame: FrameRecord, match_flag: int
) -> DerivedPhase:
    """Derive one frame's full match phase, feeding the counter + damages + flag to ``tracker``.

    The Stage 2 counterpart of :func:`derive_phase`: same counter/damage reconstruction (the counter
    is mirrored on both players, so P1's is read; damage is ``round_start_health - health``), plus
    the separately-read global ``match_flag`` (:func:`read_match_flag`) that gates in-stage vs menu.
    """
    hp = table.sanity.round_start_health
    p0, p1 = frame.players
    return tracker.update(p0.frames_since_round_start, hp - p0.health, hp - p1.health, match_flag)


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


def poll_frames(
    source: MemorySource, table: OffsetTable, count: int, *, interval: float = 0.0
) -> list[FrameRead]:
    """Poll ``count`` successive frames from ``source`` through a fresh :class:`FrameReader`.

    The unit the doctor (docs/02 §6) uses to gather several frames over time. Propagates
    :class:`~tekken_coach.reader.faults.MemoryReadError` if the process becomes unreadable
    mid-poll (a process-lost fault, docs/02 §7); it does not swallow it, so callers can classify.

    ``interval`` sleeps that many seconds *between* reads. It defaults to 0 (back-to-back) for the
    offline suite, whose scripted source advances one frame per read; against a **live** process it
    must be non-zero, because a real game only updates its frame counter every ~16.7 ms (60 fps), so
    reads faster than that all observe the *same* frame and the doctor's monotonic-frame / motion
    checks would falsely fail on a perfectly live game.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    reader = FrameReader()
    frames: list[FrameRead] = []
    for i in range(count):
        if interval > 0 and i > 0:
            time.sleep(interval)
        frames.append(reader.read_frame(source, table))
    return frames


# ``MemoryReadError`` is re-exported for callers that catch it around a poll loop.
__all__ = [
    "ARMING_COUNTER_ADVANCE",
    "MENU_CHURN_POLLS",
    "MENU_LEAVE_CHANGES",
    "ROUND_RESET_DROP",
    "STAGE_HOLD_POLLS",
    "DerivedPhase",
    "FrameRead",
    "FrameReader",
    "MatchPhaseTracker",
    "MemoryReadError",
    "PlayerStateFacts",
    "RoundPhaseTracker",
    "decode_frame",
    "decode_state",
    "derive_match_phase",
    "derive_phase",
    "phase_signal",
    "poll_frames",
    "read_match_flag",
    "read_scalar",
    "read_state_signal",
    "resolve_anchor",
    "resolve_component",
    "resolve_player_base",
    "stamp_phase",
    "table_derives_round_phase",
]
