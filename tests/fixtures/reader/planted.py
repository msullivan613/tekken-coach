"""A synthetic memory image with a *known planted layout* for the C4c discovery tests.

The re-discovery scanners/derivation must recover a ``(base, stride, {field: offset})`` layout from
raw bytes. To prove that, we plant a layout at offsets **deliberately different from the C4a
placeholder table** (the seed), encode two snapshots of a Jin-vs-Kazuya round into contiguous byte
:class:`~tekken_coach.reader.discovery.scanners.Region`\\s, and assert the derivation recovers the
planted offsets exactly. Because the planted offsets differ from the seed, a passing build+decode
round-trip proves the tool uses the *derived* layout, not the seed's.

The planted layout is collision-free with the seed's non-derived fields (derived fields live in the
free high region ≥ 0x44), so the built candidate table decodes cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

from tekken_coach.reader.discovery.scanners import Region
from tekken_coach.reader.offsets import (
    Anchor,
    FieldSpec,
    GlobalStruct,
    OffsetTable,
    PlayerStruct,
    ScalarKind,
)
from tekken_coach.schemas import (
    ActionState,
    CounterState,
    FrameRecord,
    HeatState,
    InputState,
    MatchState,
    PlayerFrame,
)
from tests.fixtures.reader.encode import encode_frame

PLANTED_MODULE = "Polaris-Win64-Shipping.exe"
PLANTED_MODULE_BASE = 0x140000000
PLANTED_STRIDE = 0x800  # 2048 — distinct from the seed's 1024
PLANTED_PLAYER_BASE_OFFSET = 0x10000  # distinct from the seed's 4096
PLANTED_GLOBAL_BASE_OFFSET = 0x200  # distinct from the seed's 0

JIN_CHAR_ID = 1
KAZUYA_CHAR_ID = 12
ROUND_START_HEALTH = 150

# The planted player layout. Derived (discoverable) fields sit in the free high region so the seed's
# non-derived fields (which the builder carries forward) never collide with them.
_PLANTED_PLAYER_OFFSETS: dict[str, tuple[int, ScalarKind]] = {
    "char_id": (0, "u32"),
    "move_frame": (8, "u32"),
    "move_id": (0x44, "u32"),
    "health": (0x48, "i32"),
    "pos_x": (0x50, "f32"),
    "pos_y": (0x54, "f32"),
    "pos_z": (0x58, "f32"),
    # non-derived fields the encoder still needs, at non-overlapping offsets:
    "facing": (0x1C, "i32"),
    "simple_state": (0x20, "u32"),
    "counter_state": (0x24, "u32"),
    "block_stun": (0x28, "bool8"),
    "hit_stun": (0x29, "bool8"),
    "stagger": (0x2A, "bool8"),
    "throw_active": (0x2B, "bool8"),
    "throw_tech": (0x2C, "bool8"),
    "thrown": (0x2D, "bool8"),
    "airborne": (0x2E, "bool8"),
    "juggle": (0x2F, "bool8"),
    "knockdown": (0x30, "bool8"),
    "wakeup": (0x31, "bool8"),
    "sidestep": (0x32, "bool8"),
    "crouch": (0x33, "bool8"),
    "heat_active": (0x34, "bool8"),
    "heat_engager_used": (0x35, "bool8"),
    "rage": (0x36, "bool8"),
    "input_valid": (0x37, "bool8"),
    "input_dir": (0x38, "u8"),
    "heat_timer_ms": (0x3C, "u32"),
    "input_buttons": (0x40, "u16"),
}

_PLANTED_GLOBAL_OFFSETS: dict[str, tuple[int, ScalarKind]] = {
    "frame_counter": (0, "u32"),
    "match_phase": (4, "u32"),
    "game_mode": (8, "u32"),
    "round": (12, "u32"),
    "timer_ms": (16, "u32"),
}


def ground_truth_table(seed: OffsetTable) -> OffsetTable:
    """The layout the derivation should recover — the seed's state_codes/sanity, planted offsets."""
    player_fields = {
        n: FieldSpec(offset=o, kind=k) for n, (o, k) in _PLANTED_PLAYER_OFFSETS.items()
    }
    global_fields = {
        n: FieldSpec(offset=o, kind=k) for n, (o, k) in _PLANTED_GLOBAL_OFFSETS.items()
    }
    return OffsetTable(
        game_version=seed.game_version,
        discovered_at=seed.discovered_at,
        notes="planted ground-truth layout for C4c discovery tests",
        global_struct=GlobalStruct(
            anchor=Anchor(module=PLANTED_MODULE, base_offset=PLANTED_GLOBAL_BASE_OFFSET),
            fields=global_fields,
        ),
        players=PlayerStruct(
            anchor=Anchor(module=PLANTED_MODULE, base_offset=PLANTED_PLAYER_BASE_OFFSET),
            stride=PLANTED_STRIDE,
            fields=player_fields,
        ),
        state_codes=seed.state_codes,
        sanity=seed.sanity,
    )


def _player(
    *, char_id: int, move_id: int, move_frame: int, pos: tuple[float, float, float], facing: int
) -> PlayerFrame:
    return PlayerFrame(
        char_id=char_id,
        move_id=move_id,
        move_frame=move_frame,
        action_state=ActionState.neutral,
        health=ROUND_START_HEALTH,
        pos=pos,
        facing=facing,
        block_stun=False,
        hit_stun=False,
        counter_state=CounterState.none,
        throw_active=False,
        airborne=False,
        juggle=False,
        heat=HeatState(active=False, timer_ms=0, engager_used=False),
        rage=False,
        input=InputState(dir=5, buttons=[]),
    )


def round_start_frame() -> FrameRecord:
    """Snapshot A: round start — both full health, both idle, P1 = Jin, P2 = Kazuya."""
    return FrameRecord(
        frame=1000,
        match_state=MatchState.in_round,
        round=1,
        timer_ms=42000,
        players=[
            _player(
                char_id=JIN_CHAR_ID, move_id=2145, move_frame=7, pos=(1.42, 0.0, -0.31), facing=1
            ),
            _player(
                char_id=KAZUYA_CHAR_ID, move_id=100, move_frame=3, pos=(-1.0, 0.0, 0.5), facing=-1
            ),
        ],
    )


def post_action_frame() -> FrameRecord:
    """Snapshot B: P1 has walked (pos_x moved) and pressed a button (move_id changed); frame ticked.

    ``move_frame`` is deliberately held constant so ``move_id`` is the unique changed plausible u32
    (move_frame also incrementing is a real ambiguity the runbook flags for calibration).
    """
    return FrameRecord(
        frame=1004,
        match_state=MatchState.in_round,
        round=1,
        timer_ms=42000,
        players=[
            _player(
                char_id=JIN_CHAR_ID, move_id=2150, move_frame=7, pos=(2.0, 0.0, -0.31), facing=1
            ),
            _player(
                char_id=KAZUYA_CHAR_ID, move_id=100, move_frame=3, pos=(-1.0, 0.0, 0.5), facing=-1
            ),
        ],
    )


def _region_from_frame(frame: FrameRecord, table: OffsetTable, base: int, size: int) -> Region:
    """Materialize a contiguous :class:`Region` at ``base`` from an encoded frame image."""
    image = encode_frame(frame, table, module_base=PLANTED_MODULE_BASE, game_mode="practice")
    buf = bytearray(size)
    for address, chunk in image.items():
        if base <= address and address + len(chunk) <= base + size:
            buf[address - base : address - base + len(chunk)] = chunk
    return Region(base=base, data=bytes(buf))


@dataclass(frozen=True)
class PlantedScan:
    """The two planted snapshots plus the ground-truth table they were encoded from."""

    player_before: Region
    player_after: Region
    global_before: Region
    global_after: Region
    ground_truth: OffsetTable


def planted_scan(seed: OffsetTable) -> PlantedScan:
    """Build both snapshots for the planted layout, ready to feed :func:`derive_layout`."""
    truth = ground_truth_table(seed)
    a = round_start_frame()
    b = post_action_frame()
    player_base = PLANTED_MODULE_BASE + PLANTED_PLAYER_BASE_OFFSET
    global_base = PLANTED_MODULE_BASE + PLANTED_GLOBAL_BASE_OFFSET
    player_size = 2 * PLANTED_STRIDE + 0x200
    global_size = 0x100
    return PlantedScan(
        player_before=_region_from_frame(a, truth, player_base, player_size),
        player_after=_region_from_frame(b, truth, player_base, player_size),
        global_before=_region_from_frame(a, truth, global_base, global_size),
        global_after=_region_from_frame(b, truth, global_base, global_size),
        ground_truth=truth,
    )
