"""Reader self-check / doctor — all five §6 conditions, pass + one per failure mode (docs/02 §6)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from tekken_coach.reader.doctor import DoctorReport, run_doctor
from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import FakeMemorySource, MemoryImage, MemoryRegion
from tekken_coach.reader.offsets import OffsetTable, select_offset_table
from tekken_coach.schemas import (
    ActionState,
    CounterState,
    FrameRecord,
    HeatState,
    MatchState,
    PlayerFrame,
)
from tests.fixtures.reader.encode import advance_on_for, encode_frame, module_base_for

REPO_OFFSETS = Path("assets/offsets")
MODULE_BASE = 0x140000000
KNOWN_CHAR_IDS = {12, 7}  # Kazuya, Jin (aligned with the C1 movemap)


@pytest.fixture
def table() -> OffsetTable:
    return select_offset_table("2.01.01", REPO_OFFSETS)


def _player(char_id: int, move_id: int, health: int, x: float) -> PlayerFrame:
    return PlayerFrame(
        char_id=char_id,
        move_id=move_id,
        move_frame=0,
        action_state=ActionState.neutral,
        health=health,
        pos=(x, 0.0, 0.0),
        facing=1,
        block_stun=False,
        hit_stun=False,
        counter_state=CounterState.none,
        throw_active=False,
        airborne=False,
        juggle=False,
        heat=HeatState(active=False, timer_ms=0, engager_used=False),
        rage=False,
        input=None,
    )


def _frame(frame: int, p1: PlayerFrame, p2: PlayerFrame) -> FrameRecord:
    return FrameRecord(
        frame=frame,
        match_state=MatchState.in_round,
        round=1,
        timer_ms=60000,
        players=[p1, p2],
    )


def _source(frames: list[FrameRecord], table: OffsetTable) -> FakeMemorySource:
    images: list[MemoryImage] = [
        encode_frame(fr, table, module_base=MODULE_BASE, game_mode="practice") for fr in frames
    ]
    return FakeMemorySource(
        images,
        module_bases=module_base_for(table, MODULE_BASE),
        advance_on=advance_on_for(table, MODULE_BASE),
    )


def _healthy_frames() -> list[FrameRecord]:
    """A good practice-mode capture: monotonic frames, known chars, round-start health,
    a stable non-garbage move id, and the dummy moving (distance changes)."""
    frames = []
    for i in range(6):
        p1 = _player(12, 2145, 200, 1.5)  # stationary attacker, jab held (stable move id)
        p2 = _player(7, 800, 200, -1.5 + 0.1 * i)  # dummy walking -> distance changes
        frames.append(_frame(100 + i, p1, p2))
    return frames


def _run(frames: list[FrameRecord], table: OffsetTable, n: int = 6) -> DoctorReport:
    return run_doctor(_source(frames, table), table, known_char_ids=KNOWN_CHAR_IDS, frames=n)


def test_doctor_passes_on_a_healthy_source(table: OffsetTable) -> None:
    report = _run(_healthy_frames(), table)
    assert report.ok, report.failures()
    names = {c.name for c in report.checks}
    assert names == {
        "char_ids_known",
        "health_plausible",
        "frame_monotonic",
        "move_id_stable",
        "positions_change",
    }


def _failed_names(report: DoctorReport) -> set[str]:
    return {c.name for c in report.failures()}


def test_unknown_char_id_fails_char_check(table: OffsetTable) -> None:
    frames = _healthy_frames()
    for fr in frames:
        fr.players[0].char_id = 999  # not a known character -> stale offsets
    report = _run(frames, table)
    assert not report.ok
    assert "char_ids_known" in _failed_names(report)


def test_implausible_health_fails_health_check(table: OffsetTable) -> None:
    frames = _healthy_frames()
    for fr in frames:
        fr.players[0].health = 300  # above the plausible max -> garbage read
    report = _run(frames, table)
    assert not report.ok
    assert "health_plausible" in _failed_names(report)


def test_non_monotonic_frame_counter_fails_monotonic_check(table: OffsetTable) -> None:
    frames = _healthy_frames()
    for fr in frames:
        fr.frame = 100  # counter frozen -> reads are not tracking a live process
    report = _run(frames, table)
    assert not report.ok
    assert "frame_monotonic" in _failed_names(report)


def test_garbage_move_id_fails_move_check(table: OffsetTable) -> None:
    frames = _healthy_frames()
    for fr in frames:
        fr.players[0].move_id = 70000  # above move_id_max -> garbage
    report = _run(frames, table)
    assert not report.ok
    assert "move_id_stable" in _failed_names(report)


def test_frozen_positions_fail_position_check(table: OffsetTable) -> None:
    frames = _healthy_frames()
    for fr in frames:
        fr.players[1].pos = (-1.5, 0.0, 0.0)  # dummy never moves -> distance constant
    report = _run(frames, table)
    assert not report.ok
    assert "positions_change" in _failed_names(report)


def test_process_read_error_is_reported_not_raised(table: OffsetTable) -> None:
    class DeadSource:
        def read(self, address: int, size: int) -> bytes:
            raise MemoryReadError("process closed mid-capture")

        def module_base(self, module: str) -> int:
            return MODULE_BASE

        def regions(self) -> Sequence[MemoryRegion]:
            return []

        def mapped_regions(self) -> Sequence[MemoryRegion]:
            return []

    report = run_doctor(DeadSource(), table, known_char_ids=KNOWN_CHAR_IDS, frames=6)
    assert not report.ok
    assert report.runbook is not None  # a failed gate points at the §4 runbook


def test_mechanical_failure_points_at_the_offset_runbook(table: OffsetTable) -> None:
    # A frozen dummy fails positions_change (a *mechanical* check) — that genuinely implies stale
    # offsets, so the report points at the §4 re-derivation runbook.
    frames = _healthy_frames()
    for fr in frames:
        fr.players[1].pos = (-1.5, 0.0, 0.0)  # dummy stops moving -> inter-player distance frozen
    report = _run(frames, table)
    assert "positions_change" in _failed_names(report)
    assert report.runbook is not None and "update-offsets" in report.runbook


def test_char_only_failure_gets_the_unlisted_note_not_the_stale_runbook(table: OffsetTable) -> None:
    # An unlisted character with the mechanical core all green is NOT stale offsets and NOT a
    # refused capture (the live run proved capture works in that exact state). The doctor points at
    # the lighter add-the-id note, never the "unknown game version / update-offsets" runbook.
    frames = _healthy_frames()
    for fr in frames:
        fr.players[0].char_id = 999
    report = _run(frames, table)
    assert _failed_names(report) == {"char_ids_known"}
    assert report.runbook is not None
    assert "update-offsets" not in report.runbook
    assert "known_char_ids" in report.runbook
