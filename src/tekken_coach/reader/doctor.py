"""Reader self-check / doctor — the sanity gate before any capture (docs/02 §6).

Before a session and after every offset update, poll a handful of frames and assert the five
conditions of docs/02 §6. A failure means the offsets are stale (a patch shifted them) — a wrong
offset silently produces garbage, so we would rather block capture than emit corrupt interactions
three matches later. The report is **data**: a failed check names the problem and points at the §4
runbook; the doctor itself prints nothing (docs/02 §2 silent-producer). C6's ``doctor`` CLI
renders it.

The five checks (docs/02 §6):

1. both character IDs resolve to known characters,
2. health reads a plausible round-start max,
3. the frame counter increases monotonically,
4. a known move yields a stable, non-garbage move id,
5. positions/distance change when the practice dummy is moved.

All five test the **mechanical core** — the anchors, the stride, the field offsets — and none of
them needs ``match_phase``. That is deliberate: the doctor goes green *incrementally*, so a build
whose phase offset is still seeded can prove its player and global anchors are right before anyone
calibrates match state. An uncalibrated phase is reported as a :attr:`DoctorReport.notes` line, not
a failed check, because it does not make any of the five answers wrong. It does stop capture — but
at the capture gate (:func:`~tekken_coach.reader.decode.read_state_signal`), which refuses an
unknown phase, not here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from tekken_coach.reader.decode import FrameRead, poll_frames
from tekken_coach.reader.faults import PATCH_RUNBOOK, MemoryReadError
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import OffsetTable
from tekken_coach.schemas import FrameRecord, MatchState

DEFAULT_DOCTOR_FRAMES = 8


@dataclass(frozen=True)
class DoctorCheck:
    """One §6 assertion's result."""

    name: str
    ok: bool
    detail: str


@dataclass
class DoctorReport:
    """The full self-check result (docs/02 §6). ``ok`` gates capture.

    ``notes`` carries what the doctor observed but does **not** gate on — today, an uncalibrated
    ``match_phase``. Folding it into ``ok`` would keep the whole reader red over a field none of the
    five checks reads, hiding the mechanical core it *can* prove.
    """

    checks: list[DoctorCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def runbook(self) -> str | None:
        """The §4 runbook when the gate failed, else ``None`` (docs/02 §6 points at §4)."""
        return None if self.ok else PATCH_RUNBOOK

    def failures(self) -> list[DoctorCheck]:
        return [c for c in self.checks if not c.ok]


def _player_distance(fr: FrameRecord) -> float:
    (ax, ay, az), (bx, by, bz) = fr.players[0].pos, fr.players[1].pos
    return math.dist((ax, ay, az), (bx, by, bz))


def _check_char_ids(frames: list[FrameRead], known_char_ids: set[int]) -> DoctorCheck:
    first = frames[0].frame
    ids = [p.char_id for p in first.players]
    unknown = [cid for cid in ids if cid not in known_char_ids]
    ok = not unknown
    detail = (
        f"both char ids resolve: {ids}"
        if ok
        else f"unknown char id(s) {unknown} not in known set — offsets likely stale"
    )
    return DoctorCheck("char_ids_known", ok, detail)


def _check_health(frames: list[FrameRead], table: OffsetTable) -> DoctorCheck:
    s = table.sanity
    first = frames[0].frame
    healths = [p.health for p in first.players]
    plausible = all(s.health_plausible_min <= h <= s.health_plausible_max for h in healths)
    at_max = all(h == s.round_start_health for h in healths)
    ok = plausible and at_max
    if ok:
        detail = f"round-start health at plausible max {s.round_start_health}: {healths}"
    elif not plausible:
        detail = (
            f"health {healths} outside plausible "
            f"[{s.health_plausible_min}, {s.health_plausible_max}] — offsets likely stale"
        )
    else:
        detail = f"health {healths} != expected round-start max {s.round_start_health}"
    return DoctorCheck("health_plausible", ok, detail)


def _check_frame_monotonic(frames: list[FrameRead]) -> DoctorCheck:
    counters = [f.frame.frame for f in frames]
    ok = all(b > a for a, b in zip(counters, counters[1:], strict=False))
    detail = (
        f"frame counter strictly increasing across {len(counters)} frames"
        if ok
        else f"frame counter not monotonic: {counters} — reads are not tracking a live process"
    )
    return DoctorCheck("frame_monotonic", ok, detail)


def _check_move_id_stable(frames: list[FrameRead], table: OffsetTable) -> DoctorCheck:
    move_max = table.sanity.move_id_max
    all_move_ids = [p.move_id for fr in frames for p in fr.frame.players]
    non_garbage = all(0 < mid < move_max for mid in all_move_ids)
    # "Stable" = some non-garbage move id persists across >= 2 consecutive frames for a player
    # (a held jab reads the same id frame-to-frame; pure noise would not).
    stable = False
    for idx in (0, 1):
        seq = [fr.frame.players[idx].move_id for fr in frames]
        if any(a == b and 0 < a < move_max for a, b in zip(seq, seq[1:], strict=False)):
            stable = True
            break
    ok = non_garbage and stable
    if ok:
        detail = "move ids in plausible range and stable across consecutive frames"
    elif not non_garbage:
        detail = f"garbage move id(s) outside (0, {move_max}) — offsets likely stale"
    else:
        detail = "no move id persisted across consecutive frames — reads look like noise"
    return DoctorCheck("move_id_stable", ok, detail)


def _check_positions_change(frames: list[FrameRead]) -> DoctorCheck:
    distances = [_player_distance(f.frame) for f in frames]
    spread = max(distances) - min(distances)
    ok = spread > 1e-6
    detail = (
        f"inter-player distance varies by {spread:.4f} across the poll"
        if ok
        else "inter-player distance never changed — positions look frozen (stale offsets)"
    )
    return DoctorCheck("positions_change", ok, detail)


def _phase_notes(frames: list[FrameRead], table: OffsetTable) -> list[str]:
    """Observations that do not gate the five checks but the user must see (docs/02 §6).

    An ``unknown`` phase means the table's ``match_phase`` offset holds a code its ``state_codes``
    map does not name — the offset is still seeded. Everything the five checks assert stays true;
    what breaks is *capture*, which reads the phase to decide whether it may record at all.
    """
    if not any(f.frame.match_state is MatchState.unknown for f in frames):
        return []
    offset = table.global_struct.fields["match_phase"].offset
    return [
        f"match_phase (+0x{offset:x}) decodes as 'unknown' — the offset is seeded, not calibrated. "
        "The checks above do not read it, so the derived anchors/offsets are still proven; but "
        "capture will REFUSE to record until the phase codes are calibrated (docs/02 §4 step 4), "
        "because a gate that cannot recognize an online match must not run."
    ]


def evaluate_frames(
    frames: Iterable[FrameRead],
    *,
    table: OffsetTable,
    known_char_ids: set[int],
) -> DoctorReport:
    """Run the five §6 checks over already-polled frames (pure; no memory access)."""
    seq = list(frames)
    if len(seq) < 2:
        raise ValueError("doctor needs at least 2 frames to check monotonicity/motion")
    checks = [
        _check_char_ids(seq, known_char_ids),
        _check_health(seq, table),
        _check_frame_monotonic(seq),
        _check_move_id_stable(seq, table),
        _check_positions_change(seq),
    ]
    return DoctorReport(checks=checks, notes=_phase_notes(seq, table))


def run_doctor(
    source: MemorySource,
    table: OffsetTable,
    *,
    known_char_ids: set[int],
    frames: int = DEFAULT_DOCTOR_FRAMES,
) -> DoctorReport:
    """Poll ``frames`` frames from ``source`` and run the §6 self-check.

    A :class:`~tekken_coach.reader.faults.MemoryReadError` while polling is itself a failed gate
    (the process was unreadable) — it is caught and reported as a failed check rather than raised,
    so a caller always gets a report to act on.
    """
    try:
        polled = poll_frames(source, table, frames)
    except MemoryReadError as exc:
        return DoctorReport(checks=[DoctorCheck("process_readable", False, str(exc))])
    return evaluate_frames(polled, table=table, known_char_ids=known_char_ids)
