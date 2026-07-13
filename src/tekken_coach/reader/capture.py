"""Capture a FrameRecord stream to a round-trippable JSON fixture (docs/02, docs/01 §4).

The capture tool is the bridge from a live (or fake) ``MemorySource`` to an on-disk
``FrameRecord`` stream: attach -> detect version -> select the offset table -> poll ``N`` frames
through :class:`~tekken_coach.reader.decode.FrameReader` -> serialize. It is how real
``FrameRecord`` fixtures get produced for C3's segmenter suite once C4c supplies real offsets.

The serialized form is a :class:`CaptureFile`: a small header (:class:`CaptureMeta`) plus the list
of :class:`~tekken_coach.schemas.FrameRecord`\\ s. Because ``frames`` is typed as
``list[FrameRecord]``, loading a capture re-validates every frame against the C0 schema — a
malformed frame cannot survive the round trip. Dropped-frame gaps (docs/02 §7) are recorded
per-frame in the header so the segmenter's gap accounting (docs/04 §4.7) survives serialization.

Pure vs. impure: :func:`run_capture`, :func:`capture_from_reads`, :func:`write_capture`, and
:func:`load_capture` are source-agnostic and offline-tested against a ``FakeMemorySource``. Only
:func:`capture_live` attaches to a real Windows process (user-run).

This is the **recording** side of the diagnostic/capture boundary (docs/02 §6). How the match phase
is sourced depends on the build (docs/02 §8, Stage 1 round-gating): the real T8 table *derives* it
per frame from the per-player ``frames_since_round_start`` counter (its global match_phase codes are
un-calibratable), while a legacy table with real global phase codes reads the match-state signal
through the *strict* decode first and refuses outright on uncalibrated codes — where
``decode_frame`` would merely report ``MatchState.unknown``. Describing a frame you cannot fully
read is diagnosis; writing it to disk is not.

Nothing here writes to the game — it reads frames and writes a *file* (docs/02 §2).
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from tekken_coach.reader.decode import (
    FrameRead,
    FrameReader,
    MatchPhaseTracker,
    derive_match_phase,
    poll_frames,
    read_match_flag,
    read_state_signal,
    stamp_phase,
    table_derives_round_phase,
)
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import OffsetTable
from tekken_coach.schemas import FrameRecord

# Bump only on a breaking change to the capture-file envelope (the FrameRecord schema has its own
# version in docs/03 §6; this versions the wrapper).
CAPTURE_FORMAT_VERSION = "1.0.0"

# Live poll cadence: a real game advances its frame + round counters every ~16.7 ms (60 fps), so a
# faster poll re-reads the same frame and the round deriver would never see the counter move. Only
# the live path uses it; the offline suite polls back-to-back (interval 0) on a scripted source.
LIVE_POLL_INTERVAL = 0.05


class CaptureMeta(BaseModel):
    """Header for a capture file: what produced this frame stream and its gap accounting."""

    format_version: str = CAPTURE_FORMAT_VERSION
    game_version: str  # the offset-table version the frames were decoded with (docs/02 §3)
    captured_at: str  # ISO-8601 UTC timestamp of the capture run
    frame_count: int
    # Per-frame dropped-frame count (docs/02 §7), aligned index-for-index with ``frames``: the
    # number of frames missed since the previous poll (0 for the first frame and any contiguous
    # poll). Preserved so the segmenter's §4.7 gap tolerance sees the same signal offline.
    gaps: list[int] = Field(default_factory=list)


class CaptureFile(BaseModel):
    """A serialized capture: header + the FrameRecord stream (round-trips losslessly)."""

    meta: CaptureMeta
    frames: list[FrameRecord]


def capture_from_reads(reads: list[FrameRead], *, game_version: str) -> CaptureFile:
    """Assemble a :class:`CaptureFile` from polled :class:`FrameRead`\\ s (pure)."""
    frames = [r.frame for r in reads]
    meta = CaptureMeta(
        game_version=game_version,
        captured_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        frame_count=len(frames),
        gaps=[r.gap for r in reads],
    )
    return CaptureFile(meta=meta, frames=frames)


def run_capture(
    source: MemorySource,
    table: OffsetTable,
    count: int,
    *,
    game_version: str,
    interval: float = 0.0,
) -> CaptureFile:
    """Poll ``count`` frames from ``source`` and assemble a :class:`CaptureFile`.

    Source-agnostic: works identically against a live :class:`WinMemorySource` and a
    ``FakeMemorySource`` (which is how it is offline-tested). Propagates
    :class:`~tekken_coach.reader.faults.MemoryReadError` if the process becomes unreadable mid-poll
    (docs/02 §7) — the caller classifies it via :func:`~tekken_coach.reader.faults.classify_fault`.

    Two ways the match phase is sourced (docs/02 §8, Stage 1 round-gating), by
    :func:`~tekken_coach.reader.decode.table_derives_round_phase`:

    * **derived** — on the real T8 build, whose global match_phase/game_mode are un-calibratable,
      the phase is derived per frame from the per-player ``frames_since_round_start`` counter plus
      the global ``match_flag`` word by a single
      :class:`~tekken_coach.reader.decode.MatchPhaseTracker`, and stamped over each frame's bogus
      seeded global reads. There is nothing to refuse — the phase is computed, not trusted.
    * **gated** — on a legacy table with real global phase codes, it reads the match-state signal
      once up front through the *strict* decode and refuses the whole capture if those codes are
      uncalibrated (docs/01 §4.3): a capture that cannot tell an online ranked match from a practice
      round must not write frames to disk.

    ``interval`` seconds between polls (0 = back-to-back, for the offline suite); a live capture
    must pass a non-zero interval so the game's frame + round counters advance between reads.
    """
    if table_derives_round_phase(table):
        reads = _poll_with_derived_phase(source, table, count, interval)
    else:
        read_state_signal(source, table)  # strict gate: legacy builds with real global phase codes
        reads = poll_frames(source, table, count, interval=interval)
    return capture_from_reads(reads, game_version=game_version)


def _poll_with_derived_phase(
    source: MemorySource, table: OffsetTable, count: int, interval: float
) -> list[FrameRead]:
    """Poll ``count`` frames, deriving + stamping the match phase on each (round-gating, §8).

    Threads one :class:`~tekken_coach.reader.decode.MatchPhaseTracker` across the poll loop (the
    only stateful thing) and replaces each frame's seeded global match_state/round with the
    verdict — the full ``menu``…``match_over`` phase, derived from the per-player counter plus the
    separately-read global ``match_flag``. Mirrors :func:`~tekken_coach.reader.decode.poll_frames`'s
    gap accounting.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    reader = FrameReader()
    tracker = MatchPhaseTracker(table.sanity.round_start_health)
    reads: list[FrameRead] = []
    for i in range(count):
        if interval > 0 and i > 0:
            time.sleep(interval)
        read = reader.read_frame(source, table)
        match_flag = read_match_flag(source, table)
        phase = derive_match_phase(tracker, table, read.frame, match_flag)
        reads.append(replace(read, frame=stamp_phase(read.frame, phase)))
    return reads


def write_capture(path: str | Path, capture: CaptureFile) -> None:
    """Serialize ``capture`` to ``path`` as pretty-printed JSON (docs/02 §2 — writes a *file*)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(capture.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_capture(path: str | Path) -> CaptureFile:
    """Load and re-validate a capture file — every frame is checked against the C0 schema."""
    return CaptureFile.model_validate_json(Path(path).read_text(encoding="utf-8"))


def capture_live(
    process_name: str,
    offsets_dir: str | Path,
    count: int,
    out_path: str | Path,
    *,
    version_override: str | None = None,
) -> CaptureFile:  # pragma: no cover - attaches to a live Windows process (user-run)
    """Attach to the running game, capture ``count`` frames, and write them to ``out_path``.

    Full live path: attach (read-only) -> detect version (or use ``version_override``) -> select
    the offset table (fail-closed on unknown version, docs/02 §3) -> poll -> serialize.
    Windows-only; the offline suite exercises every step *except* the attach via
    :func:`run_capture` on a fake source.
    """
    from tekken_coach.reader.offsets import select_offset_table  # noqa: PLC0415
    from tekken_coach.reader.version import detect_running_version  # noqa: PLC0415
    from tekken_coach.reader.win_source import WinMemorySource  # noqa: PLC0415

    source = WinMemorySource(process_name)
    version = version_override or detect_running_version(process_name)
    table = select_offset_table(version, offsets_dir)
    capture = run_capture(source, table, count, game_version=version, interval=LIVE_POLL_INTERVAL)
    write_capture(out_path, capture)
    return capture
