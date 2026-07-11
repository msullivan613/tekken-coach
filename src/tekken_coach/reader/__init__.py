"""Read-only Tekken 8 memory reader — offline core (docs/02). Chunk C4a.

Turns Tekken 8 process memory into ``FrameRecord``s (docs/03 §1) behind a read-only
:class:`~tekken_coach.reader.memory_source.MemorySource` seam. The C4a half is the
environment-independent core: the seam + fake source, the ``assets/offsets/`` format and loader,
version selection
with fail-closed behavior, the raw-bytes -> ``FrameRecord`` decoder, the doctor self-check, and the
structured failure/state signals. **C4b** adds the OS-facing half — the concrete Windows
:class:`~tekken_coach.reader.win_source.WinMemorySource` (pymem-backed, read+query handle only),
live version detection, and the ``capture``/``doctor``/``smoke`` command entry points — all behind
the same seam. The ``update-offsets`` re-discovery tool is **C4c**.

pymem is an optional (``windows`` extra) dependency: it is imported lazily, so this package imports
and the offline suite runs with pymem absent (docs/02 §3 posture); only *constructing* a
:class:`~tekken_coach.reader.win_source.WinMemorySource` needs it.

Read-only, by construction (docs/02 §2, §5)
-------------------------------------------
There is **no memory-write or input-injection primitive anywhere in this package** — not unused,
not commented out, absent. The entire bot/input half of the ancestral TekkenBot is not ported. The
``MemorySource`` seam exposes only ``read`` + ``module_base``; nothing downstream can name a write.
``tests/test_reader_readonly.py`` greps this package and fails on any write/inject token.

Licensing posture (docs/02 §5) — not legal advice
--------------------------------------------------
* **Offsets/addresses/AOB signatures are facts/data**, not copyrightable expression: they live as
  data under ``assets/offsets/`` and are used freely.
* This decoder is an **original, read-only implementation** that uses the Tekken 8 memory-layout
  *knowledge* surfaced by the MIT-root ``WAZAAAAA0/TekkenBot`` (© roguelike2d, 2017); credited in
  ``NOTICE`` / ``THIRD_PARTY_LICENSES``. No net-new source of the unlicensed ``dcep93`` fork is
  copied.
* ``update-offsets`` (C4c, :mod:`tekken_coach.reader.discovery`) is a **clean-room**
  re-implementation of the Jin-vs-Kazuya re-discovery *technique* (an idea, not protected) — it does
  not read or copy the fork's ``update_memory_address.py``.
"""

from tekken_coach.reader.capture import (
    CaptureFile,
    CaptureMeta,
    capture_from_reads,
    load_capture,
    run_capture,
    write_capture,
)
from tekken_coach.reader.decode import (
    FrameRead,
    FrameReader,
    decode_frame,
    poll_frames,
    read_state_signal,
)
from tekken_coach.reader.doctor import DoctorCheck, DoctorReport, run_doctor
from tekken_coach.reader.faults import (
    PATCH_RUNBOOK,
    FaultKind,
    MemoryReadError,
    OffsetTableError,
    ReaderError,
    ReaderFault,
    UnknownGameVersionError,
    classify_fault,
)
from tekken_coach.reader.memory_source import FakeMemorySource, MemoryImage, MemorySource
from tekken_coach.reader.offsets import (
    OffsetIndex,
    OffsetTable,
    load_offset_index,
    select_offset_table,
)
from tekken_coach.reader.probe import (
    ChangeRecord,
    PollSample,
    build_skeleton,
    change_records,
    distinct_values,
)
from tekken_coach.reader.state import SignalKind, StateSignal, classify_state
from tekken_coach.reader.version import (
    detect_running_version,
    normalize_version,
    version_from_dwords,
)
from tekken_coach.reader.win_source import GAME_PROCESS_NAME, WinMemorySource, map_pymem_error

__all__ = [
    "GAME_PROCESS_NAME",
    "PATCH_RUNBOOK",
    "CaptureFile",
    "CaptureMeta",
    "ChangeRecord",
    "DoctorCheck",
    "DoctorReport",
    "FakeMemorySource",
    "FaultKind",
    "FrameRead",
    "FrameReader",
    "MemoryImage",
    "MemoryReadError",
    "MemorySource",
    "OffsetIndex",
    "OffsetTable",
    "OffsetTableError",
    "PollSample",
    "ReaderError",
    "ReaderFault",
    "SignalKind",
    "StateSignal",
    "UnknownGameVersionError",
    "WinMemorySource",
    "build_skeleton",
    "capture_from_reads",
    "change_records",
    "classify_fault",
    "classify_state",
    "decode_frame",
    "detect_running_version",
    "distinct_values",
    "load_capture",
    "load_offset_index",
    "map_pymem_error",
    "normalize_version",
    "poll_frames",
    "read_state_signal",
    "run_capture",
    "run_doctor",
    "select_offset_table",
    "version_from_dwords",
    "write_capture",
]
