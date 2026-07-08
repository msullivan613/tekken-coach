"""Read-only Tekken 8 memory reader — offline core (docs/02). Chunk C4a.

Turns Tekken 8 process memory into ``FrameRecord``s (docs/03 §1) behind a read-only
:class:`~tekken_coach.reader.memory_source.MemorySource` seam. This is the environment-independent
half: the seam + fake source, the ``assets/offsets/`` format and typed loader, version detection
with fail-closed behavior, the raw-bytes -> ``FrameRecord`` decoder, the doctor self-check, and the
structured failure/state signals. The concrete Windows ``ReadProcessMemory`` backend and the
``update-offsets`` re-discovery tool are **C4b** (they need the game and a Windows box) and plug in
behind the same seam.

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
* ``update-offsets`` (C4b) is a **clean-room** re-implementation of the Jin-vs-Kazuya re-discovery
  *technique* (an idea, not protected) — it does not copy the fork's ``update_memory_address.py``.
"""

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
from tekken_coach.reader.state import SignalKind, StateSignal, classify_state

__all__ = [
    "PATCH_RUNBOOK",
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
    "ReaderError",
    "ReaderFault",
    "SignalKind",
    "StateSignal",
    "UnknownGameVersionError",
    "classify_fault",
    "classify_state",
    "decode_frame",
    "load_offset_index",
    "poll_frames",
    "read_state_signal",
    "run_doctor",
    "select_offset_table",
]
