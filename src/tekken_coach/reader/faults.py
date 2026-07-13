"""Reader errors and the structured fault signal C6 consumes (docs/02 §7).

The reader **classifies** failures; it does not decide capture cadence. C6 owns the policy
(docs/01 §3.2): on a live-capture fault it fails *silent and closed* and surfaces after the
match; on a clean-capture fault it stops the batch and reports. To let C6 branch without
string-matching exceptions, every failure maps to a :class:`ReaderFault` — a ``FaultKind`` plus
a human message and a ``recoverable`` hint.

Nothing here prints or renders (docs/02 §2 silent-producer): errors are raised/returned as data.
The unknown-version case additionally carries the §4 re-discovery runbook so the *caller* (a
`doctor`/CLI in C6) can present it — the reader itself never surfaces output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# The docs/02 §4 patch runbook, carried on an unknown-version failure so the caller can present
# it. Kept as data (not printed here) to honor the silent-producer invariant (docs/02 §2).
PATCH_RUNBOOK = """\
Unknown game version — capture refused (fail-closed, docs/02 §3/§4).

A wrong offset silently produces garbage FrameRecords, which is worse than not running, so the
reader will not guess with a stale table. To restore capture after a game patch:

  1. Open Tekken 8 practice mode as P1 Jin vs P2 Kazuya.
  2. Run `tekken-coach update-offsets` (C4b) — it re-discovers the addresses and writes
     assets/offsets/<version>.json for the new build.
  3. Invalidate move/frame data for the new version (see docs/05).
  4. Re-run `tekken-coach doctor` (docs/02 §6). Green -> capture is usable again.
"""

# Shown when ``char_ids_known`` is the ONLY failing check — the mechanical core (health, frame
# monotonicity, move ids, positions) all passed, so the offsets are sound and capture works. The
# remedy is not a re-derivation; it is adding an observed char id, so this must NOT print the
# stale-offsets runbook above (which the user's live run proved false — capture ran fine).
CHAR_UNLISTED_NOTE = """\
An on-screen character is not in the table's known_char_ids — but every mechanical check passed,
so the offsets are sound and capture still works (this does not refuse capture).

known_char_ids is a sanity whitelist, not the full roster. To silence this, add the observed
char_id to assets/offsets/<version>.json "known_char_ids" (values are established by observation,
docs/02 §6). Only treat it as stale offsets if a *mechanical* check (health/frames/moves/positions)
also fails.
"""


class ReaderError(Exception):
    """Base class for all reader failures."""


class UnknownGameVersionError(ReaderError):
    """The running game version has no matching offset table — fail closed (docs/02 §3/§7).

    Never falls back to a stale table. Carries the available versions and the §4 runbook.
    """

    def __init__(self, version: str, available: list[str]) -> None:
        self.version = version
        self.available = available
        self.runbook = PATCH_RUNBOOK
        super().__init__(
            f"no offset table for game version {version!r} "
            f"(have: {', '.join(available) or 'none'}). Capture refused; see runbook."
        )


class OffsetTableError(ReaderError):
    """An offset table / index file is missing or malformed."""


class MemoryReadError(ReaderError):
    """A memory read or module-base resolution failed (process gone, unmapped, denied).

    ``access_denied`` distinguishes an anti-cheat / permission failure (do not retry-hammer,
    docs/02 §7) from a transient/process-lost read error.
    """

    def __init__(self, message: str, *, access_denied: bool = False) -> None:
        self.access_denied = access_denied
        super().__init__(message)


class DecodeError(ReaderError):
    """A frame could not be decoded from the bytes the source returned."""


class FaultKind(StrEnum):
    """The category of a reader fault (docs/02 §7). C6 maps these to capture-mode policy."""

    unknown_version = "unknown_version"  # version lookup miss -> fail closed, show runbook
    stale_offsets = "stale_offsets"  # self-check detected garbage reads -> block capture
    process_lost = "process_lost"  # process not found / closed mid-capture
    access_denied = "access_denied"  # anti-cheat / permission -> report, do not retry-hammer
    read_error = "read_error"  # other read/decode failure


@dataclass(frozen=True)
class ReaderFault:
    """A structured failure signal for C6 (docs/02 §7).

    The reader emits this; C6 decides cadence (live: fail-silent-closed and surface after the
    match; clean: stop the batch and report — docs/01 §3.2). ``recoverable`` is a hint that the
    fault may clear on its own (e.g. a process temporarily gone), not a policy directive.
    """

    kind: FaultKind
    message: str
    recoverable: bool
    runbook: str | None = None


def classify_fault(exc: ReaderError) -> ReaderFault:
    """Map a :class:`ReaderError` to the structured :class:`ReaderFault` C6 branches on."""
    if isinstance(exc, UnknownGameVersionError):
        return ReaderFault(
            FaultKind.unknown_version, str(exc), recoverable=False, runbook=exc.runbook
        )
    if isinstance(exc, MemoryReadError):
        if exc.access_denied:
            return ReaderFault(FaultKind.access_denied, str(exc), recoverable=False)
        return ReaderFault(FaultKind.process_lost, str(exc), recoverable=True)
    if isinstance(exc, OffsetTableError):
        return ReaderFault(FaultKind.stale_offsets, str(exc), recoverable=False)
    return ReaderFault(FaultKind.read_error, str(exc), recoverable=False)
