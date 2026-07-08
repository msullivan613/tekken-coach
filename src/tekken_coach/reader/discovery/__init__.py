"""Offset re-discovery — the clean-room ``update-offsets`` tool (chunk C4c, docs/02 §4/§5).

Regenerates ``assets/offsets/<version>.json`` after a game patch by attaching read-only to Tekken 8
in a fixed **P1 Jin vs P2 Kazuya** practice setup, scanning memory to re-derive where each
``FrameRecord`` field lives, and writing a versioned candidate table + a diagnostic report.

Clean-room posture (docs/02 §5) — binding
-----------------------------------------
This is an **original re-implementation of the re-discovery *technique*** (open Jin vs Kazuya, scan
to re-derive addresses — an unprotected idea), not a copy of the dcep93 fork's
``update_memory_address.py`` (which is all-rights-reserved). Offsets/addresses/signatures are
facts/data, used freely (attributed in NOTICE / THIRD_PARTY_LICENSES).

Read-only (docs/02 §2)
----------------------
Scanners work over raw ``bytes`` images; the only memory access is through the read-only
:class:`~tekken_coach.reader.memory_source.MemorySource` seam; the only thing written is a *file*.
No write/inject primitive exists here (``tests/test_reader_readonly.py`` greps this package).

Pure vs. live
-------------
Scanners, derivation, builder, report, and the ``discover``/``persist`` orchestration steps are
pure and offline-tested. Only :func:`~.orchestrate.run_update_offsets` attaches to the game.
"""

from tekken_coach.reader.discovery.builder import (
    build_offset_table,
    register_version,
    write_offset_table,
)
from tekken_coach.reader.discovery.derive import (
    Confidence,
    DerivationResult,
    DerivedField,
    DiscoverySnapshots,
    derive_layout,
)
from tekken_coach.reader.discovery.manifest import ProbeManifest, ScanWindow, load_probe_manifest
from tekken_coach.reader.discovery.orchestrate import discover, persist, run_update_offsets
from tekken_coach.reader.discovery.report import (
    CALIBRATION_RUNBOOK,
    DiagnosticReport,
    build_report,
)
from tekken_coach.reader.discovery.scanners import (
    AobPattern,
    Region,
    aob_scan,
    change_scan,
    parse_aob,
    snapshot_region,
    value_scan,
)

__all__ = [
    "CALIBRATION_RUNBOOK",
    "AobPattern",
    "Confidence",
    "DerivationResult",
    "DerivedField",
    "DiagnosticReport",
    "DiscoverySnapshots",
    "ProbeManifest",
    "Region",
    "ScanWindow",
    "aob_scan",
    "build_offset_table",
    "build_report",
    "change_scan",
    "derive_layout",
    "discover",
    "load_probe_manifest",
    "parse_aob",
    "persist",
    "register_version",
    "run_update_offsets",
    "snapshot_region",
    "value_scan",
    "write_offset_table",
]
