"""Failure-mode classification — the structured signal C6 branches on (docs/02 §7)."""

from __future__ import annotations

from tekken_coach.reader.faults import (
    FaultKind,
    MemoryReadError,
    OffsetTableError,
    UnknownGameVersionError,
    classify_fault,
)


def test_unknown_version_classifies_as_fail_closed_with_runbook() -> None:
    fault = classify_fault(UnknownGameVersionError("9.9.9", ["2.01.01"]))
    assert fault.kind is FaultKind.unknown_version
    assert fault.recoverable is False
    assert fault.runbook is not None and "update-offsets" in fault.runbook


def test_process_lost_is_recoverable() -> None:
    fault = classify_fault(MemoryReadError("process gone"))
    assert fault.kind is FaultKind.process_lost
    assert fault.recoverable is True  # process may come back; C6 decides cadence


def test_access_denied_is_not_retry_hammered() -> None:
    fault = classify_fault(MemoryReadError("denied", access_denied=True))
    assert fault.kind is FaultKind.access_denied
    assert fault.recoverable is False  # anti-cheat/permission — do not retry-hammer (docs/02 §7)


def test_offset_table_error_classifies_as_stale_offsets() -> None:
    fault = classify_fault(OffsetTableError("malformed"))
    assert fault.kind is FaultKind.stale_offsets
    assert fault.recoverable is False


def test_generic_reader_error_is_read_error() -> None:
    from tekken_coach.reader.faults import DecodeError

    fault = classify_fault(DecodeError("short read"))
    assert fault.kind is FaultKind.read_error
