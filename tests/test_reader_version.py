"""Offline tests for version-string parse/normalize (docs/02 §3) — the pure detection core.

The impure "read it off the live process" step is Windows-only and user-run; here we cover the
canonicalization that turns whatever shape a Windows version takes into the ``MAJOR.MINOR.PATCH``
key used in ``assets/offsets/index.json``, and confirm an unknown version fails closed through
``select_offset_table``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.reader.faults import UnknownGameVersionError
from tekken_coach.reader.offsets import select_offset_table
from tekken_coach.reader.version import normalize_version, version_from_dwords

REPO_OFFSETS = Path("assets/offsets")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2.1.1.0", "2.01.01"),
        ("2, 0, 0", "2.00.00"),
        ("2.01.01", "2.01.01"),
        ("v2.1", "2.01.00"),
        ("2", "2.00.00"),
        ("10.11.12.13", "10.11.12"),  # major keeps all its digits; only minor/patch pad to 2
        ("  2.1.1  ", "2.01.01"),
        # Caveat: a version embedded in surrounding text mis-parses (the "8" in "Tekken8" is taken
        # as the leading group). Detection feeds a clean product-version string, not free text.
        ("Tekken8 2.01.01 build", "8.02.01"),
    ],
)
def test_normalize_version(raw: str, expected: str) -> None:
    assert normalize_version(raw) == expected


def test_normalize_version_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="no numeric version component"):
        normalize_version("not-a-version")


@pytest.mark.parametrize(
    ("ms", "ls", "expected"),
    [
        # HIWORD.LOWORD packing: MS = major<<16|minor, LS = build<<16|revision; revision dropped.
        ((2 << 16) | 1, (1 << 16) | 0, "2.01.01"),
        ((2 << 16) | 0, (0 << 16) | 0, "2.00.00"),
        ((2 << 16) | 1, (1 << 16) | 99, "2.01.01"),  # revision (99) ignored
    ],
)
def test_version_from_dwords(ms: int, ls: int, expected: str) -> None:
    assert version_from_dwords(ms, ls) == expected


def test_normalized_version_selects_the_checked_in_table() -> None:
    # The normalized string is exactly what select_offset_table keys on (docs/02 §3).
    table = select_offset_table(normalize_version("2.1.1.0"), REPO_OFFSETS)
    assert table.game_version == "2.01.01"


def test_unknown_version_fails_closed_with_runbook() -> None:
    # A build with no matching table must fail closed, never fall back to a stale table.
    with pytest.raises(UnknownGameVersionError) as exc:
        select_offset_table(normalize_version("3.0.0"), REPO_OFFSETS)
    assert "update-offsets" in exc.value.runbook
