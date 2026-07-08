"""Pure scan primitives on synthetic images (C4c, docs/02 §3/§4)."""

from __future__ import annotations

import struct

import pytest

from tekken_coach.reader.discovery.scanners import (
    Region,
    aob_scan,
    change_scan,
    parse_aob,
    snapshot_region,
    value_scan,
)
from tekken_coach.reader.memory_source import FakeMemorySource

BASE = 0x140000000


def _u32(*values: int) -> bytes:
    return b"".join(struct.pack("<I", v) for v in values)


def test_value_scan_finds_all_aligned_matches() -> None:
    # Three u32s; 150 appears at offsets 0 and 8.
    region = Region(base=BASE, data=_u32(150, 7, 150, 42))
    assert value_scan(region, 150, "u32", align=4) == [BASE + 0, BASE + 8]


def test_value_scan_respects_alignment() -> None:
    # 150 packed starting at a non-4-aligned offset is missed at align=4 but found at align=1.
    data = b"\x00" + struct.pack("<I", 150) + b"\x00\x00\x00"
    region = Region(base=BASE, data=data)
    assert value_scan(region, 150, "u32", align=4) == []
    assert value_scan(region, 150, "u32", align=1) == [BASE + 1]


def test_value_scan_i32_and_float() -> None:
    region = Region(base=BASE, data=struct.pack("<i", -13) + struct.pack("<f", 1.5))
    assert value_scan(region, -13, "i32", align=4) == [BASE + 0]
    assert value_scan(region, 1.5, "f32", align=4) == [BASE + 4]


def test_change_scan_detects_changed_and_steady() -> None:
    before = Region(base=BASE, data=_u32(1000, 5, 42))
    after = Region(base=BASE, data=_u32(1004, 5, 99))
    assert change_scan(before, after, "u32", align=4) == [BASE + 0, BASE + 8]
    assert change_scan(before, after, "u32", align=4, changed=False) == [BASE + 4]


def test_change_scan_predicate_filters_to_monotonic_increase() -> None:
    before = Region(base=BASE, data=_u32(1000, 500))
    after = Region(base=BASE, data=_u32(1004, 400))  # first increased, second decreased
    hits = change_scan(
        before, after, "u32", align=4, predicate=lambda old, new: 0 < new - old <= 100
    )
    assert hits == [BASE + 0]


def test_change_scan_requires_matching_spans() -> None:
    a = Region(base=BASE, data=_u32(1))
    b = Region(base=BASE + 4, data=_u32(1))
    with pytest.raises(ValueError, match="same address span"):
        change_scan(a, b, "u32")


def test_parse_aob_marks_wildcards() -> None:
    pat = parse_aob("48 8B ?? ?? 89")
    assert pat.needle == bytes([0x48, 0x8B, 0, 0, 0x89])
    assert pat.mask == (True, True, False, False, True)
    assert len(pat) == 5


def test_parse_aob_rejects_bad_tokens() -> None:
    with pytest.raises(ValueError, match="bad AOB token"):
        parse_aob("48 ZZ")
    with pytest.raises(ValueError, match="empty AOB"):
        parse_aob("   ")


def test_aob_scan_matches_with_wildcards() -> None:
    data = bytes([0x00, 0x48, 0x8B, 0x11, 0x22, 0x89, 0xFF, 0x48, 0x8B, 0x33, 0x44, 0x89])
    region = Region(base=BASE, data=data)
    # Two occurrences of "48 8B ?? ?? 89" (the middle two bytes differ each time).
    assert aob_scan(region, "48 8B ?? ?? 89") == [BASE + 1, BASE + 7]


def test_aob_scan_no_match() -> None:
    region = Region(base=BASE, data=bytes([0x11, 0x22, 0x33]))
    assert aob_scan(region, "48 8B") == []


def test_region_read_and_bounds() -> None:
    region = Region(base=BASE, data=_u32(7, 8))
    assert region.read(BASE + 4, 4) == struct.pack("<I", 8)
    assert region.read_scalar(BASE + 4, "u32") == 8
    assert region.read_scalar(BASE + 8, "u32") is None  # out of range -> None
    with pytest.raises(IndexError):
        region.read(BASE + 6, 4)


def test_snapshot_region_reads_through_a_memory_source() -> None:
    # A FakeMemorySource carrying one big region; snapshot_region reads it module-relative.
    blob = _u32(150, 12, 999)
    source = FakeMemorySource(
        [{BASE + 0x1000: blob}], module_bases={"game.exe": BASE}, advance_on=BASE + 0x1000
    )
    region = snapshot_region(source, "game.exe", 0x1000, len(blob))
    assert region.base == BASE + 0x1000
    assert value_scan(region, 12, "u32") == [BASE + 0x1000 + 4]


def test_snapshot_region_absolute() -> None:
    blob = _u32(1, 2)
    source = FakeMemorySource(
        [{0x5000: blob}], module_bases={"game.exe": BASE}, advance_on=0x5000
    )
    region = snapshot_region(source, "game.exe", 0x5000, len(blob), absolute=True)
    assert region.base == 0x5000
