"""PE-header bound parsing over a synthetic in-memory image (C4d, docs/02 §3).

The base scan must be *bounded* to the module's readable data sections. These tests prove the
header walk recovers those bounds from a synthetic PE32+ image with no game and no pymem, and that
a non-PE / wrong-bitness image is rejected loudly rather than scanned as garbage.
"""

from __future__ import annotations

import struct

import pytest

from tekken_coach.reader.discovery.pe import (
    SCN_CNT_INITIALIZED_DATA,
    SCN_MEM_READ,
    SCN_MEM_WRITE,
    ModuleImage,
    PeParseError,
    Reader,
    Section,
    parse_module_image,
)
from tests.fixtures.reader.planted_chain import MODULE_BASE, planted_chain


def _reader() -> Reader:
    source = planted_chain().before

    def read(rva: int, size: int) -> bytes:
        return source.read(MODULE_BASE + rva, size)

    return read


def test_parses_size_of_image_and_section_table() -> None:
    image = parse_module_image(_reader())
    assert image.size_of_image == 0x4000
    assert [s.name for s in image.sections] == [".text", ".data"]


def test_classifies_code_and_data_sections() -> None:
    image = parse_module_image(_reader())
    assert [s.name for s in image.code_sections()] == [".text"]
    # .data is the sweep target: initialized, readable, non-executable.
    data = image.data_sections()
    assert [s.name for s in data] == [".data"]
    assert data[0].rva == 0x3000
    assert data[0].end_rva == 0x4000


def test_code_section_is_never_offered_as_a_data_slot_sweep_target() -> None:
    image = parse_module_image(_reader())
    text = next(s for s in image.sections if s.name == ".text")
    assert text.is_code and not text.is_readable_data


def test_writable_vs_readonly_data_partition() -> None:
    # The root pointer is a runtime-written global, so it lives in writable .data; .rdata is the
    # (usually much larger) fallback. The image must partition the two so the sweep prefers .data.
    rw = Section(
        name=".data",
        rva=0x1000,
        virtual_size=0x100,
        characteristics=SCN_CNT_INITIALIZED_DATA | SCN_MEM_READ | SCN_MEM_WRITE,
    )
    ro = Section(
        name=".rdata",
        rva=0x2000,
        virtual_size=0x100,
        characteristics=SCN_CNT_INITIALIZED_DATA | SCN_MEM_READ,
    )
    assert rw.is_writable_data and rw.is_readable_data
    assert ro.is_readable_data and not ro.is_writable_data
    image = ModuleImage(size_of_image=0x3000, sections=(rw, ro))
    assert image.writable_data_sections() == (rw,)
    assert image.readonly_data_sections() == (ro,)
    assert set(image.data_sections()) == {rw, ro}


def _bad(buf: bytes) -> Reader:
    def read(rva: int, size: int) -> bytes:
        return buf[rva : rva + size]

    return read


def test_rejects_a_non_pe_image() -> None:
    with pytest.raises(PeParseError, match="DOS magic"):
        parse_module_image(_bad(bytes(0x100)))


def test_rejects_a_bad_nt_signature() -> None:
    buf = bytearray(0x200)
    buf[0:2] = b"MZ"
    buf[0x3C:0x40] = struct.pack("<I", 0x80)
    with pytest.raises(PeParseError, match="NT signature"):
        parse_module_image(_bad(bytes(buf)))


def test_rejects_a_32_bit_optional_header() -> None:
    buf = bytearray(0x400)
    buf[0:2] = b"MZ"
    buf[0x3C:0x40] = struct.pack("<I", 0x80)
    buf[0x80:0x84] = struct.pack("<I", 0x00004550)
    buf[0x82 + 4 : 0x84 + 4] = struct.pack("<H", 1)  # NumberOfSections
    buf[0x80 + 4 + 16 : 0x80 + 4 + 18] = struct.pack("<H", 0xE0)
    opt = 0x80 + 4 + 20
    buf[opt : opt + 2] = struct.pack("<H", 0x10B)  # PE32, not PE32+
    with pytest.raises(PeParseError, match="not PE32"):
        parse_module_image(_bad(bytes(buf)))
