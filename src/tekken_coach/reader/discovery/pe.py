"""Bound the base scan by parsing the module's in-memory PE header (docs/02 §3).

C4d locates the heap-allocated player struct by scanning the module's *static* data for the pointer
that leads to it (the code-signature technique docs/02 §3 prefers over a heap value-scan). To keep
that scan tractable it must be **bounded** to the right part of the module image — the initialized
data sections that hold global pointers, not the whole ``SizeOfImage`` span. This module derives
those bounds by parsing the PE header the loader already mapped at ``module_base``.

It is **pure logic over ``read()``** (docs/02 §2/C4d brief): a tiny ``Reader`` callable slices bytes
out of the mapped image; nothing here adds a
:class:`~tekken_coach.reader.memory_source.MemorySource` method or writes anything. That keeps the
whole parse offline-testable against a synthetic image.

Only the fields the base scan needs are parsed — ``SizeOfImage`` (the module's mapped extent) and
the section table (name, RVA, size, characteristics) — so the scanner can pick the readable
initialized-data sections to sweep and the executable ``.text`` range for RIP-relative refs.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass

# A byte-slicer over the mapped module: ``reader(rva, size)`` returns ``size`` bytes at
# ``module_base + rva``. The live tool binds this to ``lambda rva, n: source.read(base + rva, n)``;
# the offline suite binds it to a synthetic image. Pure — it only reads.
Reader = Callable[[int, int], bytes]

# PE structure constants (winnt.h). Only what the section walk needs.
_DOS_MAGIC = 0x5A4D  # "MZ"
_NT_SIGNATURE = 0x00004550  # "PE\0\0"
_E_LFANEW_OFFSET = 0x3C  # DOS header -> file offset of the NT headers
_FILE_HEADER_SIZE = 20  # IMAGE_FILE_HEADER
_OPT_MAGIC_PE32PLUS = 0x20B  # IMAGE_OPTIONAL_HEADER64
_OPT_SIZE_OF_IMAGE_OFFSET = 56  # SizeOfImage within the optional header
_SECTION_HEADER_SIZE = 40  # IMAGE_SECTION_HEADER

# Section characteristics (winnt.h IMAGE_SCN_*).
SCN_CNT_CODE = 0x00000020
SCN_CNT_INITIALIZED_DATA = 0x00000040
SCN_MEM_EXECUTE = 0x20000000
SCN_MEM_READ = 0x40000000
SCN_MEM_WRITE = 0x80000000


class PeParseError(Exception):
    """The bytes at ``module_base`` are not a PE image we can walk (bad magic / truncated)."""


@dataclass(frozen=True)
class Section:
    """One PE section header, addresses expressed as module-relative RVAs (docs/02 §3)."""

    name: str
    rva: int  # VirtualAddress — module-relative start
    virtual_size: int
    characteristics: int

    @property
    def end_rva(self) -> int:
        return self.rva + self.virtual_size

    @property
    def is_code(self) -> bool:
        return bool(self.characteristics & (SCN_CNT_CODE | SCN_MEM_EXECUTE))

    @property
    def is_readable_data(self) -> bool:
        """Initialized, readable, non-executable — where global pointers live (.data/.rdata)."""
        return (
            bool(self.characteristics & SCN_CNT_INITIALIZED_DATA)
            and bool(self.characteristics & SCN_MEM_READ)
            and not self.is_code
        )

    @property
    def is_writable_data(self) -> bool:
        """Writable initialized data (``.data``) — where a runtime-set global root pointer sits.

        The pointer we chase is a global written *at runtime* to point at a heap-allocated game
        object, so it lives in a **writable** section. Read-only data (``.rdata``: vtables, RTTI,
        string/import pointers) is static and never the mutable root — and it is usually far larger,
        so sweeping ``.data`` first is both more likely to hit and much cheaper (docs/02 §3).
        """
        return self.is_readable_data and bool(self.characteristics & SCN_MEM_WRITE)


@dataclass(frozen=True)
class ModuleImage:
    """The parsed module bounds the base scan needs: mapped extent + the section table."""

    size_of_image: int
    sections: tuple[Section, ...]

    def data_sections(self) -> tuple[Section, ...]:
        """Readable initialized-data sections to sweep for candidate global pointers (§3)."""
        return tuple(s for s in self.sections if s.is_readable_data)

    def writable_data_sections(self) -> tuple[Section, ...]:
        """Writable data sections (``.data``) — the likely, and cheap, home of the root pointer."""
        return tuple(s for s in self.sections if s.is_writable_data)

    def readonly_data_sections(self) -> tuple[Section, ...]:
        """Read-only initialized data (``.rdata``) — the fallback sweep if ``.data`` yields none."""
        return tuple(s for s in self.sections if s.is_readable_data and not s.is_writable_data)

    def code_sections(self) -> tuple[Section, ...]:
        """Executable sections (``.text``) — where RIP-relative refs to the globals live."""
        return tuple(s for s in self.sections if s.is_code)


def _u16(b: bytes, off: int) -> int:
    return int(struct.unpack_from("<H", b, off)[0])


def _u32(b: bytes, off: int) -> int:
    return int(struct.unpack_from("<I", b, off)[0])


def parse_module_image(reader: Reader) -> ModuleImage:
    """Parse the in-memory PE header at ``module_base`` into :class:`ModuleImage` (pure).

    ``reader(rva, size)`` slices the mapped image. Raises :class:`PeParseError` on a bad DOS/NT
    magic, a 32-bit (non-PE32+) optional header, or an implausible section count — an honest
    "this is not the module we expected" rather than a scan over garbage.
    """
    dos = reader(0, _E_LFANEW_OFFSET + 4)
    if _u16(dos, 0) != _DOS_MAGIC:
        raise PeParseError(f"bad DOS magic 0x{_u16(dos, 0):04x} (expected MZ)")
    e_lfanew = _u32(dos, _E_LFANEW_OFFSET)

    # NT headers: signature + IMAGE_FILE_HEADER. Read the optional header's SizeOfImage too.
    nt = reader(e_lfanew, 4 + _FILE_HEADER_SIZE + _OPT_SIZE_OF_IMAGE_OFFSET + 4)
    if _u32(nt, 0) != _NT_SIGNATURE:
        raise PeParseError(f"bad NT signature 0x{_u32(nt, 0):08x} (expected PE\\0\\0)")
    number_of_sections = _u16(nt, 4 + 2)  # IMAGE_FILE_HEADER.NumberOfSections
    size_of_optional_header = _u16(nt, 4 + 16)  # IMAGE_FILE_HEADER.SizeOfOptionalHeader
    opt_off = 4 + _FILE_HEADER_SIZE
    magic = _u16(nt, opt_off)
    if magic != _OPT_MAGIC_PE32PLUS:
        raise PeParseError(f"optional header magic 0x{magic:04x} is not PE32+ (0x20b)")
    size_of_image = _u32(nt, opt_off + _OPT_SIZE_OF_IMAGE_OFFSET)

    if not 1 <= number_of_sections <= 96:
        raise PeParseError(f"implausible section count {number_of_sections}")

    sections_rva = e_lfanew + opt_off + size_of_optional_header
    table = reader(sections_rva, number_of_sections * _SECTION_HEADER_SIZE)
    sections: list[Section] = []
    for i in range(number_of_sections):
        base = i * _SECTION_HEADER_SIZE
        name = table[base : base + 8].rstrip(b"\x00").decode("latin-1", "replace")
        sections.append(
            Section(
                name=name,
                virtual_size=_u32(table, base + 8),
                rva=_u32(table, base + 12),
                characteristics=_u32(table, base + 36),
            )
        )
    return ModuleImage(size_of_image=size_of_image, sections=tuple(sections))
