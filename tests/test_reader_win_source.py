"""Offline tests for the Windows source (docs/02 §2, §7) — everything not needing the game.

pymem is Windows-only and absent in CI, so we assert the three things that *must* hold offline:

1. The module imports with pymem absent (it must — the whole package imports on Linux).
2. :class:`WinMemorySource` satisfies the read-only :class:`MemorySource` Protocol.
3. Constructing it without pymem raises a clear, actionable error (not an obscure ImportError).
4. The pymem-exception -> :class:`MemoryReadError` mapping sets ``access_denied`` correctly, via a
   fake pymem surface (no real pymem needed).
"""

from __future__ import annotations

import importlib.util
from types import ModuleType, SimpleNamespace

import pytest

from tekken_coach.reader.faults import FaultKind, MemoryReadError, classify_fault
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.win_source import WinMemorySource, map_pymem_error

_PYMEM_ABSENT = importlib.util.find_spec("pymem") is None


# --------------------------------------------------------------------------- #
# Protocol conformance + import posture
# --------------------------------------------------------------------------- #


def test_win_source_module_imports_without_pymem() -> None:
    # Importing the module must never require pymem (lazy import inside the constructor). If this
    # test file collected at all, the import already succeeded; assert it explicitly for intent.
    import tekken_coach.reader.win_source as mod

    assert mod.GAME_PROCESS_NAME.endswith(".exe")


def test_win_source_satisfies_memory_source_protocol() -> None:
    # runtime_checkable, methods-only Protocol -> issubclass works without instantiating.
    assert issubclass(WinMemorySource, MemorySource)
    members = {m for m in dir(WinMemorySource) if not m.startswith("_")}
    # The seam's surface: read + module_base + regions, all read-side (regions is a VirtualQueryEx
    # map query, C4h Phase 1), nothing that mutates.
    assert members == {"read", "module_base", "regions"}


def test_read_only_open_flags_are_read_plus_query_only() -> None:
    # Read-only at the OS-handle level (docs/02 §2): VM_READ | QUERY_INFORMATION, never ALL_ACCESS.
    from tekken_coach.reader import win_source as ws

    assert ws.PROCESS_READ_ACCESS == ws.PROCESS_VM_READ | ws.PROCESS_QUERY_INFORMATION
    assert ws.PROCESS_READ_ACCESS == 0x0410


@pytest.mark.skipif(not _PYMEM_ABSENT, reason="pymem is installed; the no-pymem path can't be hit")
def test_instantiating_without_pymem_raises_clear_error() -> None:
    with pytest.raises(MemoryReadError) as exc:
        WinMemorySource()
    msg = str(exc.value)
    assert "pymem" in msg and "Windows-only" in msg


# --------------------------------------------------------------------------- #
# Fault mapping with a fake pymem surface
# --------------------------------------------------------------------------- #


def _fake_pymem(*, could_not_open: type[Exception]) -> ModuleType:
    """A minimal stand-in for the pymem module exposing just its ``exception`` classes."""
    mod = ModuleType("fake_pymem")
    mod.exception = SimpleNamespace(  # type: ignore[attr-defined]
        MemoryReadError=type("MemoryReadError", (Exception,), {}),
        CouldNotOpenProcess=could_not_open,
    )
    return mod


class _CouldNotOpen(Exception):
    def __init__(self, code: int | None = None) -> None:
        self.error_code = code
        super().__init__(f"could not open (code={code})")


class _WinErr(Exception):
    def __init__(self, code: int) -> None:
        self.error_code = code
        super().__init__(f"winapi error {code}")


def test_access_denied_code_maps_to_access_denied() -> None:
    pymem = _fake_pymem(could_not_open=_CouldNotOpen)
    err = map_pymem_error(_WinErr(5), pymem, context="read")
    assert err.access_denied is True
    assert classify_fault(err).kind is FaultKind.access_denied


def test_partial_copy_code_maps_to_process_lost() -> None:
    pymem = _fake_pymem(could_not_open=_CouldNotOpen)
    err = map_pymem_error(_WinErr(299), pymem, context="read")  # ERROR_PARTIAL_COPY
    assert err.access_denied is False
    assert classify_fault(err).kind is FaultKind.process_lost


def test_could_not_open_without_code_is_conservatively_access_denied() -> None:
    # An open failure with no decipherable code: don't retry-hammer (docs/02 §7).
    pymem = _fake_pymem(could_not_open=_CouldNotOpen)
    err = map_pymem_error(_CouldNotOpen(None), pymem, context="open")
    assert err.access_denied is True


def test_unknown_exception_is_not_access_denied() -> None:
    pymem = _fake_pymem(could_not_open=_CouldNotOpen)
    err = map_pymem_error(RuntimeError("boom"), pymem, context="read")
    assert err.access_denied is False
    assert classify_fault(err).kind is FaultKind.process_lost


# --------------------------------------------------------------------------- #
# read()/module_base() behavior via an injected fake handle (no live process)
# --------------------------------------------------------------------------- #


class _FakeHandle:
    """Stands in for the opened pymem read handle."""

    process_handle = 0xABCD

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def read_bytes(self, address: int, size: int) -> bytes:
        self.calls.append((address, size))
        return b"\x01\x02\x03\x04"[:size]


def _fake_pymem_with_module(base: int) -> ModuleType:
    pymem = _fake_pymem(could_not_open=_CouldNotOpen)
    pymem.process = SimpleNamespace(  # type: ignore[attr-defined]
        module_from_name=lambda handle, name: SimpleNamespace(lpBaseOfDll=base)
    )
    return pymem


def test_read_delegates_to_read_bytes() -> None:
    pymem = _fake_pymem_with_module(0x140000000)
    src = WinMemorySource(pymem_module=pymem, handle=_FakeHandle())
    assert src.read(0x1000, 4) == b"\x01\x02\x03\x04"


def test_read_short_read_raises() -> None:
    class _ShortHandle(_FakeHandle):
        def read_bytes(self, address: int, size: int) -> bytes:
            return b"\x01"  # fewer bytes than requested

    pymem = _fake_pymem_with_module(0x140000000)
    src = WinMemorySource(pymem_module=pymem, handle=_ShortHandle())
    with pytest.raises(MemoryReadError, match="short read"):
        src.read(0x1000, 4)


def test_read_maps_pymem_exception() -> None:
    class _DeadHandle(_FakeHandle):
        def read_bytes(self, address: int, size: int) -> bytes:
            raise _WinErr(5)  # access denied mid-read

    pymem = _fake_pymem_with_module(0x140000000)
    src = WinMemorySource(pymem_module=pymem, handle=_DeadHandle())
    with pytest.raises(MemoryReadError) as exc:
        src.read(0x1000, 4)
    assert exc.value.access_denied is True


def test_module_base_resolves_via_module_from_name() -> None:
    pymem = _fake_pymem_with_module(0x140000000)
    src = WinMemorySource(pymem_module=pymem, handle=_FakeHandle())
    assert src.module_base("Polaris-Win64-Shipping.exe") == 0x140000000


def test_module_base_missing_module_raises() -> None:
    pymem = _fake_pymem(could_not_open=_CouldNotOpen)
    pymem.process = SimpleNamespace(  # type: ignore[attr-defined]
        module_from_name=lambda handle, name: None
    )
    src = WinMemorySource(pymem_module=pymem, handle=_FakeHandle())
    with pytest.raises(MemoryReadError, match="module not loaded"):
        src.module_base("nope.dll")


def test_injected_win_source_is_a_memory_source_instance() -> None:
    pymem = _fake_pymem_with_module(0x1000)
    src = WinMemorySource(pymem_module=pymem, handle=_FakeHandle())
    assert isinstance(src, MemorySource)
