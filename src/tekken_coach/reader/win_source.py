"""The concrete Windows :class:`MemorySource` — read-only by construction (docs/02 §2, §5).

This is the OS-facing half of the reader (chunk C4b): it opens the running Tekken 8 process and
serves byte ranges to the decoder through the same ``MemorySource`` seam that
:class:`~tekken_coach.reader.memory_source.FakeMemorySource` serves offline. Everything above it
(decode, doctor, state, faults) is unchanged and cannot tell a real source from a fake one.

Read-only, at the OS-handle level
----------------------------------
The process is opened with ``PROCESS_VM_READ | PROCESS_QUERY_INFORMATION`` **only** — never
``PROCESS_ALL_ACCESS`` — so the handle itself grants no write capability (docs/02 §2). The class
implements exactly the two seam methods, :meth:`read` and :meth:`module_base`, backed by pymem's
``read_bytes`` and module enumeration. There is no write/inject primitive here, not even referenced
in a comment; ``tests/test_reader_readonly.py`` greps this file and fails on any such token.

pymem is Windows-only and an **optional** dependency (the ``windows`` extra). It is imported
*lazily inside the constructor*, never at module import time, so this module imports cleanly on
Linux/macOS with pymem absent and the offline suite stays green. Instantiating
:class:`WinMemorySource` without pymem raises a clear, actionable error.

Nothing in this module runs offline in CI: attaching needs a live Windows process. The fault
mapping (pymem exception -> :class:`MemoryReadError`) is factored into pure helpers that *are*
offline-tested with a fake pymem surface (``tests/test_reader_win_source.py``).
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from tekken_coach.reader.faults import MemoryReadError
from tekken_coach.reader.memory_source import MemoryRegion

# The Tekken 8 shipping executable / module name (Unreal "Polaris" project). The offset tables in
# ``assets/offsets/`` anchor against this module (docs/02 §3); it is the default attach target.
GAME_PROCESS_NAME = "Polaris-Win64-Shipping.exe"

# Windows OpenProcess access-right flags (winnt.h). We request the *minimum* needed to read:
#   PROCESS_VM_READ           (0x0010) — ReadProcessMemory
#   PROCESS_QUERY_INFORMATION (0x0400) — module enumeration + image-path query
# We deliberately never request write/operation rights, so the handle cannot mutate the process
# (docs/02 §2). This is the read-only invariant expressed at the OS-handle level.
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_READ_ACCESS = PROCESS_VM_READ | PROCESS_QUERY_INFORMATION

# Windows system error codes (winerror.h) we distinguish. ACCESS_DENIED is the anti-cheat /
# permission signal that C6 must not retry-hammer (docs/02 §7); everything else on a read is
# treated as a transient/process-lost error.
_ERROR_ACCESS_DENIED = 5

# VirtualQueryEx classification constants (winnt.h). Region enumeration (C4h Phase 1) is a
# read-only *query* of the process map — MEMORY_BASIC_INFORMATION per region, no bytes read.
_MEM_COMMIT = 0x1000  # State: backed by physical storage (as opposed to reserved/free)
_MEM_IMAGE = 0x1000000  # Type: a mapped module image (.text/.data) — swept separately via the PE
_PAGE_GUARD = 0x100  # a guard page: touching it raises, so never enumerate it as readable
_PAGE_NOACCESS = 0x01  # no access at all
# Protection bits that grant read access (any of these, minus guard/no-access).
_PAGE_READABLE = 0x02 | 0x04 | 0x08 | 0x20 | 0x40 | 0x80  # R / RW / WC / XR / XRW / XWC

# x64 user-space bounds and tractability caps for the sweep. Deliberately generous; the scan layer
# bounds its own work further. A single reserved span far larger than any heap struct's arena is
# skipped so one 4 GiB reservation cannot dominate the sweep.
_ENUM_MIN_ADDRESS = 0x10000
_ENUM_MAX_ADDRESS = 0x7FFF_FFFF_FFFF
_ENUM_MAX_REGION_BYTES = 512 * 1024 * 1024
_ENUM_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024


class _MemoryBasicInformation(ctypes.Structure):
    """The x64 ``MEMORY_BASIC_INFORMATION`` (winnt.h) that ``VirtualQueryEx`` fills in.

    Defined with **fixed-width** ``ctypes`` scalars — ``c_uint64`` for the x64 pointers/``SIZE_T``,
    ``c_uint32`` for the ``DWORD``\\ s — not ``ctypes.wintypes`` (Windows-only, would break this
    module's Linux import) and *not* ``c_ulong``: ``c_ulong`` is 4 bytes on Windows but 8 on Linux
    (LP64), which silently changes the layout off-Windows. ``DWORD`` is always 32-bit, so pinning it
    makes the struct byte-identical on every platform — this is a fixed OS ABI, not a native type.
    The ``__alignment`` members are the 8-byte padding the 64-bit layout carries (``RegionSize`` is
    8-aligned; the struct is padded to a multiple of 8 → 48 bytes). Only the actual
    ``ctypes.windll.kernel32`` call in :meth:`WinMemorySource.regions` is Windows-only, never
    reached offline; the layout itself is offline-tested (``test_reader_win_source``).

    We bind ``VirtualQueryEx`` directly rather than through a pymem helper: ``pymem`` exposes no
    ``virtual_query`` on the installed version (an ``AttributeError`` the old binding swallowed,
    yielding "0 readable regions"), and this raw binding was validated live before it landed.
    """

    _fields_ = (
        ("BaseAddress", ctypes.c_uint64),
        ("AllocationBase", ctypes.c_uint64),
        ("AllocationProtect", ctypes.c_uint32),
        ("__alignment1", ctypes.c_uint32),
        ("RegionSize", ctypes.c_uint64),
        ("State", ctypes.c_uint32),
        ("Protect", ctypes.c_uint32),
        ("Type", ctypes.c_uint32),
        ("__alignment2", ctypes.c_uint32),
    )


@dataclass(frozen=True)
class _BasicRegion:
    """The slice of ``MEMORY_BASIC_INFORMATION`` :func:`enumerate_committed_regions` needs."""

    base: int
    size: int
    state: int
    protect: int
    type: int


def _is_readable_committed(region: _BasicRegion, *, skip_image: bool) -> bool:
    """Whether a queried region is committed, readable, non-guard heap (not a module image)."""
    if region.state != _MEM_COMMIT or region.size <= 0:
        return False
    if region.protect & (_PAGE_GUARD | _PAGE_NOACCESS):
        return False
    if not region.protect & _PAGE_READABLE:
        return False
    return not (skip_image and region.type & _MEM_IMAGE)


def enumerate_committed_regions(
    query: Callable[[int], _BasicRegion | None],
    *,
    min_address: int = _ENUM_MIN_ADDRESS,
    max_address: int = _ENUM_MAX_ADDRESS,
    max_region_bytes: int = _ENUM_MAX_REGION_BYTES,
    max_total_bytes: int = _ENUM_MAX_TOTAL_BYTES,
    skip_image: bool = True,
) -> list[MemoryRegion]:
    """Walk the process map via ``query`` (a ``VirtualQueryEx`` wrapper), collecting readable heap.

    Pure over ``query(address) -> _BasicRegion | None`` so it is offline-testable with a fake map
    (``None`` ends the walk, as ``VirtualQueryEx`` returning 0 does). Bounded three ways for
    tractability — a userspace address window, a per-region ceiling (a giant reservation is skipped,
    not swept), and a running total — because the caller sweeps every returned byte. It **reads no
    memory**: it only asks the OS what is mapped.
    """
    out: list[MemoryRegion] = []
    total = 0
    address = min_address
    while address < max_address:
        region = query(address)
        if region is None:
            break
        nxt = region.base + region.size
        if nxt <= address:  # a non-advancing query would loop forever; stop defensively
            break
        readable = _is_readable_committed(region, skip_image=skip_image)
        if readable and region.size <= max_region_bytes:
            out.append(MemoryRegion(base=region.base, size=region.size))
            total += region.size
            if total >= max_total_bytes:
                break
        address = nxt
    return out


_PYMEM_MISSING_MSG = (
    "WinMemorySource is Windows-only and needs the 'pymem' package, which is not installed. "
    "Install the capture extra on a Windows machine: `pip install tekken-coach[windows]` "
    "(or `pip install pymem`). The offline pipeline does not require it."
)


def _require_pymem() -> ModuleType:
    """Import pymem lazily, raising a clear actionable error if it is absent (docs/02 §7).

    Kept out of module scope so importing :mod:`win_source` never needs pymem — the offline suite
    imports this module on Linux with pymem absent and must not fail.
    """
    try:
        import pymem  # noqa: PLC0415  (intentional lazy import — keeps the module Linux-importable)
        import pymem.exception  # noqa: PLC0415
        import pymem.process  # noqa: PLC0415
        import pymem.ressources.kernel32  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only where pymem is absent
        raise MemoryReadError(_PYMEM_MISSING_MSG) from exc
    module: ModuleType = pymem
    return module


def _error_code_of(exc: BaseException) -> int | None:
    """Best-effort extraction of a Windows error code from a pymem/ctypes exception.

    pymem's ``MemoryReadError`` / ``WinAPIError`` carry an ``error_code``; a raw ``OSError`` carries
    ``winerror``. Returning ``None`` means "no code available" (treated as not-access-denied).
    """
    for attr in ("error_code", "winerror"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    return None


def _is_access_denied(exc: BaseException, pymem_mod: ModuleType) -> bool:
    """Decide whether a pymem exception represents an access-denied / anti-cheat failure.

    Access-denied (Windows error 5) is the "do not retry-hammer" signal (docs/02 §7). We check the
    exception's error code first; a ``CouldNotOpenProcess`` with a 5 code, or any exception whose
    code is ``ERROR_ACCESS_DENIED``, maps to ``access_denied=True``. Everything else (process gone,
    partial copy, unmapped region) is a transient/process-lost read error.
    """
    code = _error_code_of(exc)
    if code == _ERROR_ACCESS_DENIED:
        return True
    could_not_open = getattr(pymem_mod.exception, "CouldNotOpenProcess", None)
    if could_not_open is not None and isinstance(exc, could_not_open):
        # An open failure with no decipherable code is conservatively reported as access-denied so
        # C6 does not retry-hammer a process that may be shielded by anti-cheat (docs/02 §7).
        return code is None or code == _ERROR_ACCESS_DENIED
    return False


def map_pymem_error(exc: BaseException, pymem_mod: ModuleType, *, context: str) -> MemoryReadError:
    """Map a pymem exception onto a :class:`MemoryReadError` with the right ``access_denied`` flag.

    This is the seam between pymem's exception surface and the reader's structured faults: the
    resulting :class:`MemoryReadError` flows to :func:`~tekken_coach.reader.faults.classify_fault`,
    which turns ``access_denied=True`` into ``FaultKind.access_denied`` (report, do not
    retry-hammer) and everything else into ``process_lost`` (docs/02 §7). Pure and offline-testable
    with a fake pymem surface.
    """
    denied = _is_access_denied(exc, pymem_mod)
    code = _error_code_of(exc)
    code_note = f" (win error {code})" if code is not None else ""
    message = f"{context}: {type(exc).__name__}: {exc}{code_note}"
    return MemoryReadError(message, access_denied=denied)


def _open_readonly(pymem_mod: ModuleType, process_name: str) -> Any:
    """Attach to ``process_name`` with a read+query-only handle and return a pymem read handle.

    We do **not** use ``Pymem(process_name)``: its default open requests ``PROCESS_ALL_ACCESS``,
    which would grant write rights we must never hold (docs/02 §2). Instead we resolve the pid,
    ``OpenProcess`` with :data:`PROCESS_READ_ACCESS` only, and hand the resulting handle to an
    otherwise-unopened ``Pymem`` object whose ``read_bytes`` we use.
    """
    entry = pymem_mod.process.process_from_name(process_name)
    if entry is None:
        raise MemoryReadError(
            f"process not found: {process_name!r} — is Tekken 8 running? (docs/02 §7)"
        )
    pid = int(entry.th32ProcessID)
    handle = pymem_mod.ressources.kernel32.OpenProcess(PROCESS_READ_ACCESS, False, pid)
    if not handle:
        code = pymem_mod.ressources.kernel32.GetLastError()
        raise MemoryReadError(
            f"OpenProcess failed for pid {pid} (win error {code}) — "
            "access denied (anti-cheat/permission?) or process exited (docs/02 §7)",
            access_denied=(code == _ERROR_ACCESS_DENIED),
        )
    pm = pymem_mod.Pymem()
    pm.process_id = pid
    pm.process_handle = handle
    return pm


class WinMemorySource:
    """Read-only Windows :class:`~tekken_coach.reader.memory_source.MemorySource` (docs/02 §2).

    Implements exactly the two seam methods. Backed by pymem's ``read_bytes`` and
    ``module_from_name``; opened with a read+query-only handle. Satisfies the ``@runtime_checkable``
    ``MemorySource`` Protocol.

    Construction attaches to the process. For offline unit tests of the fault mapping, an already
    opened pymem handle and module can be injected via ``pymem_module`` / ``handle`` so no live
    process is required; production callers pass neither.
    """

    def __init__(
        self,
        process_name: str = GAME_PROCESS_NAME,
        *,
        pymem_module: ModuleType | None = None,
        handle: Any | None = None,
    ) -> None:
        self._pymem = pymem_module if pymem_module is not None else _require_pymem()
        self._process_name = process_name
        self._pm = handle if handle is not None else _open_readonly(self._pymem, process_name)

    def read(self, address: int, size: int) -> bytes:
        """Return ``size`` bytes at ``address`` via pymem's ``read_bytes`` (docs/02 §2)."""
        try:
            data: bytes = self._pm.read_bytes(address, size)
        except Exception as exc:  # noqa: BLE001 - pymem raises several classes; we normalize them
            raise map_pymem_error(
                exc, self._pymem, context=f"read 0x{address:x} (+{size})"
            ) from exc
        if len(data) != size:
            raise MemoryReadError(f"short read at 0x{address:x}: got {len(data)}, need {size}")
        return data

    def module_base(self, module: str) -> int:
        """Return the load base of ``module`` via pymem module enumeration (docs/02 §3)."""
        try:
            info = self._pymem.process.module_from_name(self._pm.process_handle, module)
        except Exception as exc:  # noqa: BLE001 - normalize pymem's exception surface
            raise map_pymem_error(exc, self._pymem, context=f"module_base {module!r}") from exc
        if info is None:
            raise MemoryReadError(f"module not loaded: {module!r}")
        return int(info.lpBaseOfDll)

    def regions(self) -> Sequence[MemoryRegion]:  # pragma: no cover - needs a live VirtualQueryEx
        """Enumerate committed readable heap via ``VirtualQueryEx`` (read-only, docs/02 §2, C4h).

        Binds :func:`enumerate_committed_regions` to ``kernel32.VirtualQueryEx`` directly (the
        installed pymem has no ``virtual_query`` helper). This is a *query* of the process map — one
        ``MEMORY_BASIC_INFORMATION`` per region — and reads no region content; the sweep reads bytes
        afterwards through :meth:`read`. The walk logic is offline-tested via
        :func:`enumerate_committed_regions` with a fake map; only this ``VirtualQueryEx`` binding is
        live-only (and was validated against the running game before it landed).
        """
        # ``ctypes.windll`` exists only on Windows, so it is resolved here (never offline) rather
        # than at import; typeshed hides it off-win32, hence the localized ignore.
        virtual_query_ex = ctypes.windll.kernel32.VirtualQueryEx  # type: ignore[attr-defined]
        virtual_query_ex.restype = ctypes.c_size_t
        virtual_query_ex.argtypes = (
            ctypes.c_void_p,  # hProcess
            ctypes.c_void_p,  # lpAddress
            ctypes.POINTER(_MemoryBasicInformation),  # lpBuffer
            ctypes.c_size_t,  # dwLength
        )
        handle = self._pm.process_handle
        buffer_size = ctypes.sizeof(_MemoryBasicInformation)

        def query(address: int) -> _BasicRegion | None:
            mbi = _MemoryBasicInformation()
            written = virtual_query_ex(
                handle, ctypes.c_void_p(address), ctypes.byref(mbi), buffer_size
            )
            if written == 0:  # past the top of the address space (or unqueryable) — end the walk
                return None
            return _BasicRegion(
                base=int(mbi.BaseAddress),
                size=int(mbi.RegionSize),
                state=int(mbi.State),
                protect=int(mbi.Protect),
                type=int(mbi.Type),
            )

        return enumerate_committed_regions(query)
