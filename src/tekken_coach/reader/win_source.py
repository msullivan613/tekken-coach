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

from types import ModuleType
from typing import Any

from tekken_coach.reader.faults import MemoryReadError

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
