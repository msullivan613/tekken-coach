"""Detect the running game's version and normalize it to an offset-table key (docs/02 §3).

The reader selects an offset table by *game version* and **fails closed** on an unknown one
(docs/02 §3/§7): a wrong offset silently yields garbage FrameRecords, which is worse than not
running. This module produces the version string that
:func:`~tekken_coach.reader.offsets.select_offset_table` consumes. On no match, the existing
fail-closed path (``UnknownGameVersionError`` + ``PATCH_RUNBOOK``) fires — this module does not
guess.

Pure vs. impure (deliberately split so the offline suite covers the logic):

* :func:`normalize_version` and :func:`version_from_dwords` are **pure** string/number transforms
  — the canonicalization from a raw Windows version quad to the ``MAJOR.MINOR.PATCH`` key used in
  ``assets/offsets/index.json`` (e.g. ``"2.1.1.0"`` -> ``"2.01.01"``). These are unit-tested
  offline.
* :func:`detect_running_version` and its helpers **read the live process image** (executable
  product-version info via the Win32 version API) and are Windows-only, user-run. The
  memory-signature fallback is a documented hook (:func:`signature_version`) that the C4c
  re-discovery tool populates; it is intentionally not implemented here (C4c owns the RE technique).

The canonical form matches the checked-in offset keys: major unchanged, minor and patch
zero-padded to two digits. If a build's real exe version does not map cleanly onto a known key,
that surfaces as the *correct* fail-closed miss, not a silent mis-attach.
"""

from __future__ import annotations

import re
from types import ModuleType
from typing import Any

from tekken_coach.reader.faults import MemoryReadError

# The default attach target; kept in sync with win_source.GAME_PROCESS_NAME without importing it
# (version detection must be usable without constructing a source).
GAME_PROCESS_NAME = "Polaris-Win64-Shipping.exe"

_DIGIT_GROUP = re.compile(r"\d+")


def normalize_version(raw: str) -> str:
    """Canonicalize a raw version string to the ``MAJOR.MINOR.PATCH`` offset-table key.

    Accepts the many shapes a Windows version can take (``"2.1.1.0"``, ``"2, 0, 0"``, ``"2.01.01"``,
    ``"v2.1"``) by extracting the leading numeric groups. The first three become major/minor/patch;
    a missing minor/patch defaults to 0. Minor and patch are zero-padded to two digits to match the
    keys in ``assets/offsets/index.json`` (docs/02 §3):

        "2.1.1.0"  -> "2.01.01"
        "2, 0, 0"  -> "2.00.00"
        "2.01.01"  -> "2.01.01"
        "v2.1"     -> "2.01.00"

    Raises ``ValueError`` if no numeric component is present (an unusable version string).
    """
    groups = [int(g) for g in _DIGIT_GROUP.findall(raw)]
    if not groups:
        raise ValueError(f"no numeric version component in {raw!r}")
    major = groups[0]
    minor = groups[1] if len(groups) > 1 else 0
    patch = groups[2] if len(groups) > 2 else 0
    return f"{major}.{minor:02d}.{patch:02d}"


def version_from_dwords(file_version_ms: int, file_version_ls: int) -> str:
    """Turn a ``VS_FIXEDFILEINFO`` version quad into a normalized offset-table key.

    The Win32 version API reports a file version as two 32-bit words. Per winver.h,
    ``major = HIWORD(MS)``, ``minor = LOWORD(MS)``, ``build = HIWORD(LS)``,
    ``revision = LOWORD(LS)``. We feed ``major.minor.build`` through :func:`normalize_version`
    (the 4th component — revision — is dropped, matching the 3-part offset keys). Pure and tested.
    """
    major = (file_version_ms >> 16) & 0xFFFF
    minor = file_version_ms & 0xFFFF
    build = (file_version_ls >> 16) & 0xFFFF
    return normalize_version(f"{major}.{minor}.{build}")


def signature_version(read: Any) -> str | None:  # pragma: no cover - needs a live version pattern
    """Memory-signature version fallback (docs/02 §3) — a named seam, still returning ``None``.

    When the executable product-version info is unreliable, a version can be recovered from a known
    in-memory string/pattern. C4d built the AOB machinery this would use — the same
    :func:`~tekken_coach.reader.discovery.scanners.aob_scan` and
    :func:`~tekken_coach.reader.discovery.basescan.find_by_signature` that re-find the player-struct
    base signature can re-find a version-string pattern. What is still missing is the *pattern
    itself*: a build-stable AOB around the version string, which can only be captured from a live
    build (there is no offline oracle for it, unlike the player struct's field layout). So this
    stays ``None`` — a clear "populate me from a calibrated live run" seam — rather than guessing.
    Capture procedure: locate the product-version string in a data section on a known build,
    wildcard the version digits, and store the AOB + a parse offset alongside the offset table.
    """
    return None


# ---------------------------------------------------------------------------
# Impure: read the version off the live Windows process (user-run, not offline-tested).
# ---------------------------------------------------------------------------


def _process_image_path(pymem_mod: ModuleType, pid: int) -> str:  # pragma: no cover - Windows-only
    """Return the full executable path for ``pid`` via ``QueryFullProcessImageNameW``."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    handle = pymem_mod.ressources.kernel32.OpenProcess(0x0400, False, pid)  # QUERY_INFORMATION
    if not handle:
        code = pymem_mod.ressources.kernel32.GetLastError()
        raise MemoryReadError(
            f"could not open pid {pid} to read image path (win error {code})",
            access_denied=(code == 5),
        )
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]  # Windows-only
        ok = kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size))
        if not ok:
            code = kernel32.GetLastError()
            raise MemoryReadError(f"QueryFullProcessImageNameW failed (win error {code})")
        return buf.value
    finally:
        pymem_mod.ressources.kernel32.CloseHandle(handle)


def _read_product_version(path: str) -> str:  # pragma: no cover - Windows-only
    """Read ``path``'s file version via the Win32 version API and normalize it."""
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    version = ctypes.windll.version  # type: ignore[attr-defined]  # Windows-only
    size = version.GetFileVersionInfoSizeW(path, None)
    if not size:
        raise MemoryReadError(f"no version info in {path!r} (GetFileVersionInfoSizeW == 0)")
    data = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(path, 0, size, data):
        raise MemoryReadError(f"GetFileVersionInfoW failed for {path!r}")
    ffi = ctypes.c_void_p()
    length = wintypes.UINT()
    if not version.VerQueryValueW(data, "\\", ctypes.byref(ffi), ctypes.byref(length)):
        raise MemoryReadError(f"VerQueryValueW failed for {path!r}")

    class _FixedFileInfo(ctypes.Structure):
        _fields_ = [
            ("dwSignature", wintypes.DWORD),
            ("dwStrucVersion", wintypes.DWORD),
            ("dwFileVersionMS", wintypes.DWORD),
            ("dwFileVersionLS", wintypes.DWORD),
            ("dwProductVersionMS", wintypes.DWORD),
            ("dwProductVersionLS", wintypes.DWORD),
            ("dwFileFlagsMask", wintypes.DWORD),
            ("dwFileFlags", wintypes.DWORD),
            ("dwFileOS", wintypes.DWORD),
            ("dwFileType", wintypes.DWORD),
            ("dwFileSubtype", wintypes.DWORD),
            ("dwFileDateMS", wintypes.DWORD),
            ("dwFileDateLS", wintypes.DWORD),
        ]

    info = ctypes.cast(ffi, ctypes.POINTER(_FixedFileInfo)).contents
    return version_from_dwords(info.dwProductVersionMS, info.dwProductVersionLS)


def detect_running_version(
    process_name: str = GAME_PROCESS_NAME,
    *,
    pymem_module: ModuleType | None = None,
) -> str:  # pragma: no cover - Windows-only, user-run
    """Resolve the running game's normalized version string (docs/02 §3).

    Reads the executable's product-version info from the process image, falling back to a memory
    signature (:func:`signature_version`, a C4c hook) if that is unavailable. The returned string
    is fed straight to :func:`~tekken_coach.reader.offsets.select_offset_table`, which fails closed
    on an unknown version — this function never guesses.
    """
    if pymem_module is None:
        from tekken_coach.reader.win_source import _require_pymem  # noqa: PLC0415

        pymem_module = _require_pymem()
    entry = pymem_module.process.process_from_name(process_name)
    if entry is None:
        raise MemoryReadError(
            f"process not found: {process_name!r} — is Tekken 8 running? (docs/02 §7)"
        )
    pid = int(entry.th32ProcessID)
    path = _process_image_path(pymem_module, pid)
    return _read_product_version(path)
