"""The read-only invariant, enforced by construction and by grep (docs/02 §2, §5).

The reader opens the process for *read* only; the bot/input half of the ancestral TekkenBot is
removed from the port, not merely unused. These tests assert that posture two ways:

1. No write/inject primitive appears *anywhere* in the ``tekken_coach.reader`` package source.
2. The ``MemorySource`` seam exposes no mutating method — so no call site can even name one.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import tekken_coach.reader as reader_pkg
from tekken_coach.reader.memory_source import MemorySource

# Tokens that would indicate a memory-write or input-synthesis capability. Case-insensitive
# substring match against every source file in the package. If any appears, the read-only
# invariant is broken (docs/02 §2). Kept deliberately broad: OS write calls, pymem/ctypes write
# helpers, and input-injection APIs.
FORBIDDEN_TOKENS = (
    "writeprocessmemory",
    "ntwritevirtualmemory",
    "write_bytes",
    "write_int",
    "write_uint",
    "write_short",
    "write_long",
    "write_float",
    "write_double",
    "write_string",
    "write_ctype",
    "write_memory",
    "sendinput",
    "keybd_event",
    "mouse_event",
    "postmessage",
    "sendmessage",
    "setforegroundwindow",
    "pydirectinput",
    "pyautogui",
    "vkeycode",
)


def _reader_source_files() -> list[Path]:
    root = Path(inspect.getfile(reader_pkg)).parent
    return sorted(root.rglob("*.py"))


def test_reader_package_has_source_files() -> None:
    # Guard against the grep silently passing because it found nothing to scan.
    files = _reader_source_files()
    assert files, "no reader source files found to scan"


def test_no_write_or_inject_primitive_in_reader() -> None:
    offenders: dict[str, list[str]] = {}
    for path in _reader_source_files():
        text = path.read_text(encoding="utf-8").lower()
        hits = [tok for tok in FORBIDDEN_TOKENS if tok in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"read-only invariant violated — write/inject tokens found: {offenders}"


def test_memory_source_seam_exposes_no_write_method() -> None:
    # The Protocol's only public members are read + module_base. Nothing that mutates.
    members = {name for name in dir(MemorySource) if not name.startswith("_")}
    assert members == {"read", "module_base"}, f"unexpected MemorySource surface: {members}"
    # And no member name hints at writing/injecting.
    assert not any("write" in m or "inject" in m or "send" in m for m in members)
