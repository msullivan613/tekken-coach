"""Thin runnable entry points for C4b: smoke-attach, doctor, capture (docs/02 §6, docs/01).

This is the **command** layer — the polished top-level ``tekken-coach`` CLI with mode triggers is
C6 (docs/07). These are minimal, Windows-run wrappers that let a user exercise the reader against a
live game today:

* ``smoke``   — raw attach sanity: open the process read-only, resolve the module base, read a few
  bytes. Proves attach + read work *before* offsets are known (useful pre-C4c).
* ``doctor``  — attach, detect version, select the offset table, run the §6 self-check, print the
  report (and the §4 runbook on failure).
* ``capture`` — attach, detect version, poll ``N`` frames, write a FrameRecord JSON fixture.
* ``update-offsets`` — attach, re-discover offsets at the Jin-vs-Kazuya setup, and write a
  candidate ``assets/offsets/<version>.json`` + diagnostic report (C4c, docs/02 §4).

Silent-producer boundary (docs/02 §2): the reader *library* prints nothing; **this command layer**
does. All rendering lives here, not in ``decode``/``doctor``/``faults``. ``doctor``/``capture`` only
go green once C4c has written real offsets for the running build; before that they fail closed with
the runbook, which is the correct behavior.

Run on Windows (native Python, not WSL):

    python -m tekken_coach.reader.commands smoke
    python -m tekken_coach.reader.commands doctor
    python -m tekken_coach.reader.commands capture --count 300 --out captures/set1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tekken_coach.reader.faults import ReaderError, classify_fault
from tekken_coach.reader.version import GAME_PROCESS_NAME

DEFAULT_OFFSETS_DIR = "assets/offsets"
DEFAULT_MOVEMAP_DIR = "assets/movemap"


def _load_known_char_ids(movemap_dir: str | Path) -> set[int]:
    """Load the known character IDs from the movemap index (docs/02 §6 char-id check).

    Skips characters whose ``char_id`` is not yet seeded (``null``). Returns an empty set if the
    index is absent — the doctor's char-id check then simply reports all ids as unknown, which is
    an honest "offsets/movemap not ready" signal rather than a crash.
    """
    index_path = Path(movemap_dir) / "index.json"
    if not index_path.exists():
        return set()
    data = json.loads(index_path.read_text(encoding="utf-8"))
    return {int(c["char_id"]) for c in data.get("characters", []) if c.get("char_id") is not None}


def _report_fault(exc: ReaderError) -> int:
    """Print a classified reader fault and its runbook (if any); return a nonzero exit code."""
    fault = classify_fault(exc)
    print(f"error [{fault.kind.value}]: {fault.message}", file=sys.stderr)
    if fault.runbook:
        print("\n" + fault.runbook, file=sys.stderr)
    return 1


def smoke_main(args: argparse.Namespace) -> int:
    """Raw attach smoke: attach read-only, resolve the module base, read a few bytes."""
    from tekken_coach.reader.win_source import WinMemorySource  # noqa: PLC0415

    try:
        source = WinMemorySource(args.process)
        base = source.module_base(args.process)
        sample = source.read(base, 16)
    except ReaderError as exc:
        return _report_fault(exc)
    print(f"attached read-only to {args.process!r}")
    print(f"module base: 0x{base:x}")
    print(f"first 16 bytes at base: {sample.hex(' ')}")
    try:
        from tekken_coach.reader.version import detect_running_version  # noqa: PLC0415

        print(f"detected version: {detect_running_version(args.process)}")
    except ReaderError as exc:
        print(f"(version detection unavailable: {exc})", file=sys.stderr)
    print("smoke OK — attach + read work. (Offsets validated separately by `doctor`.)")
    return 0


def doctor_main(args: argparse.Namespace) -> int:
    """Attach, detect version, select the table, run the §6 self-check, and print the report."""
    from tekken_coach.reader.doctor import run_doctor  # noqa: PLC0415
    from tekken_coach.reader.offsets import select_offset_table  # noqa: PLC0415
    from tekken_coach.reader.version import detect_running_version  # noqa: PLC0415
    from tekken_coach.reader.win_source import WinMemorySource  # noqa: PLC0415

    try:
        source = WinMemorySource(args.process)
        version = args.version or detect_running_version(args.process)
        print(f"detected game version: {version}")
        table = select_offset_table(version, args.offsets)
        known = _load_known_char_ids(args.movemap)
        report = run_doctor(source, table, known_char_ids=known, frames=args.frames)
    except ReaderError as exc:
        return _report_fault(exc)

    print(f"\nreader self-check (docs/02 §6) — {'PASS' if report.ok else 'FAIL'}")
    for check in report.checks:
        print(f"  [{'ok' if check.ok else 'XX'}] {check.name}: {check.detail}")
    if not report.ok and report.runbook:
        print("\n" + report.runbook, file=sys.stderr)
        return 1
    return 0


def capture_main(args: argparse.Namespace) -> int:
    """Attach, detect version, poll ``N`` frames, and write a FrameRecord JSON fixture."""
    from tekken_coach.reader.capture import capture_live  # noqa: PLC0415

    try:
        capture = capture_live(
            args.process,
            args.offsets,
            args.count,
            args.out,
            version_override=args.version,
        )
    except ReaderError as exc:
        return _report_fault(exc)
    dropped = sum(capture.meta.gaps)
    print(
        f"captured {capture.meta.frame_count} frames (version {capture.meta.game_version}, "
        f"{dropped} dropped) -> {args.out}"
    )
    return 0


def update_offsets_main(args: argparse.Namespace) -> int:
    """Attach read-only, re-discover offsets at the Jin-vs-Kazuya setup, write a candidate table.

    Clean-room re-discovery (docs/02 §4/§5). Two techniques share this command:

    * default — C4c heap value-scan (module-relative windows).
    * ``--base-scan`` — C4d code-signature: scan the module's static data for the pointer that leads
      to the heap-allocated player struct and follow a pointer chain. This is the robust path on
      Tekken 8, whose entity struct reallocates on every character change / round (docs/02 §3).

    Prints the diagnostic report either way; returns nonzero when the confident core did not resolve
    (the report then says which anchors to widen).
    """
    from tekken_coach.reader.discovery.orchestrate import (  # noqa: PLC0415
        run_update_offsets,
        run_update_offsets_base,
    )

    def act_prompt(message: str) -> None:
        input("\n" + message + "\n")

    runner = run_update_offsets_base if args.base_scan else run_update_offsets
    try:
        table, report = runner(
            args.process,
            offsets_dir=args.offsets,
            manifest_path=args.manifest,
            version_override=args.version,
            act_prompt=act_prompt,
        )
    except ReaderError as exc:
        return _report_fault(exc)

    print("\n" + report.render())
    if table is None:
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tekken_coach.reader.commands",
        description="C4b reader entry points (smoke/doctor/capture). Windows-only; run in "
        "native Python, not WSL. The polished CLI is C6.",
    )
    parser.add_argument("--process", default=GAME_PROCESS_NAME, help="target process/module name")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="raw attach + read sanity (works before offsets exist)")
    p_smoke.set_defaults(func=smoke_main)

    p_doctor = sub.add_parser("doctor", help="run the docs/02 §6 self-check")
    p_doctor.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    p_doctor.add_argument("--movemap", default=DEFAULT_MOVEMAP_DIR)
    p_doctor.add_argument("--version", default=None, help="override detected version")
    p_doctor.add_argument("--frames", type=int, default=8)
    p_doctor.set_defaults(func=doctor_main)

    p_capture = sub.add_parser("capture", help="capture N frames to a JSON fixture")
    p_capture.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    p_capture.add_argument("--version", default=None, help="override detected version")
    p_capture.add_argument("--count", type=int, default=300, help="number of frames to poll")
    p_capture.add_argument("--out", required=True, help="output JSON fixture path")
    p_capture.set_defaults(func=capture_main)

    p_update = sub.add_parser(
        "update-offsets", help="re-discover offsets at the Jin-vs-Kazuya setup (C4c)"
    )
    p_update.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    p_update.add_argument(
        "--manifest", default="assets/offsets/probe-manifest.json", help="probe manifest path"
    )
    p_update.add_argument("--version", default=None, help="override detected version")
    p_update.add_argument(
        "--base-scan",
        action="store_true",
        help="C4d: locate the heap player struct via a static-pointer code signature + chain "
        "(robust on Tekken 8's reallocating entity struct) instead of the C4c heap value-scan",
    )
    p_update.set_defaults(func=update_offsets_main)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: object = args.func
    assert callable(func)
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
