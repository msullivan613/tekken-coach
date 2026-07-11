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
* ``probe-state`` — stream the raw encoded state words while the user performs each state, so the
  value -> meaning map can be filled in by observation (C4e, docs/02 §8). The offsets say *where*
  the state lives; only this says what it *means*.

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
from collections.abc import Iterator
from pathlib import Path

from tekken_coach.reader.decode import read_scalar, resolve_player_base
from tekken_coach.reader.faults import ReaderError, classify_fault
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.offsets import OffsetTable
from tekken_coach.reader.probe import ChangeRecord, PollSample
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
        # The table records the memory char ids it was calibrated with (docs/02 §6); prefer those
        # over the movemap index, whose ids are a different (framedata) space and may be empty.
        known = set(table.known_char_ids) or _load_known_char_ids(args.movemap)
        report = run_doctor(
            source, table, known_char_ids=known, frames=args.frames, poll_interval=args.interval
        )
    except ReaderError as exc:
        return _report_fault(exc)

    print(f"\nreader self-check (docs/02 §6) — {'PASS' if report.ok else 'FAIL'}")
    for check in report.checks:
        print(f"  [{'ok' if check.ok else 'XX'}] {check.name}: {check.detail}")
    for note in report.notes:
        print(f"  [--] {note}")
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
        run_update_offsets_derive,
        run_update_offsets_holder,
    )

    def act_prompt(message: str) -> None:
        input("\n" + message + "\n")

    def progress(message: str) -> None:
        # The scans sweep a large module/heap and can run for minutes; stream progress to stderr so
        # the run is observable (the report itself still goes to stdout). Flush so it appears live.
        print(message, file=sys.stderr, flush=True)

    common = {
        "offsets_dir": args.offsets,
        "manifest_path": args.manifest,
        "version_override": args.version,
        "act_prompt": act_prompt,
    }
    try:
        if args.holder_scan:
            print("locating the player holder by its AoB code signature (T8 v3 model)...")
            table, report = run_update_offsets_holder(
                args.process, progress=progress, debug_dir=args.debug_dir, **common
            )
        elif args.derive:
            print("deriving the player layout from behavior (heap sweep; can take a minute)...")
            table, report = run_update_offsets_derive(args.process, progress=progress, **common)
        elif args.base_scan:
            print("scanning for the player-struct pointer (this can take a minute)...")
            table, report = run_update_offsets_base(args.process, progress=progress, **common)
        else:
            table, report = run_update_offsets(args.process, **common)
    except ReaderError as exc:
        return _report_fault(exc)

    print("\n" + report.render())
    if table is None:
        return 1
    return 0


def _probe_targets(table: OffsetTable) -> list[str]:
    """The player fields ``probe-state`` watches: the encoded state words plus move context."""
    spec = table.state_codes.encoded_state
    assert spec is not None  # caller checks
    context = [n for n in ("move_id", "move_frame", "counter_state") if n in table.players.fields]
    return context + sorted(spec.flags)


def _probe_row(
    source: MemorySource, table: OffsetTable, index: int, names: list[str]
) -> tuple[int, ...]:
    """Read one player's watched fields as raw integers (under either addressing model)."""
    base = resolve_player_base(source, table, index)
    fields = table.players.fields
    return tuple(int(read_scalar(source, base + fields[n].offset, fields[n].kind)) for n in names)


def _live_samples(
    source: MemorySource, table: OffsetTable, names: list[str], interval: float, seconds: float
) -> Iterator[PollSample]:  # pragma: no cover - live loop; the pure consumers are tested
    """Poll both players forever (or for ``seconds``), yielding one :class:`PollSample` per instant.

    The thin live shell over the tested pure core
    (:func:`~tekken_coach.reader.probe.change_records`) — mirroring how ``poll_frames`` (live) feeds
    ``evaluate_frames`` (pure). The chain is re-resolved every poll: the entity struct reallocates
    on every round and character change, which is what the anchor exists to survive (docs/02 §3).
    """
    import time  # noqa: PLC0415

    started = time.monotonic()
    while seconds <= 0 or time.monotonic() - started < seconds:
        rows = tuple(_probe_row(source, table, index, names) for index in (0, 1))
        yield PollSample(t=time.monotonic() - started, rows=rows)
        time.sleep(interval)


def _format_change(record: ChangeRecord, names: list[str]) -> str:
    """Render a change record as the aligned console row (same columns as the header)."""
    row = "  ".join(f"{record.fields[n]:>18}" for n in names)
    return f"{record.t:>7.2f}  P{record.player:<5}  {row}"


def _ensure_parent_dirs(*paths: Path | None) -> None:
    """Create parent dirs for probe-state's output files (skip ``None`` and bare-filename paths).

    Without this, the documented ``--record debug/state-obs.jsonl`` invocation crashes with a raw
    ``FileNotFoundError`` when ``debug/`` is absent; ``update-offsets --debug-dir`` already mkdir's
    its output the same way.
    """
    for out in paths:
        if out is not None and out.parent != Path():
            out.parent.mkdir(parents=True, exist_ok=True)


def _skeleton_path(args: argparse.Namespace) -> Path | None:
    """Where (if anywhere) the draft skeleton goes: ``--emit-skeleton`` wins, else auto on record.

    ``--record foo.jsonl`` auto-emits ``foo.skeleton.json`` beside it, so the owner performs the
    states once and walks away with both the observation log and the annotate-me draft.
    """
    if args.emit_skeleton:
        return Path(args.emit_skeleton)
    if args.record:
        return Path(args.record).with_suffix(".skeleton.json")
    return None


def probe_state_main(args: argparse.Namespace) -> int:
    """Stream the raw encoded state words for both players (the docs/02 §8 calibration protocol).

    The scan proves *where* the state words live; only observation proves what their **values** mean
    — no round-start oracle can, because nobody is in blockstun at round start. So this prints a
    line every time any watched value changes, and the user performs each state in turn (block a
    jab, eat a jab, get staggered, get thrown, ...) and reads off the raw values to bake into
    ``assets/offsets/state-map.json``.

    ``--record <path>`` additionally appends one JSONL object per change to a reviewable log, and
    (unless ``--emit-skeleton`` overrides the location) writes a draft state-map skeleton on exit
    listing every distinct value observed per encoded field with empty flag lists — so the owner
    annotates flags next to real values instead of alt-tabbing to transcribe integers by hand.

    Read-only and derivative: it resolves the same anchor the decoder uses and reads the same
    fields. It deliberately does **not** call ``decode_frame`` — that would need the very map we are
    here to build. It also never maps a value to a flag: that is the human's judgment (docs/02 §5).
    """
    from tekken_coach.reader.offsets import select_offset_table  # noqa: PLC0415
    from tekken_coach.reader.probe import build_skeleton, change_records  # noqa: PLC0415
    from tekken_coach.reader.version import detect_running_version  # noqa: PLC0415
    from tekken_coach.reader.win_source import WinMemorySource  # noqa: PLC0415

    try:
        source = WinMemorySource(args.process)
        version = args.version or detect_running_version(args.process)
        table = select_offset_table(version, args.offsets)
    except ReaderError as exc:
        return _report_fault(exc)

    spec = table.state_codes.encoded_state
    if spec is None:
        print(
            f"offset table {version} has no encoded-state map; nothing to probe. Run "
            "`update-offsets --base-scan` first (it writes the state-word offsets into the table).",
            file=sys.stderr,
        )
        return 1

    names = _probe_targets(table)
    encoded_fields = sorted(spec.flags)
    print(f"probing {len(names)} fields x 2 players (Ctrl-C to stop): {', '.join(names)}")
    print("perform one state at a time (block a jab, eat a jab, stagger, get thrown, jump, ...)")
    print(f"\n{'time':>7}  {'player':<6}  " + "  ".join(f"{n:>18}" for n in names))

    skeleton_path = _skeleton_path(args)
    # Create parent dirs for the outputs so the documented `--record debug/...` invocation does not
    # crash when the dir is absent (mirrors update-offsets --debug-dir).
    _ensure_parent_dirs(Path(args.record) if args.record else None, skeleton_path)
    # Opened once and closed in the finally below (its lifetime spans the whole poll loop), so a
    # `with` block would not fit; flushed per line so a Ctrl-C loses at most a tail (docs/02 §8).
    record_file = open(args.record, "w", encoding="utf-8") if args.record else None  # noqa: SIM115
    observed: list[ChangeRecord] = []
    try:
        for record in change_records(
            _live_samples(source, table, names, args.interval, args.seconds), names
        ):
            print(_format_change(record, names), flush=True)
            if record_file is not None:
                # Flush per line: the run is Ctrl-C-terminated, so a lost tail is fine but a
                # session-long buffer would lose everything on interrupt.
                record_file.write(record.to_jsonl() + "\n")
                record_file.flush()
            if skeleton_path is not None:
                observed.append(record)
    except ReaderError as exc:
        return _report_fault(exc)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        if record_file is not None:
            record_file.close()

    if args.record:
        print(f"recorded observations -> {args.record}")
    if skeleton_path is not None:
        skeleton = build_skeleton(observed, encoded_fields)
        skeleton_path.write_text(json.dumps(skeleton, indent=2) + "\n", encoding="utf-8")
        print(f"draft skeleton (fill the flags, set calibrated:true) -> {skeleton_path}")

    print(
        "\nNow map the raw values into assets/offsets/state-map.json and set `calibrated: true` "
        "(docs/02 §8). Re-run `doctor` to confirm."
    )
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
    p_doctor.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="seconds between frame polls (default 0.05). Must exceed one game frame (~0.017s at "
        "60 fps) or the live frame counter looks frozen and frame_monotonic falsely fails.",
    )
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
    p_update.add_argument(
        "--derive",
        action="store_true",
        help="C4h: FULLY DERIVE the layout from behavior — seed no within-struct offset or chain. "
        "Locates the entity struct on the heap by behavior, derives every field offset + stride + "
        "Jin's id, and reverse-scans for a static path that survives a round reset. Prefer this "
        "when the seeded --base-scan offsets have gone stale (a new season/patch).",
    )
    p_update.add_argument(
        "--holder-scan",
        action="store_true",
        help="C4i: adopt the live T8 holder model — find the player holder by an AoB CODE "
        "signature in .text (RIP-relative -> a .data slot), then read TWO per-player pointer slots "
        "(holder+0x30 / +0x38) to separate allocations. This is what the current community tools "
        "use; the AoB is patch-durable (re-source the pattern only if a patch moves the function).",
    )
    p_update.add_argument(
        "--debug-dir",
        default=None,
        help="(--holder-scan only) write a JSON capture of the round-start landing and the "
        "per-sample move_id/damage series to this directory for offline diagnosis of a failed "
        "behavioral oracle.",
    )
    p_update.set_defaults(func=update_offsets_main)

    p_probe = sub.add_parser(
        "probe-state",
        help="stream the raw encoded state words while you act (docs/02 §8 state-map calibration)",
    )
    p_probe.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    p_probe.add_argument("--version", default=None, help="override detected version")
    p_probe.add_argument("--interval", type=float, default=0.05, help="poll interval, seconds")
    p_probe.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (0 = ∞)")
    p_probe.add_argument(
        "--record",
        default=None,
        help="append one JSONL object per observed change to this path (a reviewable observation "
        "log), and — unless --emit-skeleton overrides the path — auto-write a draft state-map "
        "skeleton beside it on exit. Removes the alt-tab/transcribe loop (docs/02 §8).",
    )
    p_probe.add_argument(
        "--emit-skeleton",
        default=None,
        help="write the draft state-map skeleton to this path on exit (every distinct value seen "
        "per encoded field, with empty flag lists for a human to annotate). Defaults to "
        "<record>.skeleton.json when --record is given.",
    )
    p_probe.set_defaults(func=probe_state_main)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: object = args.func
    assert callable(func)
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
