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

from tekken_coach.reader.decode import (
    DerivedPhase,
    MatchPhaseTracker,
    MemoryReadError,
    decode_frame,
    derive_match_phase,
    read_match_flag,
    resolve_anchor,
    resolve_component,
    resolve_player_base,
)
from tekken_coach.reader.faults import ReaderError, classify_fault
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.monitor import PlayerView, monitor_lines, views_of
from tekken_coach.reader.offsets import OffsetTable
from tekken_coach.reader.probe import (
    ChangeRecord,
    PollRate,
    PollSample,
    ReadPlan,
    WatchPoint,
    assemble_row,
    build_read_plan,
    due_for_beat,
    heartbeat_line,
    is_wide_sweep,
    parse_watch,
    parse_watch_behind,
)
from tekken_coach.reader.version import GAME_PROCESS_NAME
from tekken_coach.schemas import MatchState

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


def _table_points(table: OffsetTable, names: list[str]) -> list[WatchPoint]:
    """The default watch points: each named table field at its own offset/kind."""
    fields = table.players.fields
    return [WatchPoint(name=n, offset=fields[n].offset, kind=fields[n].kind) for n in names]


def _plan_address(source: MemorySource, base: int, plan: ReadPlan) -> int:
    """The address ``plan``'s block starts at, resolved **fresh** for this poll.

    Never cache a landing. The pointer is re-dereferenced every poll because the object it points at
    can be freed and reallocated between polls; a cached address would then read some other object's
    bytes and report them as this player's input — silent garbage that looks like data. Re-resolving
    costs one 8-byte read and is the difference between an observation and a fiction.
    """
    if plan.slot is None:
        return base + plan.start
    return resolve_component(source, base, plan.slot.to_component()) + plan.start


def _read_points_at(
    source: MemorySource, base: int, plans: list[ReadPlan], width: int
) -> tuple[int | float, ...]:
    """Read every watch point at struct ``base`` — one block read per object, sliced offline.

    bool8 folds to int; f32 stays float. See :func:`~tekken_coach.reader.probe.block_span` for why
    this is a block read and not a read per offset (#10's sweep managed 4.7 Hz doing the latter).
    """
    blocks = [source.read(_plan_address(source, base, plan), plan.size) for plan in plans]
    return assemble_row(plans, blocks, width)


def _watch_bases(source: MemorySource, table: OffsetTable, is_global: bool) -> list[int]:
    """The struct base(s) a poll reads: the global/match struct (one) or both players (two).

    Re-resolved every poll: the global anchor is static, but the player structs reallocate every
    round/character change — which is exactly what the anchors exist to survive (docs/02 §3).
    """
    if is_global:
        return [resolve_anchor(source, table.global_struct.anchor)]
    return [resolve_player_base(source, table, index) for index in (0, 1)]


def _live_samples(
    source: MemorySource,
    table: OffsetTable,
    plans: list[ReadPlan],
    width: int,
    interval: float,
    seconds: float,
    *,
    is_global: bool = False,
    rate: PollRate | None = None,
) -> Iterator[PollSample]:  # pragma: no cover - live loop; the pure consumers are tested
    """Poll the watched struct(s) forever (or for ``seconds``), yielding one :class:`PollSample`.

    The thin live shell over the tested pure core
    (:func:`~tekken_coach.reader.probe.change_records`) — mirroring how ``poll_frames`` (live) feeds
    ``evaluate_frames`` (pure). ``is_global`` watches the single global/match struct instead of the
    two players (for locating match_phase / game_mode, docs/02 §4).
    """
    import time  # noqa: PLC0415

    started = time.monotonic()
    while seconds <= 0 or time.monotonic() - started < seconds:
        bases = _watch_bases(source, table, is_global)
        rows = tuple(_read_points_at(source, base, plans, width) for base in bases)
        now = time.monotonic() - started
        if rate is not None:
            rate.polls += 1
            rate.elapsed = now
        yield PollSample(t=now, rows=rows)
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


def _slots_main(
    args: argparse.Namespace, source: MemorySource, table: OffsetTable
) -> int:  # pragma: no cover - live enumeration; classify_slots/format_slot_table are tested
    """``probe-state --slots``: enumerate the player struct's plausible pointer slots (#11 Stage 1).

    No protocol, no timing discipline, no presses — ~10 seconds standing in a Practice match is
    enough, because it asks a question about *structure*, not behaviour: which 8-byte slots hold a
    readable heap address, and which of those look per-player. The few polls it takes exist only to
    tell a stable slot from a churning one.

    Everything decidable lives in :mod:`tekken_coach.reader.slots`; this is the live shell that
    block-reads both structs and hands the bytes over.
    """
    import time  # noqa: PLC0415

    from tekken_coach.reader.slots import (  # noqa: PLC0415
        DEFAULT_SLOT_END,
        DEFAULT_SLOT_START,
        RegionIndex,
        classify_slots,
        format_slot_table,
    )

    start, end = DEFAULT_SLOT_START, DEFAULT_SLOT_END
    size = end - start
    print(f"enumerating pointer slots in 0x{start:x}-0x{end:x} for 2 players")
    print("stand in a Practice match; no presses needed — this reads structure, not behaviour.")

    try:
        # The pointer-validity oracle, read once: VirtualQueryEx's committed map. This is a query of
        # what is mapped — it reads no contents and adds no write path (docs/02 §2).
        regions = RegionIndex(source.regions())
        samples: list[list[bytes]] = []
        for _ in range(args.polls):
            bases = [resolve_player_base(source, table, index) for index in (0, 1)]
            # One block read per player per poll — the whole struct region at once.
            samples.append([source.read(base + start, size) for base in bases])
            time.sleep(args.interval)
    except ReaderError as exc:
        return _report_fault(exc)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 1

    findings = classify_slots(samples, regions, start=start)
    print()
    for line in format_slot_table(findings, regions, top=args.top):
        print(line)
    if args.record:
        _ensure_parent_dirs(Path(args.record))
        Path(args.record).write_text(
            json.dumps(
                {
                    "range": [start, end],
                    "polls": args.polls,
                    "slots": [
                        {
                            "offset": f.offset,
                            "values": [f"0x{v:x}" for v in f.values],
                            "plausible": list(f.plausible),
                            "stable": f.stable,
                            "per_player": f.per_player,
                            "chase": f.chase,
                        }
                        for f in findings
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nslot table -> {args.record}")
    return 0


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

    if args.slots:
        return _slots_main(args, source, table)

    if args.watch and args.watch_behind:
        print("--watch and --watch-behind are alternatives; pass one.", file=sys.stderr)
        return 1

    if args.is_global and not args.watch:
        print(
            "--global requires --watch: the global/match struct has no default field set to probe. "
            "Sweep a region to locate match_phase/game_mode, e.g. "
            "--global --watch 0xd2e0-0xd4c0:u32,0x0-0x80:u32",
            file=sys.stderr,
        )
        return 1

    spec = table.state_codes.encoded_state
    if not (args.watch or args.watch_behind) and spec is None:
        print(
            f"offset table {version} has no encoded-state map; nothing to probe. Run "
            "`update-offsets --base-scan` first (it writes the state-word offsets into the table).",
            file=sys.stderr,
        )
        return 1

    # Default: watch the table's encoded-state fields (+ move context). `--watch` overrides that
    # with ad-hoc candidate offsets (a range sweep locates a field whose seeded offset went stale,
    # docs/02 §8); `--global` points the sweep at the global/match struct instead of the players.
    if args.watch or args.watch_behind:
        flag = "--watch" if args.watch else "--watch-behind"
        parse = parse_watch if args.watch else parse_watch_behind
        try:
            points = parse(args.watch or args.watch_behind)
        except ValueError as exc:
            print(f"invalid {flag}: {exc}", file=sys.stderr)
            return 1
        names = [p.name for p in points]
        skeleton_fields = names  # summarize distinct values for every watched candidate
    else:
        assert spec is not None
        names = _probe_targets(table)
        points = _table_points(table, names)
        skeleton_fields = sorted(spec.flags)

    # One block read per object per poll instead of one per offset: #10's 5376-offset sweep ran at
    # 4.7 Hz doing 10752 syscalls a poll, too slow to resolve a 2s hold. This makes it len(plans)x2.
    plans = build_read_plan(points)
    struct_label = "the global/match struct" if args.is_global else "2 players"
    # A wide sweep's column list and per-change rows are tens of KB each — printing them floods the
    # terminal and starves the poll loop. Name the points only when they fit on screen.
    wide = is_wide_sweep(names)
    header = f"probing {len(names)} fields x {struct_label} (Ctrl-C to stop)"
    print(header if wide else f"{header}: {', '.join(names)}")
    print(f"reading {len(plans)} block(s) per struct per poll ({len(names)} points)")
    if args.is_global:
        print("move through phases: menu -> match -> round -> round over -> results -> menu")
    elif wide:
        print("follow the `input-protocol` checklist — the t below is the clock it is written in.")
    else:
        print(
            "perform one state at a time (block a jab, eat a jab, stagger, get thrown, jump, ...)"
        )
    if not wide:
        print(f"\n{'time':>7}  {'struct':<6}  " + "  ".join(f"{n:>14}" for n in names))

    skeleton_path = _skeleton_path(args)
    # Create parent dirs for the outputs so the documented `--record debug/...` invocation does not
    # crash when the dir is absent (mirrors update-offsets --debug-dir).
    _ensure_parent_dirs(Path(args.record) if args.record else None, skeleton_path)
    # Opened once and closed in the finally below (its lifetime spans the whole poll loop), so a
    # `with` block would not fit; flushed per line so a Ctrl-C loses at most a tail (docs/02 §8).
    record_file = open(args.record, "w", encoding="utf-8") if args.record else None  # noqa: SIM115
    observed: list[ChangeRecord] = []
    rate = PollRate()
    try:
        last_beat: float | None = None
        for changes, record in enumerate(
            change_records(
                _live_samples(
                    source,
                    table,
                    plans,
                    len(names),
                    args.interval,
                    args.seconds,
                    is_global=args.is_global,
                    rate=rate,
                ),
                names,
            ),
            start=1,
        ):
            # A whole-struct sweep prints a ~100 KB row per change; rendering that is slower than
            # the game, so it would wreck the very pass it is recording. Heartbeat instead.
            if not wide:
                print(_format_change(record, names), flush=True)
            elif due_for_beat(last_beat, record.t):
                last_beat = record.t
                print(heartbeat_line(record.t, changes, len(names), rate.hz), flush=True)
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

    print(rate.summary())
    if args.record:
        print(f"recorded observations -> {args.record}")
    if skeleton_path is not None:
        skeleton = build_skeleton(observed, skeleton_fields)
        skeleton_path.write_text(json.dumps(skeleton, indent=2) + "\n", encoding="utf-8")
        print(f"draft skeleton (fill the flags, set calibrated:true) -> {skeleton_path}")

    print(
        "\nNow map the raw values into assets/offsets/state-map.json and set `calibrated: true` "
        "(docs/02 §8). Re-run `doctor` to confirm."
    )
    return 0


def moveset_probe_main(args: argparse.Namespace) -> int:  # pragma: no cover - live discovery shell
    """``moveset-probe``: find the character's ``tk_moveset`` by heap shape + gate scan (brief #19).

    The #18 assumption — the header is a direct player pointer slot — was disproved live: none of
    the plausible direct slots landed on a header. So this scans the **heap** for the header's shape
    (bounded cancels/moves counts + array pointers that land in a region), gates the handful of
    survivors on the character's known ``move_id -> notation`` anchors (which read the *static*
    cancels, so an idle player is fine — brief #19 Part A), and then reverse-scans a durable
    player-relative :class:`~tekken_coach.reader.offsets.ComponentAnchor` path to record. Read-only;
    degrades gracefully at the menu like the other live commands.
    """
    import time  # noqa: PLC0415

    from tekken_coach.reader.discovery.moveset_scan import (  # noqa: PLC0415
        derive_reference_path,
        scan_moveset,
    )
    from tekken_coach.reader.moveset import gate_pairs_for  # noqa: PLC0415
    from tekken_coach.reader.offsets import select_offset_table  # noqa: PLC0415
    from tekken_coach.reader.slots import DEFAULT_SLOT_END  # noqa: PLC0415
    from tekken_coach.reader.version import detect_running_version  # noqa: PLC0415
    from tekken_coach.reader.win_source import WinMemorySource  # noqa: PLC0415

    try:
        source = WinMemorySource(args.process)
        version = args.version or detect_running_version(args.process)
        table = select_offset_table(version, args.offsets)
    except ReaderError as exc:
        return _report_fault(exc)

    pairs = gate_pairs_for(args.char)
    if pairs is None:
        print(
            f"no decoder-gate anchors recorded for {args.char!r}; the scan needs a character whose "
            "known move_id -> notation ids are in GATE_PAIRS_BY_CHAR (brief #19). Bryan is there.",
            file=sys.stderr,
        )
        return 1

    index = args.player - 1
    print(f"heap shape+gate scan for {args.char}'s tk_moveset (P{args.player}) on {version}")
    print("stand in a Practice match as the target character; idle is fine (no presses needed).\n")

    try:
        elapsed_start = time.perf_counter()
        scan = scan_moveset(source, pairs=pairs, progress=lambda m: print(m))
        elapsed = time.perf_counter() - elapsed_start
    except ReaderError as exc:
        return _report_fault(exc)
    except KeyboardInterrupt:
        print("\nstopped.")
        return 1

    print(f"\nscanned in {elapsed:.1f}s: {len(scan.survivors)} shape-survivor(s)\n")
    print(f"{'header':>18}  {'cancels':>8}  {'moves':>7}  {'gate':>6}")
    for cand in scan.candidates:
        gate = f"{sum(r.found for r in cand.gate)}/{len(cand.gate)}"
        print(
            f"0x{cand.header_addr:>16x}  {cand.header.cancels_count:>8}  "
            f"{cand.header.moves_count:>7}  {gate:>6}"
        )

    winner = scan.winner
    print()
    if winner is None:
        if len(scan.matches) == 0:
            print("no header reproduced the anchors. Confirm you are in a match as the target")
            print("character; if it persists the gate anchors may need this character's ids.")
        return 0
    print(f"MOVESET HEADER FOUND: 0x{winner.header_addr:x} (gate reproduced all anchors)")
    if len(scan.matches) > 1:
        others = ", ".join(f"0x{m.header_addr:x}" for m in scan.matches[1:])
        print(f"(also passed: {others} — one per loaded character; confirm which is this player's)")

    try:
        player_base = resolve_player_base(source, table, index)
        anchor = derive_reference_path(
            source,
            scan.buffers,
            header_addr=winner.header_addr,
            player_base=player_base,
            player_struct_span=DEFAULT_SLOT_END,
            progress=lambda m: print(m),
        )
    except ReaderError as exc:
        return _report_fault(exc)

    print()
    if anchor is not None:
        path = ", ".join(f"{o}" for o in anchor.pointer_path)
        print("record the durable path in players.moveset_slot:")
        print(
            f'  "moveset_slot": {{ "slot_offset": {anchor.slot_offset}, '
            f'"pointer_path": [{path}], "fields": {{}} }}'
        )
    else:
        print("no durable player-relative path found. Record NOTHING and let moveset-build re-run")
        print("this shape+gate scan at startup to relocate the header (slower, but self-healing).")
    return 0


def moveset_build_main(args: argparse.Namespace) -> int:  # pragma: no cover - live build shell
    """``moveset-build``: read the live moveset -> decode -> join -> write the movemap (Phase 2).

    Depends on the Phase-1 discovery: with ``players.moveset_slot`` still ``null`` it reports
    "moveset offset not yet discovered" and exits cleanly (brief #18). Owner attribution also needs
    the ``tk_move`` cancel-range layout (undocumented in our tables — see
    :class:`~tekken_coach.reader.moveset.MoveLayout`); supply it with ``--move-size`` /
    ``--cancel-ptr-offset`` / ``--cancel-count-offset`` / ``--neutral-move-id`` once the live run
    confirms them; otherwise the build reports the gap and exits. When both are present it rebuilds
    ``move_id -> notation``, merges the notations that resolve to a real ``framedata_key`` into
    ``assets/movemap/<char>.json``, and prints a hit/miss self-check against the committed ids.
    """
    from tekken_coach.framedata.loader import load_current_framedata  # noqa: PLC0415
    from tekken_coach.framedata.movemap_miner import merge_mappings  # noqa: PLC0415
    from tekken_coach.reader.decode import resolve_component  # noqa: PLC0415
    from tekken_coach.reader.moveset import (  # noqa: PLC0415
        MoveLayout,
        build_notation_map,
        self_check,
    )
    from tekken_coach.reader.offsets import select_offset_table  # noqa: PLC0415
    from tekken_coach.reader.version import detect_running_version  # noqa: PLC0415
    from tekken_coach.reader.win_source import WinMemorySource  # noqa: PLC0415

    try:
        source = WinMemorySource(args.process)
        version = args.version or detect_running_version(args.process)
        table = select_offset_table(version, args.offsets)
    except ReaderError as exc:
        return _report_fault(exc)

    slot = table.players.moveset_slot
    if slot is None:
        print(
            "moveset offset not yet discovered — players.moveset_slot is null. Run "
            "`moveset-probe` live first to find it (brief #18 Phase 1).",
            file=sys.stderr,
        )
        return 0

    layout_args = (args.move_size, args.cancel_ptr_offset, args.cancel_count_offset)
    if any(a is None for a in layout_args) or args.neutral_move_id is None:
        print(
            "tk_move cancel-range layout not confirmed. Owner attribution needs --move-size, "
            "--cancel-ptr-offset, --cancel-count-offset and --neutral-move-id (confirmed by the "
            "live self-check). Without them a string-only move would be mis-mapped, so the build "
            "declines rather than write a wrong notation (docs/05 §2.3).",
            file=sys.stderr,
        )
        return 0

    layout = MoveLayout(
        size=args.move_size,
        cancel_ptr_offset=args.cancel_ptr_offset,
        cancel_count_offset=args.cancel_count_offset,
    )
    index = args.player - 1
    try:
        base = resolve_player_base(source, table, index)
        moveset_ptr = resolve_component(source, base, slot)
        result = build_notation_map(
            source, moveset_ptr, layout, neutral_move_id=args.neutral_move_id
        )
    except ReaderError as exc:
        return _report_fault(exc)

    print(
        f"rebuilt {len(result.notation)} move_id -> notation "
        f"({len(result.collisions)} collisions, {len(result.unresolved)} unresolved)"
    )

    snapshot = load_current_framedata()
    char_fd = snapshot.get_char(args.char.lower())
    if char_fd is None:
        print(
            f"no frame-data snapshot for {args.char!r} — run `fetch-framedata {args.char}` first.",
            file=sys.stderr,
        )
        return 1

    # Only write notations that resolve to a real framedata_key; the rest are honest needs-manual.
    mappable = [(mid, note) for mid, note in result.notation.items() if note in char_fd.moves]
    unknown_key = sorted(mid for mid, note in result.notation.items() if note not in char_fd.moves)
    merge = merge_mappings(
        args.char.lower(),
        char_fd,
        snapshot.manifest.game_version or version,
        mappable,
        movemap_dir=args.movemap,
        overwrite=args.overwrite,
    )
    verb = "created" if merge.created else "updated"
    print(
        f"{verb} {merge.path}: +{len(merge.written)} new, "
        f"{len(merge.overwritten)} overwritten, {len(merge.preserved)} preserved"
    )
    if unknown_key:
        print(f"{len(unknown_key)} rebuilt notation(s) match no framedata_key (needs-manual)")

    # Self-check against the committed ids already in the movemap (Bryan's ground truth).
    from tekken_coach.framedata.loader import load_char_move_map  # noqa: PLC0415

    map_path = Path(args.movemap) / f"{args.char.lower()}.json"
    if map_path.exists():
        committed = {int(k): e.notation for k, e in load_char_move_map(map_path).moves.items()}
        rows = self_check(result.notation, committed)
        hits = sum(1 for r in rows if r.status == "HIT")
        misses = [r for r in rows if r.status == "MISS"]
        print(f"\nself-check vs {map_path}: {hits}/{len(rows)} hit")
        for r in misses:
            print(f"  MISS {r.move_id}: expected {r.expected!r}, rebuilt {r.got!r}")
    return 0


def _live_monitor_stream(
    source: MemorySource, table: OffsetTable, interval: float
) -> Iterator[
    tuple[float, DerivedPhase, int, list[PlayerView]]
]:  # pragma: no cover - live loop; consumers are tested
    """Decode both players every ``interval`` s, deriving the match phase — the monitor feed.

    Threads a single :class:`MatchPhaseTracker` (the one stateful thing) so the ``[match]`` line can
    show the derived full phase (``menu``…``match_over``) + round + counter + the raw ``match_flag``
    alongside the per-player decoded state (docs/02 §8).

    Menu-tolerant (Part A): ``match_flag`` (a module-relative global) is the liveness probe read
    first, so a genuinely-closed game still surfaces ``process_lost``; but when the player decode
    faults out of a match (a null holder slot at the menu / character select), it prints a ``menu``
    ``[match]`` line with no player views instead of crashing.
    """
    import time  # noqa: PLC0415

    tracker = MatchPhaseTracker(table.sanity.round_start_health)
    started = time.monotonic()
    while True:
        match_flag = read_match_flag(source, table)  # liveness probe; propagate if the game is gone
        try:
            frame = decode_frame(source, table)
        except MemoryReadError:
            # Alive but out of a match: show a menu line (no views), don't advance the tracker.
            yield time.monotonic() - started, DerivedPhase(MatchState.menu, 0), match_flag, []
            time.sleep(interval)
            continue
        phase = derive_match_phase(tracker, table, frame, match_flag)
        yield time.monotonic() - started, phase, match_flag, views_of(frame)
        time.sleep(interval)


def monitor_main(args: argparse.Namespace) -> int:
    """Stream the reader's DECODED view of each player — the state-map check (docs/02 §8).

    Attaches read-only, decodes both players every poll, and prints a line whenever a player's
    decoded state (``action_state`` + situational flags) changes — so you can perform each state and
    check the reader agrees (stand -> neutral, block -> blockstun, get juggled -> hitstun+juggle).
    A ``[match]`` line shows the derived match_state (``menu``…``match_over``) + round + raw counter
    + the global ``match_flag`` whenever the phase changes (round-gating verification). ``--raw``
    appends the raw encoded state words, so a mis-decode is diagnosable on the spot. ``--input``
    runs the brief #9 input-reconstruction probe: it keys change-detection on the decoded input and
    appends ``in=dir:buttons``, so pressing each button / holding each direction surfaces one line —
    the recipe for validating the ``input_valid``/``input_dir``/``input_buttons`` offsets.
    """
    from tekken_coach.reader.offsets import select_offset_table  # noqa: PLC0415
    from tekken_coach.reader.version import detect_running_version  # noqa: PLC0415
    from tekken_coach.reader.win_source import WinMemorySource  # noqa: PLC0415

    try:
        source = WinMemorySource(args.process)
        version = args.version or detect_running_version(args.process)
        table = select_offset_table(version, args.offsets)
    except ReaderError as exc:
        return _report_fault(exc)

    print(f"monitoring {version} — decoded player state (Ctrl-C to stop)")
    spec = table.state_codes.encoded_state
    if spec is None or not spec.calibrated:
        print(
            "note: this table's state map is not calibrated — states read as `neutral` and the "
            "stun/juggle flags stay empty (docs/02 §8).",
            file=sys.stderr,
        )
    try:
        for line in monitor_lines(
            _live_monitor_stream(source, table, args.interval),
            show_raw=args.raw,
            show_input=args.show_input,
        ):
            print(line, flush=True)
    except ReaderError as exc:
        return _report_fault(exc)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


def input_protocol_main(args: argparse.Namespace) -> int:
    """Print the scripted press-through pass for the input-offset re-derivation (brief #10).

    The user's whole part in re-sourcing the dead ``input_*`` offsets is one recorded pass following
    this script; everything after it is offline (``analyze-input``). Printed rather than paced so it
    can sit on a second monitor next to the game — the exact start instant does not matter, since
    ``analyze-input`` fits the script to the log.
    """
    from tekken_coach.reader.input_probe import (  # noqa: PLC0415
        PROTOCOL,
        render_checklist,
        step_windows,
    )

    total = step_windows(PROTOCOL, args.start)[-1].t1
    record = args.record
    print("Record the pass in Practice (you as P1, the P2 dummy left STANDING — its stillness is")
    print("what tells the analyzer which struct is yours), then run the sweep in another terminal.")
    print()
    print(
        "Sweep BEHIND the pointer slots (#11): the flat struct 0x0-0x1600 is a settled negative —"
    )
    print("#10 swept it twice and input is not there. Get the slots to chase from Stage 1 first:")
    print()
    print("  py -m tekken_coach.reader.commands probe-state --slots")
    print()
    print("then pass the ones it ranks CHASE (it prints this line for you, filled in):")
    print()
    print('  py -m tekken_coach.reader.commands probe-state --watch-behind "0x38:0x0-0x100:u8" \\')
    print(f"      --record {record}")
    print()
    print(f"Then follow this script ({total:.0f}s), and analyze the log offline:")
    print()
    print(f"  py -m tekken_coach.reader.commands analyze-input {record}")
    print()
    # A distinct filename per run, deliberately: #10's first recorded pass was lost to an overwrite
    # of debug/input.jsonl, and the brief that needed it back paid for the re-run in user time.
    print(f"NOTE: {record} is a fresh name on purpose — do not overwrite a previous run's log.")
    print("Pass --record to name it yourself if you are doing a second pass.")
    print()
    for line in render_checklist(PROTOCOL, args.start):
        print(line)
    return 0


def analyze_input_main(args: argparse.Namespace) -> int:
    """Rank swept offsets as ``input_dir`` / ``input_buttons`` from a recorded pass (brief #10).

    Offline and read-only: it consumes the ``probe-state --record`` JSONL, so the ranking can be
    re-run against a saved log without the game. It reports observed value sets and refuses to name
    a candidate that does not clear
    :data:`~tekken_coach.reader.input_probe.MIN_PLAUSIBLE` — a clean sweep is a real finding (the
    fields may not live on the player struct at all), not a prompt to crown the least-bad offset.
    """
    from tekken_coach.reader.input_probe import (  # noqa: PLC0415
        best_alignment,
        format_report,
        load_observation_file,
    )

    path = Path(args.record)
    if not path.exists():
        print(f"no such record: {path}", file=sys.stderr)
        return 1
    obs = load_observation_file(path)
    if not obs.fields:
        print(f"{path} has no observed changes — was the sweep watching anything?", file=sys.stderr)
        return 1
    if args.player not in obs.players:
        print(
            f"player {args.player} is not in {path} (saw {obs.players}); pass --player.",
            file=sys.stderr,
        )
        return 1
    fitted_start, fitted_scale = best_alignment(obs, acting_player=args.player, scale=args.scale)
    start = args.start if args.start is not None else fitted_start
    for line in format_report(
        obs, start=start, scale=fitted_scale, acting_player=args.player, top=args.top
    ):
        print(line)
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
    p_probe.add_argument(
        "--watch",
        default=None,
        help="watch ad-hoc raw offsets instead of the table's state fields, as comma-separated "
        "OFFSET:KIND (or START-END:KIND range) terms (e.g. 0x550:u32,0x434:u32 or a sweep "
        "0xd2e0-0xd4c0:u32). Use to locate a field whose seeded offset went stale (docs/02 §8); "
        "the skeleton then summarizes the distinct values each candidate took.",
    )
    p_probe.add_argument(
        "--watch-behind",
        dest="watch_behind",
        default=None,
        help="sweep BEHIND a pointer slot instead of inside the player struct (#11 Stage 2), as "
        'comma-separated SLOT[/HOP]:OFFSET-END:KIND terms (e.g. "0x38:0x0-0x100:u8,0x20/8:0x0-'
        '0x100:u8"). Resolves player_base+SLOT, walks each /HOP, and sweeps the range at the '
        "landing — re-resolved every poll. Use the slots --slots ranked as CHASE. A flat sweep "
        "cannot see a component: the pointer to it never changes, so a change-sweep skips it.",
    )
    p_probe.add_argument(
        "--slots",
        action="store_true",
        help="#11 Stage 1: enumerate the player struct's plausible pointer slots (non-null, "
        "aligned, landing in a committed region) and rank them by what makes one worth chasing — "
        "resolves for both players, P1/P2 point at DIFFERENT objects, stable across polls. Needs "
        "no presses and ~10s in a Practice match; its table picks --watch-behind's targets.",
    )
    p_probe.add_argument(
        "--polls",
        type=int,
        default=20,
        help="(--slots only) polls to sample before classifying (default 20) — enough to tell a "
        "stable slot from a churning one.",
    )
    p_probe.add_argument(
        "--top",
        type=int,
        default=40,
        help="(--slots only) slots to print (default 40; 0 = all). The full table always goes to "
        "--record.",
    )
    p_probe.add_argument(
        "--global",
        dest="is_global",
        action="store_true",
        help="watch the GLOBAL/match struct (one struct) instead of the two players — for locating "
        "match_phase / game_mode across menu/round/results transitions. Requires --watch.",
    )
    p_probe.set_defaults(func=probe_state_main)

    p_msprobe = sub.add_parser(
        "moveset-probe",
        help="find the tk_moveset by heap shape+gate scan + derive a durable path (brief #19)",
    )
    p_msprobe.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    p_msprobe.add_argument("--version", default=None, help="override detected version")
    p_msprobe.add_argument(
        "--char",
        default="bryan",
        help="the character being played, to key the decoder gate (default bryan).",
    )
    p_msprobe.add_argument(
        "--player", type=int, default=1, help="which player to probe (1 or 2; default 1)"
    )
    p_msprobe.set_defaults(func=moveset_probe_main)

    p_msbuild = sub.add_parser(
        "moveset-build",
        help="read the live moveset -> decode -> join -> write the movemap (brief #18 Phase 2)",
    )
    p_msbuild.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    p_msbuild.add_argument("--movemap", default=DEFAULT_MOVEMAP_DIR)
    p_msbuild.add_argument("--version", default=None, help="override detected version")
    p_msbuild.add_argument(
        "--char", required=True, help="the character to build the movemap for (e.g. bryan)"
    )
    p_msbuild.add_argument(
        "--player", type=int, default=1, help="which player's moveset to read (1 or 2; default 1)"
    )
    p_msbuild.add_argument(
        "--overwrite",
        action="store_true",
        help="replace already-mapped ids instead of preserving them (brief #16 skip-on-resume).",
    )
    p_msbuild.add_argument(
        "--move-size",
        type=lambda s: int(s, 0),
        default=None,
        help="tk_move stride (confirmed by the live self-check; owner attribution needs it).",
    )
    p_msbuild.add_argument(
        "--cancel-ptr-offset",
        type=lambda s: int(s, 0),
        default=None,
        help="offset within tk_move of the pointer to its first cancel.",
    )
    p_msbuild.add_argument(
        "--cancel-count-offset",
        type=lambda s: int(s, 0),
        default=None,
        help="offset within tk_move of its cancel count (u64).",
    )
    p_msbuild.add_argument(
        "--neutral-move-id",
        type=lambda s: int(s, 0),
        default=None,
        help="the character's neutral/standing move id (its cancel list holds the from-neutral "
        "canonical inputs).",
    )
    p_msbuild.set_defaults(func=moveset_build_main)

    p_monitor = sub.add_parser(
        "monitor",
        help="stream the reader's DECODED view of each player (verify the calibrated state map)",
    )
    p_monitor.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    p_monitor.add_argument("--version", default=None, help="override detected version")
    p_monitor.add_argument("--interval", type=float, default=0.05, help="poll interval, seconds")
    p_monitor.add_argument(
        "--raw",
        action="store_true",
        help="append the raw encoded state words to each line, so a mis-decode is diagnosable.",
    )
    p_monitor.add_argument(
        "--input",
        dest="show_input",
        action="store_true",
        help="input-reconstruction probe (brief #9): key change-detection on the decoded input and "
        "append it (in=dir:buttons), so each button press / held direction surfaces one line.",
    )
    p_monitor.set_defaults(func=monitor_main)

    p_script = sub.add_parser(
        "input-protocol",
        help="print the scripted press-through pass to follow while probe-state records (#10)",
    )
    p_script.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="shift every timestamp by N seconds, if you start the script after the probe.",
    )
    p_script.add_argument(
        "--record",
        default="debug/behind-1.jsonl",
        help="the --record path to recommend in the printed instructions. Use a DISTINCT name per "
        "run: #10's first recorded pass was lost by overwriting the previous log, and a live pass "
        "costs user time to redo.",
    )
    p_script.set_defaults(func=input_protocol_main)

    p_analyze = sub.add_parser(
        "analyze-input",
        help="rank swept offsets as input_dir/input_buttons from a probe-state --record log (#10)",
    )
    p_analyze.add_argument("record", help="the probe-state --record JSONL from the scripted pass")
    p_analyze.add_argument(
        "--start",
        type=float,
        default=None,
        help="when the script began on the probe's clock (default: fit it to the log).",
    )
    p_analyze.add_argument(
        "--player", type=int, default=1, help="the acting player (default 1; the dummy is static)"
    )
    p_analyze.add_argument(
        "--scale",
        type=float,
        default=None,
        help="pin the tempo the pass was performed at (1.15 = 15%% slower than the script) instead "
        "of fitting it. A human reading a checklist runs slow, and it compounds.",
    )
    p_analyze.add_argument("--top", type=int, default=5, help="candidates to show per role")
    p_analyze.set_defaults(func=analyze_input_main)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: object = args.func
    assert callable(func)
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
