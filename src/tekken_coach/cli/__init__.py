"""The ``tekken-coach`` command surface (docs/07 §1). Chunk C6.

One CLI orchestrates the whole pipeline (docs/00) and renders coaching to the terminal. Six
user-facing commands (docs/07 §1.1):

    live              live capture: arm → record per match → coach at match end (docs/01 §3)
    clean [replays…]  clean capture: offline replay batch → coach at session end (docs/01 §4)
    coach <log>       re-run coaching on an existing session log (no capture) — works today
    update-offsets    post-patch offset re-discovery (delegated to the reader, docs/02 §4)
    fetch-framedata   ingest a new frame-data snapshot (delegated to C1, docs/05 §3.3)
    doctor            reader self-check + data-freshness (delegated to the reader, docs/02 §6)

``live``/``clean``/``coach`` are new here; ``update-offsets``/``fetch-framedata``/``doctor`` are
already implemented elsewhere and are **delegated** — this module only registers them at the top
level. The reader's diagnostic subcommands (``smoke``/``probe-state``/``monitor``/fixture
``capture``) stay on the reader's own surface and are intentionally not re-exposed here (docs/07
§1.1 lists only the six).

⚠️ Real-game **live/clean** bring-up is blocked on the deferred round-gating (``match_phase`` /
``game_mode`` not yet calibrated on build 5.02.01 — project memory
``capture-round-gating-deferred``): the reader's state signal raises there, so live/clean cannot
fully run against the real game in this chunk. The orchestration is complete and is exercised
end-to-end with a scripted fake reader; ``coach <log>`` works with no game at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tekken_coach.cli import capture as capture_mod
from tekken_coach.cli.config import load_config, resolve_settings
from tekken_coach.cli.orchestrate import CaptureError
from tekken_coach.cli.render import Renderer
from tekken_coach.cli.source import ReaderCaptureSource
from tekken_coach.coach import coach_session
from tekken_coach.reader.commands import (
    DEFAULT_MOVEMAP_DIR,
    DEFAULT_OFFSETS_DIR,
    _report_fault,
    doctor_main,
    update_offsets_main,
)
from tekken_coach.reader.faults import ReaderError
from tekken_coach.reader.version import GAME_PROCESS_NAME
from tekken_coach.session.store import IncompatibleSchemaVersionError, load_session

# ---------------------------------------------------------------------------
# Capture commands (live / clean) — new in C6
# ---------------------------------------------------------------------------


def _capture_command(args: argparse.Namespace) -> int:
    """Shared handler for ``live`` and ``clean``: resolve settings, attach, run the pipeline."""
    try:
        settings = resolve_settings(
            mode=args.mode,
            coach=args.coach,
            user=args.user,
            char=args.char,
            out=args.out,
            config=load_config(),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    renderer = Renderer()
    if getattr(args, "replays", None):
        # v1 replay selection is manual: the user starts playback in-game and the tool detects the
        # playback-active state (docs/01 §4.2). The Wavu /api/replays selection is a v1.x extra.
        renderer.notice(
            "clean-mode replay selection is manual in v1 — start each replay playback in-game; "
            f"the {len(args.replays)} path(s) given are informational."
        )

    source = ReaderCaptureSource(args.process, args.offsets, version_override=args.version)
    try:
        assets = capture_mod.load_assets()
        capture_mod.run_capture(settings=settings, source=source, assets=assets, renderer=renderer)
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ReaderError as exc:  # unknown version / attach failure surfaces the §02 runbook
        return _report_fault(exc)
    return 0


# ---------------------------------------------------------------------------
# coach <log> — re-run coaching on an existing log (no capture); works today
# ---------------------------------------------------------------------------


def _coach_command(args: argparse.Namespace) -> int:
    """Coach an existing session ``.jsonl`` (docs/07 §1). No game, no capture."""
    path = Path(args.log)
    if not path.exists():
        print(f"error: session log not found: {path}", file=sys.stderr)
        return 1

    coach = args.coach or "skill"
    renderer = Renderer()
    try:
        if coach == "api":
            renderer.coach_result(coach_session(path))
        else:
            session = load_session(path)  # validates schema_version, gives header + interactions
            renderer.log_handoff(path, len(session.header.matches), len(session.interactions))
    except IncompatibleSchemaVersionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# fetch-framedata — ingest a snapshot (delegates to the C1 callable)
# ---------------------------------------------------------------------------


def _fetch_framedata_command(args: argparse.Namespace) -> int:
    """Register + delegate to the C1 ``fetch_framedata`` ingest callable (docs/05 §3.3)."""
    from tekken_coach.framedata.ingest import CharSpec, fetch_framedata

    def parse_spec(item: str) -> CharSpec:
        # "Name" -> slug=name.lower(); "Name:slug" -> explicit slug.
        name, _, slug = item.partition(":")
        return CharSpec(char_name=name, slug=slug or name.lower())

    result = fetch_framedata(
        [parse_spec(item) for item in args.char],
        sha=args.sha,
        repo=args.repo,
        snapshot_date=args.snapshot_date,
        game_version=args.game_version,
        repoint=args.repoint,
    )
    print(f"wrote {result.snapshot_name} -> {result.snapshot_dir}")
    for diff in result.diff:
        if diff.is_empty:
            print(f"  {diff.slug}: no change")
        else:
            print(
                f"  {diff.slug}: +{len(diff.added)} / -{len(diff.removed)} / "
                f"~{len(diff.changed)} moves"
            )
    adopted = (
        "repointed to this snapshot" if result.repointed else "unchanged (use --repoint to adopt)"
    )
    print(f"current {adopted}")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_capture_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``live`` and ``clean`` (docs/07 §1.2). CLI flags override config."""
    parser.add_argument("--coach", choices=("skill", "api"), default=None, help="coaching backend")
    parser.add_argument("--user", default=None, help="which player is the user: p1|p2 (docs/01 §5)")
    parser.add_argument("--char", default=None, help="the user's character (validated vs reads)")
    parser.add_argument("--out", default=None, help="session .jsonl path (default sessions/<ts>)")
    parser.add_argument("--process", default=GAME_PROCESS_NAME, help="target process/module name")
    parser.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR, help="offset-table directory")
    parser.add_argument("--version", default=None, help="override the detected game version")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tekken-coach",
        description="Read-only Tekken 8 coaching side-car: capture pipeline + coaching (docs/07).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_live = sub.add_parser("live", help="live capture: record per match, coach at match end")
    _add_capture_flags(p_live)
    p_live.set_defaults(func=_capture_command, mode="live")

    p_clean = sub.add_parser("clean", help="clean capture: replay batch, coach at session end")
    _add_capture_flags(p_clean)
    p_clean.add_argument("replays", nargs="*", help="replay identifiers (v1: manual selection)")
    p_clean.set_defaults(func=_capture_command, mode="clean")

    p_coach = sub.add_parser("coach", help="re-run coaching on an existing session .jsonl")
    p_coach.add_argument("log", help="path to a session .jsonl log")
    p_coach.add_argument("--coach", choices=("skill", "api"), default=None, help="coaching backend")
    p_coach.set_defaults(func=_coach_command)

    # --- delegated commands (already implemented elsewhere) --------------
    p_update = sub.add_parser("update-offsets", help="post-patch offset re-discovery (docs/02 §4)")
    _add_update_offsets_flags(p_update)
    p_update.set_defaults(func=update_offsets_main)

    p_fetch = sub.add_parser("fetch-framedata", help="ingest a new frame-data snapshot")
    _add_fetch_framedata_flags(p_fetch)
    p_fetch.set_defaults(func=_fetch_framedata_command)

    p_doctor = sub.add_parser("doctor", help="reader self-check + data-freshness report")
    _add_doctor_flags(p_doctor)
    p_doctor.set_defaults(func=doctor_main)

    return parser


def _add_doctor_flags(parser: argparse.ArgumentParser) -> None:
    """Flags the delegated ``doctor_main`` reads (mirrors reader.commands)."""
    parser.add_argument("--process", default=GAME_PROCESS_NAME, help="target process/module name")
    parser.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    parser.add_argument("--movemap", default=DEFAULT_MOVEMAP_DIR)
    parser.add_argument("--version", default=None, help="override detected version")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--interval", type=float, default=0.05, help="seconds between frame polls")


def _add_update_offsets_flags(parser: argparse.ArgumentParser) -> None:
    """Flags the delegated ``update_offsets_main`` reads (mirrors reader.commands)."""
    parser.add_argument("--process", default=GAME_PROCESS_NAME, help="target process/module name")
    parser.add_argument("--offsets", default=DEFAULT_OFFSETS_DIR)
    parser.add_argument("--manifest", default="assets/offsets/probe-manifest.json")
    parser.add_argument("--version", default=None, help="override detected version")
    parser.add_argument("--base-scan", action="store_true", help="C4d static-pointer code sig")
    parser.add_argument("--derive", action="store_true", help="C4h fully derive the layout")
    parser.add_argument("--holder-scan", action="store_true", help="C4i AoB holder model")
    parser.add_argument("--debug-dir", default=None, help="(--holder-scan) write a debug capture")


def _add_fetch_framedata_flags(parser: argparse.ArgumentParser) -> None:
    """Flags for the C1 ingest delegate (docs/05 §3.3)."""
    parser.add_argument(
        "--char",
        action="append",
        required=True,
        metavar="NAME[:slug]",
        help="a character to ingest (repeatable); slug defaults to name.lower()",
    )
    parser.add_argument("--sha", required=True, help="pinned source commit SHA (reproducible key)")
    parser.add_argument("--repo", default="pbruvoll/tekkendocs", help="source repo")
    parser.add_argument("--snapshot-date", default=None, help="YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--game-version", default=None, help="balance-patch version stamp")
    parser.add_argument(
        "--repoint",
        action="store_true",
        help="adopt the new snapshot as `current` (the approval gate, docs/05 §3.3)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = args.func
    assert callable(func)
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())
