"""CLI wiring for ``map-moves`` (brief #6). The pure miner/join is tested in test_movemap_*.

Here we only prove the subcommand parses, routes ``--from-log`` through the miner, refuses a bad
flag combination, and honours ``--movemap`` / ``--framedata`` — driving the real committed slice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tekken_coach.cli import build_parser, main

REPO_ROOT = Path(__file__).parent.parent
FRAMEDATA = REPO_ROOT / "assets" / "framedata"
SLICE = REPO_ROOT / "tests" / "fixtures" / "framedata" / "live-run-1-slice.jsonl"


def test_from_log_runs_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "map-moves",
            "--from-log",
            str(SLICE),
            "--movemap",
            str(tmp_path),
            "--framedata",
            str(FRAMEDATA),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "map-moves:" in out
    assert "needs framedata" in out  # Bryan groups surfaced, not crashed


def test_requires_exactly_one_mode(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["map-moves"])  # neither --from-log nor --live
    assert rc == 2
    assert "exactly one" in capsys.readouterr().err


def test_from_log_and_live_are_mutually_exclusive(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["map-moves", "--from-log", str(SLICE), "--live", "--char", "paul"])
    assert rc == 2
    assert "exactly one" in capsys.readouterr().err


def test_missing_log_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["map-moves", "--from-log", str(tmp_path / "nope.jsonl")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_parser_registers_map_moves() -> None:
    parser = build_parser()
    args = parser.parse_args(["map-moves", "--from-log", "x.jsonl", "--char", "paul"])
    assert args.command == "map-moves"
    assert args.from_log == "x.jsonl"
    assert args.char == "paul"
    assert args.live is False
