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
    assert "move-id groups" in out  # groups surfaced, not crashed
    # Xiaoyu (char_id 5) is unnamed by the header, so its groups surface as unresolved (not a
    # crash); Bryan's frame data now exists in the snapshot, so it no longer needs framedata.
    assert "not named by the session header" in out


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


def test_live_poll_rate_flags_default() -> None:
    """--hz defaults to 120 (Part A oversample) and --reps to 5 (Part B accumulation), brief #13."""
    parser = build_parser()
    args = parser.parse_args(["map-moves", "--live", "--char", "bryan"])
    assert args.hz == 120.0
    assert args.reps == 5


def test_live_hz_and_reps_translate_to_run_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """--hz N reaches run_live as interval=1/N, and --reps N passes straight through (brief #13)."""
    import tekken_coach.framedata.movemap_live as live

    captured: dict[str, object] = {}

    def fake_run_live(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(live, "run_live", fake_run_live)
    rc = main(
        ["map-moves", "--live", "--char", "bryan", "--user", "p1", "--hz", "200", "--reps", "8"]
    )
    assert rc == 0
    assert captured["interval"] == pytest.approx(1.0 / 200)
    assert captured["reps"] == 8
    assert captured["char"] == "bryan"
    assert captured["user_player"] == 0


def test_live_rejects_non_positive_hz(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["map-moves", "--live", "--char", "bryan", "--hz", "0"])
    assert rc == 2
    assert "--hz must be positive" in capsys.readouterr().err


def test_live_rejects_reps_below_one(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["map-moves", "--live", "--char", "bryan", "--reps", "0"])
    assert rc == 2
    assert "--reps must be at least 1" in capsys.readouterr().err
