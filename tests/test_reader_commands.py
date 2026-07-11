"""Offline tests for the command layer's source-independent bits (docs/02 §6).

The smoke/doctor/capture commands attach to a live Windows process, so their end-to-end path is
user-run. Here we cover what is exercisable offline: the movemap char-id loader, the argument
parser, and the fault-reporting helper (which prints the §4 runbook on an unknown version).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from tekken_coach.reader import commands
from tekken_coach.reader.faults import UnknownGameVersionError


def test_load_known_char_ids_from_repo_movemap() -> None:
    ids = commands._load_known_char_ids("assets/movemap")
    assert 12 in ids  # Kazuya is seeded; Paul's null id is skipped
    assert all(isinstance(i, int) for i in ids)


def test_load_known_char_ids_missing_index_is_empty(tmp_path: Path) -> None:
    assert commands._load_known_char_ids(tmp_path) == set()


def test_report_fault_prints_runbook(capsys: pytest.CaptureFixture[str]) -> None:
    code = commands._report_fault(UnknownGameVersionError("9.9.9", ["2.01.01"]))
    assert code == 1
    err = capsys.readouterr().err
    assert "unknown_version" in err
    assert "update-offsets" in err  # the §4 runbook is surfaced by the command, not the library


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        commands.build_parser().parse_args([])


def test_parser_capture_requires_out() -> None:
    with pytest.raises(SystemExit):
        commands.build_parser().parse_args(["capture"])


def test_parser_wires_subcommand_funcs() -> None:
    parser = commands.build_parser()
    for name, func in [
        ("smoke", commands.smoke_main),
        ("doctor", commands.doctor_main),
    ]:
        args = parser.parse_args([name])
        assert args.func is func
    args = parser.parse_args(["capture", "--out", "x.json", "--count", "10"])
    assert args.func is commands.capture_main
    assert args.count == 10
    assert isinstance(args, argparse.Namespace)


def test_parser_update_offsets_base_scan_flag() -> None:
    # C4d: --base-scan selects the code-signature/pointer-chain derivation; default stays C4c.
    parser = commands.build_parser()
    assert parser.parse_args(["update-offsets"]).base_scan is False
    args = parser.parse_args(["update-offsets", "--base-scan"])
    assert args.base_scan is True
    assert args.func is commands.update_offsets_main


def test_parser_wires_probe_state() -> None:
    # C4e: the executable half of the docs/02 §8 state-map calibration protocol.
    parser = commands.build_parser()
    args = parser.parse_args(["probe-state", "--seconds", "3"])
    assert args.func is commands.probe_state_main
    assert args.seconds == 3.0
    assert args.interval == 0.05  # ~3 polls per game frame; changes are what get printed
    assert args.record is None  # C4j: opt-in observation log, unchanged behaviour by default
    assert args.emit_skeleton is None


def test_parser_probe_state_record_and_skeleton_flags() -> None:
    # C4j: --record persists the observation log; --emit-skeleton overrides where the draft lands.
    parser = commands.build_parser()
    args = parser.parse_args(
        ["probe-state", "--record", "debug/obs.jsonl", "--emit-skeleton", "debug/draft.json"]
    )
    assert args.record == "debug/obs.jsonl"
    assert args.emit_skeleton == "debug/draft.json"


def test_skeleton_path_prefers_explicit_then_derives_from_record() -> None:
    parser = commands.build_parser()
    # --emit-skeleton wins outright.
    args = parser.parse_args(["probe-state", "--record", "obs.jsonl", "--emit-skeleton", "x.json"])
    assert commands._skeleton_path(args) == Path("x.json")
    # else it is derived beside the record file (.jsonl -> .skeleton.json).
    args = parser.parse_args(["probe-state", "--record", "debug/state-obs.jsonl"])
    assert commands._skeleton_path(args) == Path("debug/state-obs.skeleton.json")
    # with neither, no skeleton is emitted.
    assert commands._skeleton_path(parser.parse_args(["probe-state"])) is None


def test_format_change_renders_aligned_columns() -> None:
    from tekken_coach.reader.probe import ChangeRecord

    record = ChangeRecord(t=1.5, player=2, fields={"move_id": 133, "stun_type": 3})
    line = commands._format_change(record, ["move_id", "stun_type"])
    assert line.split() == ["1.50", "P2", "133", "3"]


def test_probe_targets_watch_the_state_words_plus_move_context() -> None:
    # The raw words alone are unreadable: "which move was I in when stun_type went to 3" is the
    # question the calibration protocol actually answers, so move_id/move_frame ride along.
    from tests.test_reader_decode_encoded import _encoded_table

    names = commands._probe_targets(_encoded_table())
    assert names[:3] == ["move_id", "move_frame", "counter_state"]
    assert "stun_type" in names and "simple_move_state" in names
