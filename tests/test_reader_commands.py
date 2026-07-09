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
