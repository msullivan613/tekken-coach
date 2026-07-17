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
    assert args.watch is None  # ad-hoc candidate-offset watch is opt-in
    assert args.is_global is False  # player-struct probe by default


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


def test_parser_wires_monitor() -> None:
    parser = commands.build_parser()
    args = parser.parse_args(["monitor", "--raw", "--interval", "0.1"])
    assert args.func is commands.monitor_main
    assert args.raw is True and args.interval == 0.1
    assert parser.parse_args(["monitor"]).raw is False  # raw is opt-in


def test_parser_probe_state_global_flag() -> None:
    parser = commands.build_parser()
    args = parser.parse_args(["probe-state", "--global", "--watch", "0xd2e0-0xd4c0:u32"])
    assert args.is_global is True
    assert args.watch == "0xd2e0-0xd4c0:u32"


def test_parser_probe_state_watch_flag() -> None:
    parser = commands.build_parser()
    args = parser.parse_args(["probe-state", "--watch", "0x434:u32,0x670:u32"])
    assert args.watch == "0x434:u32,0x670:u32"


def test_table_points_map_names_to_field_offsets() -> None:
    from tests.test_reader_decode_encoded import _encoded_table

    table = _encoded_table()
    points = commands._table_points(table, ["move_id", "stun_type"])
    fields = table.players.fields
    assert [(p.name, p.offset, p.kind) for p in points] == [
        ("move_id", fields["move_id"].offset, fields["move_id"].kind),
        ("stun_type", fields["stun_type"].offset, fields["stun_type"].kind),
    ]


def test_ensure_parent_dirs_creates_missing_dirs(tmp_path: Path) -> None:
    # The documented `--record debug/state-obs.jsonl` must not crash when `debug/` is absent.
    record = tmp_path / "debug" / "state-obs.jsonl"
    skeleton = tmp_path / "debug" / "state-obs.skeleton.json"
    commands._ensure_parent_dirs(record, skeleton, None)
    assert record.parent.is_dir()
    # A bare filename (parent == "") and None are both no-ops, not errors.
    commands._ensure_parent_dirs(Path("bare.jsonl"), None)


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


# --- input-offset re-derivation (brief #10) ------------------------------------------------------


def test_parser_wires_the_input_rederivation_commands() -> None:
    parser = commands.build_parser()
    assert parser.parse_args(["input-protocol"]).func is commands.input_protocol_main
    args = parser.parse_args(["analyze-input", "debug/input.jsonl"])
    assert args.func is commands.analyze_input_main
    assert args.record == "debug/input.jsonl"
    assert args.start is None  # default: fit the script to the log rather than trust the clocks
    assert args.player == 1


def test_input_protocol_prints_the_script_and_the_commands_around_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = commands.input_protocol_main(
        argparse.Namespace(start=0.0, record="debug/behind-1.jsonl")
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "probe-state --slots" in out  # #11 Stage 1: get the slots to chase first
    assert "probe-state --watch-behind" in out  # #11 Stage 2: how to record the pass
    assert "analyze-input" in out  # what to do with the log afterwards
    assert "dummy left STANDING" in out  # the discriminator the analyzer depends on
    assert "press+hold 1 for 2s" in out
    # A distinct record name per run: #10's first pass was lost to an overwrite, and re-running a
    # live pass spends the one resource this work is actually short of — the user's time.
    assert "debug/behind-1.jsonl" in out
    assert "do not overwrite" in out


def _analyze_args(record: object, **kwargs: object) -> argparse.Namespace:
    return argparse.Namespace(record=str(record), start=None, player=1, top=3, **kwargs)


def test_analyze_input_reports_a_missing_record_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = commands.analyze_input_main(_analyze_args(tmp_path / "nope.jsonl"))
    assert code == 1
    assert "no such record" in capsys.readouterr().err


def test_analyze_input_rejects_an_empty_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    code = commands.analyze_input_main(_analyze_args(empty))
    assert code == 1
    assert "no observed changes" in capsys.readouterr().err


def test_analyze_input_rejects_a_player_absent_from_the_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    record = tmp_path / "one.jsonl"
    record.write_text('{"t": 0.0, "player": 1, "fields": {"@0x8": 1}}\n', encoding="utf-8")
    args = _analyze_args(record)
    args.player = 2
    code = commands.analyze_input_main(args)
    assert code == 1
    assert "is not in" in capsys.readouterr().err
