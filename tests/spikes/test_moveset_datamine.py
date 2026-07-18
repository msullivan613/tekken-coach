"""Spike #15 validation: does the cancel-command decode + join reproduce the ground-truth map?

These tests prove the LOGIC that a real Bryan `tk_cancel` dump would flow through — the decode of
the documented command encoding and the from-neutral / string-continuation join — on a synthetic
fixture encoded in that exact layout. The real extract runs the identical path (see the report).
"""

from __future__ import annotations

from tests.spikes.moveset_datamine.decoder import (
    DirectionAlphabet,
    decode_command,
    join_moves,
)
from tests.spikes.moveset_datamine.fixtures import (
    BRYAN_GROUND_TRUTH,
    DIRECTION_ALPHABET,
    INPUT_SEQUENCES,
    KAZUYA_GROUND_TRUTH,
    KAZUYA_NEUTRAL_MOVE_ID,
    NEUTRAL_MOVE_ID,
    build_bryan_cancels,
    build_kazuya_cancels,
)
from tests.spikes.moveset_datamine.validate import build_validation_table, format_table


def test_command_decode_units() -> None:
    """Spot-check the raw command → motion+button decode against the documented encoding."""
    alpha = DirectionAlphabet({0x0050: "df"})
    seqs = {1: "qcb"}
    # neutral + button 1 (bit 0)
    assert decode_command(0x0000 | (0x1 << 32), alpha, seqs).notation() == "1"
    # df + button 2 (bit 1)
    assert decode_command(0x0050 | (0x2 << 32), alpha, seqs).notation() == "df+2"
    # neutral + buttons 1 and 2
    assert decode_command(0x0000 | (0x3 << 32), alpha, seqs).notation() == "1+2"
    # sequence (qcb) + button 3 (bit 2)
    assert decode_command((0x800D + 1) | (0x4 << 32), alpha, seqs).notation() == "qcb+3"
    # uncalibrated direction → no notation, never a wrong guess
    assert decode_command(0x0999 | (0x1 << 32), alpha, seqs).notation() is None


def test_bryan_all_sixteen_map() -> None:
    """The gate: every one of Bryan's 16 ground-truth ids reconstructs to its confirmed notation."""
    rows = build_validation_table(
        build_bryan_cancels(),
        BRYAN_GROUND_TRUTH,
        neutral_move_id=NEUTRAL_MOVE_ID,
        alphabet=DIRECTION_ALPHABET,
        input_sequences=INPUT_SEQUENCES,
    )
    misses = [r for r in rows if r.status != "HIT"]
    assert not misses, "not all ids mapped:\n" + format_table(rows)
    assert len(rows) == 16


def test_kazuya_anchor() -> None:
    rows = build_validation_table(
        build_kazuya_cancels(),
        KAZUYA_GROUND_TRUTH,
        neutral_move_id=KAZUYA_NEUTRAL_MOVE_ID,
        alphabet=DIRECTION_ALPHABET,
        input_sequences=INPUT_SEQUENCES,
    )
    assert [r.status for r in rows] == ["HIT"]
    assert rows[0].got == "df+2"


def test_neutral_command_wins_over_string_path() -> None:
    """The jab is also a mid-string cancel target; its neutral command must stay canonical."""
    result = join_moves(
        build_bryan_cancels(),
        neutral_move_id=NEUTRAL_MOVE_ID,
        alphabet=DIRECTION_ALPHABET,
        input_sequences=INPUT_SEQUENCES,
    )
    assert result.notation[1695] == "1"  # not "b+1,1"
    assert 1695 not in result.collisions


def test_uncalibrated_direction_degrades_to_unresolved() -> None:
    """A direction code with no alphabet entry yields no mapping — degrade, don't misattribute."""
    result = join_moves(
        build_bryan_cancels(),
        neutral_move_id=NEUTRAL_MOVE_ID,
        alphabet=DIRECTION_ALPHABET,
        input_sequences=INPUT_SEQUENCES,
    )
    assert 9001 in result.unresolved
    assert 9001 not in result.notation
