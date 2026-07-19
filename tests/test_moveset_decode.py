"""Unit tests for the production Tekken 8 cancel-command decoder + join (brief #18).

Proves the *confirmed T8* command encoding (direction low-32 bitfield, ``0xMMNNHHPP`` button hi-32)
decodes to notation, and that the game-agnostic join reconstructs ``move_id -> notation`` off the
cancel graph — from-neutral canonical, string continuation, collision reported, unknown degraded.
"""

from __future__ import annotations

from tekken_coach.framedata.moveset_decode import (
    MODE_DIRECTION_ONLY,
    MODE_NORMAL,
    Cancel,
    decode_command,
    join_moves,
)


def _cmd(direction: int, pp: int, mode: int = MODE_NORMAL) -> int:
    """Pack a command uint64 the way the game does: direction low-32, ``0xMMNNHHPP`` button."""
    return direction | (((mode << 24) | pp) << 32)


def test_direction_bits_decode_to_tokens() -> None:
    """Each documented direction bit maps to its notation token; 0x00/0x20 mean no prefix."""
    assert decode_command(_cmd(0x08, 0x02)).notation() == "df+2"  # df + 2
    assert decode_command(_cmd(0x04, 0x08)).notation() == "d+4"  # d + 4
    assert decode_command(_cmd(0x10, 0x01)).notation() == "b+1"  # b + 1
    assert decode_command(_cmd(0x00, 0x01)).notation() == "1"  # any -> no prefix
    assert decode_command(_cmd(0x20, 0x01)).notation() == "1"  # neutral -> no prefix


def test_button_bits_decode_to_notation_buttons() -> None:
    """PP bits 0x01/0x02/0x04/0x08 are buttons 1-4; multiple buttons join with '+'."""
    assert decode_command(_cmd(0x00, 0x04)).notation() == "3"
    assert decode_command(_cmd(0x00, 0x08)).notation() == "4"
    assert decode_command(_cmd(0x00, 0x01 | 0x02)).notation() == "1+2"


def test_heat_and_rage_art_are_not_buttons() -> None:
    """Heat (0x10) and Rage Art (0x40) are not buttons 1-4; a special-only engage is unresolved."""
    assert decode_command(_cmd(0x00, 0x10)).notation() is None  # Heat only
    assert decode_command(_cmd(0x00, 0x40)).notation() is None  # Rage Art only
    # A special bit alongside a real button decodes to the real button (the engage rides along).
    assert decode_command(_cmd(0x00, 0x10 | 0x01)).notation() == "1"


def test_unknown_direction_degrades_to_none() -> None:
    """A direction value that is not a modeled code yields no notation — never a wrong guess."""
    decoded = decode_command(_cmd(0x400, 0x01))
    assert decoded.unknown_direction is True
    assert decoded.notation() is None


def test_direction_only_mode_emits_a_bare_direction() -> None:
    """A direction-only command (a movement input) notates as the direction alone; else nothing."""
    assert decode_command(_cmd(0x40, 0x00, mode=MODE_DIRECTION_ONLY)).notation() == "f"
    # A bare direction in a normal-mode command with no buttons is not a notation on its own.
    assert decode_command(_cmd(0x40, 0x00, mode=MODE_NORMAL)).notation() is None


def test_join_from_neutral_and_string_continuation() -> None:
    """From-neutral commands are canonical; a move off a resolved prefix becomes a comma-string."""
    cancels = [
        Cancel(0, 1574, _cmd(0x00, 0x08)),  # neutral -> "4"
        Cancel(0, 1695, _cmd(0x00, 0x01)),  # neutral -> "1"
        Cancel(1574, 1582, _cmd(0x00, 0x04)),  # 4 -> 4,3
        Cancel(1695, 1697, _cmd(0x00, 0x02)),  # 1 -> 1,2
    ]
    result = join_moves(cancels, neutral_move_id=0)
    assert result.notation[1574] == "4"
    assert result.notation[1582] == "4,3"
    assert result.notation[1697] == "1,2"


def test_join_neutral_command_wins_over_string_path() -> None:
    """The jab is also a mid-string cancel target off b+1; its neutral "1" stays canonical."""
    cancels = [
        Cancel(0, 1705, _cmd(0x10, 0x01)),  # neutral -> b+1
        Cancel(0, 1695, _cmd(0x00, 0x01)),  # neutral -> 1 (the jab)
        Cancel(1705, 1695, _cmd(0x00, 0x01)),  # b+1 -> 1 (mid-string): must not win
    ]
    result = join_moves(cancels, neutral_move_id=0)
    assert result.notation[1695] == "1"  # not "b+1,1"
    assert 1695 not in result.collisions


def test_join_conflicting_neutral_candidates_collide() -> None:
    """Two from-neutral cancels to one move with different notations collide, not a guess."""
    cancels = [
        Cancel(0, 1991, _cmd(0x00, 0x01)),  # neutral -> 1
        Cancel(0, 1991, _cmd(0x00, 0x02)),  # neutral -> 2
    ]
    result = join_moves(cancels, neutral_move_id=0)
    assert result.collisions[1991] == ["1", "2"]
    assert 1991 not in result.notation


def test_join_undecodable_command_is_unresolved() -> None:
    """A move whose only cancel has an undecodable command is unresolved, never mis-mapped."""
    cancels = [Cancel(0, 1990, _cmd(0x400, 0x01))]  # unknown direction
    result = join_moves(cancels, neutral_move_id=0)
    assert 1990 in result.unresolved
    assert 1990 not in result.notation
