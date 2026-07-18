"""Synthetic Bryan/Kazuya cancel fixtures for the moveset-datamine spike (#15).

These stand in for the real extraction until the user produces one (see AGENT-REPORT.md). Each
cancel is encoded in the *documented* command layout (direction low-32 | button high-32), using a
concrete direction alphabet, so a green validation here proves the decode + join LOGIC end to
end on data shaped exactly like a real ``tk_cancel`` dump. The real Bryan cancels flow through the
identical ``join_moves`` path; only the direction alphabet is calibrated from live anchors.

Ground truth (the live ``move_id@0x550`` space) — assets/movemap/bryan.json + kazuya.json:

    1546→1+2  1566→2   1573→3    1574→4    1582→4,3  1600→f+3  1604→b+4  1626→DF+1
    1628→df+2 1656→b+2 1695→1    1697→1,2  1705→b+1  1725→d+4  1765→qcb+3 1779→b+3
    (Kazuya) 2145→df+2
"""

from __future__ import annotations

from .decoder import Cancel, DirectionAlphabet

# The neutral / standing move whose cancel list holds every from-neutral command.
NEUTRAL_MOVE_ID = 0

# Calibration alphabet: direction code (< 0x8000) → notation token. Arbitrary but internally
# consistent integers; on real data these are solved from the ground-truth anchors.
_DIR_CODES: dict[str, int] = {
    "f": 0x0010,
    "b": 0x0020,
    "d": 0x0030,
    "u": 0x0040,
    "df": 0x0050,
    "db": 0x0060,
    "uf": 0x0070,
    "ub": 0x0080,
    "DF": 0x0150,  # held down-forward — a distinct code from the tapped "df"
}
DIRECTION_ALPHABET = DirectionAlphabet({code: token for token, code in _DIR_CODES.items()})

# Sequence index → resolved motion (the qcb motion for Bryan's Hatchet Kick).
INPUT_SEQUENCES: dict[int, str] = {1: "qcb"}
_SEQUENCE_BASE = 0x800D

# button notation → button-field bit
_BTN_BITS: dict[str, int] = {"1": 0, "2": 1, "3": 2, "4": 3}


def _cmd(direction: int, buttons: str) -> int:
    """Pack a command uint64 from a direction code and a '+'-joined button string."""
    btn_field = 0
    for b in buttons.split("+"):
        if b:
            btn_field |= 1 << _BTN_BITS[b]
    return direction | (btn_field << 32)


def _dir(token: str) -> int:
    return 0 if token == "" else _DIR_CODES[token]


BRYAN_GROUND_TRUTH: dict[int, str] = {
    1546: "1+2",
    1566: "2",
    1573: "3",
    1574: "4",
    1582: "4,3",
    1600: "f+3",
    1604: "b+4",
    1626: "DF+1",
    1628: "df+2",
    1656: "b+2",
    1695: "1",
    1697: "1,2",
    1705: "b+1",
    1725: "d+4",
    1765: "qcb+3",
    1779: "b+3",
}


def build_bryan_cancels() -> list[Cancel]:
    """The 16 ground-truth moves as cancel rows, plus realistic noise the join must survive."""
    n = NEUTRAL_MOVE_ID
    cancels: list[Cancel] = [
        # from-neutral single-input moves: (dest, direction token, buttons)
        Cancel(n, 1546, _cmd(_dir(""), "1+2")),
        Cancel(n, 1566, _cmd(_dir(""), "2")),
        Cancel(n, 1573, _cmd(_dir(""), "3")),
        Cancel(n, 1574, _cmd(_dir(""), "4")),
        Cancel(n, 1600, _cmd(_dir("f"), "3")),
        Cancel(n, 1604, _cmd(_dir("b"), "4")),
        Cancel(n, 1626, _cmd(_dir("DF"), "1")),
        Cancel(n, 1628, _cmd(_dir("df"), "2")),
        Cancel(n, 1656, _cmd(_dir("b"), "2")),
        Cancel(n, 1695, _cmd(_dir(""), "1")),
        Cancel(n, 1705, _cmd(_dir("b"), "1")),
        Cancel(n, 1725, _cmd(_dir("d"), "4")),
        Cancel(n, 1779, _cmd(_dir("b"), "3")),
        # motion-input move via an input sequence (qcb): direction = SEQUENCE_BASE + idx
        Cancel(n, 1765, _cmd(_SEQUENCE_BASE + 1, "3")),
        # string continuations: reached FROM their prefix move, follow-up button only
        Cancel(1574, 1582, _cmd(_dir(""), "3")),  # 4 -> 4,3
        Cancel(1695, 1697, _cmd(_dir(""), "2")),  # 1 -> 1,2
        # --- realistic noise the join must not trip on ---
        # the jab (1695) is ALSO a mid-string cancel target from b+1 (1705); the neutral command
        # must still win, not collide.
        Cancel(1705, 1695, _cmd(_dir(""), "1")),
        # an uncalibrated direction (a code with no alphabet entry) on some unrelated move —
        # must degrade to unresolved, never a wrong guess.
        Cancel(n, 9001, _cmd(0x0999, "1")),
    ]
    return cancels


KAZUYA_NEUTRAL_MOVE_ID = 0
KAZUYA_GROUND_TRUTH: dict[int, str] = {2145: "df+2"}


def build_kazuya_cancels() -> list[Cancel]:
    return [Cancel(KAZUYA_NEUTRAL_MOVE_ID, 2145, _cmd(_dir("df"), "2"))]
