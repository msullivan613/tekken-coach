"""Clean-room decoder for the Tekken 8 moveset *cancel command* → input notation.

Spike #15 (moveset-datamine feasibility). This is a THROWAWAY analysis module, not
production code — it lives under ``tests/`` so the four gates type-check and exercise it, but
it ships no runtime extractor.

Everything here is built from **published structure facts** (usable with attribution), never
from copied code or a labelled move table:

* ``tk_cancel`` layout — tekkenmods.com Tekken 8 moveset docs:
  ``command`` (uint64) @ 0x00, ``move_id`` (uint16, the *destination* move) @ 0x24, size 0x28.
* Command encoding — TekkenMovesetExtractor / Kiloutre (github.com/Kiloutre):
  the command splits into a *direction* field and a *button* field.
  - direction ``< 0x8000``  → a directional input code (neutral, f, b, df, …)
  - ``0x8000..0x800d``      → special commands ([AUTO], double-tap F/B/U/D, …)
  - direction ``> 0x800d``  → ``input_sequences[direction - 0x800d]`` (a motion, e.g. qcb)
  - button bits 0-3 → attack buttons 1-4, bit 4 → Rage Art, bit 29 → "partial input".

The one value NOT published as a table is the concrete integer for each *direction* token
(what int means "df"). We model it as a calibration alphabet (``DirectionAlphabet``) that the
real Bryan extraction solves from a handful of anchors — see the report. The decode/join LOGIC
below is what these tests prove; the alphabet is a one-line calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- command field split (documented assumption: direction = low 32b, button = high 32b) -------
# The special-direction range (0x8000..0x800d) needs a >=16-bit field and the "partial input"
# flag at button bit 29 needs a ~32-bit field, so a uint64 packed as (dir_lo32 | btn_hi32<<32)
# is the layout consistent with both facts. Confirmed/adjusted by the extraction (1 line).
_DIR_MASK = 0xFFFFFFFF
_BTN_SHIFT = 32

SPECIAL_BASE = 0x8000
SEQUENCE_BASE = 0x800D  # direction > this indexes input_sequences[direction - 0x800D]

# button field bit → notation button
_BUTTON_BITS: dict[int, str] = {0: "1", 1: "2", 2: "3", 3: "4", 4: "RA"}
_PARTIAL_INPUT_BIT = 29

# Special (non-sequence) direction commands, 0x8000..0x800d.
_SPECIAL_COMMANDS: dict[int, str] = {
    0x8000: "",  # [AUTO] — auto/generic, no explicit motion
    0x8001: "f,F",  # double-tap forward
    0x8002: "b,B",  # double-tap back
    0x8003: "u,U",  # double-tap up
    0x8004: "d,D",  # double-tap down
}


class DirectionAlphabet:
    """Maps a direction *code* (< 0x8000) to its notation token (``"df"``, ``"b"``, …).

    The concrete integers are the single calibration unknown of this spike. On real data they
    are solved from the ground-truth anchors (e.g. Bryan ``1628`` is known to be ``df+2``, so
    that cancel's direction code *is* the ``df`` code). Construct with the solved mapping.
    """

    def __init__(self, code_to_token: dict[int, str]) -> None:
        self._code_to_token = dict(code_to_token)

    def token(self, code: int) -> str | None:
        """Notation token for a direction code, or ``None`` if uncalibrated (code 0 → neutral)."""
        if code == 0:
            return ""  # neutral / no direction
        return self._code_to_token.get(code)

    def is_known(self, code: int) -> bool:
        return code == 0 or code in self._code_to_token


@dataclass(frozen=True)
class DecodedCommand:
    """A decoded cancel command, before it is joined into full (possibly string) notation."""

    motion: str  # direction/motion token: "", "df", "qcb", "f,F", … ("" == neutral)
    buttons: tuple[str, ...]  # ("1",), ("1", "2"), ("RA",), …
    kind: str  # "direct" | "special" | "sequence"
    partial_input: bool = False
    uncalibrated_dir_code: int | None = None  # set when the direction code has no alphabet entry

    def notation(self) -> str | None:
        """Single-cancel notation (``"df+2"``, ``"1"``). ``None`` if the direction is unknown."""
        if self.uncalibrated_dir_code is not None:
            return None
        sep = "|" if self.partial_input else "+"
        btn = sep.join(self.buttons)
        if self.motion and btn:
            return f"{self.motion}+{btn}"
        return self.motion or btn


def _decode_buttons(button_bits: int) -> tuple[tuple[str, ...], bool]:
    buttons = tuple(tok for bit, tok in sorted(_BUTTON_BITS.items()) if button_bits & (1 << bit))
    partial = bool(button_bits & (1 << _PARTIAL_INPUT_BIT))
    return buttons, partial


def decode_command(
    command: int,
    alphabet: DirectionAlphabet,
    input_sequences: dict[int, str],
) -> DecodedCommand:
    """Decode a raw ``tk_cancel.command`` uint64 into motion + buttons.

    ``input_sequences`` maps a sequence index → its resolved motion token (e.g. ``0 -> "qcb"``).
    Resolving the motion from ``tk_input`` rows is a further documented layer, out of scope for
    the spike gate; the fixture supplies pre-resolved motions for the few sequence moves.
    """
    direction = command & _DIR_MASK
    button_bits = (command >> _BTN_SHIFT) & _DIR_MASK
    buttons, partial = _decode_buttons(button_bits)

    if direction > SEQUENCE_BASE:
        idx = direction - SEQUENCE_BASE
        motion = input_sequences.get(idx, f"seq#{idx}")
        return DecodedCommand(motion, buttons, "sequence", partial)
    if direction >= SPECIAL_BASE:
        motion = _SPECIAL_COMMANDS.get(direction, f"special#{direction:#x}")
        return DecodedCommand(motion, buttons, "special", partial)
    # direct directional input
    if not alphabet.is_known(direction):
        return DecodedCommand("", buttons, "direct", partial, uncalibrated_dir_code=direction)
    motion = alphabet.token(direction) or ""
    return DecodedCommand(motion, buttons, "direct", partial)


@dataclass(frozen=True)
class Cancel:
    """One ``tk_cancel`` row, reduced to the fields the join needs."""

    source_move_id: int  # the move this cancel belongs to (its cancel-list owner)
    dest_move_id: int  # tk_cancel.move_id @ 0x24 — the move you transition INTO
    command: int  # tk_cancel.command @ 0x00


@dataclass
class JoinResult:
    """Output of the move_id → notation join."""

    notation: dict[int, str] = field(default_factory=dict)  # move_id → reconstructed notation
    collisions: dict[int, list[str]] = field(default_factory=dict)  # move_id → competing notations
    unresolved: dict[int, str] = field(default_factory=dict)  # move_id → why (no cancel/uncalib.)


def join_moves(
    cancels: list[Cancel],
    *,
    neutral_move_id: int,
    alphabet: DirectionAlphabet,
    input_sequences: dict[int, str],
) -> JoinResult:
    """Reconstruct ``move_id → notation`` from the cancel graph.

    A move reached by a cancel whose owner is the *neutral* move is a from-neutral move — its
    notation is the decoded command (``1``, ``df+2``, ``qcb+3``). A move reached from another
    already-resolved move is a *string continuation* — its notation is the source's notation
    plus ``","`` plus the decoded follow-up (``1`` → ``1,2``). Resolution is a fixed point so a
    string resolves once its prefix does. A move with two conflicting candidates is a
    **collision**, never a guess (the honest-limit posture of docs/05 §2.3).
    """
    by_dest: dict[int, list[Cancel]] = {}
    for c in cancels:
        by_dest.setdefault(c.dest_move_id, []).append(c)

    result = JoinResult()
    # Fixed-point: keep resolving moves whose source notation is available.
    resolved: dict[int, str] = {}
    pending = set(by_dest) - {neutral_move_id}
    progressed = True
    while progressed:
        progressed = False
        for dest in sorted(pending):
            neutral_candidates: set[str] = set()
            string_candidates: set[str] = set()
            blocked = False
            for c in by_dest[dest]:
                suffix = decode_command(c.command, alphabet, input_sequences).notation()
                if suffix is None:
                    continue  # uncalibrated direction — contributes no candidate
                if c.source_move_id == neutral_move_id:
                    neutral_candidates.add(suffix)
                elif c.source_move_id in resolved:
                    string_candidates.add(f"{resolved[c.source_move_id]},{suffix}")
                elif c.source_move_id in pending:
                    blocked = True  # prefix not resolved yet — retry next sweep
            # A from-neutral command is the move's canonical notation; string-continuation paths
            # (the jab is also a cancel target mid-string) only name moves with no neutral entry.
            candidates = neutral_candidates or string_candidates
            if not candidates:
                if blocked:
                    continue
                result.unresolved[dest] = "no decodable from-neutral or resolved-prefix cancel"
                pending.discard(dest)
                progressed = True
            elif len(candidates) == 1:
                resolved[dest] = next(iter(candidates))
                pending.discard(dest)
                progressed = True
            else:
                result.collisions[dest] = sorted(candidates)
                pending.discard(dest)
                progressed = True

    result.notation.update(resolved)
    # Anything still pending after the fixed point is stuck behind an unresolved prefix.
    for dest in pending:
        result.unresolved.setdefault(dest, "prefix move never resolved")
    return result
