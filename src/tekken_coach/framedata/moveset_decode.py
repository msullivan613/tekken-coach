"""Decode a Tekken 8 moveset *cancel command* into input notation, and join the cancel graph.

Brief #18 (route B): our read-only reader walks the character's live ``cancels`` array and turns
each ``tk_cancel.command`` into notation, then joins ``move_id -> notation`` off the cancel graph.
This is the **production** decoder — it supersedes the spike's ``tests/spikes/moveset_datamine``
``decode_command``, which was T7-shaped (wrong direction/button model, an invented ``0x8000/0x800d``
range). Only the *game-agnostic* join (:func:`join_moves`) is ported from the spike; the command
decode below is built fresh from the **confirmed Tekken 8 encoding**.

Everything here is built from **published Tekken 8 structure facts** (usable with attribution),
never from copied code or a labelled move table — tekkenmods.com/documentation/Tekken_8/… docs:

* ``command`` (uint64) splits into ``direction = command & 0xFFFFFFFF`` (low 32) and
  ``button = command >> 32`` (high 32).
* **Direction** (low 32) is a *known* bitfield — no calibration unknown::

    0x00 any · 0x02 db · 0x04 d · 0x08 df · 0x10 b · 0x20 n · 0x40 f · 0x80 ub · 0x100 u · 0x200 uf

  ``0x00`` (any) and ``0x20`` (neutral) both mean **no directional prefix**. An unrecognized
  direction value decodes to *unresolved* — never a wrong guess (docs/05 §2.3 miss-tolerance).
* **Button** (high 32) is ``0xMMNNHHPP``:
  - ``PP`` (low byte) = buttons pressed: ``0x01=1 · 0x02=2 · 0x04=3 · 0x08=4 · 0x10=Heat ·
    0x20=Special · 0x40=Rage Art``. Only 1-4 are notation buttons; Heat/Special/RA are **not**.
  - ``MM`` (high byte of the 32) = mode: ``0x20=partial · 0x40=normal · 0x80=direction-only``.
  - ``HH`` (held) / ``NN`` (forbidden) — **ignored for v1** (held-direction charge moves such as a
    capitalized ``DF`` therefore decode to their tapped ``df`` form, or fall to unresolved).

The clean-room boundary (docs/02 §5): this decodes documented *structure*, and it never invents a
mapping to hit coverage — an input it cannot cleanly notate (a pure Heat/RA engage, an unknown
direction, a motion from the ``input_sequences`` array) returns ``None`` so the join reports it as
needs-manual rather than writing a wrong notation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- command field split (T8: direction = low 32b, button = high 32b) ----------------------------
_DIR_MASK = 0xFFFFFFFF
_BTN_SHIFT = 32

# Direction bitfield (low 32) -> notation token. The documented single-direction codes; combinations
# are expressed by their own code (df has 0x08, not d|f), so a value absent from this table (and not
# a no-prefix code) is an input this v1 does not model -> unresolved.
_DIRECTION_TOKENS: dict[int, str] = {
    0x02: "db",
    0x04: "d",
    0x08: "df",
    0x10: "b",
    0x40: "f",
    0x80: "ub",
    0x100: "u",
    0x200: "uf",
}

# 0x00 = any, 0x20 = neutral: both mean "no directional prefix" (a bare button input like the jab).
_NO_PREFIX_DIRECTIONS = frozenset({0x00, 0x20})

# Button field low byte (PP) -> notation button. Only 1-4 are notation buttons.
_ATTACK_BUTTON_TOKENS: tuple[tuple[int, str], ...] = (
    (0x01, "1"),
    (0x02, "2"),
    (0x04, "3"),
    (0x08, "4"),
)

# PP bits that are NOT notation buttons 1-4 (a move gated on Heat/Special/Rage Art). Their presence
# without any plain 1-4 button means this v1 cannot cleanly notate the input -> unresolved.
_SPECIAL_BUTTON_MASK = 0x10 | 0x20 | 0x40  # Heat | Special | Rage Art

# Mode byte (MM, the high byte of the button 32) values.
MODE_PARTIAL = 0x20
MODE_NORMAL = 0x40
MODE_DIRECTION_ONLY = 0x80


@dataclass(frozen=True)
class DecodedCommand:
    """A decoded ``tk_cancel.command``, before it is joined into full (possibly string) notation.

    ``direction`` is the resolved token (``""`` == no prefix); ``buttons`` are the notation buttons
    1-4 pressed. The flags record why a command may not resolve to clean notation:

    * ``unknown_direction`` — the direction bits are not a modeled code (a held/charge input or a
      value this v1 does not cover). :meth:`notation` returns ``None``.
    * ``special_only`` — the only buttons pressed are Heat/Special/Rage Art (no 1-4), so there is no
      clean notation to emit. :meth:`notation` returns ``None``.
    """

    direction: str
    buttons: tuple[str, ...]
    mode: int
    unknown_direction: bool = False
    special_only: bool = False

    def notation(self) -> str | None:
        """The single-cancel notation (``"df+2"``, ``"1"``, ``"f"``), or ``None`` if unresolved.

        ``None`` is the honest degrade: an unknown direction or a special-only engage yields no
        candidate rather than a wrong guess (docs/05 §2.3). A bare direction (no buttons) resolves
        only in ``direction-only`` mode — a movement input like a dash — and otherwise is nothing.
        """
        if self.unknown_direction or self.special_only:
            return None
        btn = "+".join(self.buttons)
        if self.direction and btn:
            return f"{self.direction}+{btn}"
        if btn:
            return btn
        # No buttons: only a direction-only movement input carries notation on its own.
        if self.direction and self.mode == MODE_DIRECTION_ONLY:
            return self.direction
        return None


def decode_command(command: int) -> DecodedCommand:
    """Decode a raw ``tk_cancel.command`` uint64 into direction + buttons (confirmed T8 encoding).

    Pure and total: every uint64 decodes to a :class:`DecodedCommand`; whether it yields *notation*
    is :meth:`DecodedCommand.notation`'s call (``None`` when unresolved).
    """
    direction_bits = command & _DIR_MASK
    button_field = (command >> _BTN_SHIFT) & _DIR_MASK
    pressed = button_field & 0xFF
    mode = (button_field >> 24) & 0xFF

    if direction_bits in _NO_PREFIX_DIRECTIONS:
        direction, unknown = "", False
    else:
        token = _DIRECTION_TOKENS.get(direction_bits)
        direction, unknown = (token, False) if token is not None else ("", True)

    buttons = tuple(tok for bit, tok in _ATTACK_BUTTON_TOKENS if pressed & bit)
    special_only = bool(pressed & _SPECIAL_BUTTON_MASK) and not buttons

    return DecodedCommand(
        direction=direction,
        buttons=buttons,
        mode=mode,
        unknown_direction=unknown,
        special_only=special_only,
    )


# ---------------------------------------------------------------------------
# The cancel-graph join (game-agnostic; ported from the #15 spike, decode swapped for T8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cancel:
    """One ``tk_cancel`` row, reduced to the fields the join needs.

    ``source_move_id`` is the move whose cancel-list this row belongs to (its owner), and
    ``dest_move_id`` (``tk_cancel.move_id`` @ 0x24) is the move it transitions *into*. Owner
    attribution is what separates a from-neutral canonical input from a mid-string follow-up.
    """

    source_move_id: int
    dest_move_id: int
    command: int


@dataclass
class JoinResult:
    """Output of the ``move_id -> notation`` join."""

    notation: dict[int, str] = field(default_factory=dict)  # move_id -> reconstructed notation
    collisions: dict[int, list[str]] = field(default_factory=dict)  # move_id -> competing notations
    unresolved: dict[int, str] = field(
        default_factory=dict
    )  # move_id -> why (no cancel/undecodable)


def join_moves(cancels: list[Cancel], *, neutral_move_id: int) -> JoinResult:
    """Reconstruct ``move_id -> notation`` from the cancel graph (ported spike logic, T8 decode).

    A move reached by a cancel whose owner is the **neutral** move is a from-neutral move — its
    notation is the decoded command (``1``, ``df+2``). A move reached from another already-resolved
    move is a **string continuation** — its notation is the source's notation plus ``","`` plus the
    decoded follow-up (``1`` -> ``1,2``). Resolution is a fixed point so a string resolves once its
    prefix does. A from-neutral command is canonical: it wins over any string path to the same move
    (the jab is also a mid-string cancel target, but stays ``1``, not ``b+1,1``). A move with two
    conflicting *from-neutral* candidates is a **collision**, reported never guessed (docs/05 §2.3).
    """
    by_dest: dict[int, list[Cancel]] = {}
    for c in cancels:
        by_dest.setdefault(c.dest_move_id, []).append(c)

    result = JoinResult()
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
                suffix = decode_command(c.command).notation()
                if suffix is None:
                    continue  # undecodable command — contributes no candidate
                if c.source_move_id == neutral_move_id:
                    neutral_candidates.add(suffix)
                elif c.source_move_id in resolved:
                    string_candidates.add(f"{resolved[c.source_move_id]},{suffix}")
                elif c.source_move_id in pending:
                    blocked = True  # prefix not resolved yet — retry next sweep
            # From-neutral is canonical; string paths only name moves with no from-neutral entry.
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
    for dest in pending:
        result.unresolved.setdefault(dest, "prefix move never resolved")
    return result
