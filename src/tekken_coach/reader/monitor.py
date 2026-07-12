"""Live state monitor — show what the reader *thinks* each player is doing (verification tool).

After the state-map is calibrated (docs/02 §8), the way to check it is to watch the game decode in
real time and eyeball it against what you actually did: stand, and it should read ``neutral``;
block, ``blockstun``; get juggled, ``hitstun`` + ``juggle``. This renders the decoded
:class:`~tekken_coach.schemas.PlayerFrame` for both players and prints a line whenever a player's
*decoded state* changes (not every frame — a held state reads as one line, not a flood).

Like :mod:`.probe`, the live ``while True`` loop is untestable in CI, so the parts worth testing are
pulled out as pure functions over already-decoded :class:`~tekken_coach.schemas.FrameRecord`\\ s:
:func:`views_of` (frame -> per-player views), :func:`format_view` (view -> console line), and
:func:`changed_views` (emit-on-change). Read-only throughout — it only decodes and prints.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from tekken_coach.schemas import FrameRecord, PlayerFrame

# The boolean situational flags a PlayerFrame carries (docs/03 §1); the rest of the situation is in
# action_state. Order fixed so a line is stable to read.
BOOL_FLAGS: tuple[str, ...] = (
    "block_stun",
    "hit_stun",
    "throw_active",
    "airborne",
    "juggle",
)


@dataclass(frozen=True)
class PlayerView:
    """The decoded state of one player, reduced to what the monitor shows and keys change on.

    ``key`` is what change-detection compares: the *decoded state* (``action_state`` + situational
    flags) **plus ``move_id``**. Including ``move_id`` is what lets a string read as its constituent
    moves — ``1,2,1`` stays ``action_state=attack`` throughout, so keying on state alone would
    collapse it to a single line; keying on ``move_id`` too surfaces each move (and makes a movement
    whose ``action_state`` never leaves neutral, e.g. a backdash, still show as a line). ``move_id``
    only changes per move, not per frame, so it does not flood; ``move_frame`` (which does change
    every frame) is deliberately **not** in the key.
    """

    player: int  # 1-based, matching the on-screen P1/P2
    char_id: int
    action_state: str
    flags: tuple[str, ...]
    move_id: int
    move_frame: int
    health: int
    raw_state: tuple[tuple[str, int], ...]

    @property
    def key(self) -> tuple[str, tuple[str, ...], int]:
        return (self.action_state, self.flags, self.move_id)


def view_of(index: int, pf: PlayerFrame) -> PlayerView:
    """Reduce a decoded :class:`PlayerFrame` to a :class:`PlayerView` (1-based ``player``)."""
    flags = tuple(name for name in BOOL_FLAGS if getattr(pf, name))
    raw = tuple(sorted((pf.raw_state or {}).items()))
    return PlayerView(
        player=index + 1,
        char_id=pf.char_id,
        action_state=pf.action_state.value,
        flags=flags,
        move_id=pf.move_id,
        move_frame=pf.move_frame,
        health=pf.health,
        raw_state=raw,
    )


def views_of(frame: FrameRecord) -> list[PlayerView]:
    """Both players' :class:`PlayerView`\\ s for a decoded frame, in P1..P2 order."""
    return [view_of(i, pf) for i, pf in enumerate(frame.players)]


def format_view(view: PlayerView, *, show_raw: bool = False) -> str:
    """Render one view as an aligned console line; ``show_raw`` appends the raw encoded words."""
    flags = " ".join(view.flags) if view.flags else "-"
    line = (
        f"P{view.player} char={view.char_id:<3} {view.action_state:<12} "
        f"move={view.move_id:<6} frame={view.move_frame:<4} hp={view.health:<4} [{flags}]"
    )
    if show_raw and view.raw_state:
        raw = " ".join(f"{name}={value}" for name, value in view.raw_state)
        line += f"  raw({raw})"
    return line


def changed_views(
    stream: Iterable[tuple[float, Sequence[PlayerView]]],
) -> Iterator[tuple[float, PlayerView]]:
    """Yield ``(t, view)`` each time a player's decoded state (:attr:`PlayerView.key`) changes.

    Emits only on change, per player independently — a state performed and held reads as a single
    line, not one per poll. The pure counterpart of the live loop (like ``probe.change_records``).
    """
    previous: dict[int, tuple[str, tuple[str, ...], int]] = {}
    for t, views in stream:
        for view in views:
            if previous.get(view.player) == view.key:
                continue
            previous[view.player] = view.key
            yield t, view
