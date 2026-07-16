"""Live state monitor — show what the reader *thinks* each player is doing (verification tool).

After the state-map is calibrated (docs/02 §8), the way to check it is to watch the game decode in
real time and eyeball it against what you actually did: stand, and it should read ``neutral``;
block, ``blockstun``; get juggled, ``hitstun`` + ``juggle``. This renders the decoded
:class:`~tekken_coach.schemas.PlayerFrame` for both players and prints a line whenever a player's
*decoded state* changes (not every frame — a held state reads as one line, not a flood). A
``[match]`` line also shows the derived match phase (``menu``…``match_over``) + round + raw counter
+ the global ``match_flag`` whenever the phase changes, so the round-gating deriver (docs/02 §8) can
be eyeballed against live play.

Like :mod:`.probe`, the live ``while True`` loop is untestable in CI, so the parts worth testing are
pulled out as pure functions over already-decoded :class:`~tekken_coach.schemas.FrameRecord`\\ s:
:func:`views_of` (frame -> per-player views), :func:`format_view` (view -> console line),
:func:`changed_views` (per-player emit-on-change), and :func:`monitor_lines` (the whole live-loop
output — phase + view lines). Read-only throughout — it only decodes and prints.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from tekken_coach.reader.decode import DerivedPhase
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
    counter: int  # frames_since_round_start — the per-round counter the phase deriver reads (§8)
    raw_state: tuple[tuple[str, int], ...]
    # Decoded per-frame input (numpad dir 1-9, sorted attack buttons), or None when unresolvable —
    # the input-reconstruction probe (brief #9) reads it to validate the input_* offsets. Not shown
    # or keyed on unless the caller opts in (``show_input`` / ``with_input``), so the default
    # state-map monitor is unchanged.
    input: tuple[int, tuple[str, ...]] | None = None

    @property
    def key(self) -> tuple[str, tuple[str, ...], int]:
        # Deliberately excludes ``counter`` (and move_frame): it ticks every frame, so keying on it
        # would flood a held state to one line per poll. It rides the line only for eyeballing.
        return (self.action_state, self.flags, self.move_id)

    @property
    def input_key(self) -> tuple[int, tuple[str, ...]] | None:
        # The change key for the input probe: emit a line whenever the decoded input changes, even
        # while the player stands still (action_state/move_id held). This is what lets the user
        # press each button / hold each direction and see the decode react, one line per input.
        return self.input


def view_of(index: int, pf: PlayerFrame) -> PlayerView:
    """Reduce a decoded :class:`PlayerFrame` to a :class:`PlayerView` (1-based ``player``)."""
    flags = tuple(name for name in BOOL_FLAGS if getattr(pf, name))
    raw = tuple(sorted((pf.raw_state or {}).items()))
    inp = (pf.input.dir, tuple(pf.input.buttons)) if pf.input is not None else None
    return PlayerView(
        player=index + 1,
        char_id=pf.char_id,
        action_state=pf.action_state.value,
        flags=flags,
        move_id=pf.move_id,
        move_frame=pf.move_frame,
        health=pf.health,
        counter=pf.frames_since_round_start,
        raw_state=raw,
        input=inp,
    )


def format_input(inp: tuple[int, tuple[str, ...]] | None) -> str:
    """Render a decoded input as ``dir:buttons`` (``6:2``), ``5:-`` for a bare direction, ``none``.

    The numpad ``dir`` and the pressed buttons are exactly the onset window a reconstructor matches
    against the character's notation (brief #9), so the probe line shows them verbatim.
    """
    if inp is None:
        return "none"
    dir_, buttons = inp
    return f"{dir_}:{'+'.join(buttons) if buttons else '-'}"


def views_of(frame: FrameRecord) -> list[PlayerView]:
    """Both players' :class:`PlayerView`\\ s for a decoded frame, in P1..P2 order."""
    return [view_of(i, pf) for i, pf in enumerate(frame.players)]


def format_view(view: PlayerView, *, show_raw: bool = False, show_input: bool = False) -> str:
    """Render one view as an aligned console line.

    ``show_raw`` appends the raw encoded state words; ``show_input`` appends the decoded input onset
    (``in=6:2``) for the input-reconstruction probe (brief #9).
    """
    flags = " ".join(view.flags) if view.flags else "-"
    line = (
        f"P{view.player} char={view.char_id:<3} {view.action_state:<12} "
        f"move={view.move_id:<6} frame={view.move_frame:<4} hp={view.health:<4} "
        f"cnt={view.counter:<5} [{flags}]"
    )
    if show_input:
        line += f"  in={format_input(view.input)}"
    if show_raw and view.raw_state:
        raw = " ".join(f"{name}={value}" for name, value in view.raw_state)
        line += f"  raw({raw})"
    return line


def _view_key(view: PlayerView, *, with_input: bool) -> object:
    """The change key for a view: the decoded input for the probe, else the decoded state."""
    return view.input_key if with_input else view.key


def _changed(previous: dict[int, object], view: PlayerView, *, with_input: bool = False) -> bool:
    """Whether ``view`` differs from the last-seen key for its player; records it if so.

    ``with_input`` selects the probe key (:attr:`PlayerView.input_key`) so a bare button press
    surfaces even while the player stands still; otherwise the state key drives change-detection.
    """
    key = _view_key(view, with_input=with_input)
    if previous.get(view.player) == key:
        return False
    previous[view.player] = key
    return True


def changed_views(
    stream: Iterable[tuple[float, Sequence[PlayerView]]],
    *,
    with_input: bool = False,
) -> Iterator[tuple[float, PlayerView]]:
    """Yield ``(t, view)`` each time a player's decoded state (:attr:`PlayerView.key`) changes.

    Emits only on change, per player independently — a state performed and held reads as a single
    line, not one per poll. The pure counterpart of the live loop (like ``probe.change_records``).
    ``with_input`` keys on the decoded input instead (the brief #9 probe), so each distinct input
    surfaces even while the player's state is held.
    """
    previous: dict[int, object] = {}
    for t, views in stream:
        for view in views:
            if _changed(previous, view, with_input=with_input):
                yield t, view


def format_phase(match_state: str, round_no: int, counter: int, match_flag: int) -> str:
    """Render the derived match phase as a console line (round-gating, docs/02 §8).

    Shows the now-fuller ``match_state`` (``menu``/``match_over`` included, Stage 2) plus the raw
    global ``match_flag`` that gates in-stage vs menu, so both are eyeballable against real play.
    """
    return f"[match] {match_state:<10} round={round_no:<2} counter={counter:<5} flag={match_flag}"


def monitor_lines(
    stream: Iterable[tuple[float, DerivedPhase, int, Sequence[PlayerView]]],
    *,
    show_raw: bool = False,
    show_input: bool = False,
) -> Iterator[str]:
    """Yield formatted monitor lines from a ``(t, phase, match_flag, views)`` stream (live loop).

    Emits a ``[match]`` line whenever the derived phase (``match_state`` + ``round``) changes, and a
    per-player line whenever a player's decoded state changes — both on-change so a held situation
    reads as one line, not a per-poll flood. The raw ``match_flag`` rides the ``[match]`` line for
    eyeballing but is deliberately **not** in the change key (like the per-round counter): in a menu
    it churns every poll, so keying on it would flood. Pure and fully testable; the live ``monitor``
    loop is a thin shell that decodes frames, derives the phase + flag, and prints these.

    ``show_input`` switches per-player change-detection to key on the decoded input (brief #9's
    input-reconstruction probe) and appends it to each line, so pressing a button / holding a
    direction surfaces one line per distinct input even while the player's state is held.
    """
    prev_phase: tuple[str, int] | None = None
    prev_views: dict[int, object] = {}
    for t, phase, match_flag, views in stream:
        pkey = (phase.match_state.value, phase.round)
        if pkey != prev_phase:
            prev_phase = pkey
            counter = views[0].counter if views else 0
            line = format_phase(phase.match_state.value, phase.round, counter, match_flag)
            yield f"{t:>7.2f}  {line}"
        for view in views:
            if _changed(prev_views, view, with_input=show_input):
                yield f"{t:>7.2f}  {format_view(view, show_raw=show_raw, show_input=show_input)}"
