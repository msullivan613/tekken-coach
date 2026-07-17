"""Re-derive the live ``input_dir`` / ``input_buttons`` player-struct offsets by observation.

The seeded input offsets (``input_valid@55``/``input_dir@56``/``input_buttons@64``) are fork-era
leftovers and read dead on 5.02.01: 79 s of live mashing decoded ``input=None`` every frame, while
``frames_since_round_start`` (a *player-struct* field) read fine — so the holder chain works and the
input offsets specifically are stale. Nothing but observation can say where they moved to.

This module is the offline half of that re-derivation. The live half is the existing sweep::

    py -m tekken_coach.reader.commands probe-state --watch "0x0-0x100:u8" --record debug/input.jsonl

which watches **both players** and records one JSONL row per change (:mod:`.probe`). The user runs
one pass following :data:`PROTOCOL` — press each button alone, then hold each direction, resting in
between — and this module ranks every swept offset by how well it correlates with that script.

**The discriminator that makes this tractable**: in Practice the user is P1 and the P2 dummy is
static, so a genuine input field changes on the acting player's struct only. Two more cut the field
hard: an ``input_buttons`` candidate must react on the *button* steps and stay at rest through the
*direction* steps (and vice versa), and both must return to a single rest value on release.

Pure and offline throughout — it consumes the recorded JSONL, so the ranking is unit-tested against
scripted records rather than the game. Nothing here maps a value to a meaning: it reports the
observed value sets per candidate and lets the human read the encoding off them (docs/02 §5 rule 2).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

# Human timing is sloppy: the user is following a printed script, not a frame-accurate trigger. So
# the first slice of every window is discarded before the value is judged — a press that lands late
# (or is released early) must not read as "the field failed to react".
SETTLE = 0.4

# A quiet baseline before the first press, so the rest value is established from a window nobody is
# touching (and so a mis-started recording is obvious in the log rather than silently skewing rest).
BASELINE = 5.0

# Cardinality ceilings per role (brief #10): a direction is one of 9 numpad values (plus rest), a
# button mask over 4 buttons with one pair pressed shows at most {0,1,2,4,8,3}. A candidate that
# takes far more distinct values across the script is churning, not encoding input.
_CARDINALITY_CEILING: dict[str, int] = {"dir": 10, "button": 8}

Value = int | float


@dataclass(frozen=True)
class Step:
    """One scripted action: hold ``label`` for ``hold`` seconds, then rest for ``rest`` seconds.

    ``role`` is ``"button"`` or ``"dir"`` — the axis this step is meant to move. It is what lets the
    ranking demand that an ``input_buttons`` candidate stay *still* through the direction steps: a
    field that reacts to everything is not the button mask.
    """

    label: str
    role: str
    hold: float = 2.0
    rest: float = 2.0


# The recorded pass, in order. Each button is pressed alone (so its bit is unambiguous), then `1+2`
# proves the mask is a bitwise OR rather than a last-button-wins code; `1` and `f` repeat late so
# the ranking can check the same action reads the same value (a map, not a drifting counter).
# Directions cover all 8 holds, which is what answers "is input_dir numpad 1-9, or a raw stick value
# needing a mapping?" — the observed value set per direction *is* the answer.
PROTOCOL: tuple[Step, ...] = (
    Step("1", "button"),
    Step("2", "button"),
    Step("3", "button"),
    Step("4", "button"),
    Step("1+2", "button"),
    Step("1 (again)", "button"),
    Step("u", "dir"),
    Step("d", "dir"),
    Step("b", "dir"),
    Step("f", "dir"),
    Step("u/f", "dir"),
    Step("d/f", "dir"),
    Step("d/b", "dir"),
    Step("u/b", "dir"),
    Step("f (again)", "dir"),
)


@dataclass(frozen=True)
class Window:
    """A stretch of the script with a known expectation: ``kind`` is ``"hold"`` or ``"rest"``."""

    t0: float
    t1: float
    kind: str
    step: Step | None  # None for the leading baseline rest


def step_windows(protocol: Sequence[Step] = PROTOCOL, start: float = 0.0) -> list[Window]:
    """Lay the protocol out on the probe's elapsed-seconds clock, from ``start``.

    ``start`` is when the user began the script relative to ``probe-state``'s ``t=0``; the recorded
    log and the script are two clocks, and :func:`best_alignment` searches this parameter rather
    than trusting the user to have started both at the same instant.
    """
    windows = [Window(start, start + BASELINE, "rest", None)]
    t = start + BASELINE
    for step in protocol:
        windows.append(Window(t, t + step.hold, "hold", step))
        t += step.hold
        windows.append(Window(t, t + step.rest, "rest", step))
        t += step.rest
    return windows


def render_checklist(protocol: Sequence[Step] = PROTOCOL, start: float = 0.0) -> Iterator[str]:
    """Render the script as a timestamped checklist the user reads while the probe records."""
    yield f"{'t':>7}  {'action':<14}  what to do"
    yield f"{start:>7.1f}  {'(baseline)':<14}  hands OFF the pad — let it settle"
    for window in step_windows(protocol, start):
        if window.kind != "hold" or window.step is None:
            continue
        verb = "press+hold" if window.step.role == "button" else "hold"
        yield (
            f"{window.t0:>7.1f}  {window.step.label:<14}  "
            f"{verb} {window.step.label} for {window.step.hold:.0f}s, then release and rest "
            f"{window.step.rest:.0f}s"
        )
    total = step_windows(protocol, start)[-1].t1
    yield f"{total:>7.1f}  {'(done)':<14}  stop the probe (Ctrl-C)"


# --- reading the recorded log -------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """The recorded sweep: per player, per watched field, the change series the probe logged."""

    series: dict[int, dict[str, list[tuple[float, Value]]]]

    @property
    def fields(self) -> list[str]:
        """Every watched field in the log, sorted by offset when the names are ``@0x…``."""
        names = {name for by_field in self.series.values() for name in by_field}
        return sorted(names, key=_name_sort_key)

    @property
    def players(self) -> list[int]:
        return sorted(self.series)


def _name_sort_key(name: str) -> tuple[int, object]:
    """Sort ``@0x38`` before ``@0x100`` (numerically), and any other name lexically after."""
    if name.startswith("@0x"):
        try:
            return (0, int(name[1:], 16))
        except ValueError:
            pass
    return (1, name)


def load_observation(lines: Iterable[str]) -> Observation:
    """Parse ``probe-state --record`` JSONL into per-player, per-field change series.

    Each row is ``{"t":…, "player":…, "fields":{name: value}}`` and carries **every** watched field
    (the probe emits a full row whenever any one of them changes), so a field's value at any instant
    is the last recorded value at or before it — a step function, reconstructed here.
    """
    series: dict[int, dict[str, list[tuple[float, Value]]]] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        by_field = series.setdefault(int(row["player"]), {})
        t = float(row["t"])
        for name, value in row["fields"].items():
            points = by_field.setdefault(name, [])
            # The probe emits on *tuple* change, so a given field repeats unchanged across rows;
            # keep only genuine transitions so the series is the field's own step function.
            if not points or points[-1][1] != value:
                points.append((t, value))
    return Observation(series=series)


def load_observation_file(path: Path) -> Observation:
    with path.open(encoding="utf-8") as handle:
        return load_observation(handle)


def _segments(
    points: Sequence[tuple[float, Value]], t0: float, t1: float
) -> list[tuple[Value, float]]:
    """The ``(value, seconds_held)`` segments a field's step function spends inside ``[t0, t1)``."""
    if not points or t1 <= t0:
        return []
    out: list[tuple[Value, float]] = []
    for index, (at, value) in enumerate(points):
        until = points[index + 1][0] if index + 1 < len(points) else float("inf")
        lo, hi = max(at, t0), min(until, t1)
        if hi > lo:
            out.append((value, hi - lo))
    return out


def window_values(points: Sequence[tuple[float, Value]], window: Window) -> tuple[Value, ...]:
    """The distinct values a field holds through ``window``, after the :data:`SETTLE` margin."""
    seen: list[Value] = []
    for value, _ in _segments(points, window.t0 + SETTLE, window.t1):
        if not seen or seen[-1] != value:
            seen.append(value)
    return tuple(seen)


def dominant_value(points: Sequence[tuple[float, Value]], window: Window) -> Value | None:
    """The value a field spends the most *time* at inside ``window`` (``None`` if never observed).

    Time-weighted rather than most-frequent: a candidate that glitches through three values for a
    frame each on the way to the held value should still read as the held value.
    """
    held: dict[Value, float] = {}
    for value, seconds in _segments(points, window.t0 + SETTLE, window.t1):
        held[value] = held.get(value, 0.0) + seconds
    if not held:
        return None
    return max(held, key=lambda value: held[value])


# --- ranking ------------------------------------------------------------------------------------

# `reacts` and `quiet_other` are not preferences to be weighed against the rest — they are the two
# NECESSARY conditions, and together they are the role discriminator: the button mask moves on the
# button steps *and stays still* through the direction steps. They multiply the score rather than
# contributing to it, because a weighted sum lets a field that fails one of them coast in on the
# others: a dead-constant offset satisfies "stable rest value", "steady", "consistent" and a
# "plausible cardinality" perfectly while never once reacting to input, and would otherwise outrank
# a real field.
#
# The remaining criteria grade a candidate that already passes the gate. `acting_only` is the
# Practice-mode discriminator: the P2 dummy is static, so its copy of a real input never moves.
_QUALITY_WEIGHTS: dict[str, float] = {
    "rest_stable": 0.35,
    "steady": 0.20,
    "consistent": 0.20,
    "acting_only": 0.20,
    "cardinality": 0.05,
}


@dataclass(frozen=True)
class CandidateScore:
    """One swept offset's fitness as ``input_dir``/``input_buttons``, with the evidence behind it.

    ``values_by_step`` is the whole point of the exercise: the observed value for each scripted
    action. That table *is* the encoding answer — read the direction rows to see whether
    ``input_dir`` is numpad 1-9 or a raw stick value, and the button rows to see whether the mask
    really is bit order ``1,2,3,4`` (:data:`~tekken_coach.reader.decode._BUTTON_BITS`).
    """

    name: str
    role: str
    score: float
    rest_value: Value | None
    values_by_step: dict[str, Value | None]
    distinct: int
    parts: dict[str, float]
    notes: tuple[str, ...]

    @property
    def acting_only(self) -> bool:
        return self.parts.get("acting_only", 0.0) == 1.0


def _fraction(hits: int, total: int, *, vacuous: float = 0.0) -> float:
    """A pass rate. ``vacuous`` is the score when there are no windows to judge (no zero division).

    It differs by criterion: a candidate with no steps of its own role has proven nothing and scores
    0, whereas one with no *other*-role steps to stay quiet through is vacuously quiet and scores 1.
    """
    return hits / total if total else vacuous


def score_candidate(
    obs: Observation,
    name: str,
    role: str,
    *,
    protocol: Sequence[Step] = PROTOCOL,
    start: float = 0.0,
    acting_player: int = 1,
) -> CandidateScore:
    """Score one watched offset as a candidate for ``role`` (``"button"`` or ``"dir"``)."""
    points = obs.series.get(acting_player, {}).get(name, [])
    windows = step_windows(protocol, start)
    rests = [w for w in windows if w.kind == "rest"]
    holds = [w for w in windows if w.kind == "hold" and w.step is not None]
    mine = [w for w in holds if w.step is not None and w.step.role == role]
    others = [w for w in holds if w.step is not None and w.step.role != role]
    notes: list[str] = []

    rest_doms = [dominant_value(points, w) for w in rests]
    seen_rest = {value for value in rest_doms if value is not None}
    rest_value = next(iter(seen_rest)) if len(seen_rest) == 1 else None
    rest_stable = 1.0 if rest_value is not None else 0.0
    if not rest_stable:
        notes.append(f"no single rest value across the gaps (saw {sorted(map(str, seen_rest))})")

    reacts = _fraction(
        sum(1 for w in mine if dominant_value(points, w) not in (None, rest_value)), len(mine)
    )
    if reacts == 0.0:
        notes.append(f"never leaves rest on the {role} steps")
    quiet_other = _fraction(
        sum(1 for w in others if dominant_value(points, w) == rest_value), len(others), vacuous=1.0
    )
    if others and quiet_other < 1.0:
        notes.append(f"also moves on the {'dir' if role == 'button' else 'button'} steps")
    steady = _fraction(sum(1 for w in mine if len(window_values(points, w)) == 1), len(mine))

    values_by_step: dict[str, Value | None] = {}
    for window in mine:
        assert window.step is not None
        values_by_step[window.step.label] = dominant_value(points, window)

    # A repeat of the same action ("1" vs "1 (again)") must read the same value — a real field maps
    # action -> value; a frame counter or a churning neighbour drifts.
    repeats = [(w.step.label, w) for w in mine if w.step is not None and "(again)" in w.step.label]
    consistent = 1.0
    for label, window in repeats:
        base = label.split(" (again)")[0]
        if base in values_by_step and values_by_step[base] != dominant_value(points, window):
            consistent = 0.0
            notes.append(f"{base!r} read differently on its repeat")

    other_points = {
        player: obs.series.get(player, {}).get(name, [])
        for player in obs.players
        if player != acting_player
    }
    moved = [player for player, pts in other_points.items() if len({v for _, v in pts}) > 1]
    acting_only = 0.0 if moved else 1.0
    if moved:
        notes.append(f"also changes on the static dummy (P{', P'.join(map(str, moved))})")

    distinct = len({value for _, value in points})
    ceiling = _CARDINALITY_CEILING[role]
    cardinality = 1.0 if distinct <= ceiling else 0.0
    if not cardinality:
        notes.append(f"{distinct} distinct values (> {ceiling} expected for {role})")

    parts = {
        "reacts": reacts,
        "quiet_other": quiet_other,
        "rest_stable": rest_stable,
        "steady": steady,
        "consistent": consistent,
        "acting_only": acting_only,
        "cardinality": cardinality,
    }
    quality = sum(_QUALITY_WEIGHTS[key] * parts[key] for key in _QUALITY_WEIGHTS)
    return CandidateScore(
        name=name,
        role=role,
        score=reacts * quiet_other * quality,
        rest_value=rest_value,
        values_by_step=values_by_step,
        distinct=distinct,
        parts=parts,
        notes=tuple(notes),
    )


def rank_for_role(
    obs: Observation,
    role: str,
    *,
    protocol: Sequence[Step] = PROTOCOL,
    start: float = 0.0,
    acting_player: int = 1,
    limit: int | None = None,
) -> list[CandidateScore]:
    """Every watched offset scored as ``role``, best first (ties broken by offset for stability)."""
    scores = [
        score_candidate(
            obs, name, role, protocol=protocol, start=start, acting_player=acting_player
        )
        for name in obs.fields
    ]
    scores.sort(key=lambda c: (-c.score, _name_sort_key(c.name)))
    return scores[:limit] if limit is not None else scores


def best_alignment(
    obs: Observation,
    *,
    protocol: Sequence[Step] = PROTOCOL,
    search: float = 4.0,
    step: float = 0.2,
    acting_player: int = 1,
) -> float:
    """Find when the user actually started the script, by fitting it to the log.

    The probe's clock and the user's reading of the checklist are two clocks — expecting them to
    start on the same instant would make the whole ranking hostage to a slow alt-tab. So the script
    is slid across ``±search`` seconds and the offset that best explains the log wins. Scored on the
    single best candidate per role, since at most a couple of offsets are real inputs and everything
    else is noise that no alignment improves.
    """
    best, best_start = -1.0, 0.0
    ticks = int(search / step)
    for tick in range(-ticks, ticks + 1):
        start = tick * step
        total = sum(
            max(
                (
                    c.score
                    for c in rank_for_role(
                        obs, role, protocol=protocol, start=start, acting_player=acting_player
                    )
                ),
                default=0.0,
            )
            for role in ("button", "dir")
        )
        if total > best:
            best, best_start = total, start
    return best_start


# --- reporting ----------------------------------------------------------------------------------

# Below this, a candidate is not evidence of anything — it is the best of a bad field, and the
# report says so rather than naming it. The gate already zeroes anything that fails the role
# discriminator outright; this additionally demands the graded criteria mostly hold, so a field that
# reacts correctly but wanders on the dummy or has no stable rest value still does not get crowned.
MIN_PLAUSIBLE = 0.55


def format_candidate(candidate: CandidateScore) -> Iterator[str]:
    """Render one candidate: its score, its rest value, and the value it read for each action."""
    yield (
        f"  {candidate.name:<10} score={candidate.score:.2f}  "
        f"rest={candidate.rest_value}  distinct={candidate.distinct}"
    )
    parts = " ".join(f"{key}={value:.2f}" for key, value in candidate.parts.items())
    yield f"    {parts}"
    if candidate.values_by_step:
        values = "  ".join(f"{label}={value}" for label, value in candidate.values_by_step.items())
        yield f"    observed: {values}"
    for note in candidate.notes:
        yield f"    - {note}"


def format_report(
    obs: Observation,
    *,
    protocol: Sequence[Step] = PROTOCOL,
    start: float = 0.0,
    acting_player: int = 1,
    top: int = 5,
) -> Iterator[str]:
    """The analyzer verdict: the top candidates per role, and whether to believe any of them."""
    yield (
        f"analyzed {len(obs.fields)} watched offsets x {len(obs.players)} players; "
        f"script aligned at t={start:.1f}s (acting player P{acting_player})"
    )
    for role, field in (("dir", "input_dir"), ("button", "input_buttons")):
        ranked = rank_for_role(
            obs, role, protocol=protocol, start=start, acting_player=acting_player, limit=top
        )
        best = ranked[0] if ranked else None
        yield ""
        if best is None or best.score < MIN_PLAUSIBLE:
            top_score = f"{best.score:.2f}" if best else "n/a"
            yield (
                f"{field}: NO CANDIDATE (best score {top_score} < {MIN_PLAUSIBLE:.2f}). "
                f"Nothing in this sweep behaves like {field} — widen the range or look off the "
                "player struct. Do not bake a guess."
            )
        else:
            yield f"{field}: best candidate {best.name} (score {best.score:.2f})"
        for candidate in ranked:
            yield from format_candidate(candidate)
