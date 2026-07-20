"""Build ``move_id -> notation`` by walking the live per-move cancel graph (brief #26).

The 2026-07-19 live run closed the last open unknown from brief #18. On 5.02.01, confirmed:

* ``tk_move + 0x098`` is **this move's cancel-run start pointer**;
* runs are **contiguous**, so move *N* owns ``[ptr(N), ptr(N+1))`` — both probed spans came out
  exact multiples of ``CANCEL_SIZE`` (360 = 9 rows, 320 = 8 rows);
* the ``tk_cancel`` layout (``command`` @ 0x00, ``move_id`` @ 0x24, stride 0x28) and the T8 command
  encoding both hold — the jab's run decoded to ``f+1+2 / 1+2 / 1+3 / 2 / 4`` and standing-2's to
  ``2+3 / 2+4 / 1+2 / 3``, which are Bryan's real follow-ups. Independently, both runs mapped
  command ``1+2`` to move **1546**, the same id the live capture protocol recorded for ``1+2``
  from neutral.

Per-move runs *are* owner attribution, so #18's open unknown is closed and the map is derivable.
This module is the walk; the decode and the join are the existing, tested
:mod:`tekken_coach.framedata.moveset_decode` — this feeds them better data, it does not re-implement
them.

Three steps, all offline-testable against the planted world:

* **Part A — :func:`read_move_cancels`.** One move's run, read via #25's
  :func:`~tekken_coach.reader.discovery.moveset_anchor.cancel_range`, returned as
  :class:`~tekken_coach.framedata.moveset_decode.Cancel` rows owned by that move. The two
  **structural** rows are excluded and *counted* rather than silently dropped: the run terminator
  (destination is the ``0x8001`` neutral alias, not a move index) and the auto-transition
  (``command == 0`` to move ``N+1`` — the jab's ``-> 1696`` recovery link). Neither is an input, so
  neither may become a graph edge.

* **Part B — :func:`find_neutral_move`.** A per-move run holds that move's *follow-ups*, so the
  from-neutral anchors are not expected in an arbitrary move's run — which makes them the test for
  *finding* the neutral move. The move whose run maps ``1 -> 1695``, ``2 -> 1566``, ``3 -> 1573``,
  ``4 -> 1574`` **is** the root. Same self-validating gate trick as #21/#25, one more time.

* **Part C — :func:`walk_cancel_graph`.** BFS from the root, collecting each reached move's rows and
  following destinations, bounded by ``max_moves`` and a visited set (the graph has cycles — a
  string can return to neutral). The collected rows go to the existing ``join_moves``.

**Many-to-one is expected and preserved.** Command ``2`` from the jab reaches destinations 1697,
1698 *and* 1699; ``3`` from standing-2 reaches 1567/1568/1569. Those are variants of one logical
move that legitimately share a notation — *not* a collision. A collision is one move_id carrying two
conflicting **from-neutral** notations, which ``join_moves`` reports and never guesses. The live
reader sees whichever variant fires, so the ids are all kept.

Read-only throughout (docs/02 §2): resolve, read, follow pointers. Non-results stay data — an
unreadable pointer, a falsified span, or a descending span yields no rows plus a recorded reason,
never an exception and never a fabricated edge (docs/05 §2.3).

Deliberately **not** routed through ``read_moveset_header`` / ``MoveLayout`` /
``read_attributed_cancels``: those assume the doc-derived header offsets that the 2026-07-19 dumps
proved wrong for 5.02.01. The anchor path (``moves_base`` + ``move_stride`` + ``+0x098``) replaces
them.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from tekken_coach.framedata.moveset_decode import Cancel
from tekken_coach.reader.discovery.moveset_anchor import (
    DEFAULT_CANCEL_PTR_OFFSET,
    MAX_PROBE_ROWS,
    REAL_MOVE_ID_MAX,
    MovesArray,
    cancel_range,
    read_cancel_row,
)
from tekken_coach.reader.memory_source import MemorySource
from tekken_coach.reader.moveset import CANCEL_SIZE, GateRow, KnownPair, RawCancel, gate_pairs

# The run terminator's command word. Its destination is the ``0x8001`` neutral **alias**, which is
# not an index into the moves array — the row closes the run, it is not an input.
TERMINATOR_COMMAND = 0x8000

# How many moves a walk visits before it stops. Bryan's moves array is > 1780 records and the
# reachable set is smaller, so this is headroom, not a target: it exists so a wrong ``ptr_offset``
# feeding junk destinations is a bounded, reportable non-result rather than an unbounded read.
DEFAULT_MAX_MOVES = 4096

# The default sweep for the root search. The from-neutral anchor ids sit in the 1500-1700 band, but
# the neutral move itself is a low index on every build we have seen, so the sweep starts at 0.
DEFAULT_SEARCH_START = 0
DEFAULT_SEARCH_END = 4096

# How many near-miss candidates :func:`find_neutral_move` reports when nothing gates fully. Enough
# to diagnose a stale anchor id (a candidate matching 4 of 5 is a much louder signal than a wall of
# 0-of-5s), short enough that the output stays readable.
DEFAULT_PARTIALS_REPORTED = 5


@dataclass(frozen=True)
class MoveCancels:
    """One move's cancel run, split into input rows and the structural rows that were excluded.

    ``cancels`` are the graph edges — the rows that represent an *input* the player can make. The
    two structural kinds are counted, never silently dropped, so a run's rows are accounted for:
    ``len(cancels) + n_terminator + n_auto_transition`` should equal ``n_rows``, and any shortfall
    is a row this reader read but did not classify.

    ``reason`` is ``None`` on a clean read (including a legitimately empty run) and otherwise names
    why no rows were produced — an unreadable pointer, a falsified span, a descending span. A
    non-result is data, so it is carried here rather than raised.
    """

    move_id: int
    cancels: tuple[Cancel, ...]
    n_rows: int
    n_terminator: int
    n_auto_transition: int
    reason: str | None = None

    @property
    def n_unclassified(self) -> int:
        """Rows read that became neither an edge nor a counted structural row (expected: 0)."""
        return self.n_rows - len(self.cancels) - self.n_terminator - self.n_auto_transition


def _is_terminator(row: RawCancel) -> bool:
    """Whether ``row`` closes the run rather than naming an input.

    Keyed on the **destination**, not the command: a destination at or above
    :data:`~tekken_coach.reader.discovery.moveset_anchor.REAL_MOVE_ID_MAX` is the neutral alias, so
    it can never be a moves-array index and therefore can never be a graph node — whatever command
    word carries it.
    """
    return row.dest_move_id >= REAL_MOVE_ID_MAX


def _is_auto_transition(row: RawCancel, move_id: int) -> bool:
    """Whether ``row`` is the no-input link into the next move (the jab's ``-> 1696`` recovery)."""
    return row.command == 0 and row.dest_move_id == move_id + 1


def read_move_cancels(
    source: MemorySource,
    moves: MovesArray,
    move_id: int,
    *,
    ptr_offset: int = DEFAULT_CANCEL_PTR_OFFSET,
    max_rows: int = MAX_PROBE_ROWS,
) -> MoveCancels:
    """Read ``move_id``'s contiguous cancel run as owner-attributed input rows (Part A, read-only).

    Bounds the run with #25's :func:`~tekken_coach.reader.discovery.moveset_anchor.cancel_range` —
    which is also what falsifies it, since a span that is not a whole multiple of
    :data:`~tekken_coach.reader.moveset.CANCEL_SIZE` cannot be a run of ``tk_cancel``\\ s — then
    reads each ``0x28`` row and keeps the ones that name an input.

    The structural rows are dropped **explicitly and counted**. ``join_moves`` would ignore them
    anyway (both decode to no notation, so they contribute no candidate), so this is belt-and-braces
    — but a row that vanishes silently is a row we cannot account for, and the terminator's
    ``32769`` destination would otherwise enter the BFS frontier as a phantom move.
    """
    rng = cancel_range(source, moves, move_id, ptr_offset=ptr_offset, max_rows=0)

    def nothing(reason: str) -> MoveCancels:
        return MoveCancels(move_id, (), 0, 0, 0, reason)

    if rng.start is None:
        return nothing(f"move {move_id}'s own cancel pointer at +0x{ptr_offset:x} is unreadable")
    if rng.next_start is None or rng.span is None:
        return nothing(
            f"move {move_id + 1}'s cancel pointer is unreadable, so this run's end is unknown"
        )
    if rng.span < 0:
        return nothing(
            f"move {move_id + 1}'s run starts before move {move_id}'s ({rng.span} bytes), so the "
            "runs are not in ascending move_id order — this run's extent is unknown"
        )
    if rng.falsified:
        return nothing(
            f"span {rng.span} is not a whole multiple of 0x{CANCEL_SIZE:x} — not a tk_cancel run"
        )

    count = min(rng.count or 0, max_rows)
    cancels: list[Cancel] = []
    n_rows = n_terminator = n_auto = 0
    for i in range(count):
        row = read_cancel_row(source, rng.start + i * CANCEL_SIZE)
        if row is None:
            break  # the run ran past its mapping — report what was readable, never guess the rest
        n_rows += 1
        if _is_terminator(row):
            n_terminator += 1
        elif _is_auto_transition(row, move_id):
            n_auto += 1
        else:
            cancels.append(
                Cancel(source_move_id=move_id, dest_move_id=row.dest_move_id, command=row.command)
            )
    return MoveCancels(move_id, tuple(cancels), n_rows, n_terminator, n_auto)


# ---------------------------------------------------------------------------
# Part B — find the neutral root by gating its run against the from-neutral anchors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeutralCandidate:
    """One candidate move id and how much of the from-neutral anchor set its run reproduced."""

    move_id: int
    gate: tuple[GateRow, ...]

    @property
    def n_matched(self) -> int:
        """How many anchors this candidate's run reproduced."""
        return sum(1 for row in self.gate if row.found)

    @property
    def full(self) -> bool:
        """Whether **every** anchor was reproduced — the only state that names a root."""
        return bool(self.gate) and self.n_matched == len(self.gate)


@dataclass(frozen=True)
class NeutralSearch:
    """The result of sweeping candidate move ids for the neutral root (Part B).

    ``full_matches`` holds **every** candidate that reproduced the whole anchor set, not just the
    first. Two roots means the sweep is wrong or the anchors are ambiguous, and that is surfaced
    rather than resolved by picking one — :attr:`root` is ``None`` unless exactly one candidate
    gated.
    """

    search_start: int
    search_end: int
    full_matches: tuple[int, ...]
    partials: tuple[NeutralCandidate, ...]
    n_scanned: int
    n_unreadable: int

    @property
    def root(self) -> int | None:
        """The neutral move id, or ``None`` when zero or more than one candidate gated fully."""
        return self.full_matches[0] if len(self.full_matches) == 1 else None

    @property
    def ambiguous(self) -> bool:
        """Whether more than one candidate reproduced the full anchor set."""
        return len(self.full_matches) > 1

    def report(self) -> str:
        """A multi-line summary naming the root, the ambiguity, or the best near-misses."""
        lines = [
            f"neutral-move search over ids [{self.search_start}, {self.search_end}): "
            f"{self.n_scanned} scanned, {self.n_unreadable} unreadable"
        ]
        if self.ambiguous:
            ids = ", ".join(str(m) for m in self.full_matches)
            lines.append(
                f"  AMBIGUOUS: {len(self.full_matches)} candidates reproduced every anchor "
                f"({ids}). "
                "Either the sweep is wrong or the anchors do not distinguish the root — reporting "
                "both rather than picking one."
            )
        elif self.root is not None:
            lines.append(f"  ROOT: move {self.root} reproduced every from-neutral anchor.")
        else:
            lines.append("  NO ROOT: no candidate reproduced the full anchor set.")
        if self.partials:
            lines.append("  best partial matches (a near-miss may mean a stale anchor id):")
            for cand in self.partials:
                missed = ", ".join(str(row.move_id) for row in cand.gate if not row.found)
                lines.append(
                    f"    move {cand.move_id}: {cand.n_matched}/{len(cand.gate)} "
                    f"(missed {missed or 'nothing'})"
                )
        return "\n".join(lines)


def find_neutral_move(
    source: MemorySource,
    moves: MovesArray,
    known_pairs: Sequence[KnownPair],
    *,
    search_start: int = DEFAULT_SEARCH_START,
    search_end: int = DEFAULT_SEARCH_END,
    ptr_offset: int = DEFAULT_CANCEL_PTR_OFFSET,
    n_partials: int = DEFAULT_PARTIALS_REPORTED,
) -> NeutralSearch:
    """Find the move whose run reproduces the from-neutral anchors — the graph's root (Part B).

    Each candidate costs one bounded pair of pointer reads plus its own run's rows, so a sweep of a
    few thousand ids is a few tens of thousands of small reads. Read-only and total: an unreadable
    candidate is counted, never raised.

    Reports every full match and the best near-misses. A candidate matching 4 of 5 anchors is how we
    would learn an anchor id has gone stale, so it is surfaced rather than collapsed into "no root".
    """
    pairs = tuple(known_pairs)
    full: list[int] = []
    candidates: list[NeutralCandidate] = []
    n_scanned = n_unreadable = 0

    for move_id in range(search_start, search_end):
        run = read_move_cancels(source, moves, move_id, ptr_offset=ptr_offset)
        if run.reason is not None:
            n_unreadable += 1
            continue
        n_scanned += 1
        if not run.cancels:
            continue
        rows = [RawCancel(command=c.command, dest_move_id=c.dest_move_id) for c in run.cancels]
        cand = NeutralCandidate(move_id, tuple(gate_pairs(rows, pairs)))
        if cand.full:
            full.append(move_id)
        elif cand.n_matched:
            candidates.append(cand)

    partials = sorted(candidates, key=lambda c: (-c.n_matched, c.move_id))[:n_partials]
    return NeutralSearch(
        search_start=search_start,
        search_end=search_end,
        full_matches=tuple(full),
        partials=tuple(partials),
        n_scanned=n_scanned,
        n_unreadable=n_unreadable,
    )


# ---------------------------------------------------------------------------
# Part C — walk the graph from the root
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CancelGraph:
    """Every cancel row reachable from the root, plus what the walk could not read (Part C)."""

    root: int
    cancels: tuple[Cancel, ...]
    reached: tuple[int, ...]  # every move id visited, sorted
    unreadable: tuple[
        tuple[int, str], ...
    ]  # (move_id, why) for each move whose run yielded nothing
    n_terminator: int
    n_auto_transition: int
    truncated: bool  # True when ``max_moves`` stopped the walk with frontier left

    def report(self) -> str:
        """A one-block summary of the walk's extent and everything it declined to read."""
        lines = [
            f"cancel graph from root {self.root}: {len(self.reached)} moves reached, "
            f"{len(self.cancels)} input cancels",
            f"  excluded {self.n_terminator} terminator + {self.n_auto_transition} "
            "auto-transition row(s) (structural, not inputs)",
        ]
        if self.truncated:
            lines.append(
                f"  TRUNCATED at max_moves={len(self.reached)} — the reachable set is larger, so "
                "this map is incomplete by construction."
            )
        if self.unreadable:
            lines.append(f"  {len(self.unreadable)} move(s) yielded no run:")
            for move_id, why in self.unreadable[:10]:
                lines.append(f"    move {move_id}: {why}")
            if len(self.unreadable) > 10:
                lines.append(f"    ... {len(self.unreadable) - 10} more")
        return "\n".join(lines)


def walk_cancel_graph(
    source: MemorySource,
    moves: MovesArray,
    root: int,
    *,
    ptr_offset: int = DEFAULT_CANCEL_PTR_OFFSET,
    max_moves: int = DEFAULT_MAX_MOVES,
) -> CancelGraph:
    """BFS the per-move cancel graph from ``root``, collecting every reachable row (read-only).

    The graph has cycles — a string can cancel back into neutral — so the visited set is what
    terminates the walk, and ``max_moves`` bounds it even if a wrong ``ptr_offset`` feeds it junk
    destinations. Truncation is reported, because a map built from a truncated walk is incomplete by
    construction and must not be presented as complete.

    Returns the rows for the existing
    :func:`~tekken_coach.framedata.moveset_decode.join_moves`; this function deliberately does not
    resolve notation itself.
    """
    cancels: list[Cancel] = []
    unreadable: list[tuple[int, str]] = []
    visited = {root}
    frontier: deque[int] = deque([root])
    n_terminator = n_auto = 0
    truncated = False

    while frontier:
        move_id = frontier.popleft()
        run = read_move_cancels(source, moves, move_id, ptr_offset=ptr_offset)
        n_terminator += run.n_terminator
        n_auto += run.n_auto_transition
        if run.reason is not None:
            unreadable.append((move_id, run.reason))
            continue
        cancels.extend(run.cancels)
        for cancel in run.cancels:
            dest = cancel.dest_move_id
            if dest in visited:
                continue
            if len(visited) >= max_moves:
                truncated = True
                break
            visited.add(dest)
            frontier.append(dest)
        if truncated:
            break

    return CancelGraph(
        root=root,
        cancels=tuple(cancels),
        reached=tuple(sorted(visited)),
        unreadable=tuple(unreadable),
        n_terminator=n_terminator,
        n_auto_transition=n_auto,
        truncated=truncated,
    )
