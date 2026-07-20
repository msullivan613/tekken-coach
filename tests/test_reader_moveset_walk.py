"""Brief #26 — build the movemap by walking the per-move cancel graph.

The confirmed 5.02.01 shape (``tk_move + 0x098`` -> a contiguous run, ``tk_cancel`` 0x28 rows, the
T8 command encoding) is planted in ``planted_cancel_graph_source``, so every step below runs offline
against data laid out exactly as the live game's: the structural-row exclusion, the self-validating
root search, the bounded BFS, and the end-to-end join through the existing (unchanged)
``join_moves``.
"""

from __future__ import annotations

from tekken_coach.framedata.moveset_decode import join_moves
from tekken_coach.reader.discovery.moveset_anchor import MovesArray
from tekken_coach.reader.moveset import BRYAN_GATE_PAIRS, KnownPair, self_check
from tekken_coach.reader.moveset_walk import (
    find_neutral_move,
    read_move_cancels,
    walk_cancel_graph,
)
from tests.fixtures.reader.planted_moveset import (
    TKM_CANCEL_PTR_OFFSET,
    TKM_LAST_MOVE,
    TKM_MISALIGNED_MOVE,
    TKM_MOVE_SIZE,
    TKM_MOVES_BASE,
    TKW_AUTO_TRANSITION_DEST,
    TKW_CANCEL_PTR_OFFSET,
    TKW_COLLISION_DEST,
    TKW_DECOY_ROOT,
    TKW_DEEP,
    TKW_EXPECTED_COLLISIONS,
    TKW_EXPECTED_NOTATION,
    TKW_EXPECTED_REACHED,
    TKW_JAB,
    TKW_MANY_TO_ONE,
    TKW_ROOT,
    TKW_RUNS,
    planted_cancel_graph_source,
    planted_tk_move_source,
)

PTR = TKW_CANCEL_PTR_OFFSET


# ---------------------------------------------------------------------------
# Part A — read_move_cancels
# ---------------------------------------------------------------------------


def test_read_move_cancels_returns_only_input_rows() -> None:
    """The root's run yields exactly its planted input cancels, all attributed to the root."""
    source, moves = planted_cancel_graph_source()
    run = read_move_cancels(source, moves, TKW_ROOT, ptr_offset=PTR)

    assert run.reason is None
    assert len(run.cancels) == len(TKW_RUNS[TKW_ROOT])
    assert {c.source_move_id for c in run.cancels} == {TKW_ROOT}
    assert {c.dest_move_id for c in run.cancels} == {d for _, _, d in TKW_RUNS[TKW_ROOT]}


def test_structural_rows_are_excluded_and_counted() -> None:
    """The terminator and the command-0 auto-transition are dropped, counted, and never edges.

    Both would decode to no notation anyway, so this is belt-and-braces — but an uncounted row is a
    row we cannot account for, and the terminator's ``32769`` destination would otherwise enter the
    BFS frontier as a phantom move.
    """
    source, moves = planted_cancel_graph_source()
    run = read_move_cancels(source, moves, TKW_JAB, ptr_offset=PTR)

    assert run.n_terminator == 1
    assert run.n_auto_transition == 1
    assert TKW_AUTO_TRANSITION_DEST not in {c.dest_move_id for c in run.cancels}
    assert all(c.dest_move_id < 0x8000 for c in run.cancels)
    # Every row read is accounted for: edges + terminator + auto-transition == rows read.
    assert run.n_rows == len(TKW_RUNS[TKW_JAB]) + 2
    assert run.n_unclassified == 0


def test_empty_run_is_clean_not_an_error() -> None:
    """A move sharing its neighbour's pointer owns no cancels — no rows, and no reason recorded."""
    source, moves = planted_cancel_graph_source()
    run = read_move_cancels(source, moves, TKW_ROOT + 1, ptr_offset=PTR)

    assert run.cancels == ()
    assert run.reason is None


def test_falsified_span_yields_no_rows_with_a_reason() -> None:
    """A span that is not a whole multiple of 0x28 cannot be a tk_cancel run — report, not read."""
    source = planted_tk_move_source()
    moves = MovesArray(slot_key=(), moves_base=TKM_MOVES_BASE, move_stride=TKM_MOVE_SIZE)
    run = read_move_cancels(source, moves, TKM_MISALIGNED_MOVE, ptr_offset=TKM_CANCEL_PTR_OFFSET)

    assert run.cancels == ()
    assert run.reason is not None
    assert "whole multiple" in run.reason


def test_unreadable_neighbour_yields_no_rows_with_a_reason() -> None:
    """The last move's run has no readable end, so its extent is unknown — data, not a crash."""
    source = planted_tk_move_source()
    moves = MovesArray(slot_key=(), moves_base=TKM_MOVES_BASE, move_stride=TKM_MOVE_SIZE)
    run = read_move_cancels(source, moves, TKM_LAST_MOVE, ptr_offset=TKM_CANCEL_PTR_OFFSET)

    assert run.cancels == ()
    assert run.reason is not None
    assert "unreadable" in run.reason


# ---------------------------------------------------------------------------
# Part B — find_neutral_move
# ---------------------------------------------------------------------------


def test_find_neutral_move_finds_the_planted_root() -> None:
    """The move whose run reproduces every from-neutral anchor is the root — self-validating."""
    source, moves = planted_cancel_graph_source()
    search = find_neutral_move(
        source, moves, BRYAN_GATE_PAIRS, search_start=0, search_end=150, ptr_offset=PTR
    )

    assert search.full_matches == (TKW_ROOT,)
    assert search.root == TKW_ROOT
    assert not search.ambiguous


def test_find_neutral_move_reports_a_second_full_match_as_ambiguity() -> None:
    """Two roots means the search is wrong or the anchors do not distinguish — surface, never
    pick."""
    source, moves = planted_cancel_graph_source()
    search = find_neutral_move(
        source, moves, BRYAN_GATE_PAIRS, search_start=0, search_end=250, ptr_offset=PTR
    )

    assert search.full_matches == (TKW_ROOT, TKW_DECOY_ROOT)
    assert search.ambiguous
    assert search.root is None  # refuses to choose
    assert "AMBIGUOUS" in search.report()


def test_find_neutral_move_reports_partials_when_nothing_gates() -> None:
    """A near-miss is diagnosable: the jab's run reproduces one anchor, so it surfaces as a partial.

    This is how a stale anchor id would announce itself — as a 4-of-5 candidate rather than a wall
    of zeroes — so the partials are reported instead of being collapsed into a bare "no root".
    """
    source, moves = planted_cancel_graph_source()
    search = find_neutral_move(
        source, moves, BRYAN_GATE_PAIRS, search_start=1690, search_end=1700, ptr_offset=PTR
    )

    assert search.full_matches == ()
    assert search.root is None
    assert [c.move_id for c in search.partials] == [TKW_JAB]
    assert search.partials[0].n_matched == 1
    assert "NO ROOT" in search.report()


def test_find_neutral_move_gates_on_the_anchors_it_is_given() -> None:
    """A wrong anchor set finds nothing — the gate is the anchors, not the fixture's shape."""
    source, moves = planted_cancel_graph_source()
    search = find_neutral_move(
        source,
        moves,
        (KnownPair(1695, "4"), KnownPair(1566, "b+3")),
        search_start=0,
        search_end=250,
        ptr_offset=PTR,
    )

    assert search.full_matches == ()


# ---------------------------------------------------------------------------
# Part C — walk_cancel_graph
# ---------------------------------------------------------------------------


def test_walk_reaches_exactly_the_planted_reachable_set() -> None:
    """BFS from the root visits every move named by an input row, and nothing named structurally."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)

    assert set(graph.reached) == TKW_EXPECTED_REACHED
    assert TKW_AUTO_TRANSITION_DEST not in graph.reached
    assert not graph.truncated


def test_walk_terminates_on_the_planted_cycle() -> None:
    """The deepest move cancels back into the root; the visited set is what stops the walk."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)

    assert TKW_ROOT in {c.dest_move_id for c in graph.cancels if c.source_move_id == TKW_DEEP}
    assert len(graph.reached) == len(set(graph.reached))


def test_walk_respects_max_moves_and_reports_truncation() -> None:
    """A bounded walk is honest about being bounded: a truncated map is incomplete by design."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR, max_moves=3)

    assert graph.truncated
    assert len(graph.reached) <= 3
    assert "TRUNCATED" in graph.report()


# ---------------------------------------------------------------------------
# End to end — walk -> the existing join -> the build gate
# ---------------------------------------------------------------------------


def test_end_to_end_rebuilds_the_expected_notation_map() -> None:
    """The planted world rebuilds move_id -> notation through the unchanged ``join_moves``."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)
    result = join_moves(list(graph.cancels), neutral_move_id=TKW_ROOT)

    assert result.notation == TKW_EXPECTED_NOTATION


def test_many_to_one_ids_all_share_one_notation() -> None:
    """One command, three destination ids: variants of one logical move, all legitimately "1,2".

    Not a collision, and not deduped away — the live reader sees whichever variant fires, so every
    id must map.
    """
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)
    result = join_moves(list(graph.cancels), neutral_move_id=TKW_ROOT)

    assert {result.notation[m] for m in TKW_MANY_TO_ONE} == {"1,2"}
    assert all(m not in result.collisions for m in TKW_MANY_TO_ONE)


def test_from_neutral_wins_over_a_string_path_to_the_same_move() -> None:
    """1573 is reachable from neutral AND off the jab; the from-neutral notation is canonical."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)
    result = join_moves(list(graph.cancels), neutral_move_id=TKW_ROOT)

    assert result.notation[1573] == "3"


def test_conflicting_from_neutral_notations_are_reported_not_guessed() -> None:
    """Two from-neutral commands to one dest is a collision — reported, and never written."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)
    result = join_moves(list(graph.cancels), neutral_move_id=TKW_ROOT)

    assert result.collisions == TKW_EXPECTED_COLLISIONS
    assert TKW_COLLISION_DEST not in result.notation


def test_self_check_hits_the_planted_ground_truth() -> None:
    """The build gate: rebuilding the planted world reproduces its committed ids with no MISS."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)
    result = join_moves(list(graph.cancels), neutral_move_id=TKW_ROOT)

    rows = self_check(result.notation, TKW_EXPECTED_NOTATION)
    assert all(row.status == "HIT" for row in rows)


def test_self_check_flags_a_wrong_mapping_as_miss() -> None:
    """A MISS is the failure that matters — rebuilding a *different* notation, not an absent one."""
    source, moves = planted_cancel_graph_source()
    graph = walk_cancel_graph(source, moves, TKW_ROOT, ptr_offset=PTR)
    result = join_moves(list(graph.cancels), neutral_move_id=TKW_ROOT)

    wrong = dict(TKW_EXPECTED_NOTATION)
    wrong[TKW_JAB] = "d+4"  # deliberately wrong ground truth for the jab
    rows = self_check(result.notation, wrong)

    misses = [r for r in rows if r.status == "MISS"]
    assert [r.move_id for r in misses] == [TKW_JAB]
    assert misses[0].got == "1"
