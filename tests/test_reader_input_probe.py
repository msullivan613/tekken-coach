"""The offline input-offset re-derivation: protocol layout + candidate ranking (brief #10).

The point of these tests is that the analyzer is judged against a *planted* recording — a synthetic
``probe-state --record`` log where we know which offset is the real input field and which are decoys
(a frame counter, a churning neighbour, a field that moves on both players). If the ranking cannot
pick the planted field out of that lineup, it cannot be trusted against the live sweep either.
"""

from __future__ import annotations

import json
import random

import pytest

from tekken_coach.reader.input_probe import (
    BASELINE,
    MIN_PLAUSIBLE,
    PROTOCOL,
    SETTLE,
    Observation,
    Step,
    _segments,
    best_alignment,
    format_report,
    load_observation,
    log_span,
    observable_duration,
    rank_for_role,
    render_checklist,
    score_candidate,
    script_duration,
    step_windows,
)

# The encoding the planted log uses: numpad directions, and a bitmask in decode._BUTTON_BITS order.
_DIRS: dict[str, int] = {"u": 8, "d": 2, "b": 4, "f": 6, "u/f": 9, "d/f": 3, "d/b": 1, "u/b": 7}
_BUTTONS: dict[str, int] = {"1": 1, "2": 2, "3": 4, "4": 8, "1+2": 3}


def _label_base(label: str) -> str:
    return label.split(" (again)")[0]


def _planted_rows(start: float = 0.0, scale: float = 1.0) -> list[dict[str, object]]:
    """A recording where ``@0x40`` is the real input_dir and ``@0x48`` the real input_buttons.

    The decoys are the ways a bogus offset can *look* right, one per discriminator: ``@0x8`` ticks
    every sample (a frame counter), ``@0x10`` is dead constant (perfect on every criterion except
    ever reacting), ``@0x18`` reacts to *everything* (both roles — it fails the role split), and
    ``@0x20`` tracks the direction perfectly but does so on **both** players, which a real input
    field cannot: the P2 dummy is untouched all pass. ``@0x30`` is the decoy the real 5.02.01 sweep
    actually produced: an "an attack button is held" flag — flawless on every criterion except that
    it reads the same value for 1/2/3/4, so it reacts to input without encoding which.
    """
    rows: list[dict[str, object]] = []
    tick = 0

    def emit(t: float, dir_: int, btn: int) -> None:
        nonlocal tick
        tick += 1
        any_press = 1 if (dir_ != 5 or btn != 0) else 0
        p1 = {
            "@0x8": tick,
            "@0x10": 7,
            "@0x18": any_press,
            "@0x20": dir_,
            "@0x30": 1 if btn else 0,
            "@0x40": dir_,
            "@0x48": btn,
        }
        # The dummy: shared globals still tick, its @0x20 mirrors P1 (the decoy), but its own copy
        # of the real input fields sits at rest — nobody is holding its pad.
        p2 = {
            "@0x8": tick,
            "@0x10": 7,
            "@0x18": 0,
            "@0x20": dir_,
            "@0x30": 0,
            "@0x40": 5,
            "@0x48": 0,
        }
        rows.append({"t": round(t, 2), "player": 1, "fields": p1})
        rows.append({"t": round(t, 2), "player": 2, "fields": p2})

    emit(start, 5, 0)  # baseline: neutral stick, no buttons
    for window in step_windows(PROTOCOL, start, scale):
        if window.step is None:
            continue
        base = _label_base(window.step.label)
        if window.kind == "hold":
            dir_ = _DIRS.get(base, 5)
            btn = _BUTTONS.get(base, 0)
            emit(window.t0 + 0.1, dir_, btn)  # pressed slightly late, inside the SETTLE margin
        else:
            emit(window.t0 + 0.1, 5, 0)  # released back to rest
    return rows


def _planted(start: float = 0.0, scale: float = 1.0) -> Observation:
    return load_observation(json.dumps(row) for row in _planted_rows(start, scale))


def _report_for(obs: Observation) -> list[str]:
    """The report exactly as `analyze-input` produces it: fit the script, then rank."""
    start, scale = best_alignment(obs)
    return list(format_report(obs, start=start, scale=scale, top=3))


# --- protocol ------------------------------------------------------------------------------------


def test_step_windows_lays_the_script_out_with_a_baseline_then_hold_rest_pairs() -> None:
    windows = step_windows((Step("1", "button", hold=2.0, rest=3.0),), start=1.0)
    assert [(w.t0, w.t1, w.kind) for w in windows] == [
        (1.0, 1.0 + BASELINE, "rest"),  # hands-off baseline establishes the rest value
        (1.0 + BASELINE, 3.0 + BASELINE, "hold"),
        (3.0 + BASELINE, 6.0 + BASELINE, "rest"),
    ]


def test_protocol_covers_every_button_and_all_eight_directions_with_repeats() -> None:
    labels = [s.label for s in PROTOCOL]
    assert {"1", "2", "3", "4", "1+2"} <= set(labels)  # each button alone + a pair for the mask
    assert {"u", "d", "b", "f", "u/f", "d/f", "d/b", "u/b"} <= set(labels)  # all 8 holds
    # A repeat per role is what lets the ranking check the action -> value map is stable.
    assert {s.role for s in PROTOCOL if "(again)" in s.label} == {"button", "dir"}


def test_render_checklist_timestamps_every_action_for_the_user_to_follow() -> None:
    lines = list(render_checklist(PROTOCOL, start=0.0))
    assert "hands OFF" in lines[1]  # the baseline instruction comes first
    body = "\n".join(lines)
    for step in PROTOCOL:
        assert step.label in body
    assert "Ctrl-C" in lines[-1]


# --- loading -------------------------------------------------------------------------------------


def test_load_observation_keeps_only_genuine_transitions_per_field() -> None:
    rows = [
        {"t": 0.0, "player": 1, "fields": {"@0x8": 1, "@0x40": 5}},
        {"t": 0.1, "player": 1, "fields": {"@0x8": 2, "@0x40": 5}},  # @0x40 unchanged -> not kept
        {"t": 0.2, "player": 1, "fields": {"@0x8": 3, "@0x40": 6}},
    ]
    obs = load_observation(json.dumps(row) for row in rows)
    assert obs.series[1]["@0x40"] == [(0.0, 5), (0.2, 6)]
    assert obs.series[1]["@0x8"] == [(0.0, 1), (0.1, 2), (0.2, 3)]


def test_observation_fields_sort_numerically_by_offset() -> None:
    obs = load_observation(
        [json.dumps({"t": 0.0, "player": 1, "fields": {"@0x100": 0, "@0x8": 0, "@0x40": 0}})]
    )
    assert obs.fields == ["@0x8", "@0x40", "@0x100"]


def test_segments_seeking_matches_a_full_scan_of_the_series() -> None:
    # _segments bisects to the window instead of walking the whole series (it is the innermost loop:
    # per window, per candidate, per role, per alignment offset). The seek must be pure speed —
    # identical output, including windows that start before / end after every recorded point.
    def full_scan(points: list[tuple[float, int]], t0: float, t1: float) -> list[tuple[int, float]]:
        if not points or t1 <= t0:
            return []
        out = []
        for i, (at, value) in enumerate(points):
            until = points[i + 1][0] if i + 1 < len(points) else float("inf")
            lo, hi = max(at, t0), min(until, t1)
            if hi > lo:
                out.append((value, hi - lo))
        return out

    rng = random.Random(3)
    for _ in range(2000):
        times = sorted({round(rng.uniform(0, 20), 2) for _ in range(rng.randrange(0, 12))})
        points = [(t, rng.randrange(4)) for t in times]
        t0 = round(rng.uniform(-2, 22), 2)
        t1 = t0 + round(rng.uniform(0, 5), 2)
        assert _segments(points, t0, t1) == full_scan(points, t0, t1)


# --- ranking -------------------------------------------------------------------------------------


def test_ranking_picks_the_planted_input_dir_and_input_buttons_out_of_the_decoys() -> None:
    obs = _planted()
    assert rank_for_role(obs, "dir")[0].name == "@0x40"
    assert rank_for_role(obs, "button")[0].name == "@0x48"


def test_the_planted_winner_reports_the_encoding_it_observed() -> None:
    # This table *is* the deliverable: it answers "numpad or raw stick?" and "which bit is which?".
    best = rank_for_role(obs := _planted(), "dir")[0]
    assert best.rest_value == 5
    assert {_label_base(k): v for k, v in best.values_by_step.items()} == _DIRS
    buttons = rank_for_role(obs, "button")[0]
    assert {_label_base(k): v for k, v in buttons.values_by_step.items()} == _BUTTONS


def test_a_field_that_reacts_to_both_roles_loses_to_the_role_specific_one() -> None:
    # @0x18 goes 1 on every press, button or direction — the role discriminator must demote it.
    both = score_candidate(_planted(), "@0x18", "button")
    real = score_candidate(_planted(), "@0x48", "button")
    assert both.score < real.score
    assert both.parts["quiet_other"] < 1.0
    assert any("also moves on the dir steps" in n for n in both.notes)


def test_a_free_running_counter_scores_low_on_every_criterion_that_matters() -> None:
    counter = score_candidate(_planted(), "@0x8", "dir")
    assert counter.score < MIN_PLAUSIBLE
    assert counter.parts["rest_stable"] == 0.0  # never returns to a rest value
    assert counter.parts["cardinality"] == 0.0  # far more values than a direction can take


def test_a_dead_constant_field_scores_zero_despite_passing_every_other_criterion() -> None:
    # The case a weighted sum gets wrong: a constant is perfectly stable, steady, consistent,
    # acting-only and plausibly-sized — it just never reacts. `reacts` gates the score for exactly
    # this reason, so it must land at 0, not at "good enough to outrank a noisy real field".
    dead = score_candidate(_planted(), "@0x10", "dir")
    assert dead.parts["reacts"] == 0.0
    assert {dead.parts[k] for k in ("rest_stable", "steady", "consistent", "acting_only")} == {1.0}
    assert dead.score == 0.0
    assert any("never leaves rest" in n for n in dead.notes)


def test_a_field_that_moves_on_the_static_dummy_is_flagged_and_demoted() -> None:
    # The Practice discriminator: the P2 dummy is untouched, so a real input field cannot move
    # there — @0x20 tracks the stick perfectly but does it on both structs.
    obs = _planted()
    shared = score_candidate(obs, "@0x20", "dir")
    assert not shared.acting_only
    assert any("static dummy" in n for n in shared.notes)
    assert shared.score < score_candidate(obs, "@0x40", "dir").score


def test_best_alignment_recovers_a_start_far_later_than_any_fixed_window_would_guess() -> None:
    # Nothing synchronizes the probe's clock with the user reading a printed checklist: they start
    # the sweep in one terminal, alt-tab, find the script, begin. 18 s is an ordinary alt-tab, and
    # a search window guessed around t=0 would score every candidate against noise and report NO
    # CANDIDATE for fields that are really there — a confident false negative. The feasible range
    # comes from the recording instead, so the start is found wherever it actually is.
    obs = _planted(start=18.0)
    recovered, scale = best_alignment(obs)
    assert abs(recovered - 18.0) <= SETTLE
    assert scale == pytest.approx(1.0, abs=0.06)  # performed on tempo, so no stretch is inferred
    assert rank_for_role(obs, "dir", start=recovered, scale=scale)[0].name == "@0x40"
    assert rank_for_role(obs, "button", start=recovered, scale=scale)[0].name == "@0x48"


def test_analyzing_a_late_started_pass_still_names_the_planted_fields() -> None:
    # The end-to-end of the above: the report must not turn a slow alt-tab into "not in the struct".
    report = "\n".join(_report_for(_planted(start=18.0)))
    assert "input_dir: best candidate @0x40" in report
    assert "input_buttons: best candidate @0x48" in report


def test_script_duration_and_log_span_bound_the_search() -> None:
    assert script_duration((Step("1", "button", hold=2.0, rest=3.0),)) == BASELINE + 5.0
    obs = _planted(start=2.6)
    first, last = log_span(obs)
    assert first == 2.6  # the recording opens at the baseline...
    assert last == pytest.approx(2.6 + script_duration() - PROTOCOL[-1].rest + 0.1)  # ...to the end


def test_best_alignment_recovers_a_late_start_and_the_ranking_survives_it() -> None:
    # The user alt-tabs: the script starts 2.6 s after the probe's t=0. Nothing should depend on
    # the two clocks matching. The SETTLE margin means a spread of nearby offsets all fit perfectly,
    # so the recovered start is only asked to land inside that margin — and, the part that actually
    # matters, to be good enough that the ranking still finds the planted fields.
    obs = _planted(start=2.6)
    recovered, scale = best_alignment(obs)
    assert abs(recovered - 2.6) <= SETTLE
    assert rank_for_role(obs, "dir", start=recovered, scale=scale)[0].name == "@0x40"
    assert rank_for_role(obs, "button", start=recovered, scale=scale)[0].name == "@0x48"


def test_a_misaligned_score_is_worse_than_the_aligned_one() -> None:
    obs = _planted(start=2.6)
    assert (
        score_candidate(obs, "@0x40", "dir", start=0.0).score
        < score_candidate(obs, "@0x40", "dir", start=2.6).score
    )


# --- reporting -----------------------------------------------------------------------------------


def test_format_report_names_the_winners_and_shows_their_evidence() -> None:
    report = "\n".join(format_report(_planted(), top=3))
    assert "input_dir: best candidate @0x40" in report
    assert "input_buttons: best candidate @0x48" in report
    assert "observed:" in report


def test_format_report_warns_when_the_recording_is_shorter_than_the_script() -> None:
    # A cut-short pass scores against windows the log never covers and reads as NO CANDIDATE —
    # which is indistinguishable from "the fields aren't here" unless the report says so.
    rows = [
        {"t": t / 10, "player": p, "fields": {"@0x8": t, "@0x10": 7}}
        for t in range(100)  # 10s of recording for a ~65s script
        for p in (1, 2)
    ]
    report = "\n".join(format_report(load_observation(json.dumps(row) for row in rows)))
    assert "the recording ends at t=10s" in report
    assert "the script's last action lands at t=63s" in report
    assert "cut short" in report


def test_format_report_does_not_cry_short_on_a_full_pass() -> None:
    # The trailing rest records nothing (nobody moves after the last release), so a COMPLETE pass's
    # log always stops before the script's nominal end. Measuring to that end would warn every run.
    assert observable_duration() == script_duration() - PROTOCOL[-1].rest
    for start in (0.0, 18.0):
        assert not any("WARNING" in line for line in _report_for(_planted(start)))


def test_format_report_refuses_to_name_a_candidate_when_the_sweep_is_clean() -> None:
    # The evidenced-negative path (brief #10 acceptance b): a sweep of nothing but a counter and a
    # dead field must say so plainly rather than crown the least-bad offset.
    rows = [
        {"t": t / 10, "player": p, "fields": {"@0x8": t, "@0x10": 7}}
        for t in range(700)
        for p in (1, 2)
    ]
    obs = load_observation(json.dumps(row) for row in rows)
    report = "\n".join(format_report(obs))
    assert "input_dir: NO CANDIDATE" in report
    assert "input_buttons: NO CANDIDATE" in report
    assert "Do not bake a guess." in report


def test_a_field_that_reacts_to_input_but_does_not_encode_which_is_not_crowned() -> None:
    # The decoy the real 5.02.01 sweep produced: an "attack button is held" flag. It is
    # acting-exclusive, role-specific, rest-stable, steady and consistent — flawless on every
    # criterion the scorer had — and it scored 0.83, above the plausibility floor, while reading the
    # SAME value for 1/2/3/4. Reacting to input is not encoding it; without a discrimination gate
    # the analyzer crowns the thing downstream of the pad instead of the pad.
    flag = score_candidate(_planted(), "@0x30", "button")
    assert flag.parts["reacts"] == 1.0  # it really does fire on every button step...
    assert flag.parts["discriminates"] == pytest.approx(0.2)  # ...with one value for five actions
    assert flag.score < MIN_PLAUSIBLE
    assert any("does not encode which" in n for n in flag.notes)
    assert rank_for_role(_planted(), "button")[0].name == "@0x48"  # the real mask still wins


def test_best_alignment_recovers_a_pass_performed_slower_than_the_script() -> None:
    # A human reading a checklist runs slow, and it compounds: the real 5.02.01 pass came in at
    # 1.15x, only 15% off, but that is 8.6s of drift by the last of 15 steps — enough to slide the
    # script's tail into the gaps between presses, so every field there reads as "never reacts".
    # The error is a rate, not a delay; a start-only fit cannot express it.
    obs = _planted(start=4.0, scale=1.15)
    start, scale = best_alignment(obs)
    assert scale == pytest.approx(1.15, abs=0.06)
    assert rank_for_role(obs, "dir", start=start, scale=scale)[0].name == "@0x40"
    assert rank_for_role(obs, "button", start=start, scale=scale)[0].name == "@0x48"


def test_a_drifting_pass_is_lost_when_the_tempo_is_pinned_wrong() -> None:
    # The failure the scale fit exists to prevent, pinned: force tempo 1.0 on a 1.15x pass and the
    # real field's reactions fall outside their windows.
    obs = _planted(start=4.0, scale=1.15)
    on_tempo = score_candidate(obs, "@0x40", "dir", start=4.0, scale=1.0)
    fitted = score_candidate(obs, "@0x40", "dir", start=4.0, scale=1.15)
    assert on_tempo.score < MIN_PLAUSIBLE < fitted.score
