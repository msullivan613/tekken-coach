"""Onset-reconstructability bucketing (brief #9 Stage 1): the pure notation classifier.

Exercises :func:`classify` against a representative slice of the Wavu ``Command`` vocabulary — the
Easy buckets (single presses, strings) and every Hard reason (stance, motion, hold, just-frame,
cancel, throw) — plus the onset primitive and the coverage roll-up.
"""

from __future__ import annotations

import pytest

from tekken_coach.framedata.input_recon import (
    Bucket,
    Reason,
    classify,
    coverage,
    onset_of,
)


@pytest.mark.parametrize(
    ("key", "dir_", "buttons"),
    [
        ("2", 5, ("2",)),  # bare button, neutral direction
        ("df+2", 3, ("2",)),  # simple directional
        ("1+2", 5, ("1", "2")),  # button combo
        ("b+1", 4, ("1",)),
        ("u+4", 8, ("4",)),
        ("f+1+2", 6, ("1", "2")),
        ("uf+3", 9, ("3",)),
        ("db+4", 1, ("4",)),
        ("1+3", 5, ("1", "3")),  # a generic throw written as buttons — reconstructable
    ],
)
def test_easy_single_moves_reconstruct_their_onset(
    key: str, dir_: int, buttons: tuple[str, ...]
) -> None:
    c = classify(key)
    assert c.bucket is Bucket.easy_single
    assert c.reason is Reason.single
    assert c.easy
    assert c.onset == (dir_, buttons)
    assert onset_of(key) == (dir_, buttons)


@pytest.mark.parametrize(
    ("key", "onset"),
    [
        ("1,2", (5, ("1",))),  # onset is the first hit
        ("df+1,2", (3, ("1",))),
        ("1,2,3", (5, ("1",))),
        ("df+3,4", (3, ("3",))),
        ("df+1,df+2", (3, ("1",))),  # each hit carries its own direction
    ],
)
def test_easy_strings_bucket_by_string_and_expose_first_hit_onset(
    key: str, onset: tuple[int, tuple[str, ...]]
) -> None:
    c = classify(key)
    assert c.bucket is Bucket.easy_string
    assert c.reason is Reason.string
    assert c.easy
    assert c.onset == onset


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        ("qcf+2", Reason.motion),  # quarter-circle motion
        ("qcb+3", Reason.motion),
        ("d,df,f", Reason.motion),  # written-out motion (direction-only sequence)
        ("f,n,d,df+1", Reason.motion),  # neutral-return sequence
        ("uf,n,4", Reason.motion),
        (
            "f,F+2",
            Reason.motion,
        ),  # dash: tap f (direction-only) then hold F — the tap reads as motion
        ("1+2*", Reason.hold),  # held button variant
        ("b+2,1*", Reason.hold),
        ("d+1+2,B", Reason.hold),  # trailing held direction
        ("f,F+2:1", Reason.just_frame),  # just-frame (checked before the hold/motion in it)
        ("d+4,2:1+2", Reason.just_frame),
        ("3,2~b", Reason.cancel),  # ~ cancel
        ("df+1~b", Reason.cancel),
        ("H.1+2", Reason.stance),  # Heat stance prefix
        ("FC.df+3+4", Reason.stance),  # full-crouch prefix
        ("ws1", Reason.stance),  # while-standing (no dot, but positional)
        ("DPD.df+2", Reason.stance),
        ("DVK.1,1,2,D", Reason.stance),
        ("Left Throw", Reason.throw),  # named throw, no button notation
        ("Back Throw.1+3", Reason.throw),
        ("(back to wall).b,b,ub", Reason.stance),  # parenthetical positional
    ],
)
def test_hard_moves_carry_the_right_reason(key: str, reason: Reason) -> None:
    c = classify(key)
    assert c.bucket is Bucket.hard
    assert c.reason is reason
    assert not c.easy
    assert c.onset is None
    assert onset_of(key) is None


def test_coverage_rolls_up_buckets_reasons_and_onsets() -> None:
    keys = ["2", "df+2", "1,2", "qcf+2", "H.1+2", "1"]
    cov = coverage(keys)
    assert cov.total == 6
    assert cov.easy_single == 3  # 2, df+2, 1
    assert cov.easy_string == 1  # 1,2
    assert cov.hard == 2  # qcf+2, H.1+2
    assert cov.easy == 4
    assert cov.easy_fraction == pytest.approx(4 / 6)
    assert cov.by_reason == {"single": 3, "string": 1, "motion": 1, "stance": 1}


def test_coverage_flags_shared_onsets_but_counts_uniques() -> None:
    # "1" and "1,2" share the onset (5, ("1",)) — the benign move-vs-its-own-string case; "df+2" is
    # alone. The collision map records the shared onset; unique_onsets counts the solo ones.
    cov = coverage(["1", "1,2", "df+2"])
    assert cov.onset_collisions == {"dir5+1": 2}
    assert cov.unique_onsets == 1  # only df+2's onset is unshared


def test_empty_coverage_is_zero_not_a_divide_by_zero() -> None:
    cov = coverage([])
    assert cov.total == 0
    assert cov.easy_fraction == 0.0
