"""The machine-layer rubric — knowledge checks as detectable patterns (docs/06 §4.1).

Each knowledge check is a **rule spec**: an ``id``, a **trigger** (a pure predicate over one
interaction's fields + its frame-data labels), and a **recurrence threshold** (how many times
across a session before it is worth coaching). This module owns only the *trigger* half; the
*recurrence* half is applied session-level in :mod:`tekken_coach.framedata.tally` — a
per-interaction function cannot see session counts (docs/06 §4.1 two-phase note).

The xref (:mod:`tekken_coach.framedata.xref`) runs every trigger against a matched interaction and
records the ones that fire in ``labels.is_knowledge_check`` / ``labels.knowledge_check_ids``.

Starter set (docs/06 §4.1). The spec lists six rows; ``ate_low`` / ``ate_mid`` is one row split into
two ids here because the coaching line and the grouping differ by the move's height.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tekken_coach.schemas import (
    DefenderReaction,
    FollowUpResult,
    Interaction,
    Labels,
    MoveProperty,
    Outcome,
    StringGap,
)

# A trigger reads one interaction's structural fields and its (already-computed) frame-data
# labels. It must not read ``labels.is_knowledge_check`` / ``knowledge_check_ids`` (those are the
# output of running the rubric); the xref passes labels with only the frame-data fields filled.
TriggerFn = Callable[[Interaction, Labels], bool]

# The default recurrence threshold: a pattern must recur this many times against one move/string
# before the tally marks it a genuine knowledge check (docs/06 §4.1, summary §1).
DEFAULT_RECURRENCE_THRESHOLD = 3


@dataclass(frozen=True)
class RubricPattern:
    """One knowledge-check pattern: id + trigger predicate + recurrence rule (docs/06 §4.1)."""

    id: str
    trigger: TriggerFn
    recurrence_threshold: int
    coaching_line: str  # the human-facing line (docs/06 §4.1); final phrasing is the LLM's job


# ---------------------------------------------------------------------------
# Trigger predicates (per interaction). Each assumes the move resolved in frame data
# (the xref only runs the rubric when frame_data_matched is true, docs/05 §4.1).
# ---------------------------------------------------------------------------


def _punish_missed(itx: Interaction, labels: Labels) -> bool:
    """Punishable move, defender did nothing (docs/06 §4.1)."""
    return bool(labels.was_punishable) and itx.outcome == Outcome.no_punish


def _respected_fake_gap(itx: Interaction, labels: Labels) -> bool:
    """Blocked an interruptible mid-string gap and did nothing — could have interrupted."""
    return (
        itx.defender_reaction == DefenderReaction.blocked
        and labels.in_string
        and labels.string_gap == StringGap.interruptible
        and itx.follow_up.result == FollowUpResult.none
    )


def _challenged_true_string(itx: Interaction, labels: Labels) -> bool:
    """Pressed inside a true (uninterruptible) string and got counter-hit."""
    return (
        labels.in_string
        and labels.string_gap == StringGap.true
        and itx.follow_up.result == FollowUpResult.got_counter_hit
    )


def _standing_duckable_high(itx: Interaction, labels: Labels) -> bool:
    """Blocked a mid-string high standing that could have been ducked to punish (docs/05 §4.1)."""
    return labels.in_string and labels.duckable_high_hit is not None


def _ate_low(itx: Interaction, labels: Labels) -> bool:
    """Got hit by a low (stood on it) on a known mix."""
    got_hit = itx.defender_reaction == DefenderReaction.hit
    return got_hit and labels.move_property == MoveProperty.low


def _ate_mid(itx: Interaction, labels: Labels) -> bool:
    """Got hit by a mid (ducked it) on a known mix."""
    got_hit = itx.defender_reaction == DefenderReaction.hit
    return got_hit and labels.move_property == MoveProperty.mid


def _mashed_into_plus(itx: Interaction, labels: Labels) -> bool:
    """Pressed after a plus-on-block move and got counter-hit (docs/06 §4.1)."""
    return (
        labels.on_block is not None
        and labels.on_block > 0
        and itx.follow_up.result == FollowUpResult.got_counter_hit
    )


# ---------------------------------------------------------------------------
# The starter rubric (docs/06 §4.1). Order is the report/priority order; the tally is order-free.
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC: tuple[RubricPattern, ...] = (
    RubricPattern(
        id="punish_missed",
        trigger=_punish_missed,
        recurrence_threshold=DEFAULT_RECURRENCE_THRESHOLD,
        coaching_line="X is -N. Punish with correct_punish.",
    ),
    RubricPattern(
        id="respected_fake_gap",
        trigger=_respected_fake_gap,
        recurrence_threshold=DEFAULT_RECURRENCE_THRESHOLD,
        coaching_line="There's a gap after hit K — you can interrupt.",
    ),
    RubricPattern(
        id="challenged_true_string",
        trigger=_challenged_true_string,
        recurrence_threshold=DEFAULT_RECURRENCE_THRESHOLD,
        coaching_line="That's a true string. Stop pressing; block it.",
    ),
    RubricPattern(
        id="standing_duckable_high",
        trigger=_standing_duckable_high,
        recurrence_threshold=DEFAULT_RECURRENCE_THRESHOLD,
        coaching_line="X is mid->high->mid — duck hit K, punish before the last hit: duck_punish.",
    ),
    RubricPattern(
        id="ate_low",
        trigger=_ate_low,
        recurrence_threshold=DEFAULT_RECURRENCE_THRESHOLD,
        coaching_line="You keep standing on the low — react to X.",
    ),
    RubricPattern(
        id="ate_mid",
        trigger=_ate_mid,
        recurrence_threshold=DEFAULT_RECURRENCE_THRESHOLD,
        coaching_line="You keep ducking the mid — react to X.",
    ),
    RubricPattern(
        id="mashed_into_plus",
        trigger=_mashed_into_plus,
        recurrence_threshold=DEFAULT_RECURRENCE_THRESHOLD,
        coaching_line="X is plus. Stop mashing after it; wait your turn.",
    ),
)


def evaluate_triggers(
    itx: Interaction, labels: Labels, rubric: tuple[RubricPattern, ...] = DEFAULT_RUBRIC
) -> list[str]:
    """Return the ids of every rubric pattern whose trigger fires for this interaction.

    Pure and order-preserving (rubric order). The recurrence threshold is **not** applied here —
    that is a session-level judgment made in the tally (docs/06 §4.1 two-phase split).
    """
    return [p.id for p in rubric if p.trigger(itx, labels)]


def thresholds(rubric: tuple[RubricPattern, ...] = DEFAULT_RUBRIC) -> dict[str, int]:
    """Map each pattern id to its recurrence threshold (used by the tally)."""
    return {p.id: p.recurrence_threshold for p in rubric}
