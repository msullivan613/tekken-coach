"""The frame-data cross-reference — a pure ``Interaction -> LabeledInteraction`` (docs/05 §4).

Consumes a segmented :class:`~tekken_coach.schemas.Interaction` plus the already-loaded C1 assets
(move maps, the current frame-data snapshot, the curated punisher profiles) and the rubric, and
produces a fully-annotated :class:`~tekken_coach.schemas.LabeledInteraction`. It is the
deterministic heart of the pipeline: **no I/O beyond the loaded assets, no LLM, no memory, no
network** — the same inputs always yield the same output (docs/03 §3, docs/00 §3).

What it computes (docs/05 §4.1):

* **Name resolution** — ``char_id`` -> ``char_name`` (via the move maps), ``move_id`` ->
  ``framedata_key`` -> move record (via :func:`~tekken_coach.framedata.loader.resolve_move`). Any
  miss degrades to ``frame_data_matched:false``: null ground truth, no knowledge check.
* **Heat selection** — if the attacker was in Heat, the move's ``heat`` overrides win.
* **Punishability** — ``was_punishable`` / ``correct_punish`` / ``punish_window`` /
  ``user_punished_correctly`` vs the *defender's* fastest relevant punisher (docs/05 §4.1, gap #2).
* **String gap (timing)** and **duckable high (height)** — kept distinct (docs/05 §4.1, gap #3).
* **Observed-vs-canonical reconciliation** — docs/05 §4.2, all three branches.
* **Knowledge-check tagging** — runs the rubric triggers (docs/06 §4.1); recurrence is deferred to
  the session-level tally (:mod:`tekken_coach.framedata.tally`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from tekken_coach.framedata.loader import resolve_move
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    FrameDataSnapshot,
)
from tekken_coach.framedata.punishers import (
    FALLBACK_STANDING_STARTUP,
    PunisherProfile,
    PunisherProfiles,
    PunisherStance,
)
from tekken_coach.framedata.rubric import DEFAULT_RUBRIC, RubricPattern, evaluate_triggers
from tekken_coach.schemas import (
    DefenderReaction,
    Interaction,
    LabeledInteraction,
    Labels,
    MoveProperty,
    Outcome,
)

# Frames of slack allowed before the segmenter's observed advantage is treated as *disagreeing*
# with the canonical on-block (docs/05 §4.2). Small: the two should match within measurement noise.
RECONCILE_TOLERANCE = 1


def label_interaction(
    interaction: Interaction,
    move_maps: Mapping[str, CharMoveMap],
    framedata: FrameDataSnapshot,
    punishers: PunisherProfiles,
    rubric: tuple[RubricPattern, ...] = DEFAULT_RUBRIC,
) -> LabeledInteraction:
    """Cross-reference one interaction into a :class:`LabeledInteraction` (docs/05 §4.1). Pure."""
    notes = list(interaction.notes)

    # --- name resolution (docs/05 §4.1) ------------------------------------
    by_id = {m.char_id: m for m in move_maps.values() if m.char_id is not None}
    attacker_map = _map_for(by_id, interaction.attacker_char_id)
    defender_map = _map_for(by_id, interaction.defender_char_id)
    attacker_char_name = _char_name(attacker_map, interaction.attacker_char_id)
    defender_char_name = _char_name(defender_map, interaction.defender_char_id)

    fd_by_name = {cfd.char_name: cfd for cfd in framedata.characters.values()}
    attacker_fd: CharFrameData | None = fd_by_name.get(attacker_char_name)

    lookup = resolve_move(interaction.attacker_move_id, attacker_map, attacker_fd)

    if not lookup.matched or lookup.move is None:
        # Miss ⇒ unlabeled interaction: null ground truth, no knowledge check (docs/05 §4.1, §6).
        labels = Labels(
            frame_data_matched=False,
            in_string=False,
            is_knowledge_check=False,
            knowledge_check_ids=[],
        )
        return _build(
            interaction, notes, lookup.notation, attacker_char_name, defender_char_name, labels
        )

    move = lookup.move
    on_block = _heat_on_block(move, interaction)
    move_property = move.hit_level

    was_punishable, punish_window, correct_punish, user_punished_correctly = _punishability(
        interaction,
        on_block,
        move_property,
        punishers.get(defender_char_name),
        notes,
        defender_char_name,
    )

    has_gap = move.is_string and move.string_gap is not None
    string_gap = move.string_gap.gap if (has_gap and move.string_gap is not None) else None
    gap_size = move.string_gap.gap_size if (has_gap and move.string_gap is not None) else None

    duckable_high_hit, duck_punish = _duckable_high(interaction, move)

    _reconcile(interaction.observed_advantage, on_block, notes)

    labels = Labels(
        frame_data_matched=True,
        on_block=on_block,
        was_punishable=was_punishable,
        punish_window=punish_window,
        correct_punish=correct_punish,
        user_punished_correctly=user_punished_correctly,
        in_string=move.is_string,
        string_gap=string_gap,
        gap_size=gap_size,
        duckable_high_hit=duckable_high_hit,
        duck_punish=duck_punish,
        move_property=move_property,
        is_knowledge_check=False,
        knowledge_check_ids=[],
    )
    ids = evaluate_triggers(interaction, labels, rubric)
    labels = labels.model_copy(
        update={"is_knowledge_check": bool(ids), "knowledge_check_ids": ids}
    )

    return _build(
        interaction, notes, lookup.notation, attacker_char_name, defender_char_name, labels
    )


def label_interactions(
    interactions: Iterable[Interaction],
    move_maps: Mapping[str, CharMoveMap],
    framedata: FrameDataSnapshot,
    punishers: PunisherProfiles,
    rubric: tuple[RubricPattern, ...] = DEFAULT_RUBRIC,
) -> list[LabeledInteraction]:
    """Label a stream of interactions (a thin, pure map over :func:`label_interaction`)."""
    return [label_interaction(i, move_maps, framedata, punishers, rubric) for i in interactions]


# ---------------------------------------------------------------------------
# Helpers (all pure)
# ---------------------------------------------------------------------------


def _map_for(by_id: dict[int, CharMoveMap], char_id: int | None) -> CharMoveMap | None:
    """Look up a move map by (possibly-None) char id; None id -> no map (docs/05 §4.1 gap #1)."""
    return by_id.get(char_id) if char_id is not None else None


def _char_name(char_map: CharMoveMap | None, char_id: int | None) -> str:
    """Resolve a display name for a character, degrading to a stable fallback (docs/05 §2.3)."""
    if char_map is not None:
        return char_map.char_name
    if char_id is not None:
        return f"char_id:{char_id}"
    return "unknown"


def _heat_on_block(move: FrameDataMove, interaction: Interaction) -> int | None:
    """Canonical on-block, with the move's Heat override applied when the attacker is in Heat."""
    if (
        interaction.context.attacker_heat
        and move.heat is not None
        and move.heat.on_block is not None
    ):
        return move.heat.on_block
    return move.on_block


def _punishability(
    interaction: Interaction,
    on_block: int | None,
    move_property: MoveProperty | None,
    profile: PunisherProfile | None,
    notes: list[str],
    defender_char_name: str,
) -> tuple[bool | None, int | None, str | None, bool | None]:
    """Compute (was_punishable, punish_window, correct_punish, user_punished_correctly).

    Judged against the *defender's* fastest relevant punisher (docs/05 §4.1). A blocked low is
    punished from crouch (``while_standing``); a blocked high/mid standing. With no curated profile
    the fallback is a coarse ``on_block <= -10`` standing default with a null ``correct_punish`` and
    a note (docs/05 §4.1, gap #2).
    """
    if on_block is None:
        return None, None, None, None

    window = -on_block  # |on_block| when punishable; the frames available to punish

    if profile is None:
        was_punishable = on_block <= -FALLBACK_STANDING_STARTUP
        notes.append(
            f"no punisher profile for {defender_char_name!r}; used coarse on_block<=-"
            f"{FALLBACK_STANDING_STARTUP} standing default, correct_punish unknown (05 §4.1)"
        )
        return was_punishable, None, None, _punished_correctly(interaction, was_punishable)

    stance = (
        PunisherStance.while_standing
        if move_property == MoveProperty.low
        else PunisherStance.standing
    )
    # A blocked low with no while-standing option falls back to standing punishers.
    if profile.fastest(stance) is None and stance == PunisherStance.while_standing:
        notes.append(
            f"{defender_char_name} has no while-standing punisher; "
            "used standing punishers for a blocked low (05 §4.1)"
        )
        stance = PunisherStance.standing

    fastest = profile.fastest(stance)
    if fastest is None:
        # Empty profile for this stance: degrade to the coarse default rather than crash.
        was_punishable = on_block <= -FALLBACK_STANDING_STARTUP
        return was_punishable, None, None, _punished_correctly(interaction, was_punishable)

    was_punishable = on_block <= -fastest.startup
    if not was_punishable:
        return False, None, None, None

    punish_window = window - fastest.startup
    correct_punish = _select_correct_punish(profile, stance, window)
    return was_punishable, punish_window, correct_punish, _punished_correctly(interaction, True)


def _punished_correctly(interaction: Interaction, was_punishable: bool | None) -> bool | None:
    """Whether the user actually took the punish. Only meaningful when the move was punishable."""
    if not was_punishable:
        return None
    return interaction.outcome == Outcome.punished


def _select_correct_punish(
    profile: PunisherProfile, stance: PunisherStance, window: int
) -> str | None:
    """Pick the strongest punisher fitting the window: prefer a launcher, then damage (05 §4.1)."""
    candidates = [p for p in profile.by_stance(stance) if p.startup <= window]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: (p.launcher, p.damage or 0, p.startup))
    return best.notation


def _duckable_high(
    interaction: Interaction, move: FrameDataMove
) -> tuple[int | None, str | None]:
    """The duckable-high (height) check (docs/05 §4.1, gap #3).

    If the string carries a curated ``duck_punish`` and the user **blocked** it (stood and ate the
    high rather than ducking), flag the missed duck-punish. If they ducked it (evaded), no flag —
    that is the correct play. The merged Interaction (03 §2) has no per-hit block/duck record yet
    (that is a C3/§04 §4.2 field), so we approximate "blocked the high standing" as
    ``defender_reaction == blocked``; ducking surfaces as ``evaded``. See the C2 report gap note.
    """
    if (
        move.is_string
        and move.duck_punish is not None
        and interaction.defender_reaction == DefenderReaction.blocked
    ):
        return move.duck_punish.after_hit, move.duck_punish.answer
    return None, None


def _reconcile(observed: int | None, canonical: int | None, notes: list[str]) -> None:
    """Observed-vs-canonical reconciliation (docs/05 §4.2), all three branches.

    * observed is null (dropped frames) -> rely on canonical only, no note.
    * observed agrees within tolerance -> use canonical, no note.
    * observed disagrees -> keep observed in the record (it stays on the Interaction), prefer
      canonical for the answer, and add a note (a persistent pattern is a stale-snapshot alarm).
    """
    if observed is None or canonical is None:
        return
    if abs(observed - canonical) > RECONCILE_TOLERANCE:
        notes.append(
            f"observed_advantage {observed} disagrees with canonical on_block {canonical} "
            f"(>{RECONCILE_TOLERANCE}f tol); using canonical, possible stale snapshot (05 §4.2)"
        )


def _build(
    interaction: Interaction,
    notes: list[str],
    attacker_move_name: str,
    attacker_char_name: str,
    defender_char_name: str,
    labels: Labels,
) -> LabeledInteraction:
    """Assemble the LabeledInteraction, carrying xref notes appended to the segmenter's notes."""
    data = interaction.model_dump()
    data["notes"] = notes
    return LabeledInteraction(
        **data,
        attacker_move_name=attacker_move_name,
        attacker_char_name=attacker_char_name,
        defender_char_name=defender_char_name,
        labels=labels,
    )
