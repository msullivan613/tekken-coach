"""Knowledge-check recurrence aggregation — ``KnowledgeCheckTally`` (docs/03 §4, docs/06 §4.1).

This is the **second phase** of the two-phase rubric split (docs/06 §4.1). The xref sets a
per-interaction ``is_knowledge_check`` from the trigger predicate alone; a pure per-interaction
function cannot know how often a habit recurs. This module does the session-level counting: it
groups labeled interactions by ``(knowledge_check_id, attacker_char, attacker_move_id, matchup)``,
counts occurrences, keeps example interaction ids, and applies each pattern's recurrence threshold
(the ≥3× rule) — the recurrence that turns "you missed this once" into a genuine knowledge check
(summary §1).

Not persisted by the pipeline; computed by the coaching layer or a pre-pass (docs/03 §4).
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from tekken_coach.framedata.rubric import (
    DEFAULT_RECURRENCE_THRESHOLD,
    DEFAULT_RUBRIC,
    RubricPattern,
    thresholds,
)
from tekken_coach.schemas import LabeledInteraction


def matchup_of(interaction: LabeledInteraction) -> str:
    """The matchup key for grouping: ``"<attacker> vs <defender>"`` (docs/03 §4)."""
    return f"{interaction.attacker_char_name} vs {interaction.defender_char_name}"


class TallyEntry(BaseModel):
    """One grouped knowledge-check count (docs/03 §4).

    Grouped by ``(knowledge_check_id, attacker_char, attacker_move_id, matchup)``. ``is_recurring``
    applies the pattern's recurrence threshold (docs/06 §4.1) — the ≥3× rule by default.
    """

    knowledge_check_id: str
    attacker_char: str
    attacker_move_id: int
    matchup: str
    count: int
    example_ids: list[str] = Field(default_factory=list)
    recurrence_threshold: int = DEFAULT_RECURRENCE_THRESHOLD

    @property
    def is_recurring(self) -> bool:
        """True once the count reaches the pattern's recurrence threshold (docs/06 §4.1)."""
        return self.count >= self.recurrence_threshold


class KnowledgeCheckTally(BaseModel):
    """The full set of grouped counts for a session (docs/03 §4)."""

    entries: list[TallyEntry] = Field(default_factory=list)

    def recurring(self) -> list[TallyEntry]:
        """Only the entries that met their recurrence threshold — the true knowledge checks."""
        return [e for e in self.entries if e.is_recurring]

    def get(
        self, knowledge_check_id: str, attacker_char: str, attacker_move_id: int, matchup: str
    ) -> TallyEntry | None:
        """Look up a single grouped entry, or ``None`` if that group never triggered."""
        for entry in self.entries:
            if (
                entry.knowledge_check_id == knowledge_check_id
                and entry.attacker_char == attacker_char
                and entry.attacker_move_id == attacker_move_id
                and entry.matchup == matchup
            ):
                return entry
        return None


def build_tally(
    interactions: Iterable[LabeledInteraction],
    rubric: tuple[RubricPattern, ...] = DEFAULT_RUBRIC,
) -> KnowledgeCheckTally:
    """Aggregate labeled interactions into a :class:`KnowledgeCheckTally` (docs/03 §4). Pure.

    Deterministic: entries are ordered by descending count, then by the group key, so the same
    inputs always produce the same tally.
    """
    thresh = thresholds(rubric)
    grouped: dict[tuple[str, str, int, str], TallyEntry] = {}
    for interaction in interactions:
        matchup = matchup_of(interaction)
        for check_id in interaction.labels.knowledge_check_ids:
            key = (check_id, interaction.attacker_char_name, interaction.attacker_move_id, matchup)
            entry = grouped.get(key)
            if entry is None:
                entry = TallyEntry(
                    knowledge_check_id=check_id,
                    attacker_char=interaction.attacker_char_name,
                    attacker_move_id=interaction.attacker_move_id,
                    matchup=matchup,
                    count=0,
                    example_ids=[],
                    recurrence_threshold=thresh.get(check_id, DEFAULT_RECURRENCE_THRESHOLD),
                )
                grouped[key] = entry
            entry.count += 1
            entry.example_ids.append(interaction.id)

    def _order(e: TallyEntry) -> tuple[int, str, str, int, str]:
        return (-e.count, e.knowledge_check_id, e.attacker_char, e.attacker_move_id, e.matchup)

    return KnowledgeCheckTally(entries=sorted(grouped.values(), key=_order))
