"""A *calibrated* encoded-state map for the offline tests (docs/02 §8).

The shipped ``assets/offsets/state-map.json`` is an empty skeleton on purpose: nothing but live
observation can say what ``stun_type == 3`` means, and shipping a guess would be worse than shipping
nothing. The offline suite therefore supplies its own filled-in map, so the decode path can be
exercised end-to-end without pretending to know the real build's values.

Its shape is what matters, and it is the shape the runbook produces: one encoded field per state
axis, each raw value naming the semantic flags it implies. The decoder unions the flags across
fields, so overlapping axes compose (``stun_type=2`` + ``complex_move_state=2`` = hit_stun in a
juggle) without the map having to enumerate the product.
"""

from __future__ import annotations

from tekken_coach.reader.offsets import EncodedStateSpec

# Raw values are arbitrary here — they stand in for whatever the live build turns out to use.
CALIBRATED_FLAGS: dict[str, dict[str, list[str]]] = {
    "simple_move_state": {"0": ["neutral"], "1": ["attack"], "2": ["recovery"], "3": ["crouch"]},
    "stun_type": {"0": [], "1": ["block_stun"], "2": ["hit_stun"], "3": ["stagger"]},
    "complex_move_state": {
        "0": [],
        "1": ["airborne"],
        "2": ["airborne", "juggle"],
        "3": ["knockdown"],
        "4": ["wakeup"],
        "5": ["sidestep"],
    },
    "throw_tech_state": {"0": [], "1": ["throw_active"], "2": ["throw_tech"], "3": ["thrown"]},
    "recovery_state": {"0": [], "1": ["recovery"]},
}


def calibrated_state_map() -> EncodedStateSpec:
    """The map the offline tests decode through."""
    return EncodedStateSpec(
        calibrated=True,
        notes="test fixture; raw values stand in for the live build's",
        flags=CALIBRATED_FLAGS,
    )
