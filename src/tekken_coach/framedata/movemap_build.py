"""Build the movemap by frame-fingerprint join (the C6 ``map-moves`` core, brief #6).

The movemap is the missing ``memory move_id -> framedata_key`` bridge: without it every
interaction resolves ``frame_data_matched:false`` and coaching is inert (project memory
``live-run-1-movemap-empty``). No external dataset publishes memory move_ids, so this module
*builds* the bridge from behaviour we already observe.

The idea (docs/05 §2.2, brief #6): every interaction carries ``(char_id, move_id,
observed_advantage)``, and the Wavu snapshot is a table of ``framedata_key -> (startup,
on_block, hit_level, …)``. So for a given ``move_id`` we **fingerprint-match** its observed
behaviour to the character's Wavu moves. A *unique* match auto-maps ``move_id -> framedata_key``;
anything ambiguous is a **collision** reported for a human to disambiguate (Stage B live confirm),
never guessed. This is version-correct by construction — it is built from *this* build's move_ids —
and needs no ToS-risky datamine.

Two honest limits, by design:

* **On_block is a coarse discriminator.** Many moves share a common on-block value, so the
  log-only path (no startup) auto-maps only the rare ``move_id`` whose observed on-block is unique
  in the snapshot; everything else is a reported collision. That is the point: a wrong mapping must
  be *structurally impossible* to auto-write (05 §2.3 miss-tolerance carries over — a collision is
  reported, never resolved by a guess).
* **Startup is the tie-breaker, and it is only observable live (Stage B).** The log carries no
  per-move startup, so ``MoveFingerprint.startup`` is ``None`` on the ``--from-log`` path and the
  join ranks on on-block alone.

Everything here is pure (no game, no I/O): the miner reads already-loaded interactions and the
snapshot, and returns a plan. The live harness (Stage B) supplies fingerprints from the reader loop
but reuses this exact join/consensus/merge core, so the decision logic is unit-tested.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from tekken_coach.framedata.models import CharFrameData, FrameDataMove, MoveMapEntry
from tekken_coach.schemas import DefenderReaction, Interaction

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

# On-block match tolerance, in frames. A move's *observed* advantage (from memory polling at ~60 Hz)
# can land a frame early or late relative to Wavu's canonical on-block, so ±1 absorbs that poll
# jitter. It deliberately does NOT admit a move that differs by ≥2 frames — the join stays tight so
# an auto-map is only ever written when the snapshot has a single move at (approximately) the
# observed value (brief #6: "auto-map iff exactly one candidate within a tight tolerance").
DEFAULT_BLOCK_TOL = 1

# Startup match tolerance, in frames (Stage B only). Startup is reported as a lower bound of a range
# in the snapshot ("i22~23") and observed from ``move_frame`` at contact, so ±2 covers the range
# width plus one frame of poll jitter. Used only to break an on-block tie when startup is observed.
DEFAULT_STARTUP_TOL = 2


# ---------------------------------------------------------------------------
# Fingerprint + candidate + join result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveFingerprint:
    """The observed behaviour of one memory ``move_id`` for one character (brief #6 §A.1).

    ``on_block`` is the consensus observed advantage over the move's **blocked** samples (the
    primary discriminator); ``None`` when there is no usable blocked reading. ``startup`` is the
    observed startup — populated only on the Stage-B live path, ``None`` from a log. The sample
    counts explain *why* a fingerprint is weak, so the miner can report an actionable reason.
    """

    char_id: int
    move_id: int
    on_block: int | None
    startup: int | None = None
    blocked_samples: int = 0  # blocked interactions with a non-null observed_advantage
    total_samples: int = 0  # all interactions seen for this (char_id, move_id)


@dataclass(frozen=True)
class Candidate:
    """One ranked frame-data candidate for a fingerprint (brief #6 §A.1)."""

    framedata_key: str
    on_block: int | None
    startup: int | None
    block_delta: int | None  # |observed on_block - candidate on_block|, or None if unscored
    startup_delta: int | None  # |observed startup - candidate startup|, when both known


JoinStatus = Literal["auto_mapped", "collision", "no_candidate", "no_signal"]


@dataclass(frozen=True)
class JoinResult:
    """The outcome of joining one fingerprint against a character's frame data (brief #6 §A.1).

    * ``auto_mapped`` — exactly one candidate survived the available discriminators;
      ``framedata_key`` is set and ``candidates`` holds that single move. Safe to write.
    * ``collision`` — two or more candidates tie; ``candidates`` lists them (ranked). Never written.
    * ``no_candidate`` — no snapshot move sits within tolerance of the observed on-block.
    * ``no_signal`` — the fingerprint had no usable blocked reading to match on.
    """

    move_id: int
    status: JoinStatus
    framedata_key: str | None
    candidates: list[Candidate]
    reason: str


def join_move(
    fingerprint: MoveFingerprint,
    char_framedata: CharFrameData,
    *,
    block_tol: int = DEFAULT_BLOCK_TOL,
    startup_tol: int = DEFAULT_STARTUP_TOL,
) -> JoinResult:
    """Rank a character's Wavu moves against one observed fingerprint (brief #6 §A.1).

    Primary discriminator is **on_block** (observed when the defender blocked); the optional
    **startup** (Stage B) breaks an on-block tie. A mapping is proposed *only* when exactly one
    candidate survives — a wrong ``move_id -> framedata_key`` is structurally impossible to
    auto-write, because a tie is returned as a ``collision`` for a human to resolve.
    """
    if fingerprint.on_block is None:
        return JoinResult(
            move_id=fingerprint.move_id,
            status="no_signal",
            framedata_key=None,
            candidates=[],
            reason=_no_signal_reason(fingerprint),
        )

    observed_block = fingerprint.on_block
    block_hits = [
        move
        for move in char_framedata.moves.values()
        if move.on_block is not None and abs(move.on_block - observed_block) <= block_tol
    ]
    if not block_hits:
        return JoinResult(
            move_id=fingerprint.move_id,
            status="no_candidate",
            framedata_key=None,
            candidates=[],
            reason=(
                f"no move in {char_framedata.char_slug} snapshot has on_block "
                f"within ±{block_tol} of observed {observed_block:+d}"
            ),
        )

    candidates = [_candidate(move, fingerprint) for move in block_hits]

    # Narrow with startup only when it is observed (Stage B). If it isolates exactly one candidate,
    # that break is decisive; if it matches several or none, we fall back to the on-block set and
    # report a collision rather than trusting a partial startup signal.
    if fingerprint.startup is not None:
        startup_hits = [
            c for c in candidates if c.startup_delta is not None and c.startup_delta <= startup_tol
        ]
        if len(startup_hits) == 1:
            chosen = startup_hits[0]
            return JoinResult(
                move_id=fingerprint.move_id,
                status="auto_mapped",
                framedata_key=chosen.framedata_key,
                candidates=[chosen],
                reason=(
                    f"unique on startup ≈{fingerprint.startup} + on_block {observed_block:+d} "
                    f"→ {chosen.framedata_key}"
                ),
            )

    ranked = _rank(candidates)
    if len(ranked) == 1:
        chosen = ranked[0]
        return JoinResult(
            move_id=fingerprint.move_id,
            status="auto_mapped",
            framedata_key=chosen.framedata_key,
            candidates=[chosen],
            reason=f"unique on_block {observed_block:+d} → {chosen.framedata_key}",
        )
    return JoinResult(
        move_id=fingerprint.move_id,
        status="collision",
        framedata_key=None,
        candidates=ranked,
        reason=(
            f"{len(ranked)} moves share on_block ≈{observed_block:+d} "
            f"(within ±{block_tol}); needs a startup read (live) to disambiguate"
        ),
    )


def _candidate(move: FrameDataMove, fingerprint: MoveFingerprint) -> Candidate:
    """Build a :class:`Candidate` from a frame-data move against a fingerprint."""
    block_delta = (
        abs(move.on_block - fingerprint.on_block)
        if move.on_block is not None and fingerprint.on_block is not None
        else None
    )
    startup_delta = (
        abs(move.startup - fingerprint.startup)
        if move.startup is not None and fingerprint.startup is not None
        else None
    )
    return Candidate(
        framedata_key=move.key,
        on_block=move.on_block,
        startup=move.startup,
        block_delta=block_delta,
        startup_delta=startup_delta,
    )


def _rank(candidates: list[Candidate]) -> list[Candidate]:
    """Order candidates deterministically: closest on-block, then startup, then key (brief #6)."""

    def sort_key(c: Candidate) -> tuple[int, int, str]:
        return (
            c.block_delta if c.block_delta is not None else 1_000,
            c.startup_delta if c.startup_delta is not None else 1_000,
            c.framedata_key,
        )

    return sorted(candidates, key=sort_key)


def _no_signal_reason(fingerprint: MoveFingerprint) -> str:
    """Explain why a fingerprint has no usable on-block reading (for the miner report)."""
    if fingerprint.total_samples == 0:
        return "no samples observed"
    if fingerprint.blocked_samples == 0:
        return "only hit/CH samples — never observed on block (no on_block reading)"
    return "blocked samples had no readable advantage, or their advantages did not agree"


# ---------------------------------------------------------------------------
# Fingerprint consensus (from interactions)
# ---------------------------------------------------------------------------


def build_fingerprint(
    char_id: int, move_id: int, interactions: list[Interaction]
) -> MoveFingerprint:
    """Distil the observed fingerprint for one ``(char_id, move_id)`` group (brief #6 §A.2).

    Consensus on-block is the **modal** ``observed_advantage`` over the move's blocked samples
    (ignoring hit/CH samples, per brief #6 §A.2). A tie between two modal values is treated as *no
    consensus* (``on_block=None``) rather than an arbitrary pick — an ambiguous reading must not
    seed a mapping.
    """
    blocked_advs = [
        i.observed_advantage
        for i in interactions
        if i.defender_reaction == DefenderReaction.blocked and i.observed_advantage is not None
    ]
    on_block = _modal(blocked_advs)
    return MoveFingerprint(
        char_id=char_id,
        move_id=move_id,
        on_block=on_block,
        startup=None,
        blocked_samples=len(blocked_advs),
        total_samples=len(interactions),
    )


def _modal(values: list[int]) -> int | None:
    """Return the unique modal value, or ``None`` when empty or tied (brief #6 §A.2 consensus)."""
    if not values:
        return None
    counts = Counter(values).most_common()
    if len(counts) == 1 or counts[0][1] > counts[1][1]:
        return counts[0][0]
    return None  # tie — no consensus


# ---------------------------------------------------------------------------
# Move-map entry construction
# ---------------------------------------------------------------------------


def entry_for(char_framedata: CharFrameData, framedata_key: str) -> MoveMapEntry:
    """Build the :class:`MoveMapEntry` for an auto-mapped ``framedata_key`` (docs/05 §2.2).

    ``notation`` is the ``framedata_key`` itself (the CSV Command notation is the human notation);
    the move's display ``name``, when present and distinct, is kept as an alias. Only a key that
    exists in the snapshot is ever passed here — the join guarantees it (brief #6 §A.1).
    """
    move = char_framedata.get(framedata_key)
    aliases: list[str] = []
    if move is not None and move.name and move.name != framedata_key:
        aliases = [move.name]
    return MoveMapEntry(notation=framedata_key, aliases=aliases, framedata_key=framedata_key)
