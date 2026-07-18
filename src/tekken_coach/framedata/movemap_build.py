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

# Live startup-match tolerances, in frames (Stage B live path only, brief #14). The LIVE join
# matches on *startup* — the contact-frame ``move_frame``, an observed event that is reliable — not
# on live on-block, which is measured from fuzzy return-to-idle animation and reads too negative for
# fast/plus moves (a +1 jab reads ≈−5). The tight band is ±1; ±2 is a fallback used only when ±1 is
# empty, since startup can read one frame high on a late poll even at 120 Hz.
DEFAULT_LIVE_STARTUP_TOL = 1
DEFAULT_LIVE_STARTUP_TOL_FALLBACK = 2

# Soft on-block preference tolerance for the LIVE join (brief #14). Live on-block reads too negative
# (only the attacker side is animation-lagged, so the two sides do not cancel), which makes the
# observed value a rough *lower bound* on the truth: a candidate whose Wavu on-block is
# ``>= observed - this tol`` is consistent with that bound and is ranked ahead of one that
# contradicts it. This NEVER filters — it only orders the startup-matched candidates, so the true
# ``1`` (+1) survives an observed −5 (brief #14 §2).
DEFAULT_LIVE_BLOCK_SOFT_TOL = 1


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


def _startup_within(
    moves: list[FrameDataMove], target_startup: int, tol: int
) -> list[FrameDataMove]:
    """Moves whose Wavu startup is within ``tol`` frames of ``target_startup`` (brief #14 live).

    Moves with no Wavu startup can't be startup-matched, so they are excluded here (they are offered
    separately, ranked last). The ``is not None`` guard also narrows the type for the subtraction.
    """
    return [m for m in moves if m.startup is not None and abs(m.startup - target_startup) <= tol]


def join_move_live(
    fingerprint: MoveFingerprint,
    char_framedata: CharFrameData,
    *,
    startup_tol: int = DEFAULT_LIVE_STARTUP_TOL,
    startup_tol_fallback: int = DEFAULT_LIVE_STARTUP_TOL_FALLBACK,
    block_soft_tol: int = DEFAULT_LIVE_BLOCK_SOFT_TOL,
) -> JoinResult:
    """Rank a character's Wavu moves against one *live* fingerprint by STARTUP (brief #14).

    The live path measures startup at a crisp event — the contact-frame ``move_frame`` — so it is
    reliable. Live on-block, by contrast, is measured from the attacker's fuzzy return-to-idle
    animation and reads far too negative for fast/plus moves (a +1 jab reads ≈−5), which used to
    *hide* the true move: :func:`join_move` hard-filters by on-block, so the +1 ``1`` fell outside a
    −5 fingerprint's candidate set entirely (brief #14). This join routes around that:

    * **Primary discriminator is startup**, within ±``startup_tol`` (falling back to
      ±``startup_tol_fallback`` only when the tight band is empty). A move whose Wavu startup is
      outside the band is ruled out — startup is the trustworthy signal.
    * **On-block never filters.** The observed value is a rough *lower bound* on the truth, so it
      only **soft-ranks**: a candidate whose Wavu on-block is ``>= observed − block_soft_tol`` is
      preferred over one that contradicts the bound, but a startup-match is never dropped for
      failing it. The true ``1`` (+1) therefore survives an observed −5.
    * Moves with **no Wavu startup** (they can't be startup-matched — e.g. later hits of a string)
      are still **offered, ranked last**, never dropped.

    Unlike :func:`join_move` (the on_block-primary log path, left unchanged), this never
    *auto-writes* a mapping in practice: the live harness always has the user confirm, so a
    shared-startup band is presented as a ranked candidate list to disambiguate, never a guess.
    """
    observed_startup = fingerprint.startup
    observed_block = fingerprint.on_block

    moves = list(char_framedata.moves.values())
    without_startup = [m for m in moves if m.startup is None]

    used_tol = startup_tol
    if observed_startup is None:
        # No startup signal (should not happen live — a contact always yields one). We cannot
        # discriminate on startup, so offer every move ranked by on-block plausibility alone.
        startup_hits: list[FrameDataMove] = []
        fallback = moves
    else:
        startup_hits = _startup_within(moves, observed_startup, startup_tol)
        if not startup_hits and startup_tol_fallback > startup_tol:
            used_tol = startup_tol_fallback
            startup_hits = _startup_within(moves, observed_startup, startup_tol_fallback)
        # Moves with a Wavu startup that missed the band are ruled out; only the no-startup moves
        # are kept as a ranked-last fallback (brief #14 §2 "offer them, ranked last").
        fallback = without_startup

    def _implausible(move: FrameDataMove) -> int:
        """0 when on-block is consistent with the observed lower bound (preferred), else 1."""
        if observed_block is None or move.on_block is None:
            return 0  # nothing to contradict — treat as plausible
        return 0 if move.on_block >= observed_block - block_soft_tol else 1

    def _startup_delta(move: FrameDataMove) -> int:
        if move.startup is None or observed_startup is None:
            return 1_000
        return abs(move.startup - observed_startup)

    # tier 0 = startup-matched (primary), tier 1 = offered-last (no Wavu startup). Within a tier:
    # on-block plausibility, then startup proximity, then key — all deterministic.
    tiered: list[tuple[int, FrameDataMove]] = [(0, m) for m in startup_hits] + [
        (1, m) for m in fallback
    ]
    ordered = sorted(
        tiered, key=lambda tm: (tm[0], _implausible(tm[1]), _startup_delta(tm[1]), tm[1].key)
    )
    candidates = [_candidate(move, fingerprint) for _, move in ordered]

    startup_label = (
        f"startup ≈{observed_startup} (±{used_tol})"
        if observed_startup is not None
        else "no startup signal"
    )

    if not candidates:
        return JoinResult(
            move_id=fingerprint.move_id,
            status="no_candidate",
            framedata_key=None,
            candidates=[],
            reason=(
                f"no move in {char_framedata.char_slug} snapshot matches {startup_label} "
                "(and none lack a Wavu startup to offer)"
            ),
        )
    if len(candidates) == 1:
        chosen = candidates[0]
        return JoinResult(
            move_id=fingerprint.move_id,
            status="auto_mapped",
            framedata_key=chosen.framedata_key,
            candidates=[chosen],
            reason=f"unique on {startup_label} → {chosen.framedata_key}",
        )
    if startup_hits:
        reason = (
            f"{len(startup_hits)} move(s) share {startup_label}; on-block is advisory only "
            "(live reads low for fast moves) — disambiguate by eye"
        )
    else:
        reason = (
            f"nothing matched {startup_label}; offering moves with no Wavu startup "
            "(e.g. later string hits) for you to pick"
        )
    return JoinResult(
        move_id=fingerprint.move_id,
        status="collision",
        framedata_key=None,
        candidates=candidates,
        reason=reason,
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
