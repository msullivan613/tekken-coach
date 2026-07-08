"""The per-exchange state machine — a deterministic ``FrameRecord`` -> ``Interaction`` stream.

This is the C3a *core*: the clean, high-frequency paths of docs/04. It turns a stream of raw
per-frame game state into bounded attacker->defender interactions by running the NEUTRAL -> COMMIT
-> CONTACT -> FOLLOWUP -> NEUTRAL machine (docs/04 §2) over the frames, deriving the four outputs
that matter (docs/04 §3):

1. **who attacked / what move** — from the COMMIT transition (first attacking move),
2. **how the defender reacted** — from the CONTACT transition kind,
3. **observed advantage** — ``defender_actionable_frame - attacker_actionable_frame`` (docs/04 §3),
4. **what the defender did with it** — the follow-up move plus a *structural* ``outcome`` guess.

**Determinism is the contract** (docs/04 §6): the same ``FrameRecord`` stream always yields the
same ``Interaction``s — no wall-clock, no set/dict iteration order, no randomness. Interaction ids
come from a monotonic per-match counter. This is what makes the hand-authored goldens meaningful.

**Scope (C3a, clean paths only).** ``defender_reaction`` is limited to
``{blocked, hit, evaded, whiff_punished}``; single-hit exchanges; pure-whiff discard; basic
sidestep/whiff-punish detection; round/match-boundary truncation (docs/04 §4.8). The docs/04 §4
edge-case catalogue — stagger, multi-hit strings and the per-hit record, throw/tech windows,
counter-hit/punish-counter, Heat transitions, dropped-frame tolerance — is **C3b** and is *not*
handled here. The segmenter also does **not** label frame data, resolve move names, or decide
punishability (that is the xref, docs/04 §5); its ``outcome`` is an explicit structural guess that
the xref confirms or corrects (docs/04 §6).
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto

from tekken_coach.schemas import (
    ActionState,
    DefenderReaction,
    FollowUp,
    FollowUpResult,
    FrameRecord,
    Interaction,
    InteractionContext,
    MatchState,
    Outcome,
    PlayerFrame,
    Wall,
)

# ---------------------------------------------------------------------------
# Tunable constants (docs/04 §2). Chosen to be sensible for 60 fps play and calibrated against the
# hand-authored fixtures; real-capture data (C4) will refine them. All live in one frozen config so
# the numbers are visible, overridable in a test, and never scattered as magic literals.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmenterConfig:
    """Thresholds and window sizes for the state machine (docs/04 §2)."""

    threat_range: float = 2.5
    """Ground-plane distance (game units) below which an attack *could* make contact, so a move
    starting here opens a COMMIT. Above it the attack is spacing/footsies, not an exchange — the
    "whiffed because spaced out is not an interaction" rule (docs/04 §4.4)."""

    lookback_frames: int = 8
    """Size of the recent-frame ring buffer (docs/04 §2). C3a only needs the immediately previous
    frame for transition detection, but the buffer is held per the spec for C3b look-back."""

    followup_window: int = 20
    """Frames after the defender first becomes *actionable* during which a move they start counts
    as their punish/mash attempt. Past it with no input, the follow-up is ``nothing``. ~20 frames
    (a third of a second) comfortably covers reaction + the fastest punisher startups (i10-i15)."""

    max_interaction_frames: int = 120
    """Hard cap (frames since contact) that force-resolves an interaction so a pathological stream
    can never leave one open forever. ~2 s at 60 fps — far longer than any clean exchange."""


DEFAULT_CONFIG = SegmenterConfig()

# Player states in which a player "could input a move" — the "actionable" definition (docs/04 §3):
# out of all stun/recovery, back to a neutral-ish stance. Combined with the block/hit-stun flag
# checks in :func:`_is_actionable` (docs/04 §4.1: trust the flags, not ``action_state`` alone).
_ACTIONABLE_STATES = frozenset(
    {ActionState.neutral, ActionState.crouch, ActionState.sidestep}
)


# ---------------------------------------------------------------------------
# Frame-level predicates (pure; operate on PlayerFrame pairs)
# ---------------------------------------------------------------------------


def _distance(a: PlayerFrame, b: PlayerFrame) -> float:
    """Ground-plane (x, z) distance between two players. Distance is not stored per frame; the
    segmenter derives it from ``pos`` (docs/03 §1). Height (y) is ignored — threat range is a
    floor-distance concept."""
    dx = a.pos[0] - b.pos[0]
    dz = a.pos[2] - b.pos[2]
    return math.hypot(dx, dz)


def _is_attacking(pf: PlayerFrame) -> bool:
    return pf.action_state is ActionState.attack


def _entered_attack(prev: PlayerFrame | None, cur: PlayerFrame) -> bool:
    """True on the frame a *new* attacking move starts — a fresh commit into ``attack`` or a new
    ``move_id`` while already attacking (a chained move). ``move_frame`` distinguishing a new move
    from the same move's next frame is the reader's cheap signal (docs/03 §1)."""
    if not _is_attacking(cur):
        return False
    if prev is None:
        return False
    return not _is_attacking(prev) or cur.move_id != prev.move_id


def _in_blockstun(pf: PlayerFrame) -> bool:
    return pf.block_stun or pf.action_state is ActionState.blockstun


def _in_hitstun(pf: PlayerFrame) -> bool:
    return pf.hit_stun or pf.action_state is ActionState.hitstun


def _entered_blockstun(prev: PlayerFrame | None, cur: PlayerFrame) -> bool:
    return _in_blockstun(cur) and (prev is None or not _in_blockstun(prev))


def _entered_hitstun(prev: PlayerFrame | None, cur: PlayerFrame) -> bool:
    return _in_hitstun(cur) and (prev is None or not _in_hitstun(prev))


def _is_actionable(pf: PlayerFrame) -> bool:
    """First-principles "could input a move" test (docs/04 §3): a neutral-ish stance with **both**
    stun flags clear. Checking the flags (not just ``action_state``) is the docs/04 §4.1 rule."""
    return (
        pf.action_state in _ACTIONABLE_STATES
        and not pf.block_stun
        and not pf.hit_stun
    )


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class _State(Enum):
    NEUTRAL = auto()
    COMMIT = auto()
    CONTACT = auto()
    FOLLOWUP = auto()


class _FollowUpPhase(Enum):
    WAITING = auto()  # defender actionable, watching for their move within the window
    ACTED = auto()  # defender started a move; watching how it resolves
    DONE = auto()  # follow-up fully known (nothing, or acted + result)


@dataclass
class _Open:
    """The single in-flight interaction and the bookkeeping to resolve it deterministically."""

    attacker: int
    defender: int
    attacker_move_id: int
    attacker_char_id: int
    defender_char_id: int
    round: int
    start_frame: int
    contact_frame: int
    context: InteractionContext
    reaction: DefenderReaction

    defender_evaded: bool = False
    attacker_actionable_frame: int | None = None
    defender_actionable_frame: int | None = None

    follow_up_phase: _FollowUpPhase = _FollowUpPhase.WAITING
    follow_up_move_id: int | None = None
    follow_up_act_frame: int | None = None
    follow_up_reaction_frames: int | None = None
    follow_up_result: FollowUpResult = FollowUpResult.none


def _parse_match_no(match_id: str) -> int:
    """Recover the match number for the ``m{n}-r{round}-i{seq}`` id from a ``...#n`` match id
    (docs/03 §2). Defaults to 1 when the id carries no ``#n`` suffix."""
    m = re.search(r"#(\d+)\s*$", match_id)
    return int(m.group(1)) if m else 1


class Segmenter:
    """Streaming, deterministic ``FrameRecord`` -> ``Interaction`` consumer (docs/04 §2).

    Feed frames one at a time with :meth:`feed`; each call returns the interactions that *closed*
    on that frame (usually none). Call :meth:`close` at end of stream to flush a still-open
    interaction. Holds a short look-back ring buffer, the current open interaction, and the small
    carried context (``attacker_pressure``) — nothing else; identical input always yields identical
    output.
    """

    def __init__(self, match_id: str, config: SegmenterConfig = DEFAULT_CONFIG) -> None:
        self._match_id = match_id
        self._match_no = _parse_match_no(match_id)
        self._cfg = config

        self._state: _State = _State.NEUTRAL
        self._open: _Open | None = None
        self._prev: FrameRecord | None = None
        self._lookback: deque[FrameRecord] = deque(maxlen=config.lookback_frames)

        self._seq = 0
        self._round: int | None = None
        self._round_start_health: dict[int, int] = {}
        # Which player left the previous interaction at frame advantage → the next interaction's
        # ``context.attacker_pressure`` if they commit (docs/04 §2). Reset each round.
        self._pressure_player: int | None = None

    # -- public API --------------------------------------------------------

    def feed(self, frame: FrameRecord) -> list[Interaction]:
        """Advance the machine by one frame; return any interactions that closed this frame."""
        emitted: list[Interaction] = []
        self._handle_frame(frame, emitted)
        self._prev = frame
        self._lookback.append(frame)
        return emitted

    def close(self) -> list[Interaction]:
        """Flush a still-open interaction at end of stream (truncated). Idempotent."""
        emitted: list[Interaction] = []
        if self._open is not None and self._prev is not None:
            emitted.append(self._finalize(self._prev.frame, truncated="stream-end"))
            self._reset_open()
        return emitted

    # -- frame handling ----------------------------------------------------

    def _handle_frame(self, fr: FrameRecord, emitted: list[Interaction]) -> None:
        # Round/match boundaries close any open interaction as-is (docs/04 §4.8): a round-ending
        # hit is still recorded, and no interaction spans a round boundary.
        if fr.match_state in (MatchState.round_over, MatchState.match_over):
            if self._open is not None:
                kind = (
                    "round-boundary"
                    if fr.match_state is MatchState.round_over
                    else "match-boundary"
                )
                emitted.append(self._finalize(fr.frame, truncated=kind))
                self._reset_open()
            return

        if fr.match_state is not MatchState.in_round:
            # pre_round / menu / replay-idle: not live play. Truncate anything open and idle.
            if self._open is not None:
                emitted.append(self._finalize(fr.frame, truncated="left-round"))
                self._reset_open()
            return

        # A new round resets carried context and re-snapshots full-health baselines.
        if fr.round != self._round:
            if self._open is not None:
                emitted.append(self._finalize(fr.frame, truncated="round-boundary"))
                self._reset_open()
            self._round = fr.round
            self._round_start_health = {i: p.health for i, p in enumerate(fr.players)}
            self._pressure_player = None

        # Sequential (not elif) so a commit can advance into contact tracking on the same frame.
        if self._state is _State.NEUTRAL:
            self._try_commit(fr)

        if self._state is _State.COMMIT:
            self._advance_commit(fr)

        if self._state in (_State.CONTACT, _State.FOLLOWUP):
            self._track(fr, emitted)

    def _try_commit(self, fr: FrameRecord) -> None:
        """NEUTRAL -> COMMIT: a player starts an attacking move within threat range (docs/04 §2)."""
        prev = self._prev
        # Deterministic tiebreak: lower player index wins if both commit on the same frame.
        for atk in (0, 1):
            dfn = 1 - atk
            atk_pf = fr.players[atk]
            prev_pf = prev.players[atk] if prev is not None else None
            if not _entered_attack(prev_pf, atk_pf):
                continue
            if _distance(fr.players[atk], fr.players[dfn]) > self._cfg.threat_range:
                continue  # spacing / footsies, not an exchange (docs/04 §4.4)
            self._open_commit(fr, atk, dfn)
            return

    def _open_commit(self, fr: FrameRecord, atk: int, dfn: int) -> None:
        atk_pf = fr.players[atk]
        dfn_pf = fr.players[dfn]
        base = self._round_start_health.get(dfn, dfn_pf.health) or 1
        context = InteractionContext(
            distance=_distance(atk_pf, dfn_pf),
            attacker_heat=atk_pf.heat.active,
            defender_heat=dfn_pf.heat.active,
            attacker_pressure=(self._pressure_player == atk),
            wall=Wall.none,  # stage-geometry wall detection is deferred (needs bounds; C3b/reader)
            defender_health_frac=dfn_pf.health / base,
        )
        self._open = _Open(
            attacker=atk,
            defender=dfn,
            attacker_move_id=atk_pf.move_id,  # first attacking move of the exchange (docs/04 §3)
            attacker_char_id=atk_pf.char_id,
            defender_char_id=dfn_pf.char_id,
            round=fr.round,
            start_frame=fr.frame,
            contact_frame=fr.frame,  # provisional; set for real when contact/evade resolves
            context=context,
            reaction=DefenderReaction.blocked,  # provisional; classified at CONTACT
        )
        self._state = _State.COMMIT

    def _advance_commit(self, fr: FrameRecord) -> None:
        """COMMIT -> CONTACT (block/hit), COMMIT -> evade+whiff, or discard back to NEUTRAL."""
        assert self._open is not None
        open_ = self._open
        prev = self._prev
        dfn_pf = fr.players[open_.defender]
        prev_dfn = prev.players[open_.defender] if prev is not None else None
        atk_pf = fr.players[open_.attacker]
        prev_atk = prev.players[open_.attacker] if prev is not None else None

        # Contact: the defender enters stun. Classify from *which* transition (docs/04 §2).
        if _entered_hitstun(prev_dfn, dfn_pf):
            open_.reaction = DefenderReaction.hit
            open_.contact_frame = fr.frame
            self._state = _State.CONTACT
            return
        if _entered_blockstun(prev_dfn, dfn_pf):
            open_.reaction = DefenderReaction.blocked
            open_.contact_frame = fr.frame
            self._state = _State.CONTACT
            return

        # The defender actively evading (sidestep) during the commit — remember it, so a subsequent
        # whiff is an ``evaded`` interaction (coachable) rather than a discarded spacing whiff.
        if dfn_pf.action_state is ActionState.sidestep:
            open_.defender_evaded = True

        # Attacker's move left its active frames with no contact registered.
        if prev_atk is not None and _is_attacking(prev_atk) and not _is_attacking(atk_pf):
            if open_.defender_evaded:
                # evaded (docs/04 §4.4): the defender sidestepped and the attacker whiffed. Open the
                # action window; a whiff-punish will upgrade this to ``whiff_punished``.
                open_.reaction = DefenderReaction.evaded
                open_.contact_frame = fr.frame
                self._state = _State.CONTACT
            else:
                # Pure spacing/neutral whiff, no defender involvement → discard (docs/04 §2).
                self._reset_open()

    def _track(self, fr: FrameRecord, emitted: list[Interaction]) -> None:
        """CONTACT/FOLLOWUP: track actionable frames + the defender action window, then resolve."""
        assert self._open is not None
        open_ = self._open
        prev = self._prev

        atk_pf = fr.players[open_.attacker]
        dfn_pf = fr.players[open_.defender]

        if open_.attacker_actionable_frame is None and _is_actionable(atk_pf):
            open_.attacker_actionable_frame = fr.frame
        if open_.defender_actionable_frame is None and _is_actionable(dfn_pf):
            open_.defender_actionable_frame = fr.frame
            if self._state is _State.CONTACT:
                self._state = _State.FOLLOWUP  # defender free → action window opens (docs/04 §2)

        if open_.defender_actionable_frame is not None:
            self._track_followup(fr, prev)

        if self._ready(fr):
            emitted.append(self._finalize(fr.frame, truncated=None))
            self._reset_open()

    def _track_followup(self, fr: FrameRecord, prev: FrameRecord | None) -> None:
        """Watch the defender's action window: did they act, and how did their move resolve?"""
        assert self._open is not None
        open_ = self._open
        assert open_.defender_actionable_frame is not None
        dfn_pf = fr.players[open_.defender]
        prev_dfn = prev.players[open_.defender] if prev is not None else None

        if open_.follow_up_phase is _FollowUpPhase.WAITING:
            if _entered_attack(prev_dfn, dfn_pf):
                open_.follow_up_phase = _FollowUpPhase.ACTED
                open_.follow_up_move_id = dfn_pf.move_id
                open_.follow_up_act_frame = fr.frame
                open_.follow_up_reaction_frames = (
                    fr.frame - open_.defender_actionable_frame
                )
            elif fr.frame - open_.defender_actionable_frame > self._cfg.followup_window:
                # Window elapsed with no input → the defender did nothing (docs/04 §2).
                open_.follow_up_result = FollowUpResult.none
                open_.follow_up_phase = _FollowUpPhase.DONE
        elif open_.follow_up_phase is _FollowUpPhase.ACTED:
            atk_pf = fr.players[open_.attacker]
            prev_atk = prev.players[open_.attacker] if prev is not None else None
            if _entered_hitstun(prev_atk, atk_pf):
                open_.follow_up_result = FollowUpResult.hit
                open_.follow_up_phase = _FollowUpPhase.DONE
                if open_.reaction is DefenderReaction.evaded:
                    # The evade's whiff punish landed (docs/04 §4.4).
                    open_.reaction = DefenderReaction.whiff_punished
            elif _entered_blockstun(prev_atk, atk_pf):
                open_.follow_up_result = FollowUpResult.blocked
                open_.follow_up_phase = _FollowUpPhase.DONE
            elif prev_dfn is not None and _is_attacking(prev_dfn) and not _is_attacking(dfn_pf):
                # The defender's move recovered without touching the attacker → it whiffed.
                open_.follow_up_result = FollowUpResult.whiffed
                open_.follow_up_phase = _FollowUpPhase.DONE

    def _ready(self, fr: FrameRecord) -> bool:
        """Emit once the follow-up is fully known *and* observed advantage is settled — or at the
        safety cap. Waiting for both actionable frames is what lets a punish still carry its
        observed advantage rather than resolving the instant the defender acts."""
        assert self._open is not None
        open_ = self._open
        if fr.frame - open_.contact_frame >= self._cfg.max_interaction_frames:
            return True
        if open_.follow_up_phase is not _FollowUpPhase.DONE:
            return False
        if (
            open_.attacker_actionable_frame is not None
            and open_.defender_actionable_frame is not None
        ):
            return True
        # A landed punish means the attacker never reached neutral — advantage is N/A but the
        # interaction is fully resolved, so emit rather than wait for the cap.
        return open_.follow_up_result is FollowUpResult.hit

    # -- emission ----------------------------------------------------------

    def _finalize(self, end_frame: int, *, truncated: str | None) -> Interaction:
        assert self._open is not None
        open_ = self._open

        observed_advantage: int | None = None
        if (
            open_.attacker_actionable_frame is not None
            and open_.defender_actionable_frame is not None
        ):
            observed_advantage = (
                open_.defender_actionable_frame - open_.attacker_actionable_frame
            )

        follow_up = FollowUp(
            move_id=open_.follow_up_move_id,
            result=open_.follow_up_result,
            reaction_frames=open_.follow_up_reaction_frames,
        )
        outcome = self._guess_outcome(open_, observed_advantage)

        self._seq += 1
        interaction = Interaction(
            id=f"m{self._match_no}-r{open_.round}-i{self._seq:03d}",
            match_id=self._match_id,
            round=open_.round,
            start_frame=open_.start_frame,
            end_frame=end_frame,
            attacker=open_.attacker,
            defender=open_.defender,
            attacker_move_id=open_.attacker_move_id,
            attacker_char_id=open_.attacker_char_id,
            defender_char_id=open_.defender_char_id,
            context=open_.context,
            defender_reaction=open_.reaction,
            observed_advantage=observed_advantage,
            outcome=outcome,
            follow_up=follow_up,
            notes=[f"truncated:{truncated}"] if truncated else [],
        )

        # Carry frame-advantage context to the next interaction (docs/04 §2).
        if observed_advantage is not None and observed_advantage > 0:
            self._pressure_player = open_.attacker
        elif observed_advantage is not None and observed_advantage < 0:
            self._pressure_player = open_.defender
        else:
            self._pressure_player = None

        return interaction

    def _guess_outcome(self, open_: _Open, advantage: int | None) -> Outcome:
        """The *structural* ``outcome`` guess (docs/04 §3, §6) — a best-effort read from state
        transitions alone. The xref (docs/05) supplies the authoritative punishability/gap
        judgment and may override this. Perspective-neutral values only: the segmenter does not
        know which side is the user, and cannot know a move's high/mid/low (so it never guesses
        ``ate_mid``/``ate_low`` — that needs frame data)."""
        reaction = open_.reaction
        result = open_.follow_up_result
        acted = open_.follow_up_phase is _FollowUpPhase.DONE and open_.follow_up_move_id is not None

        if reaction is DefenderReaction.whiff_punished:
            return Outcome.punished
        if reaction is DefenderReaction.blocked:
            if not acted:
                # Attacker minus and defender pressed nothing → missed-punish guess (docs/04 §3).
                minus = advantage is not None and advantage < 0
                return Outcome.no_punish if minus else Outcome.neutral
            if result is FollowUpResult.hit:
                return Outcome.punished
            return Outcome.bad_punish  # acted but whiffed / got blocked
        if reaction is DefenderReaction.evaded:
            # Evaded but left the whiff unpunished → a missed punish.
            return Outcome.no_punish if not acted else Outcome.neutral
        # Clean hit: whether it was an ``ate_mid``/``ate_low`` needs the move's height (xref).
        return Outcome.neutral

    def _reset_open(self) -> None:
        self._open = None
        self._state = _State.NEUTRAL


def segment_frames(
    frames: Iterable[FrameRecord],
    *,
    match_id: str,
    config: SegmenterConfig = DEFAULT_CONFIG,
) -> list[Interaction]:
    """Convenience: run a whole ``FrameRecord`` iterable through a fresh :class:`Segmenter` and
    return every interaction, flushing the tail with :meth:`Segmenter.close`."""
    seg = Segmenter(match_id, config)
    out: list[Interaction] = []
    for fr in frames:
        out.extend(seg.feed(fr))
    out.extend(seg.close())
    return out
