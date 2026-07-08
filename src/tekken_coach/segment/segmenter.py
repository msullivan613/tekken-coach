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

**Scope.** C3a delivered the clean, high-frequency paths (docs/04 §2/§3): the
``{blocked, hit, evaded, whiff_punished}`` reactions, single-hit exchanges, pure-whiff discard,
basic sidestep/whiff-punish detection, round/match-boundary truncation (docs/04 §4.8). C3b adds the
docs/04 §4 edge-case catalogue on top: stagger (§4.1), multi-hit strings and the per-hit block/duck
record (§4.2), throw-break / knockdown-wakeup tech windows (§4.3), counter-hit / punish-counter
(§4.5), Heat activation within an exchange (§4.6), and dropped-frame tolerance (§4.7). The remaining
``defender_reaction`` values (``counter_hit, stagger, thrown, throw_broke``) are wired here.

The segmenter still does **not** label frame data, resolve move names, or decide punishability
(that is the xref, docs/04 §5): move-height / duckable-high labeling is driven from the per-hit
record *in the xref*, not re-derived here. Its ``outcome`` stays an explicit, conservative
structural guess that the xref confirms or corrects (docs/04 §6); the frame-data-dependent enum
values (``ate_mid``/``ate_low``/``mashed_into_ch``/``respected_*``/``challenged_*``) remain the
xref's authoritative call. **Determinism is the contract** (docs/04 §6): same stream, same output.
"""

from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

from tekken_coach.schemas import (
    ActionState,
    CounterState,
    DefenderReaction,
    FollowUp,
    FollowUpResult,
    FrameRecord,
    Interaction,
    InteractionContext,
    MatchState,
    Outcome,
    PlayerFrame,
    StringHitRecord,
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
    can never leave one open forever. ~2 s at 60 fps — far longer than any clean exchange.
    Comfortably covers a knockdown->wakeup oki window (docs/04 §4.3), so the FOLLOWUP that extends
    to the wakeup-actionable frame still resolves inside the interaction rather than at the cap."""

    max_gap_tolerated: int = 3
    """Dropped-frame tolerance (docs/04 §4.7). A frame-counter gap of this many missed frames or
    fewer is bridged by assuming state continuity (noted ``gap-tolerated:N``); a larger gap still
    emits the interaction but forces ``observed_advantage: null`` because frame-counting across it
    is unreliable, and the xref falls back to canonical frame data (docs/04 §4.7, docs/05 §4.2)."""


DEFAULT_CONFIG = SegmenterConfig()

# Player states in which a player "could input a move" — the "actionable" definition (docs/04 §3):
# out of all stun/recovery, back to a neutral-ish stance. Combined with the block/hit-stun flag
# checks in :func:`_is_actionable` (docs/04 §4.1: trust the flags, not ``action_state`` alone).
_ACTIONABLE_STATES = frozenset({ActionState.neutral, ActionState.crouch, ActionState.sidestep})


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


def _in_stagger(pf: PlayerFrame) -> bool:
    return pf.action_state is ActionState.stagger


def _entered_stagger(prev: PlayerFrame | None, cur: PlayerFrame) -> bool:
    """docs/04 §4.1: distinguish stagger by ``action_state == stagger``, not "can't act"."""
    return _in_stagger(cur) and (prev is None or not _in_stagger(prev))


def _entered_thrown(prev: PlayerFrame | None, cur: PlayerFrame) -> bool:
    return cur.action_state is ActionState.thrown and (
        prev is None or prev.action_state is not ActionState.thrown
    )


def _entered_throw(prev: PlayerFrame | None, cur: PlayerFrame) -> bool:
    """True on the frame a throw attempt starts (``throw_active`` newly set) — the throw analogue
    of :func:`_entered_attack`, so a throw opens a COMMIT even without ``action_state==attack``."""
    return cur.throw_active and (prev is None or not prev.throw_active)


def _is_counter(pf: PlayerFrame) -> bool:
    """The defender's contact carries a counter/punish-counter marker (docs/04 §4.5)."""
    return pf.counter_state is not CounterState.none


def _offense_ended(prev: PlayerFrame, cur: PlayerFrame) -> bool:
    """True when the attacker's offense (a strike's active frames or a throw) just ended with no
    contact registered — the whiff/discard signal, covering strikes and throws (docs/04 §4.3)."""
    was = _is_attacking(prev) or prev.throw_active
    now = _is_attacking(cur) or cur.throw_active
    return was and not now


def _per_hit_reaction(dfn_pf: PlayerFrame) -> DefenderReaction | None:
    """Classify one string hit from the defender's concurrent stun (docs/04 §4.2): in hitstun ⇒
    ``hit``, in blockstun/stagger ⇒ ``blocked``, otherwise ``None`` (no stun contact this frame —
    the hit has not connected, or a high whiffed over a duck)."""
    if _in_hitstun(dfn_pf):
        return DefenderReaction.hit
    if _in_blockstun(dfn_pf) or _in_stagger(dfn_pf):
        return DefenderReaction.blocked
    return None


def _stance_crouch(prev: bool, pf: PlayerFrame) -> bool:
    """Track standing-vs-crouching across stun (docs/04 §4.2 per-hit posture). A readable stance
    (crouch / neutral / sidestep / attack) sets it; stun states are unreadable, so the last known
    stance carries — a defender who ducked to block a low stays "crouching" through the string."""
    if pf.action_state is ActionState.crouch:
        return True
    if pf.action_state in (
        ActionState.neutral,
        ActionState.sidestep,
        ActionState.attack,
    ):
        return False
    return prev


def _is_actionable(pf: PlayerFrame) -> bool:
    """First-principles "could input a move" test (docs/04 §3): a neutral-ish stance with **both**
    stun flags clear. Checking the flags (not just ``action_state``) is the docs/04 §4.1 rule."""
    return pf.action_state in _ACTIONABLE_STATES and not pf.block_stun and not pf.hit_stun


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

    # C3b edge-case bookkeeping.
    string_hits: list[StringHitRecord] = field(default_factory=list)  # per-hit record (§4.2)
    last_hit_atk_move: int | None = None  # attacker move id of the last recorded string hit (§4.2)
    defender_crouch: bool = False  # tracked standing-vs-crouching stance (§4.2)
    throw_attempt: bool = False  # attacker throw + defender in tech window seen (§4.3)
    advantage_unreliable: bool = (
        False  # a large frame gap makes advantage-counting unreliable (§4.7)
    )
    extra_notes: list[str] = field(default_factory=list)  # heat / gap diagnostics (§4.6, §4.7)
    heat_noted: bool = False  # a mid-exchange Heat activation was already noted (§4.6)


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

        # Dropped-frame tolerance and mid-exchange Heat activation apply to whatever is open
        # (docs/04 §4.7, §4.6). Checked before the state dispatch so the note lands on this frame.
        if self._open is not None:
            self._note_gap(fr)
            self._note_heat(fr)
            self._open.defender_crouch = _stance_crouch(
                self._open.defender_crouch, fr.players[self._open.defender]
            )

        # Sequential (not elif) so a commit can advance into contact tracking on the same frame.
        if self._state is _State.NEUTRAL:
            self._try_commit(fr)

        if self._state is _State.COMMIT:
            self._advance_commit(fr)

        if self._state in (_State.CONTACT, _State.FOLLOWUP):
            self._track(fr, emitted)

    def _note_gap(self, fr: FrameRecord) -> None:
        """Dropped-frame tolerance (docs/04 §4.7). A small frame-counter gap is bridged with a
        ``gap-tolerated:N`` note; a gap beyond the threshold still keeps the interaction open but
        marks its advantage unreliable so :meth:`_finalize` emits ``observed_advantage: null``."""
        assert self._open is not None
        prev = self._prev
        if prev is None:
            return
        missed = fr.frame - prev.frame - 1
        if missed <= 0:
            return
        self._open.extra_notes.append(f"gap-tolerated:{missed}")
        if missed > self._cfg.max_gap_tolerated:
            self._open.advantage_unreliable = True

    def _note_heat(self, fr: FrameRecord) -> None:
        """Detect a Heat activation *within* the interaction (docs/04 §4.6): it shifts frame
        advantage mid-exchange, so note the frame (once). Context heat at start is already captured
        in :class:`InteractionContext`; this catches the activation that happens after."""
        assert self._open is not None
        open_ = self._open
        if open_.heat_noted or open_.context.attacker_heat:
            return
        prev = self._prev
        atk_pf = fr.players[open_.attacker]
        prev_atk = prev.players[open_.attacker] if prev is not None else None
        if atk_pf.heat.active and (prev_atk is None or not prev_atk.heat.active):
            open_.extra_notes.append(f"heat-activated:{fr.frame}")
            open_.heat_noted = True

    def _try_commit(self, fr: FrameRecord) -> None:
        """NEUTRAL -> COMMIT: a player starts an attacking move within threat range (docs/04 §2)."""
        prev = self._prev
        # Deterministic tiebreak: lower player index wins if both commit on the same frame.
        for atk in (0, 1):
            dfn = 1 - atk
            atk_pf = fr.players[atk]
            prev_pf = prev.players[atk] if prev is not None else None
            # A strike (``attack``) or a throw (``throw_active``) both open a commit (docs/04 §4.3).
            if not (_entered_attack(prev_pf, atk_pf) or _entered_throw(prev_pf, atk_pf)):
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
            # Wall proximity is not derivable from ``pos`` alone: it needs per-stage bounds the
            # FrameRecord does not carry (docs/04 §4 leaves geometry to the reader). Rather than
            # invent geometry, we emit ``none`` and defer real detection to the reader (C4), which
            # knows the stage. Reader-dependent by design — see the C3b report.
            wall=Wall.none,
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
        self._open.defender_crouch = _stance_crouch(False, dfn_pf)  # initial posture (§4.2)
        self._state = _State.COMMIT

    def _advance_commit(self, fr: FrameRecord) -> None:
        """COMMIT -> CONTACT (block/hit/stagger/throw/counter), evade+whiff, or discard back."""
        assert self._open is not None
        open_ = self._open
        prev = self._prev
        dfn_pf = fr.players[open_.defender]
        prev_dfn = prev.players[open_.defender] if prev is not None else None
        atk_pf = fr.players[open_.attacker]
        prev_atk = prev.players[open_.attacker] if prev is not None else None

        # --- throw tech window (docs/04 §4.3) --------------------------------
        if atk_pf.throw_active and dfn_pf.action_state is ActionState.throw_tech_window:
            open_.throw_attempt = True
        if _entered_thrown(prev_dfn, dfn_pf):
            self._begin_contact(fr, DefenderReaction.thrown, per_hit=None)
            return
        if open_.throw_attempt and _is_actionable(dfn_pf):
            # Left the tech window actionable without being thrown → the defender broke it.
            self._begin_contact(fr, DefenderReaction.throw_broke, per_hit=None)
            return

        # --- strike contact: the defender enters stun. Classify from *which* transition, and by
        # the specific flag, not "can't act" (docs/04 §2, §4.1, §4.5). ----------
        if _entered_hitstun(prev_dfn, dfn_pf):
            # A counter/punish-counter marker on the defender's hit ⇒ counter_hit (docs/04 §4.5).
            reaction = DefenderReaction.counter_hit if _is_counter(dfn_pf) else DefenderReaction.hit
            self._begin_contact(fr, reaction, per_hit=DefenderReaction.hit)
            return
        if _entered_stagger(prev_dfn, dfn_pf):
            # Stagger is its own reaction (docs/04 §4.1); per-hit it reads as a stood-up block.
            self._begin_contact(fr, DefenderReaction.stagger, per_hit=DefenderReaction.blocked)
            return
        if _entered_blockstun(prev_dfn, dfn_pf):
            self._begin_contact(fr, DefenderReaction.blocked, per_hit=DefenderReaction.blocked)
            return

        # The defender actively evading (sidestep) during the commit — remember it, so a subsequent
        # whiff is an ``evaded`` interaction (coachable) rather than a discarded spacing whiff.
        if dfn_pf.action_state is ActionState.sidestep:
            open_.defender_evaded = True

        # Attacker's offense (strike active frames or a throw) ended with no contact registered.
        if prev_atk is not None and _offense_ended(prev_atk, atk_pf):
            if open_.defender_evaded:
                # evaded (docs/04 §4.4): the defender sidestepped and the attacker whiffed. Open the
                # action window; a whiff-punish will upgrade this to ``whiff_punished``.
                self._begin_contact(fr, DefenderReaction.evaded, per_hit=None)
            else:
                # Pure spacing/neutral whiff (or a whiffed throw), defender uninvolved → discard.
                self._reset_open()

    def _begin_contact(
        self, fr: FrameRecord, reaction: DefenderReaction, *, per_hit: DefenderReaction | None
    ) -> None:
        """Resolve COMMIT -> CONTACT: fix the reaction and the real ``contact_frame``, and record
        the first string hit (docs/04 §4.2) for stun contacts. ``per_hit=None`` for a throw/evade,
        which is not a per-hit-recorded strike."""
        assert self._open is not None
        self._open.reaction = reaction
        self._open.contact_frame = fr.frame
        self._state = _State.CONTACT
        if per_hit is not None:
            self._record_string_hit(fr, per_hit)

    def _record_string_hit(self, fr: FrameRecord, per_hit: DefenderReaction) -> None:
        """Append one per-hit record for the attacker's *current* move (docs/04 §4.2). Keyed by the
        attacker's ``move_id`` so a multi-frame active move records once and a chained next hit
        (a new ``move_id``) records again."""
        assert self._open is not None
        open_ = self._open
        open_.last_hit_atk_move = fr.players[open_.attacker].move_id
        open_.string_hits.append(
            StringHitRecord(
                hit_index=len(open_.string_hits) + 1,
                defender_reaction=per_hit,
                defender_crouching=open_.defender_crouch,
            )
        )

    def _track(self, fr: FrameRecord, emitted: list[Interaction]) -> None:
        """CONTACT/FOLLOWUP: track actionable frames + the defender action window, then resolve."""
        assert self._open is not None
        open_ = self._open
        prev = self._prev

        atk_pf = fr.players[open_.attacker]
        dfn_pf = fr.players[open_.defender]

        # Keep the interaction open across consecutive hits of the same string, recording each
        # chained hit (docs/04 §4.2). Runs until the follow-up resolves — not merely until the
        # defender is actionable — so a *ducked* high (the defender is actionable and crouching, and
        # the chained high whiffs over them) is still captured as an ``evaded`` per-hit, which is
        # what tells xref the high was ducked (correct play) rather than blocked standing.
        if open_.follow_up_phase is not _FollowUpPhase.DONE:
            self._maybe_record_string_hit(fr)

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

    def _maybe_record_string_hit(self, fr: FrameRecord) -> None:
        """Record the next hit of an open string (docs/04 §4.2). A new chained attacker ``move_id``
        while the defender is still stuck is the next hit; classify it from the defender's stun this
        frame. A high whiffing over a duck (no stun, defender crouching) is recorded ``evaded`` and
        breaks the string there; a new move that has not yet connected is left for a later frame."""
        assert self._open is not None
        open_ = self._open
        atk_pf = fr.players[open_.attacker]
        dfn_pf = fr.players[open_.defender]
        if not _is_attacking(atk_pf) or atk_pf.move_id == open_.last_hit_atk_move:
            return
        per_hit = _per_hit_reaction(dfn_pf)
        if per_hit is None:
            if open_.defender_crouch:
                per_hit = (
                    DefenderReaction.evaded
                )  # ducked the high; it whiffed and broke the string
            else:
                return  # chained move not yet connected — wait for the stun frame
        self._record_string_hit(fr, per_hit)

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
                open_.follow_up_reaction_frames = fr.frame - open_.defender_actionable_frame
            elif fr.frame - open_.defender_actionable_frame > self._cfg.followup_window:
                # Window elapsed with no input → the defender did nothing (docs/04 §2).
                open_.follow_up_result = FollowUpResult.none
                open_.follow_up_phase = _FollowUpPhase.DONE
        elif open_.follow_up_phase is _FollowUpPhase.ACTED:
            atk_pf = fr.players[open_.attacker]
            prev_atk = prev.players[open_.attacker] if prev is not None else None
            if _entered_hitstun(prev_dfn, dfn_pf):
                # The defender's own follow-up got stuffed — they were hit mid-move. A counter
                # marker ⇒ ``got_counter_hit`` (the raw signal ``mashed_into_plus`` keys on, docs/04
                # §4.5); otherwise a trade. Reaction stays the original contact's (they acted).
                open_.follow_up_result = (
                    FollowUpResult.got_counter_hit if _is_counter(dfn_pf) else FollowUpResult.traded
                )
                open_.follow_up_phase = _FollowUpPhase.DONE
            elif _entered_hitstun(prev_atk, atk_pf):
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
        # A landed follow-up (a punish, a counter-hit on the mash, a trade) means one side never
        # cleanly reached neutral — advantage is N/A but the interaction is fully resolved, so emit
        # rather than wait for the cap (docs/04 §3 null-advantage path).
        return open_.follow_up_result in (
            FollowUpResult.hit,
            FollowUpResult.got_counter_hit,
            FollowUpResult.traded,
        )

    # -- emission ----------------------------------------------------------

    def _finalize(self, end_frame: int, *, truncated: str | None) -> Interaction:
        assert self._open is not None
        open_ = self._open

        # A large frame gap makes advantage-counting unreliable → emit null and let xref fall back
        # to canonical frame data (docs/04 §4.7). Otherwise the measured gap, when both sides became
        # cleanly actionable.
        observed_advantage: int | None = None
        if (
            not open_.advantage_unreliable
            and open_.attacker_actionable_frame is not None
            and open_.defender_actionable_frame is not None
        ):
            observed_advantage = open_.defender_actionable_frame - open_.attacker_actionable_frame

        follow_up = FollowUp(
            move_id=open_.follow_up_move_id,
            result=open_.follow_up_result,
            reaction_frames=open_.follow_up_reaction_frames,
        )
        outcome = self._guess_outcome(open_, observed_advantage)

        # Per-hit record only for genuine strings (>= 2 hits); a single hit emits [] (docs/04 §4.2).
        string_hits = list(open_.string_hits) if len(open_.string_hits) >= 2 else []

        notes = list(open_.extra_notes)
        if truncated:
            notes.append(f"truncated:{truncated}")

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
            string_hits=string_hits,
            notes=notes,
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
            if result in (FollowUpResult.got_counter_hit, FollowUpResult.traded):
                # The defender's press was stuffed/counter-hit — the mash-into-counter pattern, not
                # a bad punish. Stay neutral and let xref finalize ``mashed_into_ch`` (docs/04 §6).
                return Outcome.neutral
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
