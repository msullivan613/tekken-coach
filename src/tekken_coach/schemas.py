"""Data schemas for the Tekken 8 coach pipeline.

This module is the contract seam described in docs/03-data-schemas.md. It defines every
record type and enum that flows through the pipeline:

    FrameRecord (reader -> segmenter)
    Interaction (segmenter -> xref)
    LabeledInteraction (xref -> session log)
    SessionHeader (line 1 of the .jsonl session log)

All records are Pydantic v2 models: they validate on construction, enforce enum values,
and round-trip losslessly to/from JSON. Times are integer game frames unless a field name
ends in ``_ms``. IDs use lowercase snake_case.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums (docs/03 — exact names/values are the contract)
# ---------------------------------------------------------------------------


class MatchState(StrEnum):
    """FrameRecord.match_state (03 §1).

    ``unknown`` is the phase of a frame whose raw ``match_phase`` code is not in the offset table's
    map — a build whose phase offset has not been calibrated yet. It is inert everywhere it matters:
    it is not an active phase (:mod:`tekken_coach.reader.state`), so nothing buffers or records on
    it, and the segmenter treats it as "not in round". Only the *diagnostic* decode produces it; the
    capture gate refuses an unknown phase outright (docs/02 §6).
    """

    pre_round = "pre_round"
    in_round = "in_round"
    round_over = "round_over"
    match_over = "match_over"
    replay = "replay"
    menu = "menu"
    unknown = "unknown"


class ActionState(StrEnum):
    """PlayerFrame.action_state — thin normalization the reader derives cheaply (03 §1)."""

    neutral = "neutral"
    attack = "attack"
    recovery = "recovery"
    blockstun = "blockstun"
    hitstun = "hitstun"
    stagger = "stagger"
    throw_tech_window = "throw_tech_window"
    thrown = "thrown"
    airborne = "airborne"
    knockdown = "knockdown"
    wakeup = "wakeup"
    sidestep = "sidestep"
    crouch = "crouch"


class CounterState(StrEnum):
    """PlayerFrame.counter_state — the defender's hit type (03 §1)."""

    none = "none"
    counter_hit = "counter_hit"
    punish_counter = "punish_counter"


class DefenderReaction(StrEnum):
    """Interaction.defender_reaction (03 §2).

    ``stagger`` extends the docs/03 §2 enum list for the docs/04 §4.1 edge case: a stagger is
    "its own reaction," distinct from block/hit (blocked a mid that forces a stagger, or ate a
    stagger-on-normal-hit). Added under the 1.2.0 additive-minor bump alongside ``string_hits``;
    older logs never emit it, so it is backward-compatible within the major (03 §6). See the C3b
    report note on the docs/03 enum discrepancy.
    """

    blocked = "blocked"
    hit = "hit"
    counter_hit = "counter_hit"
    whiff_punished = "whiff_punished"  # defender blocked/evaded then hit back
    evaded = "evaded"  # sidestep/backdash made it whiff
    parried = "parried"
    thrown = "thrown"
    throw_broke = "throw_broke"
    traded = "traded"
    interrupted = "interrupted"  # defender's own move beat it
    stagger = "stagger"  # forced stagger (docs/04 §4.1); extends the docs/03 §2 list


class Outcome(StrEnum):
    """Interaction.outcome — from the user's coaching perspective (03 §2)."""

    no_punish = "no_punish"  # punishable, defender did nothing
    punished = "punished"
    bad_punish = "bad_punish"  # punished but suboptimal
    respected_true = "respected_true"  # respected a real gap — correct
    respected_false = "respected_false"  # respected a fake gap — could have acted
    challenged_true = "challenged_true"  # mashed into a true string — got hit
    challenged_false = "challenged_false"  # correctly challenged a gap
    ate_low = "ate_low"
    ate_mid = "ate_mid"
    mashed_into_ch = "mashed_into_ch"
    neutral = "neutral"  # nothing coachable


class FollowUpResult(StrEnum):
    """Interaction.follow_up.result (03 §2)."""

    none = "none"
    whiffed = "whiffed"
    hit = "hit"
    blocked = "blocked"
    got_counter_hit = "got_counter_hit"
    traded = "traded"


class Wall(StrEnum):
    """Interaction.context.wall — position context (03 §2)."""

    none = "none"
    near = "near"
    splat = "splat"


class StringGap(StrEnum):
    """LabeledInteraction.labels.string_gap for string situations (03 §3).

    The ``null`` case in the spec is represented by the field being ``None``; the enum
    holds only the concrete gap kinds.
    """

    duckable = "duckable"
    interruptible = "interruptible"
    true = "true"


class MoveProperty(StrEnum):
    """LabeledInteraction.labels.move_property (03 §3)."""

    high = "high"
    mid = "mid"
    low = "low"
    throw = "throw"
    unblockable = "unblockable"


class CaptureMode(StrEnum):
    """SessionHeader.capture_mode (03 §5, 01)."""

    live = "live"
    clean = "clean"


# ---------------------------------------------------------------------------
# 1. FrameRecord — one per game frame (reader -> segmenter), 03 §1
# ---------------------------------------------------------------------------


class HeatState(BaseModel):
    """Heat system state for a player (PlayerFrame.heat, 03 §1)."""

    active: bool
    timer_ms: int
    engager_used: bool  # has this player spent their Heat engager this round


class InputState(BaseModel):
    """Resolved inputs for a player this frame (PlayerFrame.input, 03 §1)."""

    dir: int  # numpad notation direction (1-9), 5 = neutral
    buttons: list[str]  # pressed attack buttons: subset of 1,2,3,4 (+ combos)


class PlayerFrame(BaseModel):
    """Per-player slice of a FrameRecord (03 §1)."""

    char_id: int  # character ID (-> name via move map, 05)
    move_id: int  # current move/animation ID (-> name via move map, 05)
    move_frame: int  # frames elapsed within the current move (0 = just started)
    action_state: ActionState
    health: int  # current HP
    pos: tuple[float, float, float]  # [x, y, z] in game units
    facing: int  # +1 faces right, -1 faces left
    block_stun: bool  # in block recovery this frame
    hit_stun: bool  # in hit recovery this frame
    counter_state: CounterState
    throw_active: bool  # executing/attempting a throw
    airborne: bool  # feet off ground (juggle-eligible)
    juggle: bool  # in an active juggle combo
    heat: HeatState
    rage: bool  # Rage available
    input: InputState | None = None  # may be null if inputs are not resolvable this frame
    # Per-round frame counter (``frames_since_round_start``), mirrored identically on both player
    # structs — ticks at 60 fps during active play, freezes while paused, and resets to ~0 at each
    # round start. The reader derives the match phase + round index from it (Stage 1 round-gating,
    # docs/02 §8) because the real T8 build exposes no usable global match-phase enum. Additive
    # minor (03 §6): consumers ignore it; ``0`` on the legacy layout, which has no such counter.
    frames_since_round_start: int = 0
    # The raw encoded state words this frame's flags were decoded from (03 §1). Tekken 8 stores
    # encoded state (``simple_move_state``, ``stun_type``, ...), not the per-flag booleans above, so
    # the reader maps value -> meaning through a calibratable data map (02 §8). Carrying the raw
    # integers makes that map debuggable — a mis-decoded state is diagnosable from a captured
    # FrameRecord alone, and the calibration protocol reads these values directly. ``None`` on the
    # legacy boolean layout. Additive minor (03 §6): consumers ignore it; nothing downstream reads
    # it, and the segmenter keys on the decoded flags, never on the raw words.
    raw_state: dict[str, int] | None = None


class FrameRecord(BaseModel):
    """The raw, uninterpreted state of one game frame (03 §1)."""

    frame: int  # game global frame counter (monotonic within a match)
    match_state: MatchState
    round: int  # 1-based round number
    timer_ms: int  # round clock remaining, ms
    players: list[PlayerFrame] = Field(min_length=2, max_length=2)  # index 0 = P1, 1 = P2


# ---------------------------------------------------------------------------
# 2. Interaction — one per segmented exchange (segmenter -> xref), 03 §2
# ---------------------------------------------------------------------------


class InteractionContext(BaseModel):
    """Positional/state context at interaction start (Interaction.context, 03 §2)."""

    distance: float  # float at interaction start
    attacker_heat: bool
    defender_heat: bool
    attacker_pressure: bool  # attacker already had frame advantage entering
    wall: Wall
    defender_health_frac: float


class FollowUp(BaseModel):
    """What the defender did in their action window after the reaction (03 §2)."""

    move_id: int | None = None  # 0 / null = nothing
    result: FollowUpResult
    reaction_frames: int | None = None  # frames until defender acted, if they acted


class StringHitRecord(BaseModel):
    """One hit of a multi-hit string, per-hit (docs/04 §4.2).

    Additive per-hit annotation that resolves the C2 duckable-high gap: the merged Interaction
    (03 §2) carried a single ``defender_reaction``, so the xref could only approximate "blocked a
    high standing" as a whole-string block. This array pins the exact hit — its per-hit reaction
    and the defender's standing-vs-crouching posture on that hit — so the xref (05 §4.1) can cross
    it with per-hit ``hit_level`` (05 §3.2) and fire ``standing_duckable_high`` (06 §4.1) precisely.

    The "hit index at which the defender's state changed" (04 §4.2) is *derivable* from the array
    (the first hit whose ``defender_reaction``/``defender_crouching`` differs), so it is not stored
    separately. Populated only for strings (>= 2 hits); ``[]`` for single-hit interactions.
    """

    hit_index: int  # 1-based hit within the string
    defender_reaction: DefenderReaction  # per-hit: blocked | hit | evaded
    defender_crouching: bool  # was the defender crouching (ducking) on this hit


class Interaction(BaseModel):
    """A discrete, bounded exchange with an attacker, defender, outcome, follow-up (03 §2)."""

    id: str  # stable id: match-round-interaction, e.g. "m3-r2-i017"
    match_id: str
    round: int
    start_frame: int
    end_frame: int
    attacker: int  # player index who initiated
    defender: int  # the other player
    attacker_move_id: int
    # Character identity (03 §2 additive, schema minor bump). The merged Interaction carried only
    # move ids; the xref (05 §4) needs both characters — the attacker's to resolve the move, the
    # defender's to pick the punisher profile. Populated by the segmenter (C3) from
    # PlayerFrame.char_id. Optional (default None) so the addition is a genuine additive-minor
    # change: an older log that omits them still loads, and an unseeded/unknown id degrades to
    # frame_data_matched:false rather than raising (03 §6, 05 §4.1).
    attacker_char_id: int | None = None
    defender_char_id: int | None = None
    context: InteractionContext
    defender_reaction: DefenderReaction
    observed_advantage: int | None = None  # int frames; negative = attacker punishable; null if N/A
    outcome: Outcome
    follow_up: FollowUp
    # Per-hit block/duck record for multi-hit strings (docs/04 §4.2). Additive minor (03 §6):
    # Optional-with-default so a pre-1.2.0 log (no per-hit array) still loads and the xref falls
    # back to the single-``defender_reaction`` approximation (05 §4.1). Empty for single-hit
    # interactions; populated (>= 2 records) only for strings.
    string_hits: list[StringHitRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # segmenter diagnostics


# ---------------------------------------------------------------------------
# 3. LabeledInteraction — xref output (xref -> session log), 03 §3
# ---------------------------------------------------------------------------


class Labels(BaseModel):
    """Ground-truth annotations from the frame-data xref (LabeledInteraction.labels, 03 §3).

    Fields derived from a frame-data match are ``None`` when ``frame_data_matched`` is
    false (the miss-tolerant path, 05 §2.3 / §4.1). See the module/report notes on which
    fields are optional and why.
    """

    frame_data_matched: bool  # did we resolve this move in the frame-data table
    on_block: int | None = None  # ground-truth on-block advantage for the move
    was_punishable: bool | None = None  # on_block <= defender's fastest punisher startup
    punish_window: int | None = None  # frames of slack (see 05)
    correct_punish: str | None = None  # recommended punish for defender's character at this range
    user_punished_correctly: bool | None = None
    in_string: bool  # was this contact part of a multi-hit string
    string_gap: StringGap | None = None  # gap kind for string situations; null otherwise
    gap_size: int | None = None  # frames of the gap, if any
    duckable_high_hit: int | None = None  # standing-blocked duck-punishable high index (05 §4.1)
    duck_punish: str | None = None  # recommended punish after ducking that high
    move_property: MoveProperty | None = None  # high | mid | low | throw | unblockable
    is_knowledge_check: bool  # did this trip a rubric pattern (06)
    knowledge_check_ids: list[str] = Field(default_factory=list)  # which rubric pattern(s), see 06


class LabeledInteraction(Interaction):
    """An Interaction plus ground-truth annotations and resolved human-readable names (03 §3)."""

    attacker_move_name: str
    attacker_char_name: str
    defender_char_name: str
    labels: Labels


# ---------------------------------------------------------------------------
# 5. Session header — line 1 of the .jsonl session log, 03 §5
# ---------------------------------------------------------------------------


class MatchSummary(BaseModel):
    """One entry of SessionHeader.matches (03 §5)."""

    match_id: str
    opponent_char: str
    result: str  # e.g. "loss" / "win" (03 does not enumerate the values)
    rounds: int


class SessionHeader(BaseModel):
    """Header record (line 1) of a session .jsonl log (03 §5)."""

    record: Literal["session_header"] = "session_header"
    schema_version: str
    created_at: str  # ISO-8601, e.g. "2026-07-07T20:14:03Z"
    capture_mode: CaptureMode
    game_version: str  # ties log to the offset/frame-data snapshot used
    framedata_snapshot: str
    user_player: int  # which player index is the user (01 §5) — coaching pivots on this
    user_char: str
    matches: list[MatchSummary] = Field(default_factory=list)  # filled/updated as matches complete
