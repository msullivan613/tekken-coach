"""Hand-authored ``FrameRecord`` streams for the C3a segmenter goldens (docs/04 §7).

There is no memory reader yet (C4), so these streams are authored by hand: a small
:class:`Timeline` builder emits one ``FrameRecord`` per frame from human-readable *poses* (an
``action_state`` + the block/hit-stun flags), auto-tracking ``move_frame`` continuity (0 on a new
``move_id``, incrementing while the same move persists — the docs/03 §1 "new move vs next frame"
signal). Each stream is well-framed: a couple of neutral in-round frames, the exchange, then a
``round_over`` frame.

The exact frame numbers below are chosen so the derived values are clean and hand-checkable — e.g.
a 13-frame block disadvantage in :func:`blocked_no_punish` really is ``defender_actionable (120) −
attacker_actionable (133) = −13``. The companion goldens in ``goldens/`` freeze the resulting
``Interaction`` list; ``test_segmenter.py`` also asserts the load-bearing fields directly so the
goldens are not merely self-fulfilling.

Note (documented simplification, docs/04 §3): where a defender *punishes*, the stream lets the
attacker reach neutral for a few frames before the punish connects, so both actionable frames are
observable and ``observed_advantage`` stays measurable. A frame-perfect punish that interrupts
recovery (advantage then ``null``) is refined with real captures / C3b.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tekken_coach.schemas import (
    ActionState,
    CounterState,
    FrameRecord,
    HeatState,
    MatchState,
    PlayerFrame,
)

FULL_HP = 150

# Move ids used purely to give moves identity in a stream (the segmenter knows ids, not names).
NEUTRAL_MOVE = 0
BLOCK_MOVE = 1
HITSTUN_MOVE = 950


@dataclass(frozen=True)
class Pose:
    """A player's per-frame situation, minus identity/position (which the Timeline holds)."""

    move_id: int
    action_state: ActionState
    block_stun: bool = False
    hit_stun: bool = False


# Reusable poses.
IDLE = Pose(NEUTRAL_MOVE, ActionState.neutral)
SIDESTEP = Pose(NEUTRAL_MOVE, ActionState.sidestep)
BLOCKSTUN = Pose(BLOCK_MOVE, ActionState.blockstun, block_stun=True)
HITSTUN = Pose(HITSTUN_MOVE, ActionState.hitstun, hit_stun=True)


def attack(move_id: int) -> Pose:
    return Pose(move_id, ActionState.attack)


def recovery(move_id: int) -> Pose:
    """Post-active recovery of the same move (same ``move_id``, no longer ``attack``)."""
    return Pose(move_id, ActionState.recovery)


@dataclass
class Timeline:
    """Builds a frame stream for one attacker(index 0)-vs-defender(index 1) exchange."""

    atk_char: int
    dfn_char: int
    atk_x: float
    dfn_x: float
    start_frame: int = 100
    round: int = 1
    frames: list[FrameRecord] = field(default_factory=list)
    _frame: int = field(init=False)
    _move: dict[int, int | None] = field(init=False)
    _mf: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self._frame = self.start_frame
        self._move = {0: None, 1: None}
        self._mf = {0: 0, 1: 0}

    def _player(self, idx: int, char: int, x: float, pose: Pose) -> PlayerFrame:
        if pose.move_id != self._move[idx]:
            self._mf[idx] = 0
            self._move[idx] = pose.move_id
        else:
            self._mf[idx] += 1
        facing = 1 if self.atk_x <= self.dfn_x else -1
        if idx == 1:
            facing = -facing
        return PlayerFrame(
            char_id=char,
            move_id=pose.move_id,
            move_frame=self._mf[idx],
            action_state=pose.action_state,
            health=FULL_HP,
            pos=(x, 0.0, 0.0),
            facing=facing,
            block_stun=pose.block_stun,
            hit_stun=pose.hit_stun,
            counter_state=CounterState.none,
            throw_active=False,
            airborne=False,
            juggle=False,
            heat=HeatState(active=False, timer_ms=0, engager_used=False),
            rage=False,
            input=None,
        )

    def step(
        self,
        count: int,
        atk: Pose,
        dfn: Pose,
        *,
        match_state: MatchState = MatchState.in_round,
    ) -> Timeline:
        for _ in range(count):
            p_atk = self._player(0, self.atk_char, self.atk_x, atk)
            p_dfn = self._player(1, self.dfn_char, self.dfn_x, dfn)
            self.frames.append(
                FrameRecord(
                    frame=self._frame,
                    match_state=match_state,
                    round=self.round,
                    timer_ms=40000,
                    players=[p_atk, p_dfn],
                )
            )
            self._frame += 1
        return self

    def build(self) -> list[FrameRecord]:
        return self.frames


# Test char ids (real ids arrive with the reader, C4): Kazuya is 12 (real), 7 is a stand-in.
KAZUYA = 12
DEFENDER = 7

# In-range / out-of-range positions relative to DEFAULT_CONFIG.threat_range (2.5 units).
NEAR_X = 1.5
FAR_X = 4.0


def neutral_dead_time() -> list[FrameRecord]:
    """Both players idle in range; no attack ever commits → no interaction (docs/04 §1)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(15, IDLE, IDLE)  # 100-114
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 115
    return t.build()


def blocked_no_punish() -> list[FrameRecord]:
    """A −13 blocked, defender presses nothing → blocked / no_punish (docs/04 §3)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102 approach
    t.step(4, attack(800), IDLE)  # 103-106 commit at 103
    t.step(13, recovery(800), BLOCKSTUN)  # 107-119 contact (block) at 107, 13f blockstun
    t.step(13, recovery(800), IDLE)  # 120-132 defender actionable at 120
    t.step(10, IDLE, IDLE)  # 133-142 attacker actionable at 133
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 143
    return t.build()


def blocked_punished() -> list[FrameRecord]:
    """A −13 blocked, defender punishes → blocked / punished, follow_up hits (docs/04 §3)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(800), IDLE)  # 103-106 commit
    t.step(13, recovery(800), BLOCKSTUN)  # 107-119 block at 107
    t.step(3, recovery(800), IDLE)  # 120-122 defender actionable at 120
    t.step(10, recovery(800), attack(900))  # 123-132 defender acts at 123 (reaction 3)
    t.step(3, IDLE, attack(900))  # 133-135 attacker actionable at 133
    t.step(5, HITSTUN, attack(900))  # 136-140 punish connects at 136
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 141
    return t.build()


def clean_hit() -> list[FrameRecord]:
    """An open hit leaving the attacker plus → hit / neutral guess (docs/04 §3, §4.1)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(5, attack(810), IDLE)  # 103-107 commit at 103
    t.step(8, recovery(810), HITSTUN)  # 108-115 hit at 108
    t.step(8, IDLE, HITSTUN)  # 116-123 attacker actionable at 116
    t.step(22, IDLE, IDLE)  # 124-145 defender actionable at 124
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 146
    return t.build()


def spacing_whiff() -> list[FrameRecord]:
    """An attack thrown out of threat range → never commits → no interaction (docs/04 §4.4)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, FAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(10, attack(820), IDLE)  # 103-112 out of range: no commit
    t.step(3, IDLE, IDLE)  # 113-115
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 116
    return t.build()


def in_range_whiff_discard() -> list[FrameRecord]:
    """An in-range attack that misses with the defender uninvolved → discarded (docs/04 §2)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(6, attack(840), IDLE)  # 103-108 commit at 103
    t.step(7, recovery(840), IDLE)  # 109-115 whiff at 109, no evade → discard
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 116
    return t.build()


def sidestep_whiff_punish() -> list[FrameRecord]:
    """Defender sidesteps a linear attack then punishes → evaded upgraded to whiff_punished."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(1, attack(830), IDLE)  # 103 commit
    t.step(3, attack(830), SIDESTEP)  # 104-106 defender sidesteps
    t.step(2, attack(830), IDLE)  # 107-108 attacker still in active frames
    t.step(9, recovery(830), IDLE)  # 109-117 whiff/evade at 109, defender actionable at 109
    t.step(8, recovery(830), attack(930))  # 118-125 defender whiff-punishes at 118 (reaction 9)
    t.step(5, IDLE, attack(930))  # 126-130 attacker actionable at 126
    t.step(5, HITSTUN, attack(930))  # 131-135 punish connects at 131
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 136
    return t.build()


def round_boundary_truncation() -> list[FrameRecord]:
    """A round ends mid-blockstun → the open interaction is truncated, not lost (docs/04 §4.8)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(800), IDLE)  # 103-106 commit
    t.step(6, recovery(800), BLOCKSTUN)  # 107-112 block at 107, still open
    t.step(1, recovery(800), BLOCKSTUN, match_state=MatchState.round_over)  # 113 round ends
    return t.build()


# Every stream, keyed by name — property tests iterate this so no fixture is skipped.
ALL_STREAMS = {
    "neutral_dead_time": neutral_dead_time,
    "blocked_no_punish": blocked_no_punish,
    "blocked_punished": blocked_punished,
    "clean_hit": clean_hit,
    "spacing_whiff": spacing_whiff,
    "in_range_whiff_discard": in_range_whiff_discard,
    "sidestep_whiff_punish": sidestep_whiff_punish,
    "round_boundary_truncation": round_boundary_truncation,
}
