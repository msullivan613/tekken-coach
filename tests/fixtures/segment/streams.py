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

from dataclasses import dataclass, field, replace

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
    """A player's per-frame situation, minus identity/position (which the Timeline holds).

    C3b adds the fields the edge cases need: ``counter_state`` (counter/punish-counter markers,
    docs/04 §4.5), ``throw_active`` (throw attempts, §4.3), and ``heat_active`` (mid-exchange Heat
    activation, §4.6). All default to the neutral value so the C3a poses read unchanged.
    """

    move_id: int
    action_state: ActionState
    block_stun: bool = False
    hit_stun: bool = False
    counter_state: CounterState = CounterState.none
    throw_active: bool = False
    heat_active: bool = False


# Reusable poses.
IDLE = Pose(NEUTRAL_MOVE, ActionState.neutral)
SIDESTEP = Pose(NEUTRAL_MOVE, ActionState.sidestep)
CROUCH = Pose(NEUTRAL_MOVE, ActionState.crouch)  # holding down / down-back (ducking)
BLOCKSTUN = Pose(BLOCK_MOVE, ActionState.blockstun, block_stun=True)
HITSTUN = Pose(HITSTUN_MOVE, ActionState.hitstun, hit_stun=True)

# C3b edge-case poses.
STAGGER_MOVE = 940
THROWN_MOVE = 960
STAGGER = Pose(STAGGER_MOVE, ActionState.stagger)  # forced stagger (§4.1)
THROW_TECH = Pose(NEUTRAL_MOVE, ActionState.throw_tech_window)  # break window open (§4.3)
THROWN = Pose(THROWN_MOVE, ActionState.thrown)  # caught by the throw (§4.3)
KNOCKDOWN = Pose(THROWN_MOVE, ActionState.knockdown)  # on the ground (§4.3)
WAKEUP = Pose(THROWN_MOVE, ActionState.wakeup)  # getting up (§4.3)
COUNTER_HITSTUN = Pose(  # got counter-hit — hitstun carrying a counter marker (§4.5)
    HITSTUN_MOVE, ActionState.hitstun, hit_stun=True, counter_state=CounterState.counter_hit
)


def attack(move_id: int) -> Pose:
    return Pose(move_id, ActionState.attack)


def throw(move_id: int) -> Pose:
    """An active throw attempt (``throw_active``); action_state ``attack`` (§4.3)."""
    return Pose(move_id, ActionState.attack, throw_active=True)


def recovery(move_id: int) -> Pose:
    """Post-active recovery of the same move (same ``move_id``, no longer ``attack``)."""
    return Pose(move_id, ActionState.recovery)


def in_heat(pose: Pose) -> Pose:
    """The same pose but with the player in Heat (docs/04 §4.6)."""
    return replace(pose, heat_active=True)


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
            counter_state=pose.counter_state,
            throw_active=pose.throw_active,
            airborne=False,
            juggle=False,
            heat=HeatState(active=pose.heat_active, timer_ms=0, engager_used=False),
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

    def skip(self, count: int) -> Timeline:
        """Advance the frame counter without emitting frames — a dropped-frame poll gap (§4.7).
        State continuity is the segmenter's problem; the stream simply jumps ``count`` frames."""
        self._frame += count
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


# ---------------------------------------------------------------------------
# C3b edge cases (docs/04 §4.1–§4.8). One hand-authored stream per case.
# ---------------------------------------------------------------------------


def stagger_on_block() -> list[FrameRecord]:
    """§4.1: a move that forces a stagger on block → its own reaction, not blocked/hit."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(801), IDLE)  # 103-106 commit at 103
    t.step(8, recovery(801), STAGGER)  # 107-114 stagger at 107 (8f)
    t.step(6, recovery(801), IDLE)  # 115-120 defender actionable at 115 (stagger ends)
    t.step(28, IDLE, IDLE)  # 121-148 attacker actionable 121; follow-up window elapses
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 149
    return t.build()


def string_blocked_standing() -> list[FrameRecord]:
    """§4.2 golden: a mid→high→mid string jailed and blocked **standing** (never ducked). Three
    per-hit records, all ``blocked`` + not crouching — hit 2 (the high) is a duck-punishable high
    the user blocked standing, which xref (05 §4.1) flags. Entry move 100 == Paul ``df+1,1,2``."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(3, attack(100), IDLE)  # 103-105 commit hit 1 (entry) at 103
    t.step(3, recovery(100), BLOCKSTUN)  # 106-108 hit 1 (mid) blocked standing at 106
    t.step(3, attack(1001), BLOCKSTUN)  # 109-111 hit 2 (high) blocked standing at 109 (jails)
    t.step(3, attack(1002), BLOCKSTUN)  # 112-114 hit 3 (mid) blocked at 112 (jails)
    t.step(4, recovery(1002), BLOCKSTUN)  # 115-118 blockstun tail
    t.step(9, recovery(1002), IDLE)  # 119-127 defender actionable at 119
    t.step(13, IDLE, IDLE)  # 128-140 attacker actionable 128; follow-up window elapses
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 141
    return t.build()


def string_ducked_high() -> list[FrameRecord]:
    """§4.2: the same string, but the defender **ducks** the high (crouch-blocks hit 1, holds down
    so hit 2 whiffs), then punishes. Hit 2 is recorded ``evaded`` (not blocked standing), which is
    exactly the signal that tells xref this was correct play → no duck flag."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, CROUCH)  # 100-102 defender holds down-back (ducking)
    t.step(3, attack(100), CROUCH)  # 103-105 commit hit 1; defender crouching
    t.step(3, recovery(100), BLOCKSTUN)  # 106-108 hit 1 (mid) crouch-blocked at 106
    t.step(
        3, attack(1001), CROUCH
    )  # 109-111 hit 2 (high) whiffs over the duck → evaded, string breaks
    t.step(3, recovery(1001), CROUCH)  # 112-114 attacker recovers; defender still ducking
    t.step(2, recovery(1001), IDLE)  # 115-116 defender stands, actionable at 115
    t.step(10, recovery(1001), attack(966))  # 117-126 defender punishes at 117
    t.step(3, IDLE, attack(966))  # 127-129 attacker actionable 127
    t.step(5, HITSTUN, attack(966))  # 130-134 punish connects at 130
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 135
    return t.build()


def string_interrupted() -> list[FrameRecord]:
    """§4.2: a two-hit string the defender blocks, then interrupts in the gap after hit 2. The
    string closes when the defender becomes actionable between hits (they acted in a real gap)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(3, attack(110), IDLE)  # 103-105 commit hit 1
    t.step(3, recovery(110), BLOCKSTUN)  # 106-108 hit 1 blocked at 106
    t.step(3, attack(1101), BLOCKSTUN)  # 109-111 hit 2 blocked at 109 (jails)
    t.step(3, recovery(1101), IDLE)  # 112-114 gap: defender actionable at 112
    t.step(10, recovery(1101), attack(962))  # 115-124 defender interrupts at 115
    t.step(3, IDLE, attack(962))  # 125-127 attacker actionable 125
    t.step(5, HITSTUN, attack(962))  # 128-132 interrupt connects at 128
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 133
    return t.build()


def throw_broke() -> list[FrameRecord]:
    """§4.3: an attacker throw the defender breaks in the tech window → throw_broke."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(1, throw(970), IDLE)  # 103 throw commit
    t.step(3, throw(970), THROW_TECH)  # 104-106 defender in the break window
    t.step(1, throw(970), IDLE)  # 107 defender broke → actionable at 107
    t.step(26, IDLE, IDLE)  # 108-133 both neutral; window elapses
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 134
    return t.build()


def thrown() -> list[FrameRecord]:
    """§4.3: an attacker throw the defender fails to break → thrown."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(1, throw(970), IDLE)  # 103 throw commit
    t.step(2, throw(970), THROW_TECH)  # 104-105 tech window (not broken)
    t.step(10, throw(970), THROWN)  # 106-115 thrown at 106
    t.step(28, IDLE, IDLE)  # 116-143 both recover; window elapses
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 144
    return t.build()


def knockdown_wakeup() -> list[FrameRecord]:
    """§4.3: a hit that knocks down. The follow-up window extends to the **wakeup**-actionable
    frame (not a mid-knockdown frame), so ``observed_advantage`` reflects the whole oki window."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(820), IDLE)  # 103-106 commit at 103
    t.step(6, recovery(820), HITSTUN)  # 107-112 hit at 107
    t.step(10, IDLE, KNOCKDOWN)  # 113-122 knocked down; attacker actionable 113
    t.step(10, IDLE, WAKEUP)  # 123-132 waking up (still not actionable)
    t.step(24, IDLE, IDLE)  # 133-156 defender wakeup-actionable at 133; window elapses
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 157
    return t.build()


def counter_hit() -> list[FrameRecord]:
    """§4.5: the defender presses and is counter-hit by the attack → counter_hit reaction."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(5, attack(850), IDLE)  # 103-107 commit at 103
    t.step(8, recovery(850), COUNTER_HITSTUN)  # 108-115 counter-hit at 108
    t.step(30, IDLE, IDLE)  # 116-145 both recover; window elapses
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 146
    return t.build()


def mashed_into_counter() -> list[FrameRecord]:
    """§4.5: a plus-on-block move blocked, the defender mashes, and their follow-up gets
    counter-hit → follow_up.result == got_counter_hit (the raw signal mashed_into_plus keys on)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(860), IDLE)  # 103-106 commit at 103 (a plus move)
    t.step(6, recovery(860), BLOCKSTUN)  # 107-112 blocked at 107
    t.step(1, IDLE, BLOCKSTUN)  # 113 attacker actionable 113 (plus)
    t.step(2, IDLE, IDLE)  # 114-115 defender actionable 114
    t.step(5, attack(861), attack(965))  # 116-120 defender mashes 965; attacker re-presses 861
    t.step(6, recovery(861), COUNTER_HITSTUN)  # 121-126 defender's mash counter-hit at 121
    t.step(20, IDLE, IDLE)  # 127-146 recover
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 147
    return t.build()


def heat_activation() -> list[FrameRecord]:
    """§4.6: the attacker activates Heat mid-exchange → noted (it shifts advantage). Context heat
    at start is false; the activation happens after contact."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(870), IDLE)  # 103-106 commit at 103 (not in Heat)
    t.step(6, in_heat(recovery(870)), BLOCKSTUN)  # 107-112 Heat activates at 107; blocked at 107
    t.step(2, in_heat(IDLE), IDLE)  # 113-114 defender actionable 113; attacker actionable 113
    t.step(25, in_heat(IDLE), IDLE)  # 115-139 window elapses
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 140
    return t.build()


def dropped_frames_tolerated() -> list[FrameRecord]:
    """§4.7: a small poll gap (≤ threshold) mid-blockstun. State continuity is assumed, the gap is
    noted, and ``observed_advantage`` is still measured."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(800), IDLE)  # 103-106 commit at 103
    t.step(6, recovery(800), BLOCKSTUN)  # 107-112 blocked at 107
    t.skip(2)  # frames 113-114 dropped (gap of 2, tolerated)
    t.step(13, recovery(800), IDLE)  # 115-127 defender actionable at 115
    t.step(10, IDLE, IDLE)  # 128-137 attacker actionable at 128
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 138
    return t.build()


def dropped_frames_unreliable() -> list[FrameRecord]:
    """§4.7: a poll gap larger than the threshold. The interaction is still emitted, but
    ``observed_advantage`` is null (frame-counting across the gap is unreliable)."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(4, attack(800), IDLE)  # 103-106 commit at 103
    t.step(6, recovery(800), BLOCKSTUN)  # 107-112 blocked at 107
    t.skip(6)  # frames 113-118 dropped (gap of 6, beyond threshold)
    t.step(13, recovery(800), IDLE)  # 119-131 defender actionable at 119
    t.step(10, IDLE, IDLE)  # 132-141 attacker actionable at 132
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 142
    return t.build()


def attacker_pressure_carry() -> list[FrameRecord]:
    """Item 9b: two exchanges in one round. The first leaves the attacker plus (a clean hit), so
    the second opens with ``context.attacker_pressure == True`` — the frame-advantage carry."""
    t = Timeline(KAZUYA, DEFENDER, 0.0, NEAR_X)
    # Exchange 1: a clean hit, attacker +8 → carries pressure to the next commit.
    t.step(3, IDLE, IDLE)  # 100-102
    t.step(5, attack(810), IDLE)  # 103-107 commit at 103
    t.step(8, recovery(810), HITSTUN)  # 108-115 hit at 108
    t.step(8, IDLE, HITSTUN)  # 116-123 attacker actionable 116
    t.step(24, IDLE, IDLE)  # 124-147 defender actionable 124 (+8); window elapses, emits ~140
    # Exchange 2: attacker commits again while holding pressure.
    t.step(4, attack(820), IDLE)  # 148-151 commit at 148 (attacker_pressure True)
    t.step(8, recovery(820), BLOCKSTUN)  # 152-159 blocked at 152
    t.step(2, recovery(820), IDLE)  # 160-161 defender actionable 160
    t.step(22, IDLE, IDLE)  # 162-183 attacker actionable 162; window elapses (no truncation)
    t.step(1, IDLE, IDLE, match_state=MatchState.round_over)  # 184
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
    # C3b edge cases (docs/04 §4).
    "stagger_on_block": stagger_on_block,
    "string_blocked_standing": string_blocked_standing,
    "string_ducked_high": string_ducked_high,
    "string_interrupted": string_interrupted,
    "throw_broke": throw_broke,
    "thrown": thrown,
    "knockdown_wakeup": knockdown_wakeup,
    "counter_hit": counter_hit,
    "mashed_into_counter": mashed_into_counter,
    "heat_activation": heat_activation,
    "dropped_frames_tolerated": dropped_frames_tolerated,
    "dropped_frames_unreliable": dropped_frames_unreliable,
    "attacker_pressure_carry": attacker_pressure_carry,
}
