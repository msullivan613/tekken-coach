"""Stage-B live fingerprinter tests (brief #6 §B).

The interactive loop is I/O and ``# pragma: no cover``; its decision core — detecting a move's
startup at contact and its on-block advantage from the blocked exchange — lives in the pure
:class:`LiveFingerprinter`, exercised here frame-by-frame with synthetic streams.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tekken_coach.framedata.movemap_build import MoveFingerprint
from tekken_coach.framedata.movemap_live import (
    FrameObservation,
    LiveFingerprinter,
    LiveObservation,
    MoveReducer,
    PollMeter,
    _reduced_on_block,
    observation_from_frames,
    reduce_observations,
)
from tekken_coach.schemas import ActionState, CounterState, HeatState, PlayerFrame


def _obs(
    *,
    move_id: int,
    move_frame: int,
    attacker_recovering: bool,
    block: bool = False,
    hit: bool = False,
    char_id: int = 12,
    frame_clock: int | None = None,
) -> FrameObservation:
    return FrameObservation(
        attacker_char_id=char_id,
        attacker_move_id=move_id,
        attacker_move_frame=move_frame,
        attacker_recovering=attacker_recovering,
        defender_block_stun=block,
        defender_hit_stun=hit,
        frame_clock=frame_clock,
    )


def _feed(fp: LiveFingerprinter, frames: list[FrameObservation]) -> list[LiveObservation]:
    return [obs for obs in (fp.feed(f) for f in frames) if obs is not None]


def test_blocked_exchange_yields_startup_and_on_block() -> None:
    """Contact on frame 15 (startup 15); attacker recovers 3f AFTER the defender → on_block −3.

    On-block advantage is negative (attacker punishable) exactly when the attacker becomes
    actionable *later* than the defender: here the defender's blockstun ends at +5 from contact and
    the attacker recovers at +8, so ``on_block = 5 - 8 = -3``.
    """
    fp = LiveFingerprinter(12)
    frames = [_obs(move_id=2145, move_frame=f, attacker_recovering=True) for f in range(15)]
    # Contact at move_frame 15 (defender enters blockstun); since_contact starts counting after.
    frames.append(_obs(move_id=2145, move_frame=15, attacker_recovering=True, block=True))
    for i in range(1, 9):
        # Defender's blockstun ends at +5 from contact; the attacker recovers at +8.
        block = i < 5
        recovering = i < 8
        frames.append(
            _obs(move_id=2145, move_frame=15 + i, attacker_recovering=recovering, block=block)
        )

    out = _feed(fp, frames)
    assert len(out) == 1
    result = out[0]
    assert result.contacted is True
    assert result.blocked is True
    assert result.fingerprint.startup == 15
    assert result.fingerprint.on_block == -3
    assert result.fingerprint.blocked_samples == 1


def test_on_block_is_measured_in_game_frames_when_frame_clock_is_present() -> None:
    """With ``frame_clock`` set, on-block is the game-frame delta, not the poll count (#12 §4).

    The real live loop polls at ~20 Hz over a 60 fps game, so a poll-count advantage is ~3x
    under-resolved. When the shared per-round counter (``frames_since_round_start``) rides along,
    recovery is timed on it directly. Here the clock jumps by **3 game frames per poll**: contact at
    clock 100, the attacker becomes actionable at clock 130, the defender leaves blockstun at clock
    135 — so ``on_block = 135 - 130 = +5`` frames, a value the ~5 intervening polls could never
    resolve. (Poll-count fallback is covered by the other tests, which leave ``frame_clock`` None.)
    """
    fp = LiveFingerprinter(12)
    frames: list[FrameObservation] = []
    # Idle (neutral/actionable), clock 90..99.
    frames += [
        _obs(move_id=32769, move_frame=5, attacker_recovering=False, frame_clock=c)
        for c in range(90, 100)
    ]
    # Startup ramp: in an attack, no contact yet, move_frame 0..15, clock 100..115.
    frames += [
        _obs(move_id=2145, move_frame=f, attacker_recovering=True, frame_clock=100 + f)
        for f in range(16)
    ]
    # Contact at move_frame 16 (clock 116): defender enters blockstun.
    frames.append(
        _obs(move_id=2145, move_frame=16, attacker_recovering=True, block=True, frame_clock=116)
    )
    # Attacker recovering + defender blocking, clock 117..129.
    frames += [
        _obs(
            move_id=2145,
            move_frame=16 + i,
            attacker_recovering=True,
            block=True,
            frame_clock=116 + i,
        )
        for i in range(1, 14)
    ]
    # Attacker becomes actionable at clock 130 (defender still blocking).
    frames.append(
        _obs(move_id=32769, move_frame=5, attacker_recovering=False, block=True, frame_clock=130)
    )
    # Defender leaves blockstun at clock 135.
    frames.append(
        _obs(move_id=32769, move_frame=5, attacker_recovering=False, block=False, frame_clock=135)
    )

    out = _feed(fp, frames)
    assert len(out) == 1
    assert out[0].fingerprint.startup == 16
    assert out[0].fingerprint.on_block == 5  # 135 (defender recovered) - 130 (attacker recovered)


def test_hit_gives_startup_but_no_on_block() -> None:
    fp = LiveFingerprinter(12)
    frames = [_obs(move_id=500, move_frame=f, attacker_recovering=True) for f in range(12)]
    frames.append(_obs(move_id=500, move_frame=12, attacker_recovering=True, hit=True))
    out = _feed(fp, frames)
    assert len(out) == 1
    assert out[0].fingerprint.startup == 12
    assert out[0].fingerprint.on_block is None
    assert out[0].blocked is False


def test_whiff_produces_no_observation() -> None:
    """A move that never connects and then changes away yields nothing."""
    fp = LiveFingerprinter(12)
    frames = [_obs(move_id=700, move_frame=f, attacker_recovering=True) for f in range(20)]
    frames += [_obs(move_id=0, move_frame=0, attacker_recovering=False) for _ in range(3)]
    assert _feed(fp, frames) == []


def test_char_change_resets() -> None:
    """Switching characters (a new match) resets the tracker rather than mixing move ids."""
    fp = LiveFingerprinter(12)
    fp.feed(_obs(move_id=2145, move_frame=3, attacker_recovering=True))
    # Different char id mid-track → reset, no crash, no observation.
    assert fp.feed(_obs(move_id=2145, move_frame=4, attacker_recovering=True, char_id=6)) is None


def _player(**kw: object) -> PlayerFrame:
    base = dict(
        char_id=12,
        move_id=2145,
        move_frame=7,
        action_state=ActionState.attack,
        health=100,
        pos=(0.0, 0.0, 0.0),
        facing=1,
        block_stun=False,
        hit_stun=False,
        counter_state=CounterState.none,
        throw_active=False,
        airborne=False,
        juggle=False,
        heat=HeatState(active=False, timer_ms=0, engager_used=False),
        rage=False,
    )
    base.update(kw)
    return PlayerFrame(**base)  # type: ignore[arg-type]


def test_observation_projection_reads_stun_from_flags_and_state() -> None:
    """observation_from_frames maps attacker recovery + defender stun from the two player frames."""
    attacker = _player(action_state=ActionState.attack, move_id=2145, move_frame=9)
    defender = _player(action_state=ActionState.blockstun, block_stun=True)
    obs = observation_from_frames(attacker, defender)
    assert obs.attacker_move_id == 2145
    assert obs.attacker_move_frame == 9
    assert obs.attacker_recovering is True
    assert obs.defender_block_stun is True
    assert obs.defender_hit_stun is False

    neutral_attacker = _player(action_state=ActionState.neutral)
    assert observation_from_frames(neutral_attacker, defender).attacker_recovering is False


# ---------------------------------------------------------------------------
# Fixture-driven regression: a real live Bryan-f3 trace (brief #12)
# ---------------------------------------------------------------------------

_F3_TRACE = Path(__file__).parent / "fixtures" / "framedata" / "f3-live-trace.jsonl"

# Raw offsets watched in the probe-state trace (brief #12).
_OFF_CHAR = "@0x168"
_OFF_MOVE_FRAME = "@0x390"
_OFF_MOVE_ID = "@0x550"
_OFF_STUN = "@0x61c"

_NEUTRAL_MOVE_ID = 0x8001  # 32769 — the observed idle/neutral id for BOTH chars (brief #12)
_BRYAN_CHAR_ID = 7
_F3_MOVE_ID = 0x640  # 1600 — Bryan f3, i16, ~+2 on block


def _load_f3_observations() -> list[FrameObservation]:
    """Pair the two players by timestamp and project raw offsets → attacker FrameObservations.

    Bryan (char 7) is the attacker; the other player is the defender. ``attacker_recovering`` is
    "attacker is in a move" (``move_id != 32769`` neutral sentinel); the defender's block-/hit-stun
    come straight from the calibrated ``stun_type`` encoding (``1`` → block_stun; ``{3,4,12}`` →
    hit_stun). The trace has no per-round frame counter, so ``frame_clock`` stays ``None`` and
    on-block falls back to poll counts (brief #12).
    """
    by_t: dict[float, dict[int, dict[str, int]]] = {}
    for line in _F3_TRACE.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_t.setdefault(row["t"], {})[row["player"]] = row["fields"]

    observations: list[FrameObservation] = []
    for t in sorted(by_t):
        players = by_t[t]
        attacker = next((f for f in players.values() if f[_OFF_CHAR] == _BRYAN_CHAR_ID), None)
        defender = next((f for f in players.values() if f[_OFF_CHAR] != _BRYAN_CHAR_ID), None)
        if attacker is None or defender is None:
            continue
        stun = defender[_OFF_STUN]
        observations.append(
            FrameObservation(
                attacker_char_id=attacker[_OFF_CHAR],
                attacker_move_id=attacker[_OFF_MOVE_ID],
                attacker_move_frame=attacker[_OFF_MOVE_FRAME],
                attacker_recovering=attacker[_OFF_MOVE_ID] != _NEUTRAL_MOVE_ID,
                defender_block_stun=stun == 1,
                defender_hit_stun=stun in {3, 4, 12},
            )
        )
    return observations


def test_f3_live_trace_maps_to_f3_not_neutral() -> None:
    """The real Bryan-f3 trace resolves to f3 (1600, i16, ~+2), never the neutral sentinel.

    Regression for brief #12: the old boundary logic (reset on any ``move_id`` change) discarded the
    real 1600 contact when the attacker returned to neutral and then fired a spurious contact for
    the lingering block-stun, emitting ``move_id=32769, startup≈2``. The redesigned machine tracks
    the move by "attacker is in an attack" and finalizes on the attacker's return to neutral.
    """
    fp = LiveFingerprinter(_BRYAN_CHAR_ID)
    emissions = _feed(fp, _load_f3_observations())

    # Every emission is f3 — the neutral sentinel is never mistaken for a move.
    assert emissions, "expected at least one f3 observation"
    assert {e.fingerprint.move_id for e in emissions} == {_F3_MOVE_ID}
    assert all(e.fingerprint.move_id != _NEUTRAL_MOVE_ID for e in emissions)

    for e in emissions:
        assert e.contacted is True
        assert e.blocked is True
        # startup reads the i16 at the sparse (~20 Hz) contact poll: 16 on the phase-aligned rep, 17
        # on the others (brief #12: move_frame reads 16-17 at every block contact).
        assert e.fingerprint.startup in (16, 17)
        assert e.fingerprint.on_block is not None
        assert 0 <= e.fingerprint.on_block <= 3

    # The i16 truth surfaces on the phase-aligned repetition.
    assert any(e.fingerprint.startup == 16 for e in emissions)


# ---------------------------------------------------------------------------
# Multi-rep reduction (brief #13 §B): min startup, modal on-block, over N reps
# ---------------------------------------------------------------------------


def _blocked_obs(
    *, move_id: int, startup: int, on_block: int, char_id: int = 12
) -> LiveObservation:
    """One blocked-contact observation, as the live loop would emit per rep."""
    return LiveObservation(
        fingerprint=MoveFingerprint(
            char_id=char_id,
            move_id=move_id,
            on_block=on_block,
            startup=startup,
            blocked_samples=1,
            total_samples=1,
        ),
        contacted=True,
        blocked=True,
    )


def test_reduce_recovers_min_startup_and_modal_on_block() -> None:
    """Truth i16/−4; per-rep startup is one-sided high and on-block is symmetric → recover (16, −4).

    Startup detection is always late (sampled the poll *after* contact), so no rep reads below 16
    and the MIN lands on the truth. On-block is a difference of two late-sampled instants, so it
    scatters both ways around −4, and its MODE converges. This is exactly why standing-3 (i16, −4)
    surfaces once the input is accurate — the value it was missing before.
    """
    reps = [
        _blocked_obs(move_id=1573, startup=16, on_block=-4),  # phase-aligned
        _blocked_obs(move_id=1573, startup=17, on_block=-6),
        _blocked_obs(move_id=1573, startup=18, on_block=-4),
        _blocked_obs(move_id=1573, startup=17, on_block=-2),
        _blocked_obs(move_id=1573, startup=19, on_block=-4),
    ]
    fp = reduce_observations(reps)
    assert fp.startup == 16  # MIN, never the biased-high mean/median
    assert fp.on_block == -4  # MODAL over the symmetric scatter
    assert fp.blocked_samples == 5
    assert fp.total_samples == 5
    assert fp.move_id == 1573
    assert fp.char_id == 12


def test_reduce_startup_min_ignores_hit_only_reps_for_on_block() -> None:
    """A hit rep contributes startup but no on-block; on-block reduces over blocked reps only."""
    hit = LiveObservation(
        fingerprint=MoveFingerprint(
            char_id=12, move_id=800, on_block=None, startup=14, blocked_samples=0, total_samples=1
        ),
        contacted=True,
        blocked=False,
    )
    reps = [
        hit,
        _blocked_obs(move_id=800, startup=15, on_block=-2),
        _blocked_obs(move_id=800, startup=16, on_block=-2),
    ]
    fp = reduce_observations(reps)
    assert fp.startup == 14  # min over all contacted reps (hit included)
    assert fp.on_block == -2  # only the two blocked reps feed on-block
    assert fp.blocked_samples == 2
    assert fp.total_samples == 3


def test_reduce_needs_at_least_one_observation() -> None:
    with pytest.raises(ValueError):
        reduce_observations([])


def test_reduced_on_block_breaks_a_modal_tie_by_median() -> None:
    """A unique mode wins; an even tie is broken by the (lower) median of the tied modes."""
    assert _reduced_on_block([-6, -4, -4, -2]) == -4  # unique mode
    assert _reduced_on_block([-5, -4, -4, -3, -3]) == -4  # tie {-4,-3} → lower-median −4
    assert _reduced_on_block([]) is None


def test_move_reducer_accumulates_until_ready_then_reduces() -> None:
    """The accumulator holds reps per move-id and is ready once the target count is reached."""
    reducer = MoveReducer(reps=3)
    for startup, on_block in ((16, -4), (18, -6), (17, -4)):
        assert reducer.is_ready(1573) is False
        reducer.add(_blocked_obs(move_id=1573, startup=startup, on_block=on_block))
    assert reducer.count(1573) == 3
    assert reducer.is_ready(1573) is True
    fp = reducer.reduce(1573)
    assert (fp.startup, fp.on_block) == (16, -4)


def test_move_reducer_pending_flush_set_respects_min_reps_and_order() -> None:
    """`pending` is the Ctrl-C flush set: move-ids with ≥ min_reps samples, in first-seen order."""
    reducer = MoveReducer(reps=5)
    reducer.add(_blocked_obs(move_id=100, startup=12, on_block=-1))
    reducer.add(_blocked_obs(move_id=100, startup=13, on_block=-1))
    reducer.add(_blocked_obs(move_id=200, startup=20, on_block=-5))  # only one rep
    assert reducer.pending(min_reps=2) == [100]  # 200 is a rep short
    assert reducer.pending(min_reps=1) == [100, 200]  # first-seen order preserved


def test_move_reducer_reps_floor_is_one() -> None:
    """A nonsensical reps<1 is floored to 1 so a single observation is immediately ready."""
    reducer = MoveReducer(reps=0)
    reducer.add(_blocked_obs(move_id=1, startup=10, on_block=0))
    assert reducer.is_ready(1) is True


def test_poll_meter_reports_achieved_hz() -> None:
    """The meter averages recorded poll-to-poll intervals into a rate; non-positive gaps ignored."""
    meter = PollMeter()
    assert meter.hz == 0.0  # nothing recorded yet
    for _ in range(4):
        meter.record(0.01)  # 10 ms between polls → 100 Hz
    meter.record(0.0)  # a stalled/duplicate frame contributes nothing
    meter.record(-0.5)  # a clock hiccup is ignored, never negative-weighting the rate
    assert meter.polls == 4
    assert meter.hz == pytest.approx(100.0)
    assert "100 Hz" in meter.summary(target_hz=120.0)
    assert "target 120 Hz" in meter.summary(target_hz=120.0)
    assert "no cap" in meter.summary(target_hz=0.0)
