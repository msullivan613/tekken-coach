"""Stage-B live fingerprinter tests (brief #6 §B).

The interactive loop is I/O and ``# pragma: no cover``; its decision core — detecting a move's
startup at contact and its on-block advantage from the blocked exchange — lives in the pure
:class:`LiveFingerprinter`, exercised here frame-by-frame with synthetic streams.
"""

from __future__ import annotations

from tekken_coach.framedata.movemap_live import (
    FrameObservation,
    LiveFingerprinter,
    LiveObservation,
    observation_from_frames,
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
) -> FrameObservation:
    return FrameObservation(
        attacker_char_id=char_id,
        attacker_move_id=move_id,
        attacker_move_frame=move_frame,
        attacker_recovering=attacker_recovering,
        defender_block_stun=block,
        defender_hit_stun=hit,
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
