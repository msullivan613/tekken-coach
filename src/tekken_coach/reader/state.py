"""Match/replay-state classification (docs/01 §4.3).

The reader reads two raw game codes each frame — the match *phase* (pre-round / in-round / ...)
and the game *mode* (offline match / online match / replay / practice / menu) — and normalizes
them into (a) the ``FrameRecord.match_state`` enum the segmenter consumes, and (b) a coarse
:class:`StateSignal` for C6's capture-mode policy.

This module **classifies only**. It does not implement mode policy (C6 owns the live/clean
triggers and cadence, docs/01 §5). It exposes enough that clean mode can *refuse to buffer on an
online-match state* (``StateSignal.online``, defense-in-depth per docs/01 §4.3) and that live
mode can find the match-active / match-over transitions (``StateSignal.kind`` + ``match_state``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tekken_coach.schemas import MatchState

# Game-mode categories, as named in the offset table's ``state_codes.game_mode`` (docs/02 §3).
MODE_REPLAY = "replay"
MODE_ONLINE = "online_match"
MODE_OFFLINE = "offline_match"
MODE_PRACTICE = "practice"
MODE_IDLE = "idle"

# Phases that mean "a match is actively in progress" (not menu/idle).
_ACTIVE_PHASES = frozenset(
    {MatchState.pre_round, MatchState.in_round, MatchState.round_over, MatchState.match_over}
)


class SignalKind(StrEnum):
    """Coarse capture-relevant state (docs/01 §4.3)."""

    live_match = "live_match"  # a real match is being played (online or offline/practice)
    replay_playback = "replay_playback"  # an offline replay is playing back
    idle = "idle"  # menu / no active match — nothing to buffer


@dataclass(frozen=True)
class StateSignal:
    """What C6 needs to gate capture without knowing offset internals (docs/01 §4.3, §4.1).

    * ``kind`` — live_match / replay_playback / idle.
    * ``online`` — is this an *online* session? Clean mode refuses to buffer when true, as
      defense-in-depth against a misconfigured live capture touching ranked (docs/01 §4.3).
    * ``match_state`` — the normalized ``FrameRecord.match_state`` phase, so live mode can find
      the match-active -> match-over transition (docs/01 §3.1).
    """

    kind: SignalKind
    online: bool
    match_state: MatchState

    @property
    def should_buffer_clean(self) -> bool:
        """Whether clean (offline replay) capture may buffer this frame (docs/01 §4.3).

        True only for offline replay playback. Never for an online match state (the reader's
        defense-in-depth refusal); C6 still owns the actual policy.
        """
        return self.kind is SignalKind.replay_playback and not self.online


def classify_state(match_state: MatchState, game_mode: str) -> StateSignal:
    """Classify the current frame's match/replay/idle state from the normalized codes."""
    online = game_mode == MODE_ONLINE
    if game_mode == MODE_REPLAY:
        kind = SignalKind.replay_playback
    elif game_mode in (MODE_ONLINE, MODE_OFFLINE, MODE_PRACTICE) and match_state in _ACTIVE_PHASES:
        kind = SignalKind.live_match
    else:
        kind = SignalKind.idle
    return StateSignal(kind=kind, online=online, match_state=match_state)
