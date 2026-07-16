"""Frame-fingerprint join core tests (brief #6 §A.1, Stage A correctness gates).

The join is the safety-critical piece: a wrong ``move_id -> framedata_key`` must be structurally
impossible to auto-write. These tests pin the anchor (Kazuya 2145 -> df+2), prove a collision is
never auto-mapped, and exercise the startup tie-breaker and the consensus rule — all offline against
synthetic frame data so the tolerances, not any one snapshot, are under test.
"""

from __future__ import annotations

from tekken_coach.framedata.models import CharFrameData, FrameDataMove
from tekken_coach.framedata.movemap_build import (
    MoveFingerprint,
    build_fingerprint,
    entry_for,
    join_move,
)
from tekken_coach.schemas import (
    DefenderReaction,
    FollowUp,
    FollowUpResult,
    Interaction,
    InteractionContext,
    MoveProperty,
    Outcome,
    Wall,
)


def _move(
    key: str, *, on_block: int | None, startup: int | None, name: str | None = None
) -> FrameDataMove:
    return FrameDataMove(
        key=key, on_block=on_block, startup=startup, name=name, hit_level=MoveProperty.mid
    )


def _char_fd(*moves: FrameDataMove, slug: str = "kazuya") -> CharFrameData:
    return CharFrameData(char_slug=slug, char_name=slug.title(), moves={m.key: m for m in moves})


def _interaction(move_id: int, reaction: DefenderReaction, adv: int | None) -> Interaction:
    return Interaction(
        id=f"m1-r1-i{move_id}",
        match_id="M#1",
        round=1,
        start_frame=100,
        end_frame=200,
        attacker=0,
        defender=1,
        attacker_move_id=move_id,
        attacker_char_id=12,
        defender_char_id=6,
        context=InteractionContext(
            distance=1.0,
            attacker_heat=False,
            defender_heat=False,
            attacker_pressure=False,
            wall=Wall.none,
            defender_health_frac=1.0,
        ),
        defender_reaction=reaction,
        observed_advantage=adv,
        outcome=Outcome.neutral,
        follow_up=FollowUp(move_id=None, result=FollowUpResult.none, reaction_frames=None),
    )


# --- the anchor: 2145 -> df+2 (brief #6 acceptance) ---------------------------


def test_anchor_unique_on_block_auto_maps() -> None:
    """Kazuya df+2 (i15, -13, mid) uniquely matches an observed blocked -13 → auto-maps 2145."""
    fd = _char_fd(
        _move("df+2", on_block=-13, startup=15, name="Twin Pistons"),
        _move("1", on_block=-3, startup=10),
        _move("b+4", on_block=-9, startup=20),
    )
    fp = build_fingerprint(12, 2145, [_interaction(2145, DefenderReaction.blocked, -13)])
    result = join_move(fp, fd)

    assert result.status == "auto_mapped"
    assert result.framedata_key == "df+2"
    assert [c.framedata_key for c in result.candidates] == ["df+2"]


def test_second_move_at_same_on_block_is_a_collision_not_a_guess() -> None:
    """A second move at -13 makes 2145 ambiguous → collision, reported with BOTH, never written."""
    fd = _char_fd(
        _move("df+2", on_block=-13, startup=15),
        _move("hFC.4", on_block=-13, startup=20),
        _move("1", on_block=-3, startup=10),
    )
    fp = build_fingerprint(12, 2145, [_interaction(2145, DefenderReaction.blocked, -13)])
    result = join_move(fp, fd)

    assert result.status == "collision"
    assert result.framedata_key is None
    assert {c.framedata_key for c in result.candidates} == {"df+2", "hFC.4"}


def test_tolerance_admits_plus_or_minus_one_frame() -> None:
    """A move one frame off is still a candidate (±1 poll jitter), so it collides."""
    fd = _char_fd(
        _move("df+2", on_block=-13, startup=15),
        _move("f+3", on_block=-14, startup=16),  # within ±1 of observed -13
    )
    fp = build_fingerprint(12, 2145, [_interaction(2145, DefenderReaction.blocked, -13)])
    result = join_move(fp, fd)
    assert result.status == "collision"
    assert {c.framedata_key for c in result.candidates} == {"df+2", "f+3"}


def test_move_two_frames_off_is_not_a_candidate() -> None:
    """A move two frames off is outside tolerance → the observed value maps uniquely."""
    fd = _char_fd(
        _move("df+2", on_block=-13, startup=15),
        _move("f+3", on_block=-15, startup=16),  # 2 frames off — excluded
    )
    fp = build_fingerprint(12, 2145, [_interaction(2145, DefenderReaction.blocked, -13)])
    result = join_move(fp, fd)
    assert result.status == "auto_mapped"
    assert result.framedata_key == "df+2"


def test_startup_breaks_an_on_block_tie() -> None:
    """When startup is observed (Stage B), it isolates one of two same-on-block moves."""
    fd = _char_fd(
        _move("df+2", on_block=-13, startup=15),
        _move("hFC.4", on_block=-13, startup=22),
    )
    fp = MoveFingerprint(
        char_id=12, move_id=2145, on_block=-13, startup=15, blocked_samples=1, total_samples=1
    )
    result = join_move(fp, fd)
    assert result.status == "auto_mapped"
    assert result.framedata_key == "df+2"


def test_no_candidate_when_nothing_within_tolerance() -> None:
    fd = _char_fd(_move("df+2", on_block=-13, startup=15))
    fp = build_fingerprint(12, 999, [_interaction(999, DefenderReaction.blocked, -61)])
    result = join_move(fp, fd)
    assert result.status == "no_candidate"
    assert result.framedata_key is None


def test_no_signal_when_no_blocked_reading() -> None:
    fd = _char_fd(_move("df+2", on_block=-13, startup=15))
    fp = build_fingerprint(12, 999, [_interaction(999, DefenderReaction.hit, 20)])
    result = join_move(fp, fd)
    assert result.status == "no_signal"
    assert fp.on_block is None
    assert fp.blocked_samples == 0


# --- consensus (brief #6 §A.2) ------------------------------------------------


def test_consensus_uses_only_blocked_samples() -> None:
    """A move seen blocked (−13, −13) and on hit (+20) fingerprints on the blocked mode alone."""
    fp = build_fingerprint(
        12,
        2145,
        [
            _interaction(2145, DefenderReaction.blocked, -13),
            _interaction(2145, DefenderReaction.blocked, -13),
            _interaction(2145, DefenderReaction.hit, 20),
            _interaction(2145, DefenderReaction.counter_hit, 25),
        ],
    )
    assert fp.on_block == -13
    assert fp.blocked_samples == 2
    assert fp.total_samples == 4


def test_consensus_tie_is_no_consensus() -> None:
    """Two equally-frequent blocked advantages do not seed a mapping (ambiguous → None)."""
    fp = build_fingerprint(
        12,
        2145,
        [
            _interaction(2145, DefenderReaction.blocked, -13),
            _interaction(2145, DefenderReaction.blocked, -6),
        ],
    )
    assert fp.on_block is None
    assert fp.blocked_samples == 2


def test_consensus_picks_the_mode() -> None:
    fp = build_fingerprint(
        12,
        2145,
        [
            _interaction(2145, DefenderReaction.blocked, -13),
            _interaction(2145, DefenderReaction.blocked, -13),
            _interaction(2145, DefenderReaction.blocked, -6),
        ],
    )
    assert fp.on_block == -13


# --- entry construction -------------------------------------------------------


def test_entry_uses_key_as_notation_and_name_as_alias() -> None:
    fd = _char_fd(_move("df+2", on_block=-13, startup=15, name="Twin Pistons"))
    entry = entry_for(fd, "df+2")
    assert entry.notation == "df+2"
    assert entry.framedata_key == "df+2"
    assert entry.aliases == ["Twin Pistons"]


def test_entry_without_name_has_no_alias() -> None:
    fd = _char_fd(_move("df+2", on_block=-13, startup=15))
    entry = entry_for(fd, "df+2")
    assert entry.aliases == []
