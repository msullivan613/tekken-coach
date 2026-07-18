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
    join_move_live,
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


# --- live startup-primary join (brief #14) ------------------------------------
#
# The live tool reads startup accurately (a crisp contact-frame event) but on-block *too negative*
# for fast/plus moves (the attacker's return-to-idle animation lags ~10 frames). The failure the
# brief pins: Bryan standing jab `1` is i10 / +1, but live reads (startup=10, on_block=−5); the
# on_block-primary log join hard-filters by on-block and so *excludes* the real +1 `1`.


def _bryan_jab_fd() -> CharFrameData:
    """A `1` at i10/+1 plus i10 decoys at −5 that legitimately share its startup (brief #14)."""
    return _char_fd(
        _move("1", on_block=1, startup=10),  # the truth: Bryan standing jab, i10 / +1
        _move("1,2,4", on_block=-5, startup=10),  # i10 decoys at −5 (the observed on-block)
        _move("1,4,2,4", on_block=-5, startup=10),
        _move("2,4", on_block=-5, startup=10),
        _move("3", on_block=-4, startup=16),  # a slow minus move — nowhere near i10
        slug="bryan",
    )


def _live_fp(startup: int | None, on_block: int | None) -> MoveFingerprint:
    return MoveFingerprint(
        char_id=7,
        move_id=1695,
        on_block=on_block,
        startup=startup,
        blocked_samples=5,
        total_samples=5,
    )


def test_old_join_excludes_the_true_plus_move_reproducing_the_bug() -> None:
    """The on_block-primary log join drops the +1 `1` from a (startup=10, on_block=−5) read."""
    fd = _bryan_jab_fd()
    result = join_move(_live_fp(startup=10, on_block=-5), fd)
    keys = {c.framedata_key for c in result.candidates}
    assert "1" not in keys  # the bug: the real move is filtered out by on-block
    assert {
        "1,2,4",
        "1,4,2,4",
        "2,4",
    } <= keys  # only −5-ish decoys survive the hard on-block filter


def test_live_join_includes_the_true_plus_move() -> None:
    """The startup-primary live join keeps the +1 `1` despite the observed −5, and ranks it well."""
    fd = _bryan_jab_fd()
    result = join_move_live(_live_fp(startup=10, on_block=-5), fd)
    keys = [c.framedata_key for c in result.candidates]
    assert "1" in keys  # the fix: the true move survives the misleadingly-negative on-block
    assert result.candidates[0].framedata_key == "1"  # and ranks at the top
    # The i10 decoys are legitimately still offered — they share startup; the user disambiguates.
    assert set(keys) == {"1", "1,2,4", "1,4,2,4", "2,4"}
    assert "3" not in keys  # the i16 minus move is ruled out by startup


def test_live_join_soft_ranks_the_lower_bound_but_never_drops() -> None:
    """A candidate contradicting the observed on-block lower bound is ranked last, not dropped."""
    fd = _char_fd(
        _move("a", on_block=2, startup=12),  # +2 ≥ observed −5 − 1 → plausible, preferred
        _move("b", on_block=-20, startup=12),  # −20 < −6 → contradicts the lower bound, ranked last
        slug="bryan",
    )
    result = join_move_live(_live_fp(startup=12, on_block=-5), fd)
    keys = [c.framedata_key for c in result.candidates]
    assert keys == ["a", "b"]  # both offered (never dropped), plausible one first


def test_live_join_startup_tolerance_hits_at_one_frame() -> None:
    """Startup ±1 admits an i11 move for an observed i10; a 2-frame-off move is ruled out."""
    fd = _char_fd(
        _move("near", on_block=-3, startup=11),  # 1 off → in
        _move("far", on_block=-3, startup=13),  # 3 off → out
        slug="bryan",
    )
    result = join_move_live(_live_fp(startup=10, on_block=-3), fd)
    assert [c.framedata_key for c in result.candidates] == ["near"]


def test_live_join_falls_back_to_two_frames_when_one_is_empty() -> None:
    """When nothing sits within ±1, the ±2 fallback admits a 2-frame-off move (late poll)."""
    fd = _char_fd(_move("two_off", on_block=-3, startup=12), slug="bryan")
    tight = join_move_live(_live_fp(startup=10, on_block=-3), fd)
    # 2 off is outside ±1, but the fallback widens to ±2 because ±1 found nothing.
    assert [c.framedata_key for c in tight.candidates] == ["two_off"]


def test_live_join_offers_no_startup_moves_ranked_last() -> None:
    """A later-hit move with no Wavu startup can't be startup-matched — offered, ranked last."""
    fd = _char_fd(
        _move("df+1", on_block=-1, startup=13),  # startup match → tier 0
        _move("df+1,2", on_block=-8, startup=None),  # no Wavu startup → offered last, not dropped
        slug="bryan",
    )
    result = join_move_live(_live_fp(startup=13, on_block=-6), fd)
    keys = [c.framedata_key for c in result.candidates]
    assert keys == ["df+1", "df+1,2"]  # startup-matched first, no-startup move last (not dropped)


def test_live_join_no_candidate_when_startup_far_and_nothing_to_offer() -> None:
    """No move near the startup and none lacking a startup → an honest no_candidate."""
    fd = _char_fd(_move("slow", on_block=-3, startup=20), slug="bryan")
    result = join_move_live(_live_fp(startup=10, on_block=-3), fd)
    assert result.status == "no_candidate"
    assert result.candidates == []
