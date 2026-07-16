"""Passive miner + merge tests (brief #6 §A.2-4, Stage A correctness gates).

Two halves, per the brief:

* **Real-framedata auto-map + merge/idempotency** — a synthetic session whose Paul move is
  on-block-unique in the *committed* snapshot (−25 → ``1,2,3``) proves the miner writes a
  Wavu-verified mapping, that re-running is a byte-for-byte no-op, and that a curated entry is
  preserved without ``--overwrite``.
* **live-run-1 slice** — the miner over a committed trim of the real 5.02.01 log: header-driven
  character resolution, zero *wrong* auto-maps (on-block alone collides, and a collision is reported
  not guessed), and Bryan/Xiaoyu reported as needs-framedata / unresolved rather than crashing.
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.framedata.loader import load_char_move_map, load_current_framedata
from tekken_coach.framedata.movemap_miner import (
    merge_report,
    mine_session,
    resolve_char_ids,
)
from tekken_coach.schemas import (
    CaptureMode,
    DefenderReaction,
    FollowUp,
    FollowUpResult,
    InteractionContext,
    LabeledInteraction,
    Labels,
    MatchSummary,
    Outcome,
    SessionHeader,
    Wall,
)
from tekken_coach.session.store import LoadedSession, load_session

REPO_ROOT = Path(__file__).parent.parent
ASSETS = REPO_ROOT / "assets"
SLICE = REPO_ROOT / "tests" / "fixtures" / "framedata" / "live-run-1-slice.jsonl"


def _labeled(
    *,
    move_id: int,
    attacker: int,
    attacker_char_id: int,
    reaction: DefenderReaction,
    adv: int | None,
    match_id: str = "M#1",
    attacker_char_name: str = "char_id:0",
) -> LabeledInteraction:
    defender = 1 - attacker
    return LabeledInteraction(
        id=f"m1-r1-i{move_id}",
        match_id=match_id,
        round=1,
        start_frame=100,
        end_frame=200,
        attacker=attacker,
        defender=defender,
        attacker_move_id=move_id,
        attacker_char_id=attacker_char_id,
        defender_char_id=7 if attacker_char_id != 7 else 0,
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
        attacker_move_name=f"move_id:{move_id}",
        attacker_char_name=attacker_char_name,
        defender_char_name="char_id:7",
        labels=Labels(frame_data_matched=False, in_string=False, is_knowledge_check=False),
    )


def _session(*interactions: LabeledInteraction, user_char: str = "bryan") -> LoadedSession:
    header = SessionHeader(
        schema_version="1.2.0",
        created_at="2026-07-15T00:00:00Z",
        capture_mode=CaptureMode.live,
        game_version="5.02.01",
        framedata_snapshot="2026-07-07",
        user_player=0,
        user_char=user_char,
        matches=[MatchSummary(match_id="M#1", opponent_char="paul", result="win", rounds=3)],
    )
    return LoadedSession(header=header, interactions=list(interactions))


# --- header-driven character resolution (brief #6 §A.3) -----------------------


def test_resolve_char_ids_from_header_not_placeholder_names() -> None:
    """char_id -> name comes from the header even though the resolved names are placeholders."""
    session = _session(
        _labeled(
            move_id=7777, attacker=1, attacker_char_id=0, reaction=DefenderReaction.blocked, adv=-25
        ),
        _labeled(
            move_id=1695, attacker=0, attacker_char_id=7, reaction=DefenderReaction.hit, adv=None
        ),
    )
    names = resolve_char_ids(session.header, session.interactions)
    assert names == {0: "paul", 7: "bryan"}


# --- auto-map against the committed Paul snapshot -----------------------------


def test_unique_paul_move_auto_maps_to_real_key(tmp_path: Path) -> None:
    """A Paul move observed at −25 (on-block-unique in the snapshot) auto-maps to ``1,2,3``."""
    snapshot = load_current_framedata(ASSETS / "framedata")
    session = _session(
        _labeled(
            move_id=7777, attacker=1, attacker_char_id=0, reaction=DefenderReaction.blocked, adv=-25
        ),
    )
    report = mine_session(session, snapshot, only_char="paul")
    assert [(g.move_id, g.status) for g in report.auto_mapped] == [(7777, "auto_mapped")]
    assert report.auto_mapped[0].join is not None
    assert report.auto_mapped[0].join.framedata_key == "1,2,3"

    merges = merge_report(report, snapshot, movemap_dir=tmp_path)
    assert len(merges) == 1
    assert merges[0].created is True
    assert merges[0].written == [7777]

    written = load_char_move_map(tmp_path / "paul.json")
    assert written.char_id == 0  # learned the memory char_id from the observed group
    entry = written.get(7777)
    assert entry is not None and entry.framedata_key == "1,2,3"
    # every written mapping is Wavu-consistent — the key exists in the snapshot
    assert "1,2,3" in snapshot.get_char("paul").moves  # type: ignore[union-attr]


def test_merge_is_idempotent(tmp_path: Path) -> None:
    """Re-running the same mine+merge is a byte-for-byte no-op; the second pass preserves."""
    snapshot = load_current_framedata(ASSETS / "framedata")
    session = _session(
        _labeled(
            move_id=7777, attacker=1, attacker_char_id=0, reaction=DefenderReaction.blocked, adv=-25
        ),
    )
    report = mine_session(session, snapshot, only_char="paul")

    merge_report(report, snapshot, movemap_dir=tmp_path)
    first = (tmp_path / "paul.json").read_bytes()

    again = merge_report(report, snapshot, movemap_dir=tmp_path)
    assert (tmp_path / "paul.json").read_bytes() == first
    assert again[0].created is False
    assert again[0].preserved == [7777]
    assert again[0].written == []


def test_existing_curated_entry_is_preserved_without_overwrite(tmp_path: Path) -> None:
    """A curated (even wrong) entry wins unless --overwrite; then it is replaced."""
    snapshot = load_current_framedata(ASSETS / "framedata")
    # Seed a curated (deliberately wrong) mapping for 7777.
    (tmp_path / "paul.json").write_text(
        '{"char_id": 0, "char_name": "Paul", "game_version": "2.01.01", "partial": true, '
        '"moves": {"7777": {"notation": "df+2", "aliases": [], "framedata_key": "df+2"}}, '
        '"framedata_keys": []}',
        encoding="utf-8",
    )
    session = _session(
        _labeled(
            move_id=7777, attacker=1, attacker_char_id=0, reaction=DefenderReaction.blocked, adv=-25
        ),
    )
    report = mine_session(session, snapshot, only_char="paul")

    kept = merge_report(report, snapshot, movemap_dir=tmp_path)
    assert kept[0].preserved == [7777]
    assert load_char_move_map(tmp_path / "paul.json").get(7777).framedata_key == "df+2"  # type: ignore[union-attr]

    forced = merge_report(report, snapshot, movemap_dir=tmp_path, overwrite=True)
    assert forced[0].overwritten == [7777]
    assert load_char_move_map(tmp_path / "paul.json").get(7777).framedata_key == "1,2,3"  # type: ignore[union-attr]


# --- the live-run-1 slice (brief #6 tests) ------------------------------------


def test_live_run_1_slice_makes_no_wrong_auto_map(tmp_path: Path) -> None:
    """On the real 5.02.01 slice: every auto-map (if any) is Wavu-consistent; no wrong guess."""
    session = load_session(SLICE)
    snapshot = load_current_framedata(ASSETS / "framedata")
    paul = snapshot.get_char("paul")
    assert paul is not None

    report = mine_session(session, snapshot)
    # On-block alone is coarse, so this log yields collisions, not auto-maps — but the invariant we
    # guard is stronger: whatever DID auto-map must exist in Paul's snapshot (never a guess).
    for group in report.auto_mapped:
        assert group.char_slug == "paul"
        assert group.join is not None and group.join.framedata_key in paul.moves

    merges = merge_report(report, snapshot, movemap_dir=tmp_path)
    for merge in merges:
        if merge.written:
            assert (tmp_path / f"{merge.char_slug}.json").exists()


def test_live_run_1_slice_reports_collisions_and_missing_framedata() -> None:
    """The slice surfaces Paul collisions, and Bryan/Xiaoyu as needs-framedata / unresolved."""
    session = load_session(SLICE)
    snapshot = load_current_framedata(ASSETS / "framedata")
    report = mine_session(session, snapshot)

    by_move = {g.move_id: g for g in report.groups}
    # A known Paul collision (on_block −48: FC.D,U+4 / DPD.df+3+4) is reported, never auto-mapped.
    assert by_move[1809].status == "collision"
    assert by_move[1809].join is not None
    assert len(by_move[1809].join.candidates) >= 2
    # −61 matches no Paul move within tolerance.
    assert by_move[1469].status == "no_candidate"
    # A move only ever seen on hit has no on-block signal.
    assert by_move[1465].status == "no_signal"

    # Bryan (char_id 7): no snapshot -> needs_framedata; Xiaoyu (char_id 5): unnamed -> unresolved.
    bryan = [g for g in report.groups if g.char_id == 7]
    assert bryan and all(g.status == "needs_framedata" for g in bryan)
    xiaoyu = [g for g in report.groups if g.char_id == 5]
    assert xiaoyu and all(g.status == "unresolved_char" for g in xiaoyu)


def test_only_char_filter_restricts_to_target() -> None:
    session = load_session(SLICE)
    snapshot = load_current_framedata(ASSETS / "framedata")
    report = mine_session(session, snapshot, only_char="paul")
    assert report.groups  # non-empty
    assert all(g.char_slug == "paul" for g in report.groups)
