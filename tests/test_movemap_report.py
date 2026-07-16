"""``map-moves --report`` tests (brief #8 Layer 5).

Synthetic movemap + framedata → the joined ``move_id -> notation -> (startup, on_block, hit_level,
name)`` line values, the confidence tags (anchor / unique / tie-broken / broken), and the
broken-entry flag for a movemap entry whose ``framedata_key`` is absent from the snapshot. A session
log feeds the optional per-entry sample count.
"""

from __future__ import annotations

from tekken_coach.framedata.anchors import Anchors
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    FrameDataSnapshot,
    MoveMapEntry,
    SnapshotManifest,
)
from tekken_coach.framedata.movemap_report import (
    CONFIDENCE_ANCHOR,
    CONFIDENCE_BROKEN,
    CONFIDENCE_TIE,
    CONFIDENCE_UNIQUE,
    build_report,
    format_report,
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
    MoveProperty,
    Outcome,
    SessionHeader,
    Wall,
)
from tekken_coach.session.store import LoadedSession


def _move(
    key: str, *, on_block: int | None, startup: int | None, name: str | None = None
) -> FrameDataMove:
    return FrameDataMove(
        key=key, on_block=on_block, startup=startup, name=name, hit_level=MoveProperty.mid
    )


def _snapshot(slug: str, *moves: FrameDataMove) -> FrameDataSnapshot:
    char = CharFrameData(char_slug=slug, char_name=slug.title(), moves={m.key: m for m in moves})
    manifest = SnapshotManifest(
        source_repo="pbruvoll/tekkendocs",
        source_commit="deadbeef",
        source_path_template="{slug}.csv",
        fetched_at="2026-07-15T00:00:00Z",
        snapshot_date="2026-07-15",
    )
    return FrameDataSnapshot(manifest=manifest, characters={slug: char})


def _movemap(char_name: str, moves: dict[int, str]) -> dict[str, CharMoveMap]:
    entries = {
        str(mid): MoveMapEntry(notation=key, framedata_key=key) for mid, key in moves.items()
    }
    return {char_name: CharMoveMap(char_name=char_name, game_version="2.01.01", moves=entries)}


# A snapshot where df+2 is unique at -12, b+4 ties with f+3 at -9, and d+2 is unique at -5.
def _kazuya_snapshot() -> FrameDataSnapshot:
    return _snapshot(
        "kazuya",
        _move("df+2", on_block=-12, startup=14, name="Abolishing Fist"),
        _move("b+4", on_block=-9, startup=20, name="Left Splits Kick"),
        _move("f+3", on_block=-9, startup=16),  # ties b+4 on on_block
        _move("d+2", on_block=-5, startup=13, name="Tsunami Kick"),
    )


def _labeled(*, move_id: int, adv: int | None) -> LabeledInteraction:
    return LabeledInteraction(
        id=f"m1-r1-i{move_id}",
        match_id="M#1",
        round=1,
        start_frame=100,
        end_frame=200,
        attacker=0,
        defender=1,
        attacker_move_id=move_id,
        attacker_char_id=12,
        defender_char_id=0,
        context=InteractionContext(
            distance=1.0,
            attacker_heat=False,
            defender_heat=False,
            attacker_pressure=False,
            wall=Wall.none,
            defender_health_frac=1.0,
        ),
        defender_reaction=DefenderReaction.blocked,
        observed_advantage=adv,
        outcome=Outcome.neutral,
        follow_up=FollowUp(move_id=None, result=FollowUpResult.none, reaction_frames=None),
        attacker_move_name=f"move_id:{move_id}",
        attacker_char_name="char_id:12",
        defender_char_name="char_id:0",
        labels=Labels(frame_data_matched=False, in_string=False, is_knowledge_check=False),
    )


def _session(*interactions: LabeledInteraction) -> LoadedSession:
    header = SessionHeader(
        schema_version="1.2.0",
        created_at="2026-07-15T00:00:00Z",
        capture_mode=CaptureMode.live,
        game_version="5.02.01",
        framedata_snapshot="2026-07-07",
        user_player=0,
        user_char="kazuya",
        matches=[MatchSummary(match_id="M#1", opponent_char="paul", result="win", rounds=3)],
    )
    return LoadedSession(header=header, interactions=list(interactions))


# --- joined line values + tags ------------------------------------------------


def test_joined_line_values_and_anchor_tag() -> None:
    """The anchored 2145 -> df+2 joins to its snapshot fields and tags as ``anchor``."""
    move_maps = _movemap("Kazuya", {2145: "df+2"})
    anchors = Anchors.model_validate({"kazuya": {"2145": "df+2"}})
    report = build_report(move_maps, _kazuya_snapshot(), anchors)

    [char] = report.chars
    [entry] = char.entries
    assert entry.move_id == 2145
    assert entry.notation == "df+2"
    assert (entry.startup, entry.on_block, entry.hit_level, entry.name) == (
        14,
        -12,
        "mid",
        "Abolishing Fist",
    )
    assert entry.is_anchor and not entry.anchor_conflict
    assert entry.confidence == CONFIDENCE_ANCHOR


def test_unique_vs_tie_broken_tags() -> None:
    """A move alone at its on_block tags ``unique``; one sharing an on_block tags ``tie-broken``."""
    move_maps = _movemap("Kazuya", {300: "d+2", 200: "b+4"})
    report = build_report(move_maps, _kazuya_snapshot(), Anchors.model_validate({}))
    by_id = {e.move_id: e for e in report.chars[0].entries}

    assert by_id[300].on_block_unique is True
    assert by_id[300].rivals == 0
    assert by_id[300].confidence == CONFIDENCE_UNIQUE

    assert by_id[200].on_block_unique is False
    assert by_id[200].rivals == 1  # f+3 shares -9
    assert by_id[200].confidence == CONFIDENCE_TIE


def test_broken_entry_flags_dangling_framedata_key() -> None:
    """A movemap entry whose framedata_key is absent from the snapshot is flagged ``broken``."""
    move_maps = _movemap("Kazuya", {999: "no-such-move"})
    report = build_report(move_maps, _kazuya_snapshot(), Anchors.model_validate({}))

    [entry] = report.chars[0].entries
    assert entry.move is None
    assert entry.on_block is None and entry.startup is None
    assert entry.confidence == CONFIDENCE_BROKEN
    assert report.broken == [entry]


def test_anchor_conflict_is_flagged() -> None:
    """A map that binds an anchored id to a different key surfaces as an ``anchor!`` conflict."""
    move_maps = _movemap("Kazuya", {2145: "b+4"})  # anchor says df+2
    anchors = Anchors.model_validate({"kazuya": {"2145": "df+2"}})
    report = build_report(move_maps, _kazuya_snapshot(), anchors)

    [entry] = report.chars[0].entries
    assert entry.is_anchor and entry.anchor_conflict
    assert entry.confidence == "anchor!"


# --- sample counts from a log -------------------------------------------------


def test_sample_count_from_session_log() -> None:
    """When a log is passed, an entry shows how many blocked samples backed it."""
    move_maps = _movemap("Kazuya", {2145: "df+2"})
    anchors = Anchors.model_validate({"kazuya": {"2145": "df+2"}})
    session = _session(
        _labeled(move_id=2145, adv=-12),
        _labeled(move_id=2145, adv=-12),
        _labeled(move_id=2145, adv=-12),
    )
    report = build_report(move_maps, _kazuya_snapshot(), anchors, session=session)
    assert report.chars[0].entries[0].blocked_samples == 3


def test_no_session_means_no_sample_count() -> None:
    move_maps = _movemap("Kazuya", {2145: "df+2"})
    report = build_report(move_maps, _kazuya_snapshot(), Anchors.model_validate({}))
    assert report.chars[0].entries[0].blocked_samples is None


# --- filtering + rendering ----------------------------------------------------


def test_only_char_filter() -> None:
    move_maps = {
        **_movemap("Kazuya", {2145: "df+2"}),
        **_movemap("Paul", {7777: "1,2,3"}),
    }
    snapshot = _kazuya_snapshot()
    report = build_report(move_maps, snapshot, Anchors.model_validate({}), only_char="kazuya")
    assert [c.char_slug for c in report.chars] == ["kazuya"]


def test_format_report_renders_tag_and_join() -> None:
    move_maps = _movemap("Kazuya", {2145: "df+2"})
    anchors = Anchors.model_validate({"kazuya": {"2145": "df+2"}})
    lines = format_report(build_report(move_maps, _kazuya_snapshot(), anchors))
    text = "\n".join(lines)
    assert "2145 -> df+2 -> (i14, -12, mid, Abolishing Fist)  [anchor]" in text
    assert "anchor 2145 -> df+2: ok (map agrees)" in text
