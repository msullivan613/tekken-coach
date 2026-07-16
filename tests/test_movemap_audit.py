"""``map-moves --audit`` drift-alarm tests (brief #8 Layer 2).

A synthetic log where a mapped move_id is consistently observed far from its notation's canonical
on-block → one drift finding (with the mis-map/stale-snapshot hint from the gap magnitude); observed
within tolerance → none; a mapped-but-never-blocked id → ``unobserved``. Plus a crash/clean check
over the committed real-log slice (nothing mapped is observed there yet, so zero drift).
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.framedata.anchors import Anchors
from tekken_coach.framedata.loader import load_current_framedata, load_move_maps
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    FrameDataSnapshot,
    MoveMapEntry,
    SnapshotManifest,
)
from tekken_coach.framedata.movemap_audit import (
    KIND_CONSISTENT,
    audit_session,
    format_audit,
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
from tekken_coach.session.store import LoadedSession, load_session

REPO_ROOT = Path(__file__).parent.parent
ASSETS = REPO_ROOT / "assets"
SLICE = REPO_ROOT / "tests" / "fixtures" / "framedata" / "live-run-1-slice.jsonl"


def _snapshot() -> FrameDataSnapshot:
    moves = {
        "df+2": FrameDataMove(
            key="df+2", on_block=-12, startup=14, name="Abolishing Fist", hit_level=MoveProperty.mid
        ),
        "d+2": FrameDataMove(key="d+2", on_block=-5, startup=13, hit_level=MoveProperty.mid),
    }
    char = CharFrameData(char_slug="kazuya", char_name="Kazuya", moves=moves)
    manifest = SnapshotManifest(
        source_repo="pbruvoll/tekkendocs",
        source_commit="deadbeef",
        source_path_template="{slug}.csv",
        fetched_at="2026-07-15T00:00:00Z",
        snapshot_date="2026-07-15",
    )
    return FrameDataSnapshot(manifest=manifest, characters={"kazuya": char})


def _movemap(moves: dict[int, str]) -> dict[str, CharMoveMap]:
    entries = {
        str(mid): MoveMapEntry(notation=key, framedata_key=key) for mid, key in moves.items()
    }
    return {"Kazuya": CharMoveMap(char_name="Kazuya", game_version="2.01.01", moves=entries)}


def _labeled(*, move_id: int, adv: int | None, i: int) -> LabeledInteraction:
    return LabeledInteraction(
        id=f"m1-r1-i{i}",
        match_id="M#1",
        round=1,
        start_frame=100 + i,
        end_frame=200 + i,
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


_NO_ANCHORS = Anchors.model_validate({})


# --- drift detection ----------------------------------------------------------


def test_consistent_far_off_observation_is_one_drift_finding() -> None:
    """2145 mapped to df+2 (canonical -12) but consistently observed at -20 → one drift finding."""
    session = _session(*[_labeled(move_id=2145, adv=-20, i=i) for i in range(5)])
    report = audit_session(session, _snapshot(), _movemap({2145: "df+2"}), _NO_ANCHORS)

    assert len(report.drift) == 1
    finding = report.drift[0]
    assert finding.move_id == 2145
    assert finding.observed_on_block == -20
    assert finding.canonical_on_block == -12
    assert finding.delta == 8
    assert finding.blocked_samples == 5
    assert finding.likely is not None and "mis-map" in finding.likely  # Δ8 > 6 → different move
    assert not report.consistent


def test_small_consistent_gap_hints_stale_snapshot() -> None:
    """A small but out-of-tolerance gap (Δ4) reads as a stale snapshot, not a mis-map."""
    session = _session(*[_labeled(move_id=2145, adv=-16, i=i) for i in range(4)])
    report = audit_session(session, _snapshot(), _movemap({2145: "df+2"}), _NO_ANCHORS)
    assert report.drift[0].delta == 4
    assert report.drift[0].likely is not None and "stale snapshot" in report.drift[0].likely


def test_observation_within_tolerance_is_not_drift() -> None:
    """Observed -13 vs canonical -12 (Δ1 ≤ tol 2) is consistent — a single frame is noise."""
    session = _session(*[_labeled(move_id=2145, adv=-13, i=i) for i in range(4)])
    report = audit_session(session, _snapshot(), _movemap({2145: "df+2"}), _NO_ANCHORS)
    assert not report.drift
    assert [c.kind for c in report.consistent] == [KIND_CONSISTENT]


def test_gap_exactly_at_tolerance_is_consistent() -> None:
    """A gap of exactly the tolerance does not fire — only a strictly larger gap is drift."""
    session = _session(*[_labeled(move_id=2145, adv=-14, i=i) for i in range(3)])  # Δ2 == tol
    report = audit_session(session, _snapshot(), _movemap({2145: "df+2"}), _NO_ANCHORS)
    assert not report.drift
    assert report.consistent[0].kind == KIND_CONSISTENT


def test_mapped_but_unobserved_move_id_is_reported() -> None:
    """A mapped move never seen blocked in the log lands in ``unobserved`` — can't be checked."""
    session = _session(_labeled(move_id=2145, adv=-12, i=0))
    report = audit_session(session, _snapshot(), _movemap({2145: "df+2", 300: "d+2"}), _NO_ANCHORS)
    assert [u.move_id for u in report.unobserved] == [300]
    assert report.consistent[0].move_id == 2145


def test_drift_ranked_by_delta_then_samples() -> None:
    """Two drifting ids rank biggest-gap-first (the most-suspect binding on top)."""
    session = _session(
        *[_labeled(move_id=2145, adv=-16, i=i) for i in range(10)],  # Δ4, many samples
        *[_labeled(move_id=300, adv=-15, i=100 + i) for i in range(3)],  # Δ10, fewer samples
    )
    report = audit_session(session, _snapshot(), _movemap({2145: "df+2", 300: "d+2"}), _NO_ANCHORS)
    assert [f.move_id for f in report.drift] == [300, 2145]  # Δ10 ranks above Δ4


# --- rendering + the real-log slice -------------------------------------------


def test_format_audit_reports_drift_and_clean_state() -> None:
    drifting = _session(*[_labeled(move_id=2145, adv=-20, i=i) for i in range(5)])
    text = "\n".join(
        format_audit(audit_session(drifting, _snapshot(), _movemap({2145: "df+2"}), _NO_ANCHORS))
    )
    assert "DRIFT" in text
    assert "observed -20 vs canonical -12" in text

    clean = _session(*[_labeled(move_id=2145, adv=-12, i=i) for i in range(5)])
    clean_text = "\n".join(
        format_audit(audit_session(clean, _snapshot(), _movemap({2145: "df+2"}), _NO_ANCHORS))
    )
    assert "no drift" in clean_text


def test_audit_over_real_slice_is_clean_and_does_not_crash() -> None:
    """The committed 5.02.01 slice: nothing mapped is observed yet, so zero drift, no crash."""
    session = load_session(SLICE)
    snapshot = load_current_framedata(ASSETS / "framedata")
    move_maps = load_move_maps(ASSETS / "movemap")
    anchors = Anchors.model_validate({})

    report = audit_session(session, snapshot, move_maps, anchors)
    assert report.drift == []  # kazuya (the only mapped id) never appears in this bryan/paul slice
