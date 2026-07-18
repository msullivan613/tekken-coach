"""C2 xref tests (docs/05 §4, docs/06 §4.1, docs/03 §3).

The cross-reference is a pure function: fixture ``Interaction``s -> asserted
``LabeledInteraction``s, fully offline. These tests drive it against the **committed** Paul/Kazuya
frame-data snapshot and punisher profiles (so the golden ``df+1,1,2`` duck-the-high and the curated
string gaps are exercised against real data), supplying fixture move maps with char ids (docs/05
§4.1 gap #1: real ids arrive with the reader, C4). Every rubric trigger and all three §4.2
reconciliation branches are covered.
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.framedata.loader import load_current_framedata
from tekken_coach.framedata.models import (
    CharFrameData,
    CharMoveMap,
    FrameDataMove,
    FrameDataSnapshot,
    HeatOverride,
    MoveMapEntry,
    SnapshotManifest,
)
from tekken_coach.framedata.punishers import (
    Punisher,
    PunisherProfile,
    PunisherProfiles,
    PunisherStance,
    load_punisher_profiles,
)
from tekken_coach.framedata.xref import RECONCILE_TOLERANCE, label_interaction
from tekken_coach.schemas import (
    DefenderReaction,
    FollowUp,
    FollowUpResult,
    Interaction,
    LabeledInteraction,
    MoveProperty,
    Outcome,
    StringGap,
    StringHitRecord,
)
from tekken_coach.segment import segment_frames
from tests.factories import make_interaction
from tests.fixtures.segment import streams

REPO_ROOT = Path(__file__).parent.parent
ASSETS = REPO_ROOT / "assets"

# Fixture char ids (docs/05 §4.1 gap #1). Kazuya's 12 is real; Paul's 7 is a test-only id — the real
# id lands with the reader (C4), which is why the committed movemap keeps Paul's char_id null.
KAZUYA_ID = 12
PAUL_ID = 7

# Fixture move_ids -> Paul framedata_keys (the committed Paul movemap ships no ids; docs/05 §2.3).
PAUL_MOVES = {
    "100": ("df+1,1,2", "df+1,1,2"),  # mid->high->mid, curated duck_punish (golden duckable high)
    "101": ("f+3,1", "f+3,1"),  # curated interruptible gap
    "102": ("1,2", "1,2"),  # curated true string
    "103": ("d+4", "d+4"),  # low (-31)
    "104": ("df+2", "df+2"),  # mid (-8), not punishable
    "105": ("f+1+2", "f+1+2"),  # plus on block (+3)
}


def _move_maps() -> dict[str, CharMoveMap]:
    kazuya = CharMoveMap(
        char_id=KAZUYA_ID,
        char_name="Kazuya",
        game_version="2.01.01",
        partial=True,
        moves={"2145": MoveMapEntry(notation="df+2", framedata_key="df+2")},
    )
    paul = CharMoveMap(
        char_id=PAUL_ID,
        char_name="Paul",
        game_version="2.01.01",
        partial=True,
        moves={
            mid: MoveMapEntry(notation=notation, framedata_key=key)
            for mid, (notation, key) in PAUL_MOVES.items()
        },
    )
    return {"Kazuya": kazuya, "Paul": paul}


def _framedata() -> FrameDataSnapshot:
    return load_current_framedata(ASSETS / "framedata")


def _punishers() -> PunisherProfiles:
    return load_punisher_profiles(ASSETS / "punishers")


def _label(interaction: Interaction) -> LabeledInteraction:
    return label_interaction(interaction, _move_maps(), _framedata(), _punishers())


def _paul_attacks(move_id: int, **overrides: object) -> Interaction:
    """A Paul-attacks-Kazuya interaction, with clean defaults (no observed advantage)."""
    base = make_interaction().model_copy(
        update={
            "attacker_char_id": PAUL_ID,
            "defender_char_id": KAZUYA_ID,
            "attacker_move_id": move_id,
            "observed_advantage": None,
        }
    )
    return base.model_copy(update=overrides)


def _kazuya_attacks(move_id: int, **overrides: object) -> Interaction:
    base = make_interaction().model_copy(
        update={
            "attacker_char_id": KAZUYA_ID,
            "defender_char_id": PAUL_ID,
            "attacker_move_id": move_id,
            "observed_advantage": None,
        }
    )
    return base.model_copy(update=overrides)


# ---------------------------------------------------------------------------
# name resolution + punishability (docs/05 §4.1)
# ---------------------------------------------------------------------------


def test_resolves_names_and_punishability() -> None:
    """Kazuya df+2 (-12) blocked by Paul: resolves names, marks punishable, recommends a punish."""
    itx = _kazuya_attacks(
        2145, defender_reaction=DefenderReaction.blocked, outcome=Outcome.no_punish
    )
    labeled = label_interaction(itx, _move_maps(), _framedata(), _punishers())

    assert labeled.attacker_char_name == "Kazuya"
    assert labeled.defender_char_name == "Paul"
    assert labeled.attacker_move_name == "df+2"
    assert labeled.labels.frame_data_matched is True
    assert labeled.labels.on_block == -12
    assert labeled.labels.move_property is MoveProperty.mid
    assert labeled.labels.was_punishable is True
    # Paul's fastest standing punisher is i10 -> window slack = |−12| − 10 = 2.
    assert labeled.labels.punish_window == 2
    # Strongest option that fits a 12f window: 1,2 (i10, higher damage than b+1).
    assert labeled.labels.correct_punish == "1,2"
    assert labeled.labels.user_punished_correctly is False  # they did nothing


def test_low_uses_while_standing_punisher() -> None:
    """A blocked low is punished from crouch (while-standing), not standing (docs/05 §4.1)."""
    itx = _paul_attacks(103, defender_reaction=DefenderReaction.blocked, outcome=Outcome.neutral)
    labeled = _label(itx)
    # Paul d+4 is -31; Kazuya's ws launcher (ws2, i16) punishes it from crouch.
    assert labeled.labels.was_punishable is True
    assert labeled.labels.correct_punish == "ws2"


def test_punishability_null_when_on_block_null() -> None:
    """A matched move whose on_block is null leaves punish fields null (not a crash)."""
    snap = _snapshot(
        CharFrameData(
            char_slug="paul",
            char_name="Paul",
            moves={"x": FrameDataMove(key="x", on_block=None, hit_level=MoveProperty.mid)},
        )
    )
    maps = {"Paul": _paul_map_for({"9": ("x", "x")}), "Kazuya": _move_maps()["Kazuya"]}
    itx = _paul_attacks(9)
    labeled = label_interaction(itx, maps, snap, _punishers())
    assert labeled.labels.frame_data_matched is True
    assert labeled.labels.on_block is None
    assert labeled.labels.was_punishable is None
    assert labeled.labels.correct_punish is None


def test_missing_punisher_profile_falls_back_to_coarse_default() -> None:
    """No profile for the defender -> coarse -10 default, null correct_punish, a note (05 §4.1)."""
    empty = PunisherProfiles(profiles={})
    itx = _kazuya_attacks(
        2145, defender_reaction=DefenderReaction.blocked, outcome=Outcome.no_punish
    )
    labeled = label_interaction(itx, _move_maps(), _framedata(), empty)
    assert labeled.labels.was_punishable is True  # -12 <= -10 coarse default
    assert labeled.labels.correct_punish is None
    assert labeled.labels.punish_window is None
    assert any("no punisher profile" in n for n in labeled.notes)


# ---------------------------------------------------------------------------
# duckable high (height) vs string gap (timing) — kept distinct (docs/05 §4.1 gap #3)
# ---------------------------------------------------------------------------


def test_standing_duckable_high_golden() -> None:
    """Golden: Paul df+1,1,2 blocked standing -> flag the missed duck-punish (committed data)."""
    itx = _paul_attacks(100, defender_reaction=DefenderReaction.blocked)
    labeled = _label(itx)
    assert labeled.labels.in_string is True
    assert labeled.labels.duckable_high_hit == 2
    assert labeled.labels.duck_punish == "df+1 (i13)"
    assert "standing_duckable_high" in labeled.labels.knowledge_check_ids


def test_ducked_high_is_not_flagged() -> None:
    """If the user ducked the high (evaded), it is correct play — no flag (docs/05 §4.1)."""
    itx = _paul_attacks(100, defender_reaction=DefenderReaction.evaded)
    labeled = _label(itx)
    assert labeled.labels.duckable_high_hit is None
    assert labeled.labels.duck_punish is None
    assert "standing_duckable_high" not in labeled.labels.knowledge_check_ids


def _hit(index: int, reaction: DefenderReaction, *, crouch: bool) -> StringHitRecord:
    return StringHitRecord(hit_index=index, defender_reaction=reaction, defender_crouching=crouch)


def test_segmenter_string_to_xref_flags_duckable_high_end_to_end() -> None:
    """End-to-end: the real segmenter `string_blocked_standing` output (a mid→high→mid string
    blocked standing, three per-hit records) retargeted to Paul → xref reads the per-hit record and
    flags duckable_high_hit=2. Proves the C3b record wires through the pipeline, not just a unit."""
    (seg_itx,) = segment_frames(streams.string_blocked_standing(), match_id="e2e#1")
    # Entry move 100 == Paul df+1,1,2 in the fixture move map; retarget the char ids to Paul.
    itx = seg_itx.model_copy(update={"attacker_char_id": PAUL_ID, "defender_char_id": KAZUYA_ID})
    assert [(h.hit_index, h.defender_crouching) for h in itx.string_hits] == [
        (1, False),
        (2, False),
        (3, False),
    ]
    labeled = _label(itx)
    assert labeled.labels.duckable_high_hit == 2
    assert "standing_duckable_high" in labeled.labels.knowledge_check_ids


def test_duckable_high_from_per_hit_record_flags_blocked_standing() -> None:
    """C3b: the per-hit record drives an exact call. Paul df+1,1,2 (mid→high→mid) with hit 2 (the
    high) blocked STANDING → duckable_high_hit set, cross-referenced against per-hit hit_level."""
    itx = _paul_attacks(
        100,
        defender_reaction=DefenderReaction.blocked,
        string_hits=[
            _hit(1, DefenderReaction.blocked, crouch=False),
            _hit(2, DefenderReaction.blocked, crouch=False),
            _hit(3, DefenderReaction.blocked, crouch=False),
        ],
    )
    labeled = _label(itx)
    assert labeled.labels.duckable_high_hit == 2
    assert labeled.labels.duck_punish == "df+1 (i13)"
    assert "standing_duckable_high" in labeled.labels.knowledge_check_ids


def test_duckable_high_not_flagged_when_high_was_ducked() -> None:
    """C3b: the same string, but hit 2 (the high) was ducked (evaded, crouching) → no flag, even
    though the overall reaction is `blocked` (hit 1). The per-hit record OVERRIDES the fallback,
    which would have flagged on `defender_reaction == blocked`."""
    itx = _paul_attacks(
        100,
        defender_reaction=DefenderReaction.blocked,
        string_hits=[
            _hit(1, DefenderReaction.blocked, crouch=True),
            _hit(2, DefenderReaction.evaded, crouch=True),
        ],
    )
    labeled = _label(itx)
    assert labeled.labels.duckable_high_hit is None
    assert labeled.labels.duck_punish is None
    assert "standing_duckable_high" not in labeled.labels.knowledge_check_ids


def test_duckable_high_fallback_used_when_no_per_hit_record() -> None:
    """C3b: a pre-1.2.0 log (no string_hits) still resolves via the retained approximation —
    `defender_reaction == blocked` on a duck_punish string → flag."""
    itx = _paul_attacks(100, defender_reaction=DefenderReaction.blocked)  # string_hits defaults []
    assert itx.string_hits == []
    labeled = _label(itx)
    assert labeled.labels.duckable_high_hit == 2  # fallback path
    assert "standing_duckable_high" in labeled.labels.knowledge_check_ids


def test_string_gap_is_timing_not_height() -> None:
    """The curated interruptible gap surfaces as string_gap, independent of the duck check."""
    itx = _paul_attacks(101, defender_reaction=DefenderReaction.blocked)
    labeled = _label(itx)
    assert labeled.labels.in_string is True
    assert labeled.labels.string_gap is StringGap.interruptible
    assert labeled.labels.gap_size == 1
    assert labeled.labels.duckable_high_hit is None  # distinct from the height check


# ---------------------------------------------------------------------------
# Heat selection (docs/05 §4.1)
# ---------------------------------------------------------------------------


def test_heat_override_selected_when_attacker_in_heat() -> None:
    snap = _snapshot(
        CharFrameData(
            char_slug="paul",
            char_name="Paul",
            moves={
                "qcf+2": FrameDataMove(
                    key="qcf+2",
                    on_block=-13,
                    hit_level=MoveProperty.mid,
                    heat=HeatOverride(on_block=-5),
                )
            },
        )
    )
    maps = {"Paul": _paul_map_for({"9": ("qcf+2", "qcf+2")}), "Kazuya": _move_maps()["Kazuya"]}

    in_heat = _paul_attacks(9).model_copy(
        update={"context": _paul_attacks(9).context.model_copy(update={"attacker_heat": True})}
    )
    labeled_heat = label_interaction(in_heat, maps, snap, _punishers())
    assert labeled_heat.labels.on_block == -5  # Heat override wins

    no_heat = _paul_attacks(9).model_copy(
        update={"context": _paul_attacks(9).context.model_copy(update={"attacker_heat": False})}
    )
    labeled_plain = label_interaction(no_heat, maps, snap, _punishers())
    assert labeled_plain.labels.on_block == -13  # canonical


# ---------------------------------------------------------------------------
# observed vs canonical reconciliation — all three branches (docs/05 §4.2)
# ---------------------------------------------------------------------------


def test_reconcile_agreement_uses_canonical_no_note() -> None:
    itx = _kazuya_attacks(
        2145,
        defender_reaction=DefenderReaction.blocked,
        observed_advantage=-12,  # equals canonical
    )
    labeled = _label(itx)
    assert labeled.labels.on_block == -12
    assert not any("disagrees" in n for n in labeled.notes)


def test_reconcile_disagreement_keeps_observed_and_notes() -> None:
    itx = _kazuya_attacks(
        2145,
        defender_reaction=DefenderReaction.blocked,
        observed_advantage=-4,  # far from canonical -12
    )
    labeled = _label(itx)
    assert labeled.labels.on_block == -12  # canonical still used for the answer
    assert labeled.observed_advantage == -4  # observed preserved on the record
    assert any("disagrees" in n for n in labeled.notes)


def test_reconcile_null_observed_uses_canonical_only() -> None:
    itx = _kazuya_attacks(2145, defender_reaction=DefenderReaction.blocked, observed_advantage=None)
    labeled = _label(itx)
    assert labeled.labels.on_block == -12
    assert not any("disagrees" in n for n in labeled.notes)


def test_reconcile_tolerance_boundary() -> None:
    """A difference exactly at tolerance agrees; one frame past it disagrees."""
    at_tol = _kazuya_attacks(
        2145,
        defender_reaction=DefenderReaction.blocked,
        observed_advantage=-12 - RECONCILE_TOLERANCE,
    )
    assert not any("disagrees" in n for n in _label(at_tol).notes)
    past_tol = _kazuya_attacks(
        2145,
        defender_reaction=DefenderReaction.blocked,
        observed_advantage=-12 - RECONCILE_TOLERANCE - 1,
    )
    assert any("disagrees" in n for n in _label(past_tol).notes)


# ---------------------------------------------------------------------------
# degradation: unknown move / unknown char (docs/05 §4.1, §6)
# ---------------------------------------------------------------------------


def test_unknown_move_id_is_unlabeled_no_crash() -> None:
    itx = _kazuya_attacks(999999, defender_reaction=DefenderReaction.blocked)
    labeled = _label(itx)
    assert labeled.attacker_move_name == "move_id:999999"
    assert labeled.attacker_char_name == "Kazuya"  # char still resolved
    assert labeled.labels.frame_data_matched is False
    assert labeled.labels.on_block is None
    assert labeled.labels.move_property is None
    assert labeled.labels.is_knowledge_check is False
    assert labeled.labels.knowledge_check_ids == []


def test_unknown_char_id_is_unlabeled_no_crash() -> None:
    itx = _kazuya_attacks(2145, attacker_char_id=999, defender_reaction=DefenderReaction.blocked)
    labeled = _label(itx)
    assert labeled.attacker_char_name == "char_id:999"
    assert labeled.labels.frame_data_matched is False
    assert labeled.labels.is_knowledge_check is False


def test_none_char_id_degrades_to_unknown() -> None:
    itx = _kazuya_attacks(2145, attacker_char_id=None, defender_reaction=DefenderReaction.blocked)
    labeled = _label(itx)
    assert labeled.attacker_char_name == "unknown"
    assert labeled.labels.frame_data_matched is False


def test_determinism_same_input_same_output() -> None:
    itx = _paul_attacks(100, defender_reaction=DefenderReaction.blocked)
    a = _label(itx)
    b = _label(itx)
    assert a == b


# ---------------------------------------------------------------------------
# per-pattern triggers (docs/06 §4.1) — one fixture proving each trigger fires at the xref level
# ---------------------------------------------------------------------------


def test_trigger_punish_missed() -> None:
    itx = _kazuya_attacks(
        2145, defender_reaction=DefenderReaction.blocked, outcome=Outcome.no_punish
    )
    assert "punish_missed" in _label(itx).labels.knowledge_check_ids


def test_trigger_respected_fake_gap() -> None:
    itx = _paul_attacks(
        101,
        defender_reaction=DefenderReaction.blocked,
        outcome=Outcome.respected_false,
        follow_up=FollowUp(move_id=0, result=FollowUpResult.none, reaction_frames=None),
    )
    assert "respected_fake_gap" in _label(itx).labels.knowledge_check_ids


def test_trigger_challenged_true_string() -> None:
    itx = _paul_attacks(
        102,
        defender_reaction=DefenderReaction.counter_hit,
        outcome=Outcome.challenged_true,
        follow_up=FollowUp(move_id=5, result=FollowUpResult.got_counter_hit, reaction_frames=8),
    )
    assert "challenged_true_string" in _label(itx).labels.knowledge_check_ids


def test_trigger_standing_duckable_high() -> None:
    itx = _paul_attacks(100, defender_reaction=DefenderReaction.blocked)
    assert "standing_duckable_high" in _label(itx).labels.knowledge_check_ids


def test_trigger_ate_low() -> None:
    itx = _paul_attacks(103, defender_reaction=DefenderReaction.hit, outcome=Outcome.ate_low)
    assert "ate_low" in _label(itx).labels.knowledge_check_ids


def test_trigger_ate_mid() -> None:
    itx = _paul_attacks(104, defender_reaction=DefenderReaction.hit, outcome=Outcome.ate_mid)
    assert "ate_mid" in _label(itx).labels.knowledge_check_ids


def test_trigger_mashed_into_plus() -> None:
    itx = _paul_attacks(
        105,
        defender_reaction=DefenderReaction.blocked,
        outcome=Outcome.mashed_into_ch,
        follow_up=FollowUp(move_id=5, result=FollowUpResult.got_counter_hit, reaction_frames=6),
    )
    labeled = _label(itx)
    assert labeled.labels.on_block == 3  # plus on block
    assert "mashed_into_plus" in labeled.labels.knowledge_check_ids


def test_unmatched_move_sets_no_knowledge_check() -> None:
    """Can't judge what we can't identify: a miss never trips a rubric pattern (docs/05 §4.1)."""
    itx = _kazuya_attacks(
        424242, defender_reaction=DefenderReaction.blocked, outcome=Outcome.no_punish
    )
    assert _label(itx).labels.is_knowledge_check is False


# ---------------------------------------------------------------------------
# small builders for the in-test framedata used by the on_block-null / Heat tests
# ---------------------------------------------------------------------------


def _paul_map_for(moves: dict[str, tuple[str, str]]) -> CharMoveMap:
    return CharMoveMap(
        char_id=PAUL_ID,
        char_name="Paul",
        game_version="2.01.01",
        partial=True,
        moves={m: MoveMapEntry(notation=n, framedata_key=k) for m, (n, k) in moves.items()},
    )


def _snapshot(char_fd: CharFrameData) -> FrameDataSnapshot:
    manifest = SnapshotManifest(
        source_repo="test",
        source_commit="deadbeef",
        source_path_template="t",
        fetched_at="2026-07-07T00:00:00Z",
        snapshot_date="2026-07-07",
    )
    return FrameDataSnapshot(manifest=manifest, characters={char_fd.char_slug: char_fd})


def test_char_name_join_is_case_insensitive() -> None:
    """A lowercase-snapshot char_name still joins a capitalized move-map name (brief #17 §B).

    Regression for the raw-scrape swap: the move map says ``"Paul"`` but the fresh snapshot's
    ``char_name`` is lowercase ``"paul"``. The join must normalize case so the move resolves rather
    than degrading to ``frame_data_matched=False``.
    """
    snap = _snapshot(
        CharFrameData(
            char_slug="paul",
            char_name="paul",  # lowercase, as a fresh scrape produces
            moves={"df+2": FrameDataMove(key="df+2", on_block=-9, hit_level=MoveProperty.mid)},
        )
    )
    maps = {"Paul": _paul_map_for({"9": ("df+2", "df+2")})}  # capitalized move-map name
    labeled = label_interaction(_paul_attacks(9), maps, snap, _punishers())

    assert labeled.labels.frame_data_matched is True
    assert labeled.attacker_move_name == "df+2"
    assert labeled.labels.on_block == -9


def test_profile_helpers_are_stance_aware() -> None:
    """PunisherProfile.fastest / by_stance select by stance (docs/05 §4.1)."""
    profile = PunisherProfile(
        char_name="Test",
        punishers=[
            Punisher(startup=10, notation="1", stance=PunisherStance.standing),
            Punisher(
                startup=15, notation="ws2", stance=PunisherStance.while_standing, launcher=True
            ),
        ],
    )
    assert profile.fastest(PunisherStance.standing) is not None
    assert profile.fastest(PunisherStance.standing).startup == 10  # type: ignore[union-attr]
    assert len(profile.by_stance(PunisherStance.while_standing)) == 1


def test_punisher_profiles_get_is_case_insensitive(tmp_path: Path) -> None:
    """A profile keyed ``"Paul"`` is found by ``"paul"`` and vice versa (brief #17 §B)."""
    (tmp_path / "paul.json").write_text(
        PunisherProfile(
            char_name="Paul", punishers=[Punisher(startup=10, notation="1,2")]
        ).model_dump_json(),
        encoding="utf-8",
    )
    profiles = load_punisher_profiles(tmp_path)
    assert profiles.get("Paul") is not None
    assert profiles.get("paul") is not None
    assert profiles.get("PAUL") is not None
