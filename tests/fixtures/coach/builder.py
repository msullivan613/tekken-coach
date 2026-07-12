"""Build a realistic sample session log by running the real pipeline (C5 validation, docs/06).

The coach consumes a ``.jsonl`` of :class:`~tekken_coach.schemas.LabeledInteraction`s. Rather than
hand-authoring labels (which could drift from what the pipeline actually emits), this builder feeds
hand-built :class:`~tekken_coach.schemas.Interaction`s through the **real frame-data xref** (C2,
:func:`~tekken_coach.framedata.xref.label_interaction`) against the committed Paul/Kazuya frame-data
snapshot and punisher profiles. So every ``labels`` block — ``was_punishable``, ``correct_punish``,
``string_gap``, ``knowledge_check_ids`` — is genuine pipeline output, not a guess.

The scenario: the **user is Kazuya (P1, the defender)** learning the Paul matchup, so coaching is
about what Paul keeps getting away with. It deliberately exercises several knowledge-check patterns
at coachable recurrence plus a couple of one-offs and neutral exchanges, so the coaching layer has
something to rank, de-duplicate, and ignore (docs/06 §4.2, §5):

* ``punish_missed``          — Paul d+4 (-31 low) blocked, never ws-punished  ·  4×
* ``challenged_true_string`` — mashed inside Paul's 1,2 true string           ·  3×
* ``standing_duckable_high`` — blocked Paul df+1,1,2's high standing           ·  3×
* ``mashed_into_plus``       — pressed after Paul's plus-on-block f+1+2        ·  3×
* ``respected_fake_gap``     — stood on an interruptible gap in f+3,1          ·  1× (one-off)
* neutral                    — blocked Paul's df+2 (-8), nothing coachable     ·  2×
"""

from __future__ import annotations

from pathlib import Path

from tekken_coach.framedata.loader import load_current_framedata
from tekken_coach.framedata.models import CharMoveMap, FrameDataSnapshot, MoveMapEntry
from tekken_coach.framedata.punishers import PunisherProfiles, load_punisher_profiles
from tekken_coach.framedata.xref import label_interaction
from tekken_coach.schemas import (
    CaptureMode,
    DefenderReaction,
    FollowUp,
    FollowUpResult,
    Interaction,
    InteractionContext,
    LabeledInteraction,
    MatchSummary,
    Outcome,
    SessionHeader,
    Wall,
)
from tekken_coach.session.store import SCHEMA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
ASSETS = REPO_ROOT / "assets"

# Fixture char ids (docs/05 §4.1 gap #1). Kazuya's 12 is real; Paul's 7 is a test-only id (the
# committed Paul movemap keeps char_id null until the reader lands it), matching the C2 xref tests.
KAZUYA_ID = 12
PAUL_ID = 7

MATCH_ID = "2026-07-11T19:30:00Z#1"

# Fixture move_id -> (notation, framedata_key) for Paul, mirroring tests/test_framedata_xref.py so
# the same curated moves (the golden duckable high, the true string, the interruptible gap) resolve.
PAUL_MOVES: dict[str, tuple[str, str]] = {
    "100": ("df+1,1,2", "df+1,1,2"),  # mid->high->mid, curated duckable high
    "101": ("f+3,1", "f+3,1"),  # curated interruptible gap
    "102": ("1,2", "1,2"),  # curated true string
    "103": ("d+4", "d+4"),  # low, -31 (while-standing punishable)
    "104": ("df+2", "df+2"),  # mid, -8 (not punishable)
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


def _paul_attacks(
    *,
    iid: str,
    round_: int,
    start_frame: int,
    move_id: int,
    defender_reaction: DefenderReaction,
    outcome: Outcome,
    follow_up: FollowUp,
    defender_health_frac: float,
) -> Interaction:
    """A Paul-attacks-Kazuya interaction (user = Kazuya, the defender)."""
    return Interaction(
        id=iid,
        match_id=MATCH_ID,
        round=round_,
        start_frame=start_frame,
        end_frame=start_frame + 40,
        attacker=1,  # Paul (P2)
        defender=0,  # Kazuya (P1) — the user
        attacker_move_id=move_id,
        attacker_char_id=PAUL_ID,
        defender_char_id=KAZUYA_ID,
        context=InteractionContext(
            distance=1.2,
            attacker_heat=False,
            defender_heat=False,
            attacker_pressure=True,
            wall=Wall.none,
            defender_health_frac=defender_health_frac,
        ),
        defender_reaction=defender_reaction,
        observed_advantage=None,
        outcome=outcome,
        follow_up=follow_up,
    )


_NO_FOLLOW_UP = FollowUp(move_id=0, result=FollowUpResult.none, reaction_frames=None)


def _raw_interactions() -> list[Interaction]:
    """The hand-built exchanges, before labeling. Ids are monotonic per match (docs/03 §2)."""
    out: list[Interaction] = []
    n = 0

    def nxt(**kw: object) -> None:
        nonlocal n
        n += 1
        out.append(_paul_attacks(iid=f"m1-r{kw['round_']}-i{n:03d}", **kw))  # type: ignore[arg-type]

    # punish_missed — Paul d+4 (-31 low) blocked, no ws-punish taken (4×, across rounds).
    for round_, sf, hp in [(1, 1200, 0.92), (1, 1900, 0.74), (2, 3100, 0.61), (3, 5200, 0.40)]:
        nxt(
            round_=round_,
            start_frame=sf,
            move_id=103,
            defender_reaction=DefenderReaction.blocked,
            outcome=Outcome.no_punish,
            follow_up=_NO_FOLLOW_UP,
            defender_health_frac=hp,
        )

    # challenged_true_string — mashed inside Paul's 1,2 true string, got counter-hit (3×).
    for round_, sf, hp in [(1, 2200, 0.68), (2, 3600, 0.52), (3, 5600, 0.28)]:
        nxt(
            round_=round_,
            start_frame=sf,
            move_id=102,
            defender_reaction=DefenderReaction.counter_hit,
            outcome=Outcome.challenged_true,
            follow_up=FollowUp(
                move_id=2145, result=FollowUpResult.got_counter_hit, reaction_frames=8
            ),
            defender_health_frac=hp,
        )

    # standing_duckable_high — blocked Paul df+1,1,2's high standing, missed the duck-punish (3×).
    for round_, sf, hp in [(1, 1500, 0.85), (2, 4000, 0.47), (3, 5900, 0.19)]:
        nxt(
            round_=round_,
            start_frame=sf,
            move_id=100,
            defender_reaction=DefenderReaction.blocked,
            outcome=Outcome.no_punish,
            follow_up=_NO_FOLLOW_UP,
            defender_health_frac=hp,
        )

    # mashed_into_plus — pressed after Paul's plus-on-block f+1+2, got counter-hit (3×).
    for round_, sf, hp in [(2, 3300, 0.58), (2, 4300, 0.44), (3, 6100, 0.12)]:
        nxt(
            round_=round_,
            start_frame=sf,
            move_id=105,
            defender_reaction=DefenderReaction.blocked,
            outcome=Outcome.mashed_into_ch,
            follow_up=FollowUp(
                move_id=2145, result=FollowUpResult.got_counter_hit, reaction_frames=6
            ),
            defender_health_frac=hp,
        )

    # respected_fake_gap — stood on the interruptible gap in Paul f+3,1 (1×, a one-off to ignore).
    nxt(
        round_=2,
        start_frame=3800,
        move_id=101,
        defender_reaction=DefenderReaction.blocked,
        outcome=Outcome.respected_false,
        follow_up=_NO_FOLLOW_UP,
        defender_health_frac=0.50,
    )

    # neutral — blocked Paul df+2 (-8 mid), nothing coachable (2×).
    for round_, sf, hp in [(1, 1700, 0.80), (3, 5400, 0.33)]:
        nxt(
            round_=round_,
            start_frame=sf,
            move_id=104,
            defender_reaction=DefenderReaction.blocked,
            outcome=Outcome.neutral,
            follow_up=_NO_FOLLOW_UP,
            defender_health_frac=hp,
        )

    out.sort(key=lambda i: i.start_frame)
    return out


def build_labeled() -> list[LabeledInteraction]:
    """Run the raw exchanges through the real C2 xref to produce labeled interactions."""
    move_maps = _move_maps()
    framedata = _framedata()
    punishers = _punishers()
    return [label_interaction(i, move_maps, framedata, punishers) for i in _raw_interactions()]


def build_header() -> SessionHeader:
    return SessionHeader(
        schema_version=SCHEMA_VERSION,
        created_at="2026-07-11T19:30:00Z",
        capture_mode=CaptureMode.clean,
        game_version="2.01.01",
        framedata_snapshot="2026-07-07",
        user_player=0,
        user_char="Kazuya",
        matches=[
            MatchSummary(match_id=MATCH_ID, opponent_char="Paul", result="loss", rounds=3),
        ],
    )


def render_jsonl() -> str:
    """The full session log as ``.jsonl``: header line then one labeled interaction per line."""
    lines = [build_header().model_dump_json()]
    lines.extend(i.model_dump_json() for i in build_labeled())
    return "\n".join(lines) + "\n"


def write_sample(path: str | Path) -> Path:
    """Write the sample session log to ``path`` (creating parents), returning the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_jsonl(), encoding="utf-8")
    return p


SAMPLE_PATH = REPO_ROOT / "samples" / "sample-session.jsonl"


if __name__ == "__main__":
    written = write_sample(SAMPLE_PATH)
    print(f"wrote {written}")
