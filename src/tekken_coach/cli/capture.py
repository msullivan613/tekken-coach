"""Capture wiring — assets, header, and the run loop that drives the orchestrator (docs/00 §4).

The CLI command handlers ([__init__][tekken_coach.cli]) resolve settings and construct a
:class:`~tekken_coach.cli.source.CaptureSource`; this module turns those into a running pipeline:
load the C1/C2 assets, build the :class:`~tekken_coach.schemas.SessionHeader`, wire a
:class:`~tekken_coach.cli.orchestrate.CaptureOrchestrator`, pump the poll stream, and finalize the
header's :class:`~tekken_coach.schemas.MatchSummary` list on close (docs/03 §5).

:func:`run_capture` takes an *already-constructed* source, so the exact same run loop drives the
real reader and a scripted fake-reader stream in tests (the plan's test strategy). It reads only
:class:`~tekken_coach.cli.source.Poll`s — nothing here knows the game exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tekken_coach.cli.config import Settings
from tekken_coach.cli.orchestrate import (
    CaptureError,
    CaptureOrchestrator,
    CharResolver,
    Labeler,
    policy_for,
)
from tekken_coach.cli.render import Renderer
from tekken_coach.cli.source import CaptureSource
from tekken_coach.coach import coach_session
from tekken_coach.framedata.loader import (
    DEFAULT_FRAMEDATA_DIR,
    DEFAULT_MOVEMAP_DIR,
    load_current_framedata,
    load_move_maps,
)
from tekken_coach.framedata.models import CharMoveMap, FrameDataSnapshot
from tekken_coach.framedata.punishers import (
    DEFAULT_PUNISHERS_DIR,
    PunisherProfiles,
    load_punisher_profiles,
)
from tekken_coach.framedata.xref import label_interaction
from tekken_coach.schemas import Interaction, LabeledInteraction, SessionHeader
from tekken_coach.session.store import SCHEMA_VERSION, SessionWriter


@dataclass(frozen=True)
class Assets:
    """The loaded C1/C2 assets a capture run cross-references against (docs/05)."""

    move_maps: dict[str, CharMoveMap]
    framedata: FrameDataSnapshot
    punishers: PunisherProfiles

    @property
    def framedata_snapshot(self) -> str:
        """The snapshot date stamped on the header, tying the log to the frame-data it used."""
        return self.framedata.manifest.snapshot_date

    def labeler(self) -> Labeler:
        """A pure ``Interaction -> LabeledInteraction`` bound to these assets (docs/05 §4)."""

        def label(interaction: Interaction) -> LabeledInteraction:
            return label_interaction(interaction, self.move_maps, self.framedata, self.punishers)

        return label

    def char_resolver(self) -> CharResolver:
        """A ``char_id -> name`` resolver from the move maps; ``char:<id>`` on a miss (docs/05)."""
        by_id = {m.char_id: m.char_name for m in self.move_maps.values() if m.char_id is not None}

        def resolve(char_id: int) -> str:
            return by_id.get(char_id, f"char:{char_id}")

        return resolve


def load_assets(
    *,
    movemap_dir: str | Path = DEFAULT_MOVEMAP_DIR,
    framedata_dir: str | Path = DEFAULT_FRAMEDATA_DIR,
    punishers_dir: str | Path = DEFAULT_PUNISHERS_DIR,
) -> Assets:
    """Load the move maps, current frame-data snapshot, and punisher profiles (docs/05)."""
    return Assets(
        move_maps=load_move_maps(movemap_dir),
        framedata=load_current_framedata(framedata_dir),
        punishers=load_punisher_profiles(punishers_dir),
    )


def _require_user_identity(settings: Settings) -> None:
    """A capture that does not know which player is the user cannot be coached (docs/01 §5).

    Checked *before* attaching to the game so a bare ``tekken-coach live`` fails fast on a
    misconfiguration without ever touching the process.
    """
    if settings.user_player is None:
        raise CaptureError("capture needs the user's side: pass --user p1|p2 (or set in config).")
    if settings.char is None:
        raise CaptureError("capture needs the user's character: pass --char <name> (docs/01 §5).")


def build_header(
    settings: Settings, *, game_version: str, framedata_snapshot: str
) -> SessionHeader:
    """Build the session header (line 1 of the log) from resolved settings (docs/03 §5)."""
    _require_user_identity(settings)
    assert settings.user_player is not None and settings.char is not None  # _require checked
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return SessionHeader(
        schema_version=SCHEMA_VERSION,
        created_at=created_at,
        capture_mode=settings.mode,
        game_version=game_version,
        framedata_snapshot=framedata_snapshot,
        user_player=settings.user_player,
        user_char=settings.char,
        matches=[],
    )


def run_capture(
    *,
    settings: Settings,
    source: CaptureSource,
    assets: Assets,
    renderer: Renderer,
) -> Path:
    """Run one capture session to completion and return the written log path (docs/00 §4).

    Wires the mode policy, session writer, and orchestrator, then pumps ``source.polls()`` until the
    stream ends (or Ctrl-C). Coaching fires on the mode's cadence (live: per match; clean: once at
    the batch end) through the orchestrator's reporter. On close, the header's ``matches`` are
    finalized in place (docs/03 §5). Never renders mid-match (docs/01 §3.2) — the orchestrator only
    calls the reporter outside a recording unit.
    """
    _require_user_identity(settings)  # fail before attach on a missing --user/--char
    source.attach()
    header = build_header(
        settings, game_version=source.game_version, framedata_snapshot=assets.framedata_snapshot
    )
    assert settings.user_player is not None and settings.char is not None  # checked above
    writer = SessionWriter(settings.out, header)

    # The reporter reads live counts from the orchestrator, which does not exist yet — hold it in a
    # one-slot box so the closure resolves it at call time (the orchestrator only reports outside a
    # recording unit, so it is always populated by then).
    box: list[CaptureOrchestrator] = []

    def report() -> None:
        orch = box[0]
        if settings.coach == "api":
            renderer.coach_result(coach_session(settings.out))
        else:
            renderer.capture_handoff(
                settings.out, len(writer.header.matches), orch.interaction_count
            )

    orch = CaptureOrchestrator(
        policy=policy_for(settings.mode),
        writer=writer,
        labeler=assets.labeler(),
        char_resolver=assets.char_resolver(),
        user_player=settings.user_player,
        user_char=settings.char,
        reporter=report,
    )
    box.append(orch)

    try:
        for poll in source.polls():
            orch.process(poll)
    except KeyboardInterrupt:
        pass  # a live session is ended by Ctrl-C; fall through to a clean finalize
    finally:
        orch.finish()  # closes any open unit and fires the mode's end-of-session coaching
        writer.close()
        source.close()

    if orch.online_refused:
        renderer.notice(
            f"clean mode refused {orch.online_refused} online-match frames (docs/01 §4.3)."
        )
    _finalize_header(settings.out, writer.header)
    return settings.out


def _finalize_header(path: Path, header: SessionHeader) -> None:
    """Rewrite line 1 of the log with the finalized header (its ``matches`` now filled, docs/03 §5).

    The writer stamps the header on open for crash-safety (a crashed session still has a readable
    header); the per-match summaries are only known at close, so this upgrades line 1 in place
    without disturbing the append-only body.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    lines[0] = header.model_dump_json()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
