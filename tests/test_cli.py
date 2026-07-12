"""C6 CLI / orchestration tests (docs/07, docs/01).

The whole capture pipeline is driven by a **fake reader** — a :class:`ScriptedCaptureSource`
replaying a hand-built ``(FrameRecord, StateSignal)`` stream — so the live/clean lifecycles, the
triggers, the round-end flush, and the coaching cadence are exercised end-to-end with no game (the
plan's test strategy). ``coach <log>`` is tested against the committed sample log. Real-game
live/clean bring-up is blocked on the deferred round-gating and is out of scope here.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from tekken_coach.cli import capture as capture_mod
from tekken_coach.cli import main
from tekken_coach.cli.config import Settings, resolve_settings
from tekken_coach.cli.orchestrate import (
    CaptureError,
    CaptureOrchestrator,
    CharacterMismatchError,
    clean_policy,
    live_policy,
)
from tekken_coach.cli.render import SKILL_HANDOFF, Renderer
from tekken_coach.cli.source import Poll, ScriptedCaptureSource
from tekken_coach.coach import BACKEND_API, BACKEND_SKILL_FALLBACK, CoachResult
from tekken_coach.reader.state import SignalKind, StateSignal
from tekken_coach.schemas import CaptureMode, FrameRecord, MatchState
from tekken_coach.session.store import SessionWriter, load_session
from tests.fixtures.coach import builder
from tests.fixtures.segment import streams

KAZUYA = "Kazuya"  # P1 in the streams (char_id 12, resolves via the committed movemap)


# ---------------------------------------------------------------------------
# Poll-stream helpers (the fake reader's script)
# ---------------------------------------------------------------------------


def _live(frame: FrameRecord) -> Poll:
    """A live-match poll carrying the frame's own phase (docs/01 §3.1)."""
    return Poll(frame=frame, signal=StateSignal(SignalKind.live_match, False, frame.match_state))


def _replay(frame: FrameRecord) -> Poll:
    """An offline replay-playback poll — the only clean-mode buffering state (docs/01 §4.3)."""
    return Poll(
        frame=frame, signal=StateSignal(SignalKind.replay_playback, False, frame.match_state)
    )


def _idle(frame: FrameRecord) -> Poll:
    """A menu/idle poll — closes any open recording unit (not capturing in either mode)."""
    return Poll(frame=frame, signal=StateSignal(SignalKind.idle, False, MatchState.menu))


def _online(frame: FrameRecord) -> Poll:
    """An online-match poll — clean mode must refuse it (docs/01 §4.3 defense-in-depth)."""
    return Poll(frame=frame, signal=StateSignal(SignalKind.live_match, True, frame.match_state))


def _one_match() -> list[FrameRecord]:
    """A stream that yields exactly one blocked/no_punish interaction (docs/04 §3)."""
    return streams.blocked_no_punish()


def _assets() -> capture_mod.Assets:
    return capture_mod.load_assets()


def _settings(out: Path, *, mode: str = "clean", coach: str = "skill") -> Settings:
    return resolve_settings(mode=mode, coach=coach, user="p1", char=KAZUYA, out=str(out), config={})


# ---------------------------------------------------------------------------
# Config precedence (docs/07 §1.2)
# ---------------------------------------------------------------------------


def test_defaults_are_clean_and_skill() -> None:
    s = resolve_settings(mode=None, coach=None, user=None, char=None, out=None, config={})
    assert s.mode is CaptureMode.clean
    assert s.coach == "skill"


def test_flag_overrides_config_overrides_default() -> None:
    config = {"mode": "live", "coach": "api", "user": "p2", "char": "Paul"}
    # No flags → config wins over the built-in defaults.
    s = resolve_settings(mode=None, coach=None, user=None, char=None, out=None, config=config)
    assert (s.mode, s.coach, s.user_player, s.char) == (CaptureMode.live, "api", 1, "Paul")
    # A flag beats config.
    s2 = resolve_settings(
        mode="clean", coach="skill", user="p1", char="Kazuya", out=None, config=config
    )
    assert (s2.mode, s2.coach, s2.user_player, s2.char) == (CaptureMode.clean, "skill", 0, "Kazuya")


def test_user_side_mapping_and_validation() -> None:
    assert (
        resolve_settings(
            mode=None, coach=None, user="p2", char=None, out=None, config={}
        ).user_player
        == 1
    )
    with pytest.raises(ValueError, match="user side"):
        resolve_settings(mode=None, coach=None, user="p3", char=None, out=None, config={})


def test_invalid_mode_and_coach_raise() -> None:
    with pytest.raises(ValueError, match="invalid mode"):
        resolve_settings(mode="bogus", coach=None, user=None, char=None, out=None, config={})
    with pytest.raises(ValueError, match="coach backend"):
        resolve_settings(mode=None, coach="bogus", user=None, char=None, out=None, config={})


# ---------------------------------------------------------------------------
# Renderer (docs/07 §3) — TTY-aware, degrades to plain ASCII
# ---------------------------------------------------------------------------


def test_skill_handoff_is_plain_when_not_a_tty() -> None:
    buf = io.StringIO()
    Renderer(buf, color=False).capture_handoff(Path("sessions/x.jsonl"), 3, 128)
    out = buf.getvalue()
    assert "\033[" not in out  # no ANSI when degraded
    assert "Session recorded: sessions/x.jsonl" in out
    assert "(3 matches, 128 interactions)" in out
    assert SKILL_HANDOFF in out


def test_skill_handoff_uses_color_on_a_tty() -> None:
    buf = io.StringIO()
    Renderer(buf, color=True).capture_handoff(Path("s.jsonl"), 1, 1)
    assert "\033[" in buf.getvalue()  # ANSI present when color is on
    assert "(1 match, 1 interactions)" in buf.getvalue()  # singular "match"


def test_coach_result_renders_report_and_fallback() -> None:
    buf = io.StringIO()
    Renderer(buf, color=False).coach_result(CoachResult(BACKEND_API, "THE REPORT", "ok"))
    assert "THE REPORT" in buf.getvalue()

    buf2 = io.StringIO()
    Renderer(buf2, color=False).coach_result(
        CoachResult(BACKEND_SKILL_FALLBACK, None, "no credential; use the skill")
    )
    assert "no credential; use the skill" in buf2.getvalue()


# ---------------------------------------------------------------------------
# Clean capture (docs/01 §4): buffer replay playback, coach once at batch end
# ---------------------------------------------------------------------------


def test_clean_capture_writes_log_and_coaches_once(tmp_path: Path) -> None:
    out = tmp_path / "clean.jsonl"
    frames = _one_match()
    script = [_replay(f) for f in frames] + [_idle(frames[-1])]
    buf = io.StringIO()

    capture_mod.run_capture(
        settings=_settings(out, mode="clean"),
        source=ScriptedCaptureSource(script),
        assets=_assets(),
        renderer=Renderer(buf, color=False),
    )

    session = load_session(out)
    assert session.header.capture_mode is CaptureMode.clean
    assert len(session.interactions) >= 1
    assert len(session.header.matches) == 1  # one replay = one match summary (finalized on close)
    # Clean coaches exactly once, at the batch end → a single skill hand-off line.
    assert buf.getvalue().count(SKILL_HANDOFF) == 1


def test_clean_capture_refuses_online_frames(tmp_path: Path) -> None:
    out = tmp_path / "clean.jsonl"
    frames = _one_match()
    # Two online frames interleaved: clean mode must refuse (not buffer) them (docs/01 §4.3).
    script = [
        _online(frames[0]),
        *(_replay(f) for f in frames),
        _online(frames[0]),
        _idle(frames[-1]),
    ]
    orch = _run_orch(clean_policy(), out, script)
    assert orch.online_refused == 2


# ---------------------------------------------------------------------------
# Live capture (docs/01 §3): record per match, coach at each match end
# ---------------------------------------------------------------------------


def test_live_capture_coaches_per_match(tmp_path: Path) -> None:
    out = tmp_path / "live.jsonl"
    m1, m2 = _one_match(), _one_match()
    script = (
        [_live(f) for f in m1]
        + [_idle(m1[-1])]  # match 1 over → coach in the downtime
        + [_live(f) for f in m2]
        + [_idle(m2[-1])]  # match 2 over → coach again
    )
    buf = io.StringIO()
    capture_mod.run_capture(
        settings=_settings(out, mode="live"),
        source=ScriptedCaptureSource(script),
        assets=_assets(),
        renderer=Renderer(buf, color=False),
    )
    session = load_session(out)
    assert session.header.capture_mode is CaptureMode.live
    assert len(session.header.matches) == 2
    # Live coaches once per match → two hand-off lines.
    assert buf.getvalue().count(SKILL_HANDOFF) == 2


# ---------------------------------------------------------------------------
# Hard invariants (acceptance criteria)
# ---------------------------------------------------------------------------


def _run_orch(policy: Any, out: Path, script: list[Poll]) -> CaptureOrchestrator:
    """Drive a raw :class:`CaptureOrchestrator` (skipping the CLI) with a spy reporter that asserts
    the no-mid-match invariant on every call. Returns the orchestrator for further assertions."""
    assets = _assets()
    header = builder.build_header()
    header.matches.clear()
    header.capture_mode = policy.mode
    writer = SessionWriter(out, header)
    box: list[CaptureOrchestrator] = []
    calls: list[bool] = []

    def reporter() -> None:
        # The invariant: output is only ever produced outside a recording unit (docs/01 §3.2).
        calls.append(box[0].is_recording)

    orch = CaptureOrchestrator(
        policy=policy,
        writer=writer,
        labeler=assets.labeler(),
        char_resolver=assets.char_resolver(),
        user_player=0,
        user_char=KAZUYA,
        reporter=reporter,
    )
    box.append(orch)
    for poll in script:
        orch.process(poll)
    orch.finish()
    writer.close()
    orch.report_calls = calls  # type: ignore[attr-defined]  # stash for the test
    return orch


def test_no_output_is_emitted_mid_match(tmp_path: Path) -> None:
    """The hard invariant (docs/01 §3.2): the reporter never fires while recording."""
    m1, m2 = _one_match(), _one_match()
    script = [_live(f) for f in m1] + [_idle(m1[-1])] + [_live(f) for f in m2] + [_idle(m2[-1])]
    orch = _run_orch(live_policy(), tmp_path / "live.jsonl", script)
    calls: list[bool] = orch.report_calls  # type: ignore[attr-defined]
    assert calls == [False, False]  # two matches, each reported strictly outside the unit


def test_orchestrator_never_branches_on_capture_mode() -> None:
    """Mode-agnostic below the trigger (docs/01 §5): the shared driver names no ``CaptureMode``."""
    import inspect

    src = inspect.getsource(CaptureOrchestrator)
    assert "CaptureMode" not in src


def test_character_mismatch_is_a_hard_error(tmp_path: Path) -> None:
    """A configured-vs-observed character mismatch aborts capture (docs/01 §5)."""
    out = tmp_path / "x.jsonl"
    frames = _one_match()  # P1 is Kazuya (char_id 12)
    script = [_replay(f) for f in frames]
    with pytest.raises(CharacterMismatchError):
        # Claim the user is Paul while P1 reads as Kazuya → inverted coaching, refused.
        capture_mod.run_capture(
            settings=resolve_settings(
                mode="clean", coach="skill", user="p1", char="Paul", out=str(out), config={}
            ),
            source=ScriptedCaptureSource(script),
            assets=_assets(),
            renderer=Renderer(io.StringIO(), color=False),
        )


def test_missing_user_identity_fails_before_attach(tmp_path: Path) -> None:
    """A bare capture with no --user/--char is a hard error, raised before touching the game."""
    source = ScriptedCaptureSource([])
    with pytest.raises(CaptureError):
        capture_mod.run_capture(
            settings=resolve_settings(
                mode="clean",
                coach="skill",
                user=None,
                char=None,
                out=str(tmp_path / "x"),
                config={},
            ),
            source=source,
            assets=_assets(),
            renderer=Renderer(io.StringIO(), color=False),
        )
    assert source.attached is False  # never attached


# ---------------------------------------------------------------------------
# Round-end flush (docs/00 §4)
# ---------------------------------------------------------------------------


def test_round_end_flushes_before_close(tmp_path: Path) -> None:
    """Interactions hit disk on the round_over flush, before the session closes (docs/00 §4)."""
    out = tmp_path / "flush.jsonl"
    assets = _assets()
    header = builder.build_header()
    header.matches.clear()
    writer = SessionWriter(out, header)
    orch = CaptureOrchestrator(
        policy=clean_policy(),
        writer=writer,
        labeler=assets.labeler(),
        char_resolver=assets.char_resolver(),
        user_player=0,
        user_char=KAZUYA,
        reporter=lambda: None,
    )
    for poll in (_replay(f) for f in _one_match()):
        orch.process(poll)
    # The round_over frame triggered a flush; the interaction is on disk even though we have not
    # closed the writer or finished the session.
    on_disk = out.read_text(encoding="utf-8").splitlines()
    assert len(on_disk) >= 2  # header + at least one flushed interaction
    writer.close()


# ---------------------------------------------------------------------------
# coach <log> (docs/07 §1) — works today, no capture
# ---------------------------------------------------------------------------


def test_coach_command_skill_handoff(capsys: pytest.CaptureFixture[str]) -> None:
    builder.write_sample(builder.SAMPLE_PATH)
    rc = main(["coach", str(builder.SAMPLE_PATH)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Session log:" in out
    assert SKILL_HANDOFF in out


def test_coach_command_api_path_mocked(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    builder.write_sample(builder.SAMPLE_PATH)

    def fake_coach(_path: Any) -> CoachResult:
        return CoachResult(BACKEND_API, "MOCKED COACHING REPORT", "ok")

    monkeypatch.setattr("tekken_coach.cli.coach_session", fake_coach)
    rc = main(["coach", str(builder.SAMPLE_PATH), "--coach", "api"])
    assert rc == 0
    assert "MOCKED COACHING REPORT" in capsys.readouterr().out


def test_coach_command_missing_log(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["coach", "does/not/exist.jsonl"])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Command surface (docs/07 §1.1) — six commands, delegates wired
# ---------------------------------------------------------------------------


def test_cli_package_adds_no_write_or_inject_primitive() -> None:
    """C6 adds no memory-write / input-injection primitive (invariant checklist; docs/02 §2)."""
    import inspect

    import tekken_coach.cli as cli_pkg
    from tests.test_reader_readonly import FORBIDDEN_TOKENS

    root = Path(inspect.getfile(cli_pkg)).parent
    files = sorted(root.rglob("*.py"))
    assert files, "no cli source files found to scan"
    offenders = {
        p.name: hits
        for p in files
        if (hits := [t for t in FORBIDDEN_TOKENS if t in p.read_text(encoding="utf-8").lower()])
    }
    assert not offenders, f"cli must stay read-only — write/inject tokens found: {offenders}"


def test_command_surface_registers_six_commands() -> None:
    import argparse

    from tekken_coach.cli import build_parser
    from tekken_coach.reader.commands import doctor_main, update_offsets_main

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == {
        "live",
        "clean",
        "coach",
        "update-offsets",
        "fetch-framedata",
        "doctor",
    }
    # The delegated commands dispatch to the reader's own mains (not re-authored here).
    assert sub.choices["doctor"].get_default("func") is doctor_main
    assert sub.choices["update-offsets"].get_default("func") is update_offsets_main
