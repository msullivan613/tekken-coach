"""C5 coaching-layer tests (docs/06).

Two backends share one authored body of domain content: the Skill (``skill/``) and the API
backend (``tekken_coach.coach.api``). These tests prove the **single source of truth** (the API
prompt is assembled from the ``skill/`` files, not a Python copy), the API request shape (model,
adaptive thinking, effort, prompt-cached rubric prefix + event-log suffix), the graceful auth
fallback, and that the whole thing runs end-to-end against the committed sample log with a
**mocked** Anthropic client (no network, no ``anthropic`` install required).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tekken_coach.coach import (
    BACKEND_API,
    BACKEND_SKILL_FALLBACK,
    MODEL,
    build_rubric_and_instructions,
    coach_session,
    read_event_log_text,
)
from tekken_coach.coach.prompt import ASSETS_DIR, SKILL_DIR, SKILL_SOURCES, _strip_frontmatter
from tekken_coach.framedata.tally import build_tally
from tekken_coach.schemas import CaptureMode, SessionHeader
from tekken_coach.session.store import IncompatibleSchemaVersionError, load_session
from tests.fixtures.coach import builder

SAMPLE = builder.SAMPLE_PATH


# ---------------------------------------------------------------------------
# Mock Anthropic client (no network; no `anthropic` install needed)
# ---------------------------------------------------------------------------


class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Response:
    def __init__(self, blocks: list[_Block]) -> None:
        self.content = blocks


class _RecordingMessages:
    """Captures the kwargs of the single ``create`` call and returns a canned response."""

    def __init__(self, response: _Response) -> None:
        self._response = response
        self.captured: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> _Response:
        self.captured = kwargs
        return self._response


class _RecordingClient:
    def __init__(self, response: _Response) -> None:
        self.messages = _RecordingMessages(response)


# ---------------------------------------------------------------------------
# Single source of truth (docs/06 §3) — the acceptance criterion
# ---------------------------------------------------------------------------


def test_system_prompt_assembled_from_every_skill_source() -> None:
    """RUBRIC_AND_INSTRUCTIONS is the concatenation of the ``skill/`` files — no second copy.

    For each skill source, its on-disk (frontmatter-stripped) body must appear verbatim in the
    assembled system prompt. This is what proves the API backend is not a hand-copied rubric.
    """
    system_prompt = build_rubric_and_instructions()
    for rel in SKILL_SOURCES:
        raw = (SKILL_DIR / rel).read_text(encoding="utf-8")
        body = _strip_frontmatter(raw) if rel == "SKILL.md" else raw
        assert body.strip() in system_prompt, f"{rel} body missing from assembled system prompt"


def test_system_prompt_covers_each_named_source_by_content() -> None:
    """Distinctive content from each docs/06 §3 source is present (a coarse, readable check)."""
    system_prompt = build_rubric_and_instructions()
    # One distinctive marker per skill source (SKILL.md, rubric, output-format, reading-the-log).
    for marker in (
        "knowledge checks first",  # SKILL.md
        "frequency × round-impact × learnability",  # rubric.md
        "Top knowledge checks",  # output-format.md
        "user_player",  # reading-the-log.md
    ):
        assert marker in system_prompt


def test_system_prompt_strips_skill_frontmatter() -> None:
    """The Claude Code Skill frontmatter (name/description) is loader metadata, not instructions."""
    system_prompt = build_rubric_and_instructions()
    assert not system_prompt.lstrip().startswith("---")
    assert "description: >-" not in system_prompt


# ---------------------------------------------------------------------------
# API backend request shape (docs/06 §3)
# ---------------------------------------------------------------------------


def test_api_request_shape_and_report_extraction() -> None:
    response = _Response([_Block("thinking", ""), _Block("text", "THE COACHING REPORT")])
    client = _RecordingClient(response)
    result = coach_session(SAMPLE, client_factory=lambda: client)

    assert result.backend == BACKEND_API
    assert result.report == "THE COACHING REPORT"  # thinking block skipped, text kept

    kwargs = client.messages.captured
    assert kwargs["model"] == MODEL == "claude-opus-4-8"
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "high"}

    # Stable, prompt-cached system prefix == the assembled rubric+instructions.
    system = kwargs["system"]
    assert system[0]["text"] == build_rubric_and_instructions()
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    # Volatile user suffix == the raw event-log text.
    assert kwargs["messages"][0]["role"] == "user"
    assert kwargs["messages"][0]["content"] == SAMPLE.read_text(encoding="utf-8")


def test_multiple_text_blocks_are_joined() -> None:
    response = _Response([_Block("text", "part one\n"), _Block("text", "part two")])
    result = coach_session(SAMPLE, client_factory=lambda: _RecordingClient(response))
    assert result.report == "part one\npart two"


# ---------------------------------------------------------------------------
# Auth fallback (docs/06 §3) — never crash on missing/invalid credential
# ---------------------------------------------------------------------------


def test_fallback_when_no_credential_construction_fails() -> None:
    def factory() -> Any:
        raise RuntimeError("The api_key client option must be set")

    result = coach_session(SAMPLE, client_factory=factory)
    assert result.backend == BACKEND_SKILL_FALLBACK
    assert result.report is None
    assert "ANTHROPIC_API_KEY" in result.message
    assert "skill" in result.message.lower()


def test_fallback_on_authentication_error_from_create() -> None:
    class AuthenticationError(Exception):
        # Mimic anthropic.AuthenticationError for _is_auth_error's module+name match.
        __module__ = "anthropic"

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            raise AuthenticationError("invalid x-api-key")

    class _Client:
        messages = _Messages()

    result = coach_session(SAMPLE, client_factory=lambda: _Client())
    assert result.backend == BACKEND_SKILL_FALLBACK
    assert result.report is None


def test_non_auth_error_propagates() -> None:
    """A genuine API error is a bug, not a missing credential — it must not be swallowed."""

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            raise ValueError("boom")

    class _Client:
        messages = _Messages()

    with pytest.raises(ValueError, match="boom"):
        coach_session(SAMPLE, client_factory=lambda: _Client())


# ---------------------------------------------------------------------------
# Event-log reading + version gate (docs/03 §6)
# ---------------------------------------------------------------------------


def test_read_event_log_returns_raw_text() -> None:
    assert read_event_log_text(SAMPLE) == SAMPLE.read_text(encoding="utf-8")


def test_read_event_log_rejects_incompatible_major(tmp_path: Path) -> None:
    header = SessionHeader(
        schema_version="2.0.0",
        created_at="2026-07-11T00:00:00Z",
        capture_mode=CaptureMode.clean,
        game_version="2.01.01",
        framedata_snapshot="2026-07-07",
        user_player=0,
        user_char="Kazuya",
    )
    log = tmp_path / "future.jsonl"
    log.write_text(header.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(IncompatibleSchemaVersionError):
        read_event_log_text(log)


# ---------------------------------------------------------------------------
# The committed sample session (docs/06 deliverable #3)
# ---------------------------------------------------------------------------


def test_committed_sample_matches_builder() -> None:
    """The committed sample is exactly what the builder regenerates — no silent drift."""
    assert SAMPLE.read_text(encoding="utf-8") == builder.render_jsonl()


def test_sample_exercises_recurring_and_one_off_checks() -> None:
    """The sample has several recurring knowledge checks plus a one-off and neutral exchanges."""
    session = load_session(SAMPLE)
    assert session.header.user_char == "Kazuya"
    assert len(session.interactions) == 16

    tally = build_tally(session.interactions)
    recurring = {e.knowledge_check_id for e in tally.recurring()}
    assert {
        "punish_missed",
        "challenged_true_string",
        "standing_duckable_high",
        "mashed_into_plus",
    } <= recurring

    # A one-off must NOT clear the recurrence threshold (it's noise the coach should drop).
    fake_gap = tally.get("respected_fake_gap", "Paul", 101, "Paul vs Kazuya")
    assert fake_gap is not None
    assert fake_gap.count == 1
    assert fake_gap.is_recurring is False


def test_sample_labels_are_real_ground_truth() -> None:
    """Spot-check that the sample carries genuine resolved labels the coach will cite verbatim."""
    session = load_session(SAMPLE)
    punish_missed = next(
        i for i in session.interactions if "punish_missed" in i.labels.knowledge_check_ids
    )
    assert punish_missed.attacker_move_name == "d+4"
    assert punish_missed.labels.on_block == -31
    assert punish_missed.labels.correct_punish == "ws2"

    duckable = next(
        i for i in session.interactions if "standing_duckable_high" in i.labels.knowledge_check_ids
    )
    assert duckable.labels.duck_punish == "df+1 (i13)"


def test_end_to_end_api_path_against_sample() -> None:
    """The full API path runs against the sample log with a mocked client (no network)."""
    response = _Response([_Block("text", "1. You never punish d+4 (6x)...")])
    result = coach_session(SAMPLE, client_factory=lambda: _RecordingClient(response))
    assert result.backend == BACKEND_API
    assert result.report is not None and "d+4" in result.report


# ---------------------------------------------------------------------------
# Skill bundle structure (docs/06 §2)
# ---------------------------------------------------------------------------


def test_skill_bundle_present_and_progressive_disclosure() -> None:
    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    # Frontmatter names it and says WHEN Claude Code should load it.
    assert skill_md.lstrip().startswith("---")
    assert "name: tekken-coach" in skill_md
    assert "description:" in skill_md

    # References exist and are pointed at on demand (progressive disclosure), not inlined.
    for ref in ("rubric.md", "output-format.md", "reading-the-log.md"):
        assert (SKILL_DIR / "references" / ref).is_file()
        assert ref in skill_md  # SKILL.md references it rather than front-loading its content


def test_assets_dir_is_the_shared_repo_assets() -> None:
    """Both backends resolve the same assets/ (no committed skill/assets symlink — docs/06 §2)."""
    assert ASSETS_DIR.name == "assets"
    assert (ASSETS_DIR / "movemap").is_dir()
    assert not (SKILL_DIR / "assets").exists()  # deliberately no symlink
